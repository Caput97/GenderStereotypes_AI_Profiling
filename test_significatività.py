import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from itertools import product
from pathlib import Path
import time
from scipy.stats import t

# ----------------------------
# PARAMETERS (EDIT THESE)
# ----------------------------

N_SENTENCES = 1000

# Annotators per group
ANNOTATORS_PER_GROUP = 40

# Sentences per annotator
SENTENCES_PER_ANNOTATOR = 50

# True probabilities
P_YOUNG_MEN = 0.50
P_OLD_MEN   = 0.60

P_YOUNG_WOMEN = 0.55
P_OLD_WOMEN   = 0.65

# Random effect strengths
SENTENCE_SD = 0.7
ANNOTATOR_SD = 0.7

# Simulation settings
N_SIMULATIONS = 200
ALPHA = 0.05

# Run mode: "single" or "grid"
RUN_MODE = "grid"

# Grid settings
GRID_ANNOTATORS_PER_GROUP = [50, 60, 70, 80, 90, 100]
GRID_SENTENCES_PER_ANNOTATOR = [70, 90, 110, 130, 150]
GRID_P_YOUNG_MEN = [0.50]
GRID_P_OLD_MEN = [0.55, 0.60, 0.65]
GRID_ANNOTATOR_SD = [0.7]
GRID_SENTENCE_SD = [0.7]

# Optional reproducibility seed
BASE_SEED = 123

# Progress print frequency
SIM_PROGRESS_STEPS = 10

# ----------------------------
# NEW: STUDY GOALS
# ----------------------------

GROUPS = ["young_men", "old_men", "young_women", "old_women"]

# Key comparisons you care about.
# You can add/remove comparisons here.
# = [
#    ("old_men", "young_men"),
#    ("old_women", "young_women"),
#]

PRIMARY_COMPARISONS = [
    ("old_men", "young_men"),
    ("old_women", "young_women"),
    ("young_women", "young_men"),
    ("old_women", "old_men"),
]

# Precision target for each group:
# max allowed half-width of the confidence interval
MAX_GROUP_CI_HALF_WIDTH = 0.05

# If True, divide alpha by number of primary comparisons
# to be more conservative on multiple testing.
USE_BONFERRONI = True

# ----------------------------
# HELPER FUNCTIONS
# ----------------------------

def logit(p):
    return np.log(p / (1 - p))

def inv_logit(x):
    return 1 / (1 + np.exp(-x))

def format_seconds(seconds):
    seconds = max(0, int(round(seconds)))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"

def get_true_group_probabilities(p_young_men, p_old_men, p_young_women, p_old_women):
    return {
        "young_men": p_young_men,
        "old_men": p_old_men,
        "young_women": p_young_women,
        "old_women": p_old_women,
    }

# ----------------------------
# NEW: GROUP PRECISION
# ----------------------------

def compute_group_precision(df, alpha=0.05):
    """
    For each group:
    - compute each annotator's mean label
    - estimate group mean across annotators
    - build a t-based confidence interval on the group mean
    """
    annotator_means = (
        df.groupby(["group", "annotator"], as_index=False)["label"]
        .mean()
        .rename(columns={"label": "annotator_mean"})
    )

    group_stats = {}

    for group_name in GROUPS:
        vals = annotator_means.loc[
            annotator_means["group"] == group_name,
            "annotator_mean"
        ].to_numpy()

        n = len(vals)
        mean_hat = np.mean(vals) if n > 0 else np.nan

        if n > 1:
            sd = np.std(vals, ddof=1)
            se = sd / np.sqrt(n)
            tcrit = t.ppf(1 - alpha / 2, df=n - 1)
            half_width = tcrit * se
            ci_low = max(0.0, mean_hat - half_width)
            ci_high = min(1.0, mean_hat + half_width)
        else:
            sd = np.nan
            se = np.nan
            half_width = np.nan
            ci_low = np.nan
            ci_high = np.nan

        group_stats[group_name] = {
            "n_annotators": n,
            "mean_hat": mean_hat,
            "sd_annotator_means": sd,
            "se": se,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "ci_half_width": half_width,
            "precision_ok": (half_width <= MAX_GROUP_CI_HALF_WIDTH) if np.isfinite(half_width) else False,
        }

    return group_stats

# ----------------------------
# NEW: COMPARISON TESTS
# ----------------------------

def test_pairwise_comparison(df, target_group, reference_group, alpha):
    """
    Fit a logistic regression on only the two groups involved in the contrast.
    Returns p-value and significance flag.
    """
    df_pair = df[df["group"].isin([target_group, reference_group])].copy()

    try:
        model = smf.logit(
            f"label ~ C(group, Treatment(reference='{reference_group}'))",
            data=df_pair
        ).fit(disp=0, method="lbfgs", maxiter=200)
    except Exception:
        return {
            "p_value": 1.0,
            "significant": False,
            "fit_failed": True,
        }

    coef_name = f"C(group, Treatment(reference='{reference_group}'))[T.{target_group}]"
    pval = model.pvalues.get(coef_name, 1.0)

    return {
        "p_value": pval,
        "significant": bool(pval < alpha),
        "fit_failed": False,
    }

# ----------------------------
# SIMULATION
# ----------------------------

def run_simulation(
    annotators_per_group,
    sentences_per_annotator,
    p_young_men,
    p_old_men,
    p_young_women,
    p_old_women,
    sentence_sd,
    annotator_sd,
    n_sentences,
    n_simulations,
    alpha,
    seed=None,
    progress_label=None,
    show_progress=True,
):
    rng = np.random.default_rng(seed)
    sim_start = time.perf_counter()

    true_probs = get_true_group_probabilities(
        p_young_men, p_old_men, p_young_women, p_old_women
    )

    # Multiple-testing alpha for primary comparisons
    comparison_alpha = alpha / len(PRIMARY_COMPARISONS) if USE_BONFERRONI else alpha

    # Store simulation-level outcomes
    overall_successes = []
    precision_successes = []
    comparison_successes = []

    # Per-comparison success rates
    comparison_success_dict = {
        f"{target}_vs_{ref}": []
        for target, ref in PRIMARY_COMPARISONS
    }

    # Per-group summary metrics across simulations
    group_half_widths = {g: [] for g in GROUPS}
    group_mean_hats = {g: [] for g in GROUPS}
    group_precision_flags = {g: [] for g in GROUPS}

    n_fit_failures = 0
    progress_every = max(1, n_simulations // SIM_PROGRESS_STEPS)

    if show_progress:
        label = f"[{progress_label}] " if progress_label else ""
        print(f"{label}Starting {n_simulations} simulations...")

    for sim_idx in range(n_simulations):
        sentence_effects = rng.normal(0, sentence_sd, n_sentences)

        data = []
        annotator_id = 0

        groups = [
            ("young_men", p_young_men),
            ("old_men", p_old_men),
            ("young_women", p_young_women),
            ("old_women", p_old_women),
        ]

        for group_name, base_p in groups:
            for _ in range(annotators_per_group):
                annotator_effect = rng.normal(0, annotator_sd)

                sentences_sampled = rng.choice(
                    n_sentences,
                    sentences_per_annotator,
                    replace=False
                )

                for s in sentences_sampled:
                    log_odds = (
                        logit(base_p)
                        + sentence_effects[s]
                        + annotator_effect
                    )

                    p = inv_logit(log_odds)
                    label_val = rng.binomial(1, p)

                    data.append({
                        "label": label_val,
                        "group": group_name,
                        "sentence": s,
                        "annotator": annotator_id
                    })

                annotator_id += 1

        df = pd.DataFrame(data)

        # ----------------------------
        # OBJECTIVE 1: precision per group
        # ----------------------------
        group_stats = compute_group_precision(df, alpha=alpha)

        for g in GROUPS:
            group_half_widths[g].append(group_stats[g]["ci_half_width"])
            group_mean_hats[g].append(group_stats[g]["mean_hat"])
            group_precision_flags[g].append(group_stats[g]["precision_ok"])

        precision_success = all(group_stats[g]["precision_ok"] for g in GROUPS)
        precision_successes.append(precision_success)

        # ----------------------------
        # OBJECTIVE 2: key comparisons
        # ----------------------------
        sim_comparison_results = []
        for target, ref in PRIMARY_COMPARISONS:
            res = test_pairwise_comparison(
                df=df,
                target_group=target,
                reference_group=ref,
                alpha=comparison_alpha
            )

            if res["fit_failed"]:
                n_fit_failures += 1

            key = f"{target}_vs_{ref}"
            comparison_success_dict[key].append(res["significant"])
            sim_comparison_results.append(res["significant"])

        comparisons_success = all(sim_comparison_results)
        comparison_successes.append(comparisons_success)

        # ----------------------------
        # OVERALL success
        # ----------------------------
        overall_success = precision_success and comparisons_success
        overall_successes.append(overall_success)

        sim_done = sim_idx + 1
        if show_progress and (sim_done % progress_every == 0 or sim_done == n_simulations):
            elapsed = time.perf_counter() - sim_start
            avg_per_sim = elapsed / sim_done
            eta = avg_per_sim * (n_simulations - sim_done)
            label = f"[{progress_label}] " if progress_label else ""
            print(
                f"{label}Sim {sim_done}/{n_simulations} | "
                f"Elapsed: {format_seconds(elapsed)} | "
                f"ETA: {format_seconds(eta)}"
            )

    # Aggregate outputs
    summary = {
        "overall_success_rate": float(np.mean(overall_successes)),
        "precision_success_rate": float(np.mean(precision_successes)),
        "comparison_success_rate": float(np.mean(comparison_successes)),
        "fit_failures": int(n_fit_failures),
        "comparison_alpha_used": comparison_alpha,
    }

    for target, ref in PRIMARY_COMPARISONS:
        key = f"{target}_vs_{ref}"
        summary[f"power_{key}"] = float(np.mean(comparison_success_dict[key]))

    for g in GROUPS:
        summary[f"mean_estimate_{g}"] = float(np.nanmean(group_mean_hats[g]))
        summary[f"mean_ci_half_width_{g}"] = float(np.nanmean(group_half_widths[g]))
        summary[f"precision_rate_{g}"] = float(np.mean(group_precision_flags[g]))
        summary[f"true_p_{g}"] = true_probs[g]

    return summary

# ----------------------------
# GRID
# ----------------------------

def run_parameter_grid():
    grid_start = time.perf_counter()
    grid_rows = []

    combinations = list(product(
        GRID_ANNOTATORS_PER_GROUP,
        GRID_SENTENCES_PER_ANNOTATOR,
        GRID_P_YOUNG_MEN,
        GRID_P_OLD_MEN,
        GRID_ANNOTATOR_SD,
        GRID_SENTENCE_SD,
    ))

    n_combinations = len(combinations)
    print(f"\nStarting grid with {n_combinations} combinations...")

    for i, (
        annotators_per_group,
        sentences_per_annotator,
        p_young_men,
        p_old_men,
        annotator_sd,
        sentence_sd,
    ) in enumerate(combinations):

        combo_idx = i + 1
        combo_start = time.perf_counter()

        combo_label = (
            f"combo {combo_idx}/{n_combinations}: "
            f"ann={annotators_per_group}, sent={sentences_per_annotator}, "
            f"young={p_young_men:.2f}, old={p_old_men:.2f}, "
            f"ann_sd={annotator_sd:.2f}, sent_sd={sentence_sd:.2f}"
        )
        print(f"\nRunning {combo_label}")

        summary = run_simulation(
            annotators_per_group=annotators_per_group,
            sentences_per_annotator=sentences_per_annotator,
            p_young_men=p_young_men,
            p_old_men=p_old_men,
            p_young_women=P_YOUNG_WOMEN,
            p_old_women=P_OLD_WOMEN,
            sentence_sd=sentence_sd,
            annotator_sd=annotator_sd,
            n_sentences=N_SENTENCES,
            n_simulations=N_SIMULATIONS,
            alpha=ALPHA,
            seed=BASE_SEED + i,
            progress_label=f"Combo {combo_idx}/{n_combinations}",
            show_progress=True,
        )

        combo_elapsed = time.perf_counter() - combo_start
        grid_elapsed = time.perf_counter() - grid_start
        avg_per_combo = grid_elapsed / combo_idx
        grid_eta = avg_per_combo * (n_combinations - combo_idx)

        print(
            f"Finished combo {combo_idx}/{n_combinations} | "
            f"Overall success: {summary['overall_success_rate']:.3f} | "
            f"Precision success: {summary['precision_success_rate']:.3f} | "
            f"Comparison success: {summary['comparison_success_rate']:.3f} | "
            f"Combo time: {format_seconds(combo_elapsed)} | "
            f"Grid ETA: {format_seconds(grid_eta)}"
        )

        row = {
            "annotators_per_group": annotators_per_group,
            "sentences_per_annotator": sentences_per_annotator,
            "p_young_men": p_young_men,
            "p_old_men": p_old_men,
            "annotator_sd": annotator_sd,
            "sentence_sd": sentence_sd,
            **summary
        }
        grid_rows.append(row)

    results_df = pd.DataFrame(grid_rows)

    sort_cols = [
        "annotators_per_group",
        "sentences_per_annotator",
        "p_young_men",
        "p_old_men",
        "annotator_sd",
        "sentence_sd",
    ]
    results_df = results_df.sort_values(sort_cols).reset_index(drop=True)

    total_elapsed = time.perf_counter() - grid_start
    print(f"\nGrid completed in {format_seconds(total_elapsed)}")

    return results_df

# ----------------------------
# RUN
# ----------------------------

if __name__ == "__main__":
    all_start = time.perf_counter()

    if RUN_MODE == "single":
        summary = run_simulation(
            annotators_per_group=ANNOTATORS_PER_GROUP,
            sentences_per_annotator=SENTENCES_PER_ANNOTATOR,
            p_young_men=P_YOUNG_MEN,
            p_old_men=P_OLD_MEN,
            p_young_women=P_YOUNG_WOMEN,
            p_old_women=P_OLD_WOMEN,
            sentence_sd=SENTENCE_SD,
            annotator_sd=ANNOTATOR_SD,
            n_sentences=N_SENTENCES,
            n_simulations=N_SIMULATIONS,
            alpha=ALPHA,
            seed=BASE_SEED,
            progress_label="Single run",
            show_progress=True,
        )

        print("\nSingle-run summary:\n")
        for k, v in summary.items():
            if isinstance(v, float):
                print(f"{k}: {v:.4f}")
            else:
                print(f"{k}: {v}")

    else:
        results_df = run_parameter_grid()

        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 250)

        print("\nGrid results:\n")
        print(results_df.to_string(index=False))

        out_path = Path.cwd() / "power_grid_results.csv"
        results_df.to_csv(out_path, index=False)
        print(f"\nSaved CSV: {out_path}")

    all_elapsed = time.perf_counter() - all_start
    print(f"\nTotal runtime: {format_seconds(all_elapsed)}")