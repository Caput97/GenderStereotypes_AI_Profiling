#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Analyse all LLM files with a hierarchical annotator+item bootstrap."""

import glob
import json
import os
import warnings
from itertools import combinations

import numpy as np
from scipy import stats

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
ANNOTATIONS_FILE = "/home/dtesta/GSI-detect_LLM_Socioprofiling/Forms/annotation_results_noGSno.json"
LLM_FOLDER       = "/home/dtesta/GenderStereotypes_AI_Profiling/LLMs_GSscores/124Matched"
OUTPUT_FILE      = "/home/dtesta/GenderStereotypes_AI_Profiling/llm_profiling_bootstrap.txt"
OUTPUT_JSON = "/home/dtesta/GenderStereotypes_AI_Profiling/llm_profiling_bootstrap.json"

N_BOOT = 10_000
ALPHA = 0.05
SEED = 42

GROUP_DEFS = {
    "Donne under 35": ("F", "under35"),
    "Donne over 35": ("F", "over35"),       # age >= 35
    "Uomini under 35": ("M", "under35"),
    "Uomini over 35": ("M", "over35"),
}

SEP = "=" * 100
SEP2 = "-" * 100

# -----------------------------------------------------------------------------
# LOADING AND PREPARATION
# -----------------------------------------------------------------------------
def get_group(data, gender, age_bracket):
    members = []
    for entry in data:
        try:
            sex = entry["demographics"]["sex"]
            age = int(entry["demographics"]["age"])
            annotation = entry["annotation"]
        except (KeyError, TypeError, ValueError):
            continue
        if sex != gender:
            continue
        if age_bracket == "under35" and age >= 35:
            continue
        if age_bracket == "over35" and age < 35:
            continue
        if isinstance(annotation, dict):
            members.append(entry)
    return members


def build_group_matrix(members, item_order):
    matrix = np.full((len(members), len(item_order)), np.nan)
    item_index = {item: i for i, item in enumerate(item_order)}
    for row, member in enumerate(members):
        for item, score in member["annotation"].items():
            if item in item_index:
                try:
                    matrix[row, item_index[item]] = float(score)
                except (TypeError, ValueError):
                    pass
    return matrix


def column_mean(matrix):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(matrix, axis=0)


def load_llm_file(path, item_order):
    """Supports JSONL, a JSON list, or {item_id: score}."""
    records = []
    try:
        with open(path, encoding="utf-8") as file:
            parsed = json.load(file)
        if isinstance(parsed, list):
            records = parsed
        elif isinstance(parsed, dict) and isinstance(parsed.get("data"), list):
            records = parsed["data"]
        elif isinstance(parsed, dict) and "id" in parsed and "gs_value" in parsed:
            records = [parsed]
        elif isinstance(parsed, dict):
            records = [{"id": key, "gs_value": value} for key, value in parsed.items()]
    except json.JSONDecodeError:
        with open(path, encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid JSON at line {line_number}: {error}") from error

    scores = {}
    for record in records:
        if not isinstance(record, dict) or "id" not in record or "gs_value" not in record:
            continue
        try:
            scores[str(record["id"])] = float(record["gs_value"])
        except (TypeError, ValueError):
            pass

    vector = np.array([scores.get(item, np.nan) for item in item_order], dtype=float)
    if np.isfinite(vector).sum() < 3:
        raise ValueError("Fewer than three LLM scores match the human item IDs")
    return vector

# -----------------------------------------------------------------------------
# METRICS
# -----------------------------------------------------------------------------
def is_constant(vector):
    return len(vector) == 0 or np.allclose(vector, vector[0])


def calculate_metrics(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 3 or is_constant(x) or is_constant(y):
        return {"pearson": np.nan, "p_pearson": np.nan,
                "spearman": np.nan, "p_spearman": np.nan,
                "mae": np.nan, "n": len(x)}
    pearson, p_pearson = stats.pearsonr(x, y)
    spearman, p_spearman = stats.spearmanr(x, y)
    return {
        "pearson": float(pearson),
        "p_pearson": float(p_pearson),
        "spearman": float(spearman),
        "p_spearman": float(p_spearman),
        "mae": float(np.mean(np.abs(x - y))),
        "n": int(len(x)),
    }


def calculate_values(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 3 or is_constant(x) or is_constant(y):
        return np.nan, np.nan, np.nan
    return (
        float(stats.pearsonr(x, y)[0]),
        float(stats.spearmanr(x, y)[0]),
        float(np.mean(np.abs(x - y))),
    )


def ci(values):
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan, np.nan
    return tuple(np.percentile(values, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)]))


def bootstrap_p(differences):
    """Approximate two-sided bootstrap p-value; CI is the primary result."""
    differences = differences[np.isfinite(differences)]
    if not len(differences):
        return np.nan
    left = (np.sum(differences <= 0) + 1) / (len(differences) + 1)
    right = (np.sum(differences >= 0) + 1) / (len(differences) + 1)
    return min(1.0, 2 * min(left, right))


def holm_adjust(p_values):
    adjusted = {name: np.nan for name in p_values}
    ordered = sorted(
        [(name, p) for name, p in p_values.items() if np.isfinite(p)],
        key=lambda pair: pair[1],
    )
    running_max = 0.0
    m = len(ordered)
    for rank, (name, p_value) in enumerate(ordered):
        running_max = max(running_max, min(1.0, (m - rank) * p_value))
        adjusted[name] = running_max
    return adjusted

# -----------------------------------------------------------------------------
# HIERARCHICAL BOOTSTRAP FOR ONE MODEL
# -----------------------------------------------------------------------------
def bootstrap_model(llm_vector, group_matrices, group_profiles, seed):
    rng = np.random.default_rng(seed)
    groups = list(group_matrices)

    valid_mask = np.isfinite(llm_vector)
    for profile in group_profiles.values():
        valid_mask &= np.isfinite(profile)
    valid_items = np.flatnonzero(valid_mask)
    if len(valid_items) < 3:
        raise ValueError("Fewer than three items are shared by the LLM and all groups")

    point = {group: calculate_metrics(llm_vector, group_profiles[group]) for group in groups}
    boot = {
        metric: np.full((N_BOOT, len(groups)), np.nan)
        for metric in ("pearson", "spearman", "mae")
    }

    for b in range(N_BOOT):
        sampled_items = rng.choice(valid_items, size=len(valid_items), replace=True)
        sampled_llm = llm_vector[sampled_items]

        for group_index, group in enumerate(groups):
            matrix = group_matrices[group]
            sampled_rows = rng.integers(0, matrix.shape[0], size=matrix.shape[0])
            sampled_profile = column_mean(matrix[sampled_rows])
            pearson, spearman, mae = calculate_values(
                sampled_llm,
                sampled_profile[sampled_items],
            )
            boot["pearson"][b, group_index] = pearson
            boot["spearman"][b, group_index] = spearman
            boot["mae"][b, group_index] = mae

    intervals = {}
    for group_index, group in enumerate(groups):
        intervals[group] = {}
        for metric in ("pearson", "spearman", "mae"):
            low, high = ci(boot[metric][:, group_index])
            intervals[group][metric] = {
                "estimate": point[group][metric],
                "ci_low": float(low),
                "ci_high": float(high),
            }

    # Selection frequency: highest correlation or lowest MAE in each replicate.
    selection_frequency = {metric: {} for metric in ("pearson", "spearman", "mae")}
    for metric in ("pearson", "spearman"):
        valid_rows = np.all(np.isfinite(boot[metric]), axis=1)
        winners = np.argmax(boot[metric][valid_rows], axis=1)
        for group_index, group in enumerate(groups):
            selection_frequency[metric][group] = float(np.mean(winners == group_index))
    valid_rows = np.all(np.isfinite(boot["mae"]), axis=1)
    winners = np.argmin(boot["mae"][valid_rows], axis=1)
    for group_index, group in enumerate(groups):
        selection_frequency["mae"][group] = float(np.mean(winners == group_index))

    # Reviewer-focused test: paired comparisons between Pearson correlations.
    pairwise_pearson = {}
    raw_p = {}
    for index_1, index_2 in combinations(range(len(groups)), 2):
        group_1, group_2 = groups[index_1], groups[index_2]
        name = f"{group_1} vs {group_2}"
        differences = boot["pearson"][:, index_1] - boot["pearson"][:, index_2]
        low, high = ci(differences)
        p_value = bootstrap_p(differences)
        pairwise_pearson[name] = {
            "group_1": group_1,
            "group_2": group_2,
            "delta": point[group_1]["pearson"] - point[group_2]["pearson"],
            "ci_low": float(low),
            "ci_high": float(high),
            "p_raw": float(p_value),
        }
        raw_p[name] = p_value

    adjusted = holm_adjust(raw_p)
    for name, p_value in adjusted.items():
        pairwise_pearson[name]["p_holm"] = float(p_value)

    best_pearson = max(groups, key=lambda group: point[group]["pearson"])
    best_spearman = max(groups, key=lambda group: point[group]["spearman"])
    best_mae = min(groups, key=lambda group: point[group]["mae"])

    best_vs_others = []
    supported_against_all = True
    for alternative in groups:
        if alternative == best_pearson:
            continue
        direct = f"{best_pearson} vs {alternative}"
        reverse = f"{alternative} vs {best_pearson}"
        if direct in pairwise_pearson:
            result = pairwise_pearson[direct]
            delta, low, high = result["delta"], result["ci_low"], result["ci_high"]
        else:
            result = pairwise_pearson[reverse]
            delta, low, high = -result["delta"], -result["ci_high"], -result["ci_low"]
        significant = bool(np.isfinite(low) and low > 0)
        supported_against_all &= significant
        best_vs_others.append({
            "alternative": alternative,
            "delta": float(delta),
            "ci_low": float(low),
            "ci_high": float(high),
            "p_holm": result["p_holm"],
            "supported": significant,
        })

    return {
        "n_items": len(valid_items),
        "point": point,
        "intervals": intervals,
        "pairwise_pearson": pairwise_pearson,
        "selection_frequency": selection_frequency,
        "assignment": {
            "best_pearson": best_pearson,
            "best_spearman": best_spearman,
            "best_mae": best_mae,
            "pearson_spearman_agree": best_pearson == best_spearman,
            "supported_against_all": supported_against_all,
            "best_vs_others": best_vs_others,
        },
    }

# -----------------------------------------------------------------------------
# REPORT
# -----------------------------------------------------------------------------
def fmt(value, digits=4):
    return "N/A" if value is None or not np.isfinite(value) else f"{value:.{digits}f}"


def fmt_p(value):
    if value is None or not np.isfinite(value):
        return "N/A"
    return "<.001" if value < 0.001 else f"{value:.3f}"


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def add_model_to_report(lines, model_name, result):
    lines.extend(["", SEP, f"LLM: {model_name}", SEP])
    lines.append(f"Shared items used: {result['n_items']}")
    lines.extend(["", "METRICS AND BOOTSTRAP CONFIDENCE INTERVALS", SEP2])

    for group in GROUP_DEFS:
        p = result["point"][group]
        i = result["intervals"][group]
        lines.append(
            f"{group:<23} | "
            f"Pearson {fmt(p['pearson'])} (p={fmt_p(p['p_pearson'])}), "
            f"CI [{fmt(i['pearson']['ci_low'])}, {fmt(i['pearson']['ci_high'])}] | "
            f"Spearman {fmt(p['spearman'])} (p={fmt_p(p['p_spearman'])}), "
            f"CI [{fmt(i['spearman']['ci_low'])}, {fmt(i['spearman']['ci_high'])}] | "
            f"MAE {fmt(p['mae'])}, CI [{fmt(i['mae']['ci_low'])}, {fmt(i['mae']['ci_high'])}]"
        )

    lines.extend(["", "PAIRWISE PEARSON COMPARISONS", SEP2])
    lines.append(
        f"{'Comparison':<49} {'Delta r':>9} {'CI low':>9} "
        f"{'CI high':>9} {'p Holm':>8}"
    )
    for name, comparison in result["pairwise_pearson"].items():
        lines.append(
            f"{name:<49} {fmt(comparison['delta']):>9} "
            f"{fmt(comparison['ci_low']):>9} {fmt(comparison['ci_high']):>9} "
            f"{fmt_p(comparison['p_holm']):>8}"
        )

    lines.extend(["", "BOOTSTRAP SELECTION FREQUENCY", SEP2])
    for group in GROUP_DEFS:
        lines.append(
            f"{group:<23} | Pearson {result['selection_frequency']['pearson'][group]:.3f} | "
            f"Spearman {result['selection_frequency']['spearman'][group]:.3f} | "
            f"MAE {result['selection_frequency']['mae'][group]:.3f}"
        )

    assignment = result["assignment"]
    lines.extend(["", "ASSIGNMENT", SEP2])
    lines.append(f"Best Pearson : {assignment['best_pearson']}")
    lines.append(f"Best Spearman: {assignment['best_spearman']}")
    lines.append(f"Best MAE     : {assignment['best_mae']}")
    lines.append(
        "Pearson/Spearman agree: "
        + ("YES" if assignment["pearson_spearman_agree"] else "NO")
    )
    lines.append(
        "Best Pearson group better than all alternatives by 95% CI: "
        + ("YES" if assignment["supported_against_all"] else "NO")
    )
    for comparison in assignment["best_vs_others"]:
        lines.append(
            f"  vs {comparison['alternative']:<22}: Δr={fmt(comparison['delta'])}, "
            f"CI [{fmt(comparison['ci_low'])}, {fmt(comparison['ci_high'])}], "
            f"p_Holm={fmt_p(comparison['p_holm'])}"
        )

# -----------------------------------------------------------------------------
# MAIN: ALL LLM FILES ARE ANALYSED AUTOMATICALLY
# -----------------------------------------------------------------------------
def main():
    with open(ANNOTATIONS_FILE, encoding="utf-8") as file:
        human_data = json.load(file)

    item_order = sorted({
        item for entry in human_data for item in entry.get("annotation", {})
    })
    group_members = {
        group: get_group(human_data, gender, age_bracket)
        for group, (gender, age_bracket) in GROUP_DEFS.items()
    }
    for group, members in group_members.items():
        if not members:
            raise ValueError(f"No annotators found for {group}")

    group_matrices = {
        group: build_group_matrix(members, item_order)
        for group, members in group_members.items()
    }
    group_profiles = {
        group: column_mean(matrix) for group, matrix in group_matrices.items()
    }

    llm_files = sorted(
        glob.glob(os.path.join(LLM_FOLDER, "*.jsonl"))
        + glob.glob(os.path.join(LLM_FOLDER, "*.json"))
    )
    if not llm_files:
        raise FileNotFoundError(f"No JSON/JSONL files found in {LLM_FOLDER}")

    results = {}
    print(f"Found {len(llm_files)} LLM files")

    # The bootstrap function works on one vector, but this loop calls it for
    # every file and collects all models in the same final report.
    for model_index, path in enumerate(llm_files):
        model_name = os.path.splitext(os.path.basename(path))[0]
        print(f"[{model_index + 1}/{len(llm_files)}] {model_name}")
        try:
            llm_vector = load_llm_file(path, item_order)
            results[model_name] = bootstrap_model(
                llm_vector,
                group_matrices,
                group_profiles,
                seed=SEED + model_index,
            )
        except Exception as error:
            print(f"  SKIPPED: {error}")

    if not results:
        raise RuntimeError("No LLM was analysed successfully")

    lines = [
        SEP,
        "SOCIO-DEMOGRAPHIC LLM PROFILING — HIERARCHICAL BOOTSTRAP",
        SEP,
        f"Bootstrap replicates per model: {N_BOOT}",
        f"Confidence level: {(1 - ALPHA) * 100:.1f}%",
        "",
        "Human group sizes:",
    ]
    for group, members in group_members.items():
        lines.append(f"  {group:<23}: {len(members)}")
    for model_name, result in results.items():
        add_model_to_report(lines, model_name, result)

    lines.extend(["", SEP, "CROSS-MODEL SUMMARY", SEP])
    lines.append(
        f"{'LLM':<42} {'Best Pearson':<23} {'Best freq.':>10} "
        f"{'P/S agree':>10} {'Supported':>11}"
    )
    lines.append(SEP2)
    for model_name, result in results.items():
        assignment = result["assignment"]
        best = assignment["best_pearson"]
        frequency = result["selection_frequency"]["pearson"][best]
        lines.append(
            f"{model_name:<42} {best:<23} {frequency:>10.3f} "
            f"{('YES' if assignment['pearson_spearman_agree'] else 'NO'):>10} "
            f"{('YES' if assignment['supported_against_all'] else 'NO'):>11}"
        )

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        file.write("\n".join(lines))
    with open(OUTPUT_JSON, "w", encoding="utf-8") as file:
        json.dump(json_safe(results), file, ensure_ascii=False, indent=2)

    print(f"Text report saved to: {OUTPUT_FILE}")
    print(f"JSON results saved to: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
