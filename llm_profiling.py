"""
Socio-demographic profiling di LLM
Confronta i punteggi di ogni LLM sui 124 item con i profili dei 4 gruppi umani
(Donne/Uomini × Under35/Over35) usando Pearson, Spearman e MAE.

Come usare:
  1. Metti tutti i file JSONL degli LLM in una cartella (es. ./llm_files/)
  2. Metti il file degli annotatori umani (annotation_results_noGSno.json)
     nella stessa cartella dello script
  3. Cambia LLM_FOLDER qui sotto se necessario
  4. python3 llm_profiling.py
"""

import json
import os
import glob
import numpy as np
from scipy import stats
from collections import defaultdict

# ─────────────────────────────────────────────
# CONFIGURAZIONE — modifica questi path
# ─────────────────────────────────────────────
ANNOTATIONS_FILE = "/home/dtesta/GenderStereotypes_AI_Profiling/Forms/annotation_results_noGSno.json"
LLM_FOLDER       = "/home/dtesta/GSI-detect_LLM_Socioprofiling/LLMs/124Matched"
OUTPUT_FILE      = "/home/dtesta/GSI-detect_LLM_Socioprofiling/LLM_profiling/llm_profiling.txt"

# ─────────────────────────────────────────────
# UTILITÀ
# ─────────────────────────────────────────────
SEP  = "=" * 70
SEP2 = "-" * 70

lines = []
def w(*args):
    lines.append(" ".join(str(a) for a in args))

def get_group(data, gender, age_bracket):
    result = []
    for d in data:
        sex = d["demographics"]["sex"]
        try:
            age = int(d["demographics"]["age"])
        except (ValueError, KeyError):
            continue
        if sex != gender:
            continue
        if age_bracket == "under35" and age >= 35:
            continue
        if age_bracket == "over35" and age < 35:
            continue
        result.append(d)
    return result

def group_mean_vector(members, item_order):
    scores = defaultdict(list)
    for d in members:
        for item, score in d["annotation"].items():
            scores[item].append(score)
    return np.array([np.mean(scores[item]) for item in item_order])

def load_llm_file(path, item_order):
    scores = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            scores[d["id"]] = float(d["gs_value"])
    return np.array([scores.get(item, np.nan) for item in item_order])

def compute_metrics(vec_llm, vec_group):
    """Pearson, Spearman, MAE + p-value per entrambe le correlazioni."""
    mask = ~(np.isnan(vec_llm) | np.isnan(vec_group))
    if mask.sum() < 3:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    a, b = vec_llm[mask], vec_group[mask]
    pearson,  p_pearson  = stats.pearsonr(a, b)
    spearman, p_spearman = stats.spearmanr(a, b)
    mae = np.mean(np.abs(a - b))
    return pearson, p_pearson, spearman, p_spearman, mae

def fmt_p(p):
    """Formatta il p-value in modo leggibile."""
    if np.isnan(p):
        return "  N/A"
    if p < 0.001:
        return "<.001"
    return f"{p:.3f}"

def assign_label(metrics_dict, group_names):
    """
    Assegna label demografica basandosi su Pearson e Spearman.
    Criterio:
      - Candidato primario: gruppo con Pearson più alta
      - Conferma: stesso gruppo ha anche Spearman più alta
      - Se concordano → label pulita
      - Se discordano → label con asterisco (*) = da discutere
    Restituisce: (label, is_ambiguous, best_pearson, best_spearman, best_mae)
    """
    best_p = max(group_names, key=lambda g: metrics_dict[g]["pearson"])
    best_s = max(group_names, key=lambda g: metrics_dict[g]["spearman"])
    best_m = min(group_names, key=lambda g: metrics_dict[g]["mae"])

    if best_p == best_s:
        # Pearson e Spearman concordano → assegnazione robusta
        label = best_p
        ambiguous = False
    else:
        # Discordano → usa Pearson come criterio primario, segnala con *
        label = best_p
        ambiguous = True

    return label, ambiguous, best_p, best_s, best_m

# ─────────────────────────────────────────────
# CARICAMENTO DATI UMANI
# ─────────────────────────────────────────────
with open(ANNOTATIONS_FILE) as f:
    human_data = json.load(f)

ITEM_ORDER = sorted(set(k for d in human_data for k in d["annotation"].keys()))

# ─────────────────────────────────────────────
# COSTRUZIONE PROFILI DEI 4 GRUPPI UMANI
# ─────────────────────────────────────────────
group_defs = {
    "Donne under 35":  ("F", "under35"),
    "Donne over 35":   ("F", "over35"),
    "Uomini under 35": ("M", "under35"),
    "Uomini over 35":  ("M", "over35"),
}

human_profiles = {}
human_sizes    = {}
for gname, (gender, bracket) in group_defs.items():
    members = get_group(human_data, gender, bracket)
    human_profiles[gname] = group_mean_vector(members, ITEM_ORDER)
    human_sizes[gname]    = len(members)

GROUP_NAMES = list(group_defs.keys())

# ─────────────────────────────────────────────
# CARICAMENTO FILE LLM
# ─────────────────────────────────────────────
llm_files = sorted(glob.glob(os.path.join(LLM_FOLDER, "*.jsonl")) +
                   glob.glob(os.path.join(LLM_FOLDER, "*.json")))

if not llm_files:
    print(f"ATTENZIONE: nessun file trovato in '{LLM_FOLDER}'")
    print("Modifica LLM_FOLDER nello script con il path corretto.")
    exit(1)

# ─────────────────────────────────────────────
# CALCOLO METRICHE PER OGNI LLM
# ─────────────────────────────────────────────
results = {}

for fpath in llm_files:
    llm_name = os.path.splitext(os.path.basename(fpath))[0]
    vec_llm  = load_llm_file(fpath, ITEM_ORDER)
    n_found  = int(np.sum(~np.isnan(vec_llm)))

    metrics = {}
    for gname, vec_group in human_profiles.items():
        p, pp, s, ps, m = compute_metrics(vec_llm, vec_group)
        metrics[gname] = {
            "pearson": p, "p_pearson": pp,
            "spearman": s, "p_spearman": ps,
            "mae": m
        }

    results[llm_name] = {"metrics": metrics, "n_items": n_found}

# ─────────────────────────────────────────────
# OUTPUT
# ─────────────────────────────────────────────
w(SEP)
w("SOCIO-DEMOGRAPHIC PROFILING DI LLM")
w("Confronto con profili dei gruppi umani (Donne/Uomini × Under/Over 35)")
w(SEP)
w()
w("Metriche usate:")
w("  • Pearson r  : correlazione lineare tra i vettori di punteggi (124 item)")
w("  • Spearman ρ : correlazione sui ranghi (robusta a distribuzioni non normali)")
w("  • MAE        : errore assoluto medio item per item (più basso = più vicino)")
w("  • p          : p-value della correlazione (N=124 item)")
w()
w("Profili umani di riferimento:")
w(f"  {'Gruppo':<22}  {'N annotatori':>13}")
w(f"  {'-'*22}  {'-'*13}")
for gname, n in human_sizes.items():
    w(f"  {gname:<22}  {n:>13}")
w()

# ── Dettaglio per ogni LLM ──────────────────
w(SEP)
w("RISULTATI PER OGNI LLM")
w(SEP)

for llm_name, data in results.items():
    metrics = data["metrics"]
    n_items = data["n_items"]
    w()
    w(f"  {llm_name}  (item: {n_items}/124)")
    w(f"  {SEP2}")
    w(f"  {'Gruppo':<22}  {'Pearson':>8}  {'p':>6}  {'Spearman':>9}  {'p':>6}  {'MAE':>7}")
    w(f"  {'-'*22}  {'-'*8}  {'-'*6}  {'-'*9}  {'-'*6}  {'-'*7}")
    for gname in GROUP_NAMES:
        m = metrics[gname]
        w(f"  {gname:<22}  {m['pearson']:>8.4f}  {fmt_p(m['p_pearson']):>6}  "
          f"{m['spearman']:>9.4f}  {fmt_p(m['p_spearman']):>6}  {m['mae']:>7.4f}")

# ── Tabelle comparative ─────────────────────
for metric_key, metric_label in [
    ("pearson",  "PEARSON"),
    ("spearman", "SPEARMAN"),
    ("mae",      "MAE"),
]:
    w()
    w(SEP)
    w(f"TABELLA COMPARATIVA — {metric_label} per tutti gli LLM")
    w(SEP)
    w()
    llm_names = list(results.keys())
    col = 10
    w("  " + " " * 22 + "  " + "  ".join(f"{n[:col]:<{col}}" for n in llm_names))
    w("  " + "-" * (22 + 2 + (col + 2) * len(llm_names)))
    for gname in GROUP_NAMES:
        row = f"  {gname:<22}  "
        for llm_name in llm_names:
            v = results[llm_name]["metrics"][gname][metric_key]
            row += f"{v:>{col}.4f}  "
        w(row)

# ── Tabella di assegnazione finale ──────────
w()
w(SEP)
w("TABELLA DI ASSEGNAZIONE DEMOGRAFICA")
w(SEP)
w()
w("  Criterio: Pearson come metrica primaria, Spearman come conferma.")
w("  [*] = Pearson e Spearman discordano → caso da discutere nel testo")
w("        usando MAE come argomento aggiuntivo.")
w()
w(f"  {'LLM':<45}  {'Profilo assegnato':<26}  {'Pearson→':<22}  {'Spearman→':<22}  {'MAE min':>7}")
w(f"  {'-'*45}  {'-'*26}  {'-'*22}  {'-'*22}  {'-'*7}")

for llm_name, data in results.items():
    metrics  = data["metrics"]
    label, ambiguous, best_p, best_s, best_m = assign_label(metrics, GROUP_NAMES)
    flag     = " [*]" if ambiguous else ""
    mae_val  = metrics[best_m]["mae"]
    w(f"  {llm_name:<45}  {label+flag:<26}  {best_p:<22}  {best_s:<22}  {mae_val:>7.4f}")

w()
w("  Legenda colonne:")
w("  Pearson→  : gruppo con Pearson più alta")
w("  Spearman→ : gruppo con Spearman più alta")
w("  MAE min   : MAE del gruppo con distanza assoluta minima")
w()
w(SEP)
w("NOTE METODOLOGICHE")
w(SEP)
w()
w("  • I punteggi continui e discreti degli LLM sono trattati uniformemente")
w("    come valori numerici reali (0.0–1.0).")
w("  • I profili umani sono costruiti come media per item del gruppo.")
w("  • Pearson e Spearman più alte = più simile nella direzione dei giudizi.")
w("  • MAE più bassa = più vicino in termini di valori assoluti.")
w("  • I p-value si riferiscono a N=124 item. Con questo N anche correlazioni")
w("    moderate risultano significative; il p-value indica affidabilità")
w("    statistica, non entità dell'effetto.")
w()
w(SEP)
w("FINE ANALISI")
w(SEP)

# ─────────────────────────────────────────────
# SALVATAGGIO
# ─────────────────────────────────────────────
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print(f"File salvato: {OUTPUT_FILE}")
print(f"LLM analizzati: {len(results)}")
