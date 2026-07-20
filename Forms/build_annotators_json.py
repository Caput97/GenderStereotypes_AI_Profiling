#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_annotators_json.py
========================

Costruisce (o aggiorna in modo incrementale) un file JSON in cui OGNI annotatore
e' un item, a partire da una coppia di file CSV:

    *_FormData.csv      -> annotazioni (annotator_id, prolific_id, sentence_id, rating)
    *_Demographics.csv  -> dati demografici (annotator_id, prolific_id, sex, age, ...)

Comportamento incrementale
--------------------------
- La prima volta il file JSON viene creato.
- Le volte successive (nuove coppie di file) il file esistente viene letto e AGGIORNATO:
    * agli annotatori NUOVI viene assegnato un ID_annotator progressivo che riparte
      dall'ultimo numero gia' presente (non da zero);
    * gli annotatori GIA' presenti vengono SALTATI (si tiene la prima versione).
- Identita' di un annotatore (regola a due livelli):
    * Se il prolific_id e' disponibile su ENTRAMBI (nuovo ed esistente), fa fede quello:
        - stesso annotator_id + stesso prolific_id  = stessa persona  -> saltato
        - stesso annotator_id + prolific_id diverso = persone diverse -> aggiunti entrambi
    * Se il prolific_id manca da almeno un lato (non usabile come discriminante),
      si confrontano le ANNOTAZIONI a parita' di annotator_id:
        - stesso annotator_id + stesse annotazioni  = stessa persona  -> saltato
        - stesso annotator_id + annotazioni diverse = persone diverse -> aggiunti entrambi

Struttura di un item nel JSON
-----------------------------
{
    "ID_annotator": 1,
    "annotator_id": "GS5",
    "prolific_id": "69289ebed21f39d1fb97d278",     # solo se disponibile
    "demographics": {                               # solo se c'e' match nel Demographics
        "sex": "...",
        "age": ...,
        "education": "...",
        "provenance": "...",
        "political_position": "...",
        "sexual_orientation": "..."
    },
    "annotation": {
        "GS0943": 0.51,
        "GS0522": 0.76,
        ...
    }
}

Uso
---
    python build_annotators_json.py \
        --formdata     path/al/..._FormData.csv \
        --demographics path/al/..._Demographics.csv \
        --json         path/output/annotators.json

--demographics e' opzionale: se omesso o se il file non esiste, gli item vengono
creati comunque (senza il campo "demographics", che potra' essere aggiunto in futuro).
"""

import argparse
import json
import os
import math
import pandas as pd


# ============================================================================
# CONFIGURAZIONE  (modifica qui se cambiano nomi colonne o regole di pulizia)
# ============================================================================

# Colonne nel FormData
COL_ANNOTATOR_ID = "annotator_id"
COL_PROLIFIC_ID  = "prolific_id"
COL_SENTENCE_ID  = "sentence_id"
COL_RATING       = "rating"

# Colonne demografiche da estrarre dal Demographics (nell'ordine desiderato).
# Vengono usate solo quelle effettivamente presenti nel file.
DEMOGRAPHIC_COLUMNS = [
    "sex",
    "age",
    "education",
    "provenance",
    "political_position",
    "sexual_orientation",
]

# Pulizia dati FormData
DROP_EXACT_DUPLICATES = True          # rimuove righe identiche su TUTTE le colonne

# sentence_id da NON inserire affatto nel JSON (item da scartare).
# Vuoto = non si scarta nulla. (Le frasi di controllo "no" vengono MANTENUTE.)
EXCLUDE_SENTENCE_IDS = set()

# sentence_id da rinominare in sequenza, per annotatore.
# Ogni occorrenza di un id qui elencato viene mantenuta e ricodificata con il prefisso
# indicato + un numero progressivo (1, 2, 3, ...) nell'ordine di comparsa.
# Es. le 5 frasi di controllo "no" -> GSno1, GSno2, GSno3, GSno4, GSno5.
# Cosi' restano in linea con le altre annotazioni ma con codice univoco.
RENAME_SEQUENTIAL_IDS = {"no": "GSno"}

# sentence_id "speciali" di cui tenere TUTTE le occorrenze SENZA rinominarle
# (condividerebbero quindi la stessa chiave). Vuoto di default.
KEEP_ALL_OCCURRENCES_IDS = set()


# ============================================================================
# FUNZIONI DI SUPPORTO
# ============================================================================

def _clean_str(value):
    """Converte in stringa pulita; restituisce '' per NaN/None/vuoti."""
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    s = str(value).strip()
    return "" if s.lower() in ("", "nan", "none") else s


def _to_native(value):
    """Converte tipi numpy/pandas in tipi nativi Python per la serializzazione JSON."""
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if hasattr(value, "item"):          # numpy scalar
        return value.item()
    return value


def identity_key(annotator_id, prolific_id):
    """Chiave (annotator_id, prolific_id) usata quando il prolific_id e' presente."""
    return (_clean_str(annotator_id), _clean_str(prolific_id))


def annotation_map(annotation_obj):
    """Converte 'annotation' in un dict {sid: rating} confrontabile.
    Supporta sia il nuovo formato {sid: rating, ...}, sia il vecchio formato
    [{sid: rating}, ...] per compatibilita' con JSON gia' esistenti."""
    if not annotation_obj:
        return {}
    if isinstance(annotation_obj, dict):
        return dict(annotation_obj)

    out = {}
    for entry in annotation_obj:
        for sid, rating in entry.items():
            out[sid] = rating
    return out


def demographics_map(demographics_obj):
    """Converte 'demographics' in un dict {col: val} confrontabile.
    Supporta sia il nuovo formato {col: val, ...}, sia il vecchio formato
    [{col: val}, ...] per compatibilita' con JSON gia' esistenti."""
    if not demographics_obj:
        return {}
    if isinstance(demographics_obj, dict):
        return {
            col: "" if val is None else str(val).strip()
            for col, val in demographics_obj.items()
        }

    out = {}
    for entry in demographics_obj:
        for col, val in entry.items():
            out[col] = "" if val is None else str(val).strip()
    return out


def is_same_annotator(new_aid, new_pid, new_ann_map, new_demo_map, existing_record):
    """
    Decide se un annotatore in arrivo coincide con uno gia' presente.

    existing_record = {"aid":..., "pid":..., "ann": {sid: rating}, "demo": {col: val}}

    Regola a livelli:
      1) serve sempre lo stesso annotator_id;
      2) se ENTRAMBI hanno prolific_id  -> match SOLO se i prolific_id coincidono;
      3) se il prolific_id manca da almeno un lato -> si guardano le annotazioni:
           - annotazioni diverse                       -> persone diverse (no match);
           - annotazioni uguali -> si usa il pacchetto DEMOGRAFICO come discriminante:
               * demografici disponibili e diversi     -> persone diverse (no match);
               * demografici disponibili e uguali      -> stessa persona (match);
               * demografici non disponibili da un lato -> non si puo' discriminare oltre:
                 a parita' di annotator_id e annotazioni si considera la stessa persona.
    """
    if _clean_str(new_aid) != existing_record["aid"]:
        return False

    e_pid = existing_record["pid"]
    if new_pid != "" and e_pid != "":
        return new_pid == e_pid

    # prolific_id non utilizzabile -> confronto sulle annotazioni
    if new_ann_map != existing_record["ann"]:
        return False  # stesse iniziali ma annotazioni diverse -> persone diverse

    # annotazioni identiche -> discriminante finale = demografici
    e_demo = existing_record.get("demo") or {}
    if new_demo_map and e_demo:
        return new_demo_map == e_demo   # uguali -> stessa persona; diversi -> persone diverse
    # demografici mancanti da almeno un lato: a parita' di id + annotazioni -> stessa persona
    return True


def load_existing_json(json_path):
    """Carica il JSON esistente (lista di item). Se non esiste, restituisce lista vuota."""
    if not os.path.exists(json_path):
        return []
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(
            f"Il file {json_path} non contiene una lista JSON di annotatori."
        )
    return data


def build_demographics_lookup(demographics_path):
    """
    Costruisce un dizionario { (annotator_id, prolific_id): {col: val, ...} }
    a partire dal file Demographics. Restituisce {} se il file manca.
    """
    if not demographics_path or not os.path.exists(demographics_path):
        return {}

    demo = pd.read_csv(demographics_path, dtype=str, keep_default_na=False)
    demo.columns = [c.strip() for c in demo.columns]

    if COL_ANNOTATOR_ID not in demo.columns:
        raise ValueError(
            f"Il Demographics non contiene la colonna '{COL_ANNOTATOR_ID}'. "
            f"Colonne trovate: {list(demo.columns)}"
        )

    has_prolific = COL_PROLIFIC_ID in demo.columns
    present_demo_cols = [c for c in DEMOGRAPHIC_COLUMNS if c in demo.columns]

    lookup = {}
    for _, row in demo.iterrows():
        aid = _clean_str(row.get(COL_ANNOTATOR_ID, ""))
        pid = _clean_str(row.get(COL_PROLIFIC_ID, "")) if has_prolific else ""
        if aid == "" and pid == "":
            continue
        demo_dict = {col: _to_native(row.get(col, "")) for col in present_demo_cols}
        lookup[identity_key(aid, pid)] = demo_dict

    return lookup


def lookup_demographics(demo_lookup, annotator_id, prolific_id):
    """
    Cerca i demografici per un annotatore con doppio filtro:
    1) match esatto su (annotator_id, prolific_id);
    2) fallback su solo annotator_id se nel JSON/Demographics il prolific_id manca.
    Restituisce None se nessun match.
    """
    if not demo_lookup:
        return None
    key = identity_key(annotator_id, prolific_id)
    if key in demo_lookup:
        return demo_lookup[key]
    # fallback: match sul solo annotator_id (prolific_id assente da un lato)
    aid = _clean_str(annotator_id)
    candidates = [v for (a, p), v in demo_lookup.items() if a == aid and (p == "" )]
    if len(candidates) == 1:
        return candidates[0]
    # ulteriore fallback: prolific_id vuoto nell'item ma presente nel demo
    if _clean_str(prolific_id) == "":
        aid_matches = [v for (a, p), v in demo_lookup.items() if a == aid]
        if len(aid_matches) == 1:
            return aid_matches[0]
    return None


def build_annotation_dict(group):
    """
    Da un gruppo di righe (un annotatore) costruisce il dizionario 'annotation':
    {sentence_id: rating, ...}, mantenendo l'ordine di prima comparsa.

    - sentence_id in RENAME_SEQUENTIAL_IDS (es. "no"): ogni occorrenza viene tenuta
      e ricodificata con prefisso + numero progressivo (GSno1, GSno2, ...).
    - sentence_id in KEEP_ALL_OCCURRENCES_IDS: in un dizionario non possono esistere
      chiavi duplicate; quindi, se presenti, viene mantenuta l'ultima occorrenza.
    - sentence_id normali: una sola voce per id (le ripetizioni sono la stessa
      annotazione loggata piu' volte -> deduplicate).
    """
    annotation = {}
    seen = set()
    rename_counter = {}
    for _, row in group.iterrows():
        sid = _clean_str(row[COL_SENTENCE_ID])
        if sid == "" or sid in EXCLUDE_SENTENCE_IDS:
            continue
        rating = _to_native(row[COL_RATING])

        if sid in RENAME_SEQUENTIAL_IDS:
            rename_counter[sid] = rename_counter.get(sid, 0) + 1
            new_sid = f"{RENAME_SEQUENTIAL_IDS[sid]}{rename_counter[sid]}"
            annotation[new_sid] = rating
            continue

        if sid in KEEP_ALL_OCCURRENCES_IDS:
            annotation[sid] = rating
            continue

        if sid in seen:
            continue
        seen.add(sid)
        annotation[sid] = rating
    return annotation


# ============================================================================
# CORE
# ============================================================================

def process(formdata_path, demographics_path, json_path):
    # --- 1. Carica FormData ---------------------------------------------------
    if not os.path.exists(formdata_path):
        raise FileNotFoundError(f"FormData non trovato: {formdata_path}")

    df = pd.read_csv(formdata_path)
    df.columns = [c.strip() for c in df.columns]

    for col in (COL_ANNOTATOR_ID, COL_SENTENCE_ID, COL_RATING):
        if col not in df.columns:
            raise ValueError(
                f"Il FormData non contiene la colonna '{col}'. "
                f"Colonne trovate: {list(df.columns)}"
            )
    has_prolific = COL_PROLIFIC_ID in df.columns

    if DROP_EXACT_DUPLICATES:
        before = len(df)
        df = df.drop_duplicates()
        print(f"[pulizia] righe duplicate esatte rimosse: {before - len(df)}")

    # --- 2. Carica JSON esistente + stato incrementale ------------------------
    items = load_existing_json(json_path)
    existing_records = []   # [{"aid":..., "pid":..., "ann": {...}}, ...]
    max_id = 0
    for it in items:
        existing_records.append({
            "aid": _clean_str(it.get(COL_ANNOTATOR_ID, "")),
            "pid": _clean_str(it.get(COL_PROLIFIC_ID, "")),
            "ann": annotation_map(it.get("annotation", [])),
            "demo": demographics_map(it.get("demographics", [])),
        })
        max_id = max(max_id, int(it.get("ID_annotator", 0)))
    next_id = max_id + 1
    print(f"[stato]   item gia' presenti nel JSON: {len(items)} | prossimo ID_annotator: {next_id}")

    # --- 3. Lookup demografici ------------------------------------------------
    demo_lookup = build_demographics_lookup(demographics_path)
    if demo_lookup:
        print(f"[demo]    righe demografiche caricate: {len(demo_lookup)}")
    else:
        print("[demo]    nessun file Demographics fornito/valido: item senza 'demographics'.")

    # --- 4. Processa ogni annotatore -----------------------------------------
    added, skipped = 0, 0
    # ordine stabile: per prima comparsa nel file
    annotator_order = list(dict.fromkeys(zip(
        df[COL_ANNOTATOR_ID],
        df[COL_PROLIFIC_ID] if has_prolific else [None] * len(df),
    )))

    for aid_raw, pid_raw in annotator_order:
        aid = _clean_str(aid_raw)
        pid = _clean_str(pid_raw) if has_prolific else ""

        # righe di questo annotatore
        if has_prolific:
            mask = (df[COL_ANNOTATOR_ID] == aid_raw) & (df[COL_PROLIFIC_ID] == pid_raw)
        else:
            mask = (df[COL_ANNOTATOR_ID] == aid_raw)
        group = df[mask]

        # annotazioni costruite subito: servono per il confronto quando manca il prolific_id
        annotation = build_annotation_dict(group)
        ann_map = annotation_map(annotation)

        # demografici del nuovo annotatore: servono come discriminante finale
        # (stesso annotator_id + stesse annotazioni -> si confrontano i demografici)
        demo_dict = lookup_demographics(demo_lookup, aid, pid)
        demo_map = demographics_map(demo_dict)

        # gia' presente? (regola a tre livelli: prolific_id -> annotazioni -> demografici)
        if any(is_same_annotator(aid, pid, ann_map, demo_map, rec) for rec in existing_records):
            skipped += 1
            continue

        # costruzione item nell'ordine voluto:
        # ID_annotator, annotator_id, [prolific_id], [demographics], annotation
        item = {"ID_annotator": next_id, COL_ANNOTATOR_ID: aid}
        if pid:
            item[COL_PROLIFIC_ID] = pid

        if demo_dict is not None:
            item["demographics"] = demo_dict

        item["annotation"] = annotation

        items.append(item)
        existing_records.append({"aid": aid, "pid": pid, "ann": ann_map, "demo": demo_map})
        next_id += 1
        added += 1

    # --- 5. Salva -------------------------------------------------------------
    out_dir = os.path.dirname(os.path.abspath(json_path))
    os.makedirs(out_dir, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)

    print(f"[fatto]   annotatori aggiunti: {added} | saltati (gia' presenti): {skipped}")
    print(f"[fatto]   totale item nel JSON: {len(items)}")
    print(f"[fatto]   salvato in: {json_path}")
    return items


def main():
    parser = argparse.ArgumentParser(
        description="Costruisce/aggiorna un JSON di annotatori da FormData + Demographics CSV."
    )
    parser.add_argument("--formdata", required=True, help="Percorso al file *_FormData.csv")
    parser.add_argument("--demographics", default=None, help="Percorso al file *_Demographics.csv (opzionale)")
    parser.add_argument("--json", required=True, help="Percorso del file JSON di output (creato o aggiornato)")
    args = parser.parse_args()
    process(args.formdata, args.demographics, args.json)


if __name__ == "__main__":
    main()
