#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set

EXCLUDE_CODES = {"GSno1", "GSno2", "GSno3", "GSno4", "GSno5"}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_gs_codes(annotation_file: Path) -> Set[str]:
    """
    Extract GS codes from the first item's `annotation` field.
    The annotation keys are assumed to be the same for every annotator/item.
    Excludes GSno1-GSno5 automatically.
    """
    data = load_json(annotation_file)

    if not isinstance(data, list) or not data:
        raise ValueError("The annotation file must be a non-empty JSON list.")

    first_item = data[0]
    if not isinstance(first_item, dict) or "annotation" not in first_item:
        raise ValueError("The first item must contain an 'annotation' field.")

    annotation = first_item["annotation"]
    if not isinstance(annotation, dict):
        raise ValueError("The 'annotation' field must be a dictionary.")

    gs_codes = {
        key for key in annotation.keys()
        if key.startswith("GS") and key not in EXCLUDE_CODES
    }

    return gs_codes


def read_input_records(input_file: Path) -> List[Dict[str, Any]]:
    """
    Read either .jsonl or .json.
    - .jsonl: one JSON object per line
    - .json: either a list of objects or a single object
    """
    suffix = input_file.suffix.lower()

    if suffix == ".jsonl":
        records = []
        with input_file.open("r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON at line {line_number} in {input_file}: {e}")
                if not isinstance(obj, dict):
                    raise ValueError(f"Line {line_number} in {input_file} is not a JSON object.")
                records.append(obj)
        return records

    if suffix == ".json":
        data = load_json(input_file)
        if isinstance(data, list):
            if not all(isinstance(item, dict) for item in data):
                raise ValueError(f"All items in {input_file} must be JSON objects.")
            return data
        if isinstance(data, dict):
            return [data]
        raise ValueError(f"Unsupported JSON structure in {input_file}. Expected list or object.")

    raise ValueError(f"Unsupported input extension for {input_file}. Use .json or .jsonl.")


def write_output_records(records: List[Dict[str, Any]], output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_file.suffix.lower()

    if suffix == ".jsonl":
        with output_file.open("w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return

    if suffix == ".json":
        with output_file.open("w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        return

    raise ValueError(f"Unsupported output extension for {output_file}. Use .json or .jsonl.")


def make_output_path(input_file: Path, suffix: str, output_dir: Path | None = None) -> Path:
    output_name = f"{input_file.stem}{suffix}{input_file.suffix}"
    if output_dir is not None:
        return output_dir / output_name
    return input_file.with_name(output_name)


def filter_file(input_file: Path, gs_codes: Set[str], suffix: str, output_dir: Path | None = None) -> Dict[str, Any]:
    records = read_input_records(input_file)
    matched = [record for record in records if str(record.get("id")) in gs_codes]

    output_file = make_output_path(input_file, suffix=suffix, output_dir=output_dir)
    write_output_records(matched, output_file)

    found_ids = {str(record.get("id")) for record in matched}
    missing_codes = sorted(gs_codes - found_ids)

    return {
        "input_file": str(input_file),
        "output_file": str(output_file),
        "input_records": len(records),
        "matched_records": len(matched),
        "missing_codes": missing_codes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Filter one or more JSON/JSONL files, keeping only records whose 'id' "
            "matches the GS codes extracted from an annotation file. "
            "GSno1-GSno5 are excluded automatically."
        )
    )
    parser.add_argument(
        "--annotation-file",
        required=True,
        help="JSON annotation file containing an 'annotation' dictionary with GS codes.",
    )
    parser.add_argument(
        "--input-files",
        nargs="+",
        required=True,
        help="One or more .json or .jsonl files to filter.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory. If omitted, each output file is saved next to its input file.",
    )
    parser.add_argument(
        "--suffix",
        default="_matched124",
        help="Suffix to add before the file extension. Default: _matched124",
    )

    args = parser.parse_args()

    annotation_file = Path(args.annotation_file)
    input_files = [Path(path) for path in args.input_files]
    output_dir = Path(args.output_dir) if args.output_dir else None

    gs_codes = extract_gs_codes(annotation_file)

    print(f"GS codes extracted, excluding GSno1-GSno5: {len(gs_codes)}")
    if len(gs_codes) != 124:
        print("WARNING: expected 124 GS codes, but found", len(gs_codes))

    print("-" * 60)

    for input_file in input_files:
        result = filter_file(
            input_file=input_file,
            gs_codes=gs_codes,
            suffix=args.suffix,
            output_dir=output_dir,
        )

        print(f"Input file: {result['input_file']}")
        print(f"Output file: {result['output_file']}")
        print(f"Input records: {result['input_records']}")
        print(f"Matched records written: {result['matched_records']}")

        if result["missing_codes"]:
            print(f"Missing GS codes in this file: {len(result['missing_codes'])}")
            print(", ".join(result["missing_codes"]))
        else:
            print("Missing GS codes in this file: 0")

        print("-" * 60)


if __name__ == "__main__":
    main()
