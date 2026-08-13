#!/usr/bin/env python3
"""Export EgoDex task objects to per-demo objects.json for the intent processor.

Reads the ``llm_objects`` attribute (falling back to the ``object`` attribute,
e.g. ``"object:stapler, color:black"``) from each EgoDex HDF5 and writes an
``objects.json`` into the corresponding processed demo directory. The intent
processor (Stage A) uses this file as the Grounding-DINO object prompt.

Mirrors b/generate_narration_csv.py.

Example:
    python b/export_egodex_objects.py --task basic_pick_place
    python b/export_egodex_objects.py --task basic_pick_place --dry-run
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

try:
    import h5py
except ImportError as exc:  # pragma: no cover
    raise SystemExit(f"h5py is required: {exc}") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export EgoDex llm_objects to per-demo objects.json",
    )
    parser.add_argument("--task", type=str, default="basic_pick_place")
    parser.add_argument("--egodex-root", type=Path, default=Path("/home/a26160/DATA/test"))
    parser.add_argument(
        "--processed-root", type=Path, default=Path("/home/a26160/DATA/test_phantom_processed")
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def read_objects_from_hdf5(hdf5_path: Path) -> list:
    with h5py.File(hdf5_path, "r") as f:
        attrs = dict(f.attrs)

    objs = attrs.get("llm_objects", None)
    if objs is not None:
        objs = [str(o).strip() for o in np.atleast_1d(objs) if str(o).strip() and str(o).lower() != "none"]
        if objs:
            return objs

    # Fallback: parse the free-form `object` attribute, e.g. "object:stapler, color:black".
    obj_field = str(attrs.get("object", "")).strip()
    if obj_field:
        for part in obj_field.split(","):
            part = part.strip()
            if part.lower().startswith("object:"):
                name = part.split(":", 1)[1].strip()
                if name:
                    return [name]

    raise ValueError(f"no object annotation in {hdf5_path}")


def main() -> None:
    args = parse_args()

    task_dir = args.egodex_root / args.task
    demo_root = args.processed_root / f"egodex_{args.task}"

    if not task_dir.is_dir():
        raise SystemExit(f"Error: task directory not found: {task_dir}")
    if not demo_root.is_dir():
        raise SystemExit(f"Error: processed directory not found: {demo_root}")

    hdf5_files = sorted(task_dir.glob("*.hdf5"), key=lambda p: int(p.stem))
    if not hdf5_files:
        raise SystemExit(f"Error: no HDF5 files found in {task_dir}")

    created = skipped_existing = skipped_no_dir = failed = 0
    for hdf5_path in hdf5_files:
        episode = hdf5_path.stem
        out_path = demo_root / episode / "objects.json"

        if not out_path.parent.is_dir():
            skipped_no_dir += 1
            continue
        if out_path.exists() and not args.overwrite:
            skipped_existing += 1
            continue

        try:
            objects = read_objects_from_hdf5(hdf5_path)
        except ValueError as exc:
            print(f"skip {episode}: {exc}")
            failed += 1
            continue

        payload = {"objects": objects, "prompt": objects[0]}
        if args.dry_run:
            print(f"would write {out_path}: {payload}")
        else:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w") as f:
                json.dump(payload, f, indent=2)
            print(f"wrote {out_path}: {objects}")
        created += 1

    print(
        f"done: {created} written, {skipped_existing} already exist, "
        f"{skipped_no_dir} missing processed dir, {failed} failed "
        f"(from {len(hdf5_files)} HDF5 files)"
    )
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
