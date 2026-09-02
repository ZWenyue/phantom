#!/usr/bin/env python3
"""Generate narration.csv for LeRobot export from EgoDex HDF5 metadata.

Reads language instructions from HDF5 file attributes (llm_description, then
description) and writes narration.csv into each processed episode directory.

Example:
    python b/generate_narration_csv.py
    python b/generate_narration_csv.py --task basic_pick_place --dry-run
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

try:
    import h5py
except ImportError as exc:
    raise SystemExit(f"h5py is required: {exc}") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate narration.csv files from EgoDex HDF5 attributes",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="basic_pick_place",
        help="EgoDex task name (default: basic_pick_place)",
    )
    parser.add_argument(
        "--egodex-root",
        type=Path,
        default=Path("/home/a26160/DATA/test"),
        help="Root directory containing task HDF5 files",
    )
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=Path("/home/a26160/DATA/test_phantom_processed"),
        help="Root directory of Phantom processed demos",
    )
    parser.add_argument(
        "--field",
        type=str,
        default="llm_description",
        choices=("llm_description", "description"),
        help="HDF5 attribute to use as narration text (default: llm_description)",
    )
    parser.add_argument("--demo", type=int, default=None, help="Export a single episode id")
    parser.add_argument(
        "--max-demos",
        type=int,
        default=None,
        help="Process at most N demos (numeric id order; default: all)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing narration.csv files",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions without writing files",
    )
    return parser.parse_args()


def read_narration_from_hdf5(hdf5_path: Path, field: str) -> str:
    with h5py.File(hdf5_path, "r") as f:
        attrs = dict(f.attrs)
    text = str(attrs.get(field, "")).strip()
    if not text or text.lower() == "none":
        text = str(attrs.get("description", "")).strip()
    if not text:
        raise ValueError(f"no narration text in {hdf5_path}")
    return text


def write_narration_csv(path: Path, narration: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["key", "value"])
        writer.writerow(["narration", narration])


def main() -> None:
    args = parse_args()

    task_dir = args.egodex_root / args.task
    demo_root = args.processed_root / f"egodex_{args.task}"

    if not task_dir.is_dir():
        raise SystemExit(f"Error: task directory not found: {task_dir}")
    if not demo_root.is_dir():
        if args.demo is not None:
            demo_root.mkdir(parents=True, exist_ok=True)
        else:
            raise SystemExit(f"Error: processed directory not found: {demo_root}")

    hdf5_files = sorted(
        task_dir.glob("*.hdf5"),
        key=lambda p: int(p.stem),
    )
    if args.demo is not None:
        hdf5_files = [task_dir / f"{args.demo}.hdf5"]
        if not hdf5_files[0].is_file():
            raise SystemExit(f"Error: missing {hdf5_files[0]}")
    elif args.max_demos is not None:
        if args.max_demos <= 0:
            raise SystemExit("--max-demos must be a positive integer")
        hdf5_files = hdf5_files[: args.max_demos]
    if not hdf5_files:
        raise SystemExit(f"Error: no HDF5 files found in {task_dir}")

    created = 0
    skipped_existing = 0
    skipped_no_processed = 0
    failed = 0

    for hdf5_path in hdf5_files:
        episode = hdf5_path.stem
        out_path = demo_root / episode / "narration.csv"

        if not out_path.parent.is_dir():
            if args.demo is not None:
                out_path.parent.mkdir(parents=True, exist_ok=True)
            else:
                print(f"skip {episode}: processed dir not found ({out_path.parent})")
                skipped_no_processed += 1
                continue

        if out_path.exists() and not args.overwrite:
            skipped_existing += 1
            continue

        try:
            narration = read_narration_from_hdf5(hdf5_path, args.field)
        except ValueError as exc:
            print(f"skip {episode}: {exc}")
            failed += 1
            continue

        if args.dry_run:
            print(f"would write {out_path}: {narration[:80]}{'...' if len(narration) > 80 else ''}")
        else:
            write_narration_csv(out_path, narration)
            print(f"wrote {out_path}")

        created += 1

    print(
        f"done: {created} written, {skipped_existing} already exist, "
        f"{skipped_no_processed} missing processed dir, {failed} failed "
        f"(from {len(hdf5_files)} HDF5 files)"
    )
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
