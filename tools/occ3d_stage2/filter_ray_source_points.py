#!/usr/bin/env python3
"""Filter per-point ray source files by semantic labels."""

import argparse
import json
from pathlib import Path

import numpy as np


def parse_labels(text):
    if not text:
        return set()
    return {int(item.strip()) for item in text.split(",") if item.strip()}


def parse_args():
    parser = argparse.ArgumentParser(description="Filter ray_source_points_with_origin .npy files.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--exclude-labels",
        default="",
        help="Comma-separated semantic labels to remove from free-ray source.",
    )
    parser.add_argument(
        "--include-labels",
        default="",
        help="Optional comma-separated whitelist. Applied before exclusions.",
    )
    parser.add_argument("--summary-output", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    include_labels = parse_labels(args.include_labels)
    exclude_labels = parse_labels(args.exclude_labels)
    files = sorted(input_dir.glob("*.npy"))
    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "include_labels": sorted(include_labels),
        "exclude_labels": sorted(exclude_labels),
        "num_files": len(files),
        "input_rows": 0,
        "output_rows": 0,
        "removed_rows": 0,
        "label_counts_input": {},
        "label_counts_output": {},
        "files": [],
    }

    for path in files:
        data = np.load(path)
        if data.ndim != 2 or data.shape[1] != 7:
            raise ValueError(f"Expected shape (N, 7), got {data.shape}: {path}")
        labels = data[:, 6].astype(np.int64)
        keep = np.ones(labels.shape[0], dtype=bool)
        if include_labels:
            keep &= np.isin(labels, np.asarray(sorted(include_labels), dtype=np.int64))
        if exclude_labels:
            keep &= ~np.isin(labels, np.asarray(sorted(exclude_labels), dtype=np.int64))

        filtered = data[keep]
        np.save(output_dir / path.name, filtered)

        unique_in, counts_in = np.unique(labels, return_counts=True)
        unique_out, counts_out = np.unique(filtered[:, 6].astype(np.int64), return_counts=True) if filtered.size else ([], [])
        for label, count in zip(unique_in, counts_in):
            key = str(int(label))
            summary["label_counts_input"][key] = summary["label_counts_input"].get(key, 0) + int(count)
        for label, count in zip(unique_out, counts_out):
            key = str(int(label))
            summary["label_counts_output"][key] = summary["label_counts_output"].get(key, 0) + int(count)

        file_summary = {
            "file": path.name,
            "input_rows": int(data.shape[0]),
            "output_rows": int(filtered.shape[0]),
            "removed_rows": int(data.shape[0] - filtered.shape[0]),
        }
        summary["files"].append(file_summary)
        summary["input_rows"] += file_summary["input_rows"]
        summary["output_rows"] += file_summary["output_rows"]
        summary["removed_rows"] += file_summary["removed_rows"]

    summary_output = Path(args.summary_output)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_output, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: summary[k] for k in ["num_files", "input_rows", "output_rows", "removed_rows"]}, indent=2))


if __name__ == "__main__":
    main()
