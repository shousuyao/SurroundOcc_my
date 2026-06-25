#!/usr/bin/env python3
"""Apply conservative flat-class height filling to dense candidate files."""

import argparse
import json
from pathlib import Path

import numpy as np

from surface_completion import fill_flat_height_holes


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--stats-output", required=True)
    parser.add_argument("--occ-size", nargs=3, type=int, default=(200, 200, 16))
    parser.add_argument("--radius", type=int, default=1)
    parser.add_argument("--min-neighbors", type=int, default=5)
    parser.add_argument("--max-z-spread", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(input_dir.glob("*.npy"))
    if not files:
        raise FileNotFoundError("No candidate .npy files under {}".format(input_dir))

    file_stats = []
    for path in files:
        voxels = np.load(path)
        filled, stats = fill_flat_height_holes(
            voxels,
            occ_size=np.asarray(args.occ_size, dtype=np.int64),
            radius=args.radius,
            min_neighbors=args.min_neighbors,
            max_z_spread=args.max_z_spread,
            flat_labels=(11, 12, 13, 14),
        )
        np.save(output_dir / path.name, filled)
        file_stats.append({
            "file": path.name,
            "input_voxels": int(voxels.shape[0]),
            "output_voxels": int(filled.shape[0]),
            **stats,
        })

    report = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "files": len(file_stats),
        "added_total": int(sum(item["added_total"] for item in file_stats)),
        "file_stats": file_stats,
    }
    stats_output = Path(args.stats_output)
    stats_output.parent.mkdir(parents=True, exist_ok=True)
    with stats_output.open("w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({"files": report["files"], "added_total": report["added_total"]}, indent=2))


if __name__ == "__main__":
    main()
