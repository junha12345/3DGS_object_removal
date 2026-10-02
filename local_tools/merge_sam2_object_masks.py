#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def read_mapping(mapping_path):
    rows = []
    with open(mapping_path, "r") as f:
        header = next(f).strip().split("\t")
        idx_i = header.index("sam2_index")
        orig_i = header.index("original_frame")
        for line in f:
            parts = line.strip().split("\t")
            if not parts or len(parts) <= max(idx_i, orig_i):
                continue
            rows.append((int(parts[idx_i]), parts[orig_i]))
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Merge SAM2 binary object masks into class-index PNG masks for object-aware 3DGS training."
    )
    parser.add_argument("--sam2_mask_root", required=True, help="Directory containing object_001/, object_002/, ... and _sam2_work/frame_mapping.tsv")
    parser.add_argument("--output_dir", required=True, help="Output class-index mask directory")
    parser.add_argument("--mapping", default=None, help="Optional frame_mapping.tsv path")
    parser.add_argument("--object", action="append", default=[], help="object_dir:class_id, e.g. object_001:1. Can repeat. Default: all object_* dirs sorted with numeric class ids.")
    parser.add_argument("--threshold", type=int, default=127)
    args = parser.parse_args()

    root = Path(args.sam2_mask_root)
    mapping_path = Path(args.mapping) if args.mapping else root / "_sam2_work" / "frame_mapping.tsv"
    if not mapping_path.exists():
        raise SystemExit(f"mapping file not found: {mapping_path}")

    if args.object:
        object_specs = []
        for spec in args.object:
            name, class_id = spec.split(":")
            object_specs.append((root / name, int(class_id)))
    else:
        object_dirs = sorted([p for p in root.iterdir() if p.is_dir() and p.name.startswith("object_")])
        object_specs = [(p, idx + 1) for idx, p in enumerate(object_dirs)]

    if not object_specs:
        raise SystemExit(f"no object mask directories found in {root}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = read_mapping(mapping_path)

    written = 0
    for sam2_idx, original_name in rows:
        merged = None
        for obj_dir, class_id in object_specs:
            mask_path = obj_dir / f"{sam2_idx:05d}.png"
            if not mask_path.exists():
                continue
            mask = np.array(Image.open(mask_path).convert("L"))
            if merged is None:
                merged = np.zeros(mask.shape, dtype=np.uint8)
            merged[mask > args.threshold] = class_id

        if merged is None:
            continue
        out_name = Path(original_name).with_suffix(".png").name
        Image.fromarray(merged).save(out_dir / out_name)
        written += 1

    print(f"objects: {[(p.name, cid) for p, cid in object_specs]}")
    print(f"written class masks: {written}")
    print(f"output: {out_dir}")


if __name__ == "__main__":
    main()
