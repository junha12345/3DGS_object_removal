#!/usr/bin/env python3
import argparse
import re
from pathlib import Path

import numpy as np


PLY_TYPES = {
    "char": "i1",
    "int8": "i1",
    "uchar": "u1",
    "uint8": "u1",
    "short": "<i2",
    "int16": "<i2",
    "ushort": "<u2",
    "uint16": "<u2",
    "int": "<i4",
    "int32": "<i4",
    "uint": "<u4",
    "uint32": "<u4",
    "float": "<f4",
    "float32": "<f4",
    "double": "<f8",
    "float64": "<f8",
}


def read_header(path: Path):
    with path.open("rb") as f:
        data = f.read()
    marker = b"end_header\n"
    end = data.find(marker)
    if end < 0:
        # Some PLYs may use CRLF.
        marker = b"end_header\r\n"
        end = data.find(marker)
    if end < 0:
        raise ValueError("Could not find PLY end_header")
    header_end = end + len(marker)
    header = data[:header_end].decode("ascii", errors="strict")
    body = data[header_end:]
    return header, body


def parse_vertex_layout(header: str):
    lines = header.splitlines()
    if not lines or lines[0].strip() != "ply":
        raise ValueError("Not a PLY file")

    fmt = None
    vertex_count = None
    vertex_props = []
    current_element = None

    for line in lines:
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "format":
            fmt = parts[1]
        elif parts[0] == "element":
            current_element = parts[1]
            if current_element == "vertex":
                vertex_count = int(parts[2])
        elif parts[0] == "property" and current_element == "vertex":
            if parts[1] == "list":
                raise ValueError("List properties in vertex are not supported")
            prop_type, prop_name = parts[1], parts[2]
            if prop_type not in PLY_TYPES:
                raise ValueError(f"Unsupported PLY property type: {prop_type}")
            vertex_props.append((prop_name, PLY_TYPES[prop_type]))

    if fmt != "binary_little_endian":
        raise ValueError(f"Only binary_little_endian PLY is supported, got: {fmt}")
    if vertex_count is None:
        raise ValueError("No vertex element found")
    if not vertex_props:
        raise ValueError("No vertex properties found")

    dtype = np.dtype(vertex_props)
    return lines, vertex_count, dtype


def replace_vertex_count(lines, new_count: int):
    out = []
    replaced = False
    for line in lines:
        if line.startswith("element vertex "):
            out.append(f"element vertex {new_count}")
            replaced = True
        else:
            out.append(line)
    if not replaced:
        raise ValueError("Could not replace vertex count in header")
    return ("\n".join(out) + "\n").encode("ascii")


def main():
    parser = argparse.ArgumentParser(
        description="Filter a 3DGS binary PLY using per-Gaussian object_logit_* argmax."
    )
    parser.add_argument("--input", required=True, help="Input point_cloud.ply")
    parser.add_argument("--output", required=True, help="Output filtered point_cloud.ply")
    parser.add_argument(
        "--remove-label",
        type=int,
        action="append",
        default=[],
        help=(
            "Remove vertices whose argmax(object_logit_*) equals this label. "
            "Can be repeated, e.g. --remove-label 1 --remove-label 2."
        ),
    )
    parser.add_argument(
        "--keep-label",
        type=int,
        action="append",
        default=[],
        help=(
            "Keep only vertices whose argmax(object_logit_*) equals this label. "
            "Can be repeated. Mutually exclusive with --remove-label."
        ),
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=None,
        help=(
            "Optional softmax confidence threshold. With --remove-label, remove only "
            "when confidence >= threshold. With --keep-label, keep only when confidence >= threshold."
        ),
    )
    args = parser.parse_args()
    if bool(args.remove_label) == bool(args.keep_label):
        raise ValueError("Specify exactly one of --remove-label or --keep-label")

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    header, body = read_header(input_path)
    lines, vertex_count, dtype = parse_vertex_layout(header)

    expected_bytes = vertex_count * dtype.itemsize
    if len(body) < expected_bytes:
        raise ValueError(
            f"PLY body is too small: expected at least {expected_bytes} bytes, got {len(body)}"
        )
    if len(body) > expected_bytes:
        raise ValueError(
            "This script currently expects vertex-only PLY files. "
            f"Found {len(body) - expected_bytes} trailing bytes after vertices."
        )

    vertices = np.frombuffer(body, dtype=dtype, count=vertex_count).copy()

    logit_names = sorted(
        [name for name in dtype.names if re.fullmatch(r"object_logit_\d+", name)],
        key=lambda x: int(x.rsplit("_", 1)[1]),
    )
    if not logit_names:
        raise ValueError("No object_logit_* fields found in PLY")
    selected_labels = args.remove_label or args.keep_label
    for label in selected_labels:
        if label < 0 or label >= len(logit_names):
            raise ValueError(
                f"labels must be in [0, {len(logit_names) - 1}], got {label}"
            )

    logits = np.stack([vertices[name] for name in logit_names], axis=1)
    labels = np.argmax(logits, axis=1)
    selected = np.isin(labels, selected_labels)

    if args.min_confidence is not None:
        stable = logits - logits.max(axis=1, keepdims=True)
        probs = np.exp(stable)
        probs /= probs.sum(axis=1, keepdims=True)
        confidence = probs[np.arange(len(labels)), labels]
        selected &= confidence >= args.min_confidence

    if args.keep_label:
        keep = selected
        removed = ~keep
        action = "Kept labels"
    else:
        removed = selected
        keep = ~removed
        action = "Removed labels"
    filtered = vertices[keep]

    new_header = replace_vertex_count(lines, len(filtered))
    with output_path.open("wb") as f:
        f.write(new_header)
        f.write(filtered.tobytes())

    counts = np.bincount(labels, minlength=len(logit_names))
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Object logit fields: {', '.join(logit_names)}")
    print(f"Original vertices: {len(vertices)}")
    print("Argmax counts:")
    for idx, count in enumerate(counts):
        print(f"  {idx}: {int(count)}")
    print(f"{action} {selected_labels}: {int(selected.sum())}")
    print(f"Removed vertices: {int(removed.sum())}")
    print(f"Kept vertices: {len(filtered)}")


if __name__ == "__main__":
    main()
