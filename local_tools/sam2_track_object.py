#!/usr/bin/env python3
import argparse
import contextlib
import os
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from sam2.build_sam import build_sam2_video_predictor


def parse_point(text):
    vals = [float(v) for v in text.split(",")]
    if len(vals) != 3:
        raise argparse.ArgumentTypeError("point must be x,y,label where label is 1 foreground or 0 background")
    x, y, label = vals
    return [x, y], int(label)


def parse_box(text):
    vals = [float(v) for v in text.split(",")]
    if len(vals) != 4:
        raise argparse.ArgumentTypeError("box must be x1,y1,x2,y2")
    return vals


def source_frames(input_dir):
    exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    frames = [p for p in Path(input_dir).iterdir() if p.suffix in exts and p.is_file()]
    return sorted(frames)


def prepare_numeric_jpg_frames(input_dir, work_dir, max_frames=None):
    frames = source_frames(input_dir)
    if max_frames is not None:
        frames = frames[:max_frames]
    if not frames:
        raise RuntimeError(f"no image frames found in {input_dir}")

    out_dir = Path(work_dir) / "frames"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mapping_path = Path(work_dir) / "frame_mapping.tsv"
    with mapping_path.open("w") as f:
        f.write("sam2_index\tsam2_frame\toriginal_frame\n")
        for i, src in enumerate(frames):
            dst = out_dir / f"{i:05d}.jpg"
            if src.suffix.lower() in {".jpg", ".jpeg"}:
                os.symlink(src.resolve(), dst)
            else:
                Image.open(src).convert("RGB").save(dst, quality=95)
            f.write(f"{i}\t{dst.name}\t{src.name}\n")

    return out_dir, frames, mapping_path


def save_mask(mask_tensor, out_path):
    mask = (mask_tensor.detach().cpu().numpy() > 0.0).astype(np.uint8) * 255
    if mask.ndim == 3:
        mask = mask[0]
    Image.fromarray(mask).save(out_path)


def main():
    parser = argparse.ArgumentParser(
        description="Track one object through 3DGS/COLMAP frames with SAM2 and save per-frame masks."
    )
    parser.add_argument("--input_dir", required=True, help="Frame directory, e.g. data/scene/images")
    parser.add_argument("--output_dir", required=True, help="Output directory for masks")
    parser.add_argument("--work_dir", default=None, help="Temporary numeric-frame directory; default: <output_dir>/_sam2_work")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model_cfg", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument("--prompt_frame", type=int, default=0, help="SAM2 frame index after sorting input frames, zero-based")
    parser.add_argument("--obj_id", type=int, default=1)
    parser.add_argument("--box", type=parse_box, default=None, help="x1,y1,x2,y2 in original image pixel coordinates")
    parser.add_argument("--point", type=parse_point, action="append", default=[], help="x,y,label; label 1=foreground, 0=background. Can repeat.")
    parser.add_argument("--reverse", action="store_true", help="Also propagate backward from prompt_frame")
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--offload_video_to_cpu", action="store_true")
    parser.add_argument("--offload_state_to_cpu", action="store_true")
    args = parser.parse_args()

    if args.box is None and not args.point:
        raise SystemExit("Provide at least --box x1,y1,x2,y2 or --point x,y,label")

    out_dir = Path(args.output_dir)
    mask_dir = out_dir / f"object_{args.obj_id:03d}"
    mask_dir.mkdir(parents=True, exist_ok=True)
    work_dir = Path(args.work_dir) if args.work_dir else out_dir / "_sam2_work"
    work_dir.mkdir(parents=True, exist_ok=True)

    frames_dir, frames, mapping_path = prepare_numeric_jpg_frames(args.input_dir, work_dir, args.max_frames)
    if not (0 <= args.prompt_frame < len(frames)):
        raise SystemExit(f"--prompt_frame must be in [0, {len(frames)-1}], got {args.prompt_frame}")

    print(f"input frames: {len(frames)}")
    print(f"sam2 frames:  {frames_dir}")
    print(f"mapping:      {mapping_path}")
    print(f"masks:        {mask_dir}")
    print(f"device:       {args.device}")

    predictor = build_sam2_video_predictor(args.model_cfg, args.checkpoint, device=args.device)
    state = predictor.init_state(
        video_path=str(frames_dir),
        offload_video_to_cpu=args.offload_video_to_cpu,
        offload_state_to_cpu=args.offload_state_to_cpu,
    )

    points = None
    labels = None
    if args.point:
        points = np.array([p for p, _ in args.point], dtype=np.float32)
        labels = np.array([label for _, label in args.point], dtype=np.int32)

    autocast_ctx = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if str(args.device).startswith("cuda") and torch.cuda.is_available()
        else contextlib.nullcontext()
    )
    with torch.inference_mode(), autocast_ctx:
        predictor.add_new_points_or_box(
            state,
            frame_idx=args.prompt_frame,
            obj_id=args.obj_id,
            points=points,
            labels=labels,
            box=args.box,
        )

        seen = set()
        for frame_idx, obj_ids, masks in predictor.propagate_in_video(state, start_frame_idx=args.prompt_frame):
            obj_pos = list(obj_ids).index(args.obj_id)
            save_mask(masks[obj_pos], mask_dir / f"{frame_idx:05d}.png")
            seen.add(frame_idx)

        if args.reverse and args.prompt_frame > 0:
            for frame_idx, obj_ids, masks in predictor.propagate_in_video(
                state, start_frame_idx=args.prompt_frame, reverse=True
            ):
                obj_pos = list(obj_ids).index(args.obj_id)
                save_mask(masks[obj_pos], mask_dir / f"{frame_idx:05d}.png")
                seen.add(frame_idx)

    print(f"saved masks:  {len(seen)} / {len(frames)}")
    print("done")


if __name__ == "__main__":
    main()
