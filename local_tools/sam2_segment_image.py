#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch

from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2


def color_for_idx(idx):
    rng = np.random.default_rng(idx + 12345)
    return tuple(int(v) for v in rng.integers(40, 256, size=3))


def main():
    parser = argparse.ArgumentParser(description="Run SAM2 automatic segmentation on one image.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model_cfg", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--points_per_side", type=int, default=24)
    parser.add_argument("--points_per_batch", type=int, default=32)
    parser.add_argument("--pred_iou_thresh", type=float, default=0.86)
    parser.add_argument("--stability_score_thresh", type=float, default=0.92)
    parser.add_argument("--min_mask_region_area", type=int, default=200)
    parser.add_argument("--max_masks", type=int, default=80)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    masks_dir = out_dir / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)

    image = Image.open(args.image).convert("RGB")
    image_np = np.array(image)

    print(f"image:  {args.image}")
    print(f"size:   {image.size}")
    print(f"device: {args.device}")
    print(f"out:    {out_dir}")

    model = build_sam2(args.model_cfg, args.checkpoint, device=args.device)
    generator = SAM2AutomaticMaskGenerator(
        model,
        points_per_side=args.points_per_side,
        points_per_batch=args.points_per_batch,
        pred_iou_thresh=args.pred_iou_thresh,
        stability_score_thresh=args.stability_score_thresh,
        min_mask_region_area=args.min_mask_region_area,
        output_mode="binary_mask",
    )

    autocast_ctx = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if str(args.device).startswith("cuda") and torch.cuda.is_available()
        else torch.no_grad()
    )
    with torch.inference_mode(), autocast_ctx:
        anns = generator.generate(image_np)

    anns = sorted(anns, key=lambda x: x["area"], reverse=True)[: args.max_masks]
    print(f"masks:  {len(anns)}")

    overlay = image.convert("RGBA")
    overlay_arr = np.array(overlay).astype(np.float32)
    label = np.zeros(image_np.shape[:2], dtype=np.uint16)
    meta = []

    for idx, ann in enumerate(anns, start=1):
        mask = ann["segmentation"].astype(bool)
        color = color_for_idx(idx)
        alpha = 0.45
        overlay_arr[mask, 0] = overlay_arr[mask, 0] * (1 - alpha) + color[0] * alpha
        overlay_arr[mask, 1] = overlay_arr[mask, 1] * (1 - alpha) + color[1] * alpha
        overlay_arr[mask, 2] = overlay_arr[mask, 2] * (1 - alpha) + color[2] * alpha
        label[np.logical_and(mask, label == 0)] = idx

        Image.fromarray((mask.astype(np.uint8) * 255)).save(masks_dir / f"mask_{idx:03d}.png")
        meta.append(
            {
                "id": idx,
                "area": int(ann["area"]),
                "bbox_xywh": [float(v) for v in ann["bbox"]],
                "predicted_iou": float(ann["predicted_iou"]),
                "stability_score": float(ann["stability_score"]),
                "color_rgb": color,
            }
        )

    overlay_img = Image.fromarray(np.clip(overlay_arr, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(overlay_img)
    for item in meta:
        x, y, w, h = item["bbox_xywh"]
        draw.rectangle([x, y, x + w, y + h], outline=item["color_rgb"] + (255,), width=2)
        draw.text((x, y), str(item["id"]), fill=(255, 255, 255, 255))

    overlay_img.convert("RGB").save(out_dir / "overlay.png")
    Image.fromarray(label).save(out_dir / "label.png")
    with open(out_dir / "masks.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"overlay: {out_dir / 'overlay.png'}")
    print(f"label:   {out_dir / 'label.png'}")
    print(f"meta:    {out_dir / 'masks.json'}")


if __name__ == "__main__":
    main()
