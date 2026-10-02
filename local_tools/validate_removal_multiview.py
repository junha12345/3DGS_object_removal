"""Validate Gaussian removal using visibility-weighted evidence from 2D masks.

The derivative of rendered RGB w.r.t. each Gaussian's override color is its
alpha-compositing contribution. Masked RGB gradients therefore accumulate
foreground and protected-region evidence without modifying the rasterizer or
retraining the scene. Geometry, color, opacity, and semantic logits stay fixed.
"""
import argparse
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from gaussian_renderer import render
from scene.cameras import MiniCam
from scene.gaussian_model import GaussianModel
from utils.graphics_utils import focal2fov, getProjectionMatrix

PIPE = SimpleNamespace(compute_cov3D_python=False, convert_SHs_python=False,
                       debug=False, antialiasing=False)
ATTRIBUTES = ['_xyz', '_features_dc', '_features_rest', '_opacity',
              '_scaling', '_rotation', '_object_logits']


def camera_from_json(spec, width=None):
    scale = min(1.0, width / spec['width']) if width else 1.0
    w, h = round(spec['width'] * scale), round(spec['height'] * scale)
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3], c2w[:3, 3] = spec['rotation'], spec['position']
    view = torch.tensor(np.linalg.inv(c2w), device='cuda').T.contiguous()
    fovx = focal2fov(spec['fx'], spec['width'])
    fovy = focal2fov(spec['fy'], spec['height'])
    proj = getProjectionMatrix(0.01, 100.0, fovx, fovy).T.cuda()
    return MiniCam(w, h, fovy, fovx, 0.01, 100.0, view, view @ proj)


def mask_regions(mask_dir, spec, camera, labels, margin):
    # Match the existing training loader: nearest-neighbor resize to the camera.
    path = mask_dir / Path(spec['img_name']).with_suffix('.png').name
    mask = np.asarray(Image.open(path))
    mask = cv2.resize(mask, (camera.image_width, camera.image_height),
                      interpolation=cv2.INTER_NEAREST)
    target = np.isin(mask, labels).astype(np.uint8)
    radius = max(1, round(margin * camera.image_width / spec['width']))
    kernel = np.ones((2 * radius + 1, 2 * radius + 1), np.uint8)
    interior = cv2.erode(target, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)
    exterior = 1 - cv2.dilate(target, kernel)
    return target, interior, exterior


def contribution_votes(model, spec, mask_dir, labels, width, margin):
    camera = camera_from_json(spec, width)
    target, inside, outside = mask_regions(mask_dir, spec, camera, labels, margin)
    colors = torch.zeros((len(model.get_xyz), 3), device='cuda', requires_grad=True)
    image = render(camera, model, PIPE, torch.zeros(3, device='cuda'),
                   override_color=colors, clamp_output=False)['render']
    weights = torch.tensor(np.stack([inside, outside, np.ones_like(target)]),
                           dtype=torch.float32, device='cuda')
    votes = torch.autograd.grad(image, colors, grad_outputs=weights)[0].detach()
    assert torch.isfinite(votes).all(), 'Non-finite rendering contribution'
    assert (votes >= -1e-6).all(), 'Negative rendering contribution'
    assert (votes[:, 0] + votes[:, 1] <= votes[:, 2] + 2e-3).all(), 'Masks must be disjoint'
    return votes


def subset(model, keep):
    result = GaussianModel(model.max_sh_degree)
    result.active_sh_degree = model.active_sh_degree
    for name in ATTRIBUTES:
        setattr(result, name, torch.nn.Parameter(getattr(model, name)[keep].detach(),
                                                 requires_grad=False))
    return result


def rgb(model, spec):
    with torch.no_grad():
        camera = camera_from_json(spec)
        image = render(camera, model, PIPE, torch.zeros(3, device='cuda'))['render']
    return image.permute(1, 2, 0).cpu().numpy()


def save_rgb(image, path):
    Image.fromarray((image * 255).round().clip(0, 255).astype(np.uint8)).save(path)


def region_metrics(reference, edited, target, protected):
    error = np.abs(reference - edited).mean(axis=2)
    inside = target.astype(bool)
    outside = protected.astype(bool)
    return dict(protected_mae=float(error[outside].mean()),
                protected_changed_fraction=float((error[outside] > 0.03).mean()),
                target_change_mae=float(error[inside].mean()),
                protected_pixels=int(outside.sum()), target_pixels=int(inside.sum()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--mask-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--iteration', type=int, default=15000)
    parser.add_argument('--labels', type=int, nargs='+', default=[1, 3])
    parser.add_argument('--views', type=int, default=48)
    parser.add_argument('--vote-width', type=int, default=640)
    parser.add_argument('--margin', type=int, default=8)
    parser.add_argument('--support-threshold', type=float, default=0.8)
    parser.add_argument('--min-positive-views', type=int, default=3)
    parser.add_argument('--max-negative-fraction', type=float, default=0.1)
    parser.add_argument('--eval-frames', nargs='+', default=[
        'frame_000107.jpg', 'frame_000146.jpg', 'frame_000219.jpg',
        'frame_000292.jpg', 'frame_000375.jpg', 'frame_000417.jpg'])
    args = parser.parse_args()
    assert torch.cuda.is_available(), 'CUDA required'
    args.output_dir.mkdir(parents=True, exist_ok=True)
    assets = args.output_dir / 'assets'
    assets.mkdir(exist_ok=True)
    specs = json.loads((args.model_dir / 'cameras.json').read_text())
    heldout = set(args.eval_frames)
    candidates = [s for s in specs if s['img_name'] not in heldout]
    indices = np.linspace(0, len(candidates) - 1, args.views, dtype=int)
    views = [candidates[i] for i in indices]
    assert len({s['img_name'] for s in views}) == args.views
    assert not heldout.intersection(s['img_name'] for s in views)
    original = args.model_dir / f'point_cloud/iteration_{args.iteration}/point_cloud.ply'
    model = GaussianModel(3)
    model.load_ply(str(original))
    for name in ATTRIBUTES:
        getattr(model, name).requires_grad_(False)
    n = len(model.get_xyz)
    sums = torch.zeros((n, 3), device='cuda')
    visible = torch.zeros(n, dtype=torch.int32, device='cuda')
    positive = visible.clone()
    negative = visible.clone()
    for i, spec in enumerate(views):
        votes = contribution_votes(model, spec, args.mask_dir, args.labels,
                                   args.vote_width, args.margin)
        observed = votes[:, 2] >= 0.01
        reliable = votes[:, 0] + votes[:, 1]
        ratio = votes[:, 0] / reliable.clamp_min(1e-8)
        sums += votes
        visible += observed.int()
        positive += (observed & (ratio >= 0.7) & (reliable >= 0.01)).int()
        negative += (observed & (ratio <= 0.3) & (reliable >= 0.01)).int()
        if (i + 1) % 4 == 0:
            print(f'Voting {i + 1}/{len(views)}: {spec["img_name"]}', flush=True)
    support = sums[:, 0] / (sums[:, 0] + sums[:, 1]).clamp_min(1e-8)
    negative_fraction = negative.float() / visible.clamp_min(1)
    mask_verified = ((support >= args.support_threshold) &
                     (positive >= args.min_positive_views) &
                     (negative_fraction <= args.max_negative_fraction))
    class_label = model.get_object_logits.argmax(dim=1)
    label_candidate = torch.zeros(n, dtype=torch.bool, device='cuda')
    for label in args.labels:
        label_candidate |= class_label == label
    selections = dict(label_only=label_candidate,
                      multiview=label_candidate & mask_verified,
                      mask_only_diagnostic=mask_verified)
    np.savez_compressed(args.output_dir / 'evidence.npz',
                        contribution=sums.cpu().numpy(), support=support.cpu().numpy(),
                        visible_views=visible.cpu().numpy(), positive_views=positive.cpu().numpy(),
                        negative_views=negative.cpu().numpy(),
                        label_candidate=label_candidate.cpu().numpy(),
                        verified=mask_verified.cpu().numpy())
    metadata = dict(method='Visibility-aware masked color gradients; frozen scene',
                    parameters={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                    vote_frames=[s['img_name'] for s in views],
                    evaluation_frames=args.eval_frames, original_gaussians=n,
                    deleted_gaussians={}, per_view_metrics={}, aggregate_metrics={})
    evaluation = [next(s for s in specs if s['img_name'] == name) for name in args.eval_frames]
    references = {}
    masks = {}
    for spec in evaluation:
        stem = Path(spec['img_name']).stem
        reference = rgb(model, spec)
        save_rgb(reference, assets / f'{stem}_before.png')
        references[stem] = reference
        camera = camera_from_json(spec)
        target, interior, protected = mask_regions(args.mask_dir, spec, camera,
                                                   args.labels, args.margin)
        masks[stem] = (target, protected)
        Image.fromarray(target * 255).save(assets / f'{stem}_target_mask.png')
    for method, remove in selections.items():
        count = int(remove.sum())
        metadata['deleted_gaussians'][method] = count
        print(f'{method}: removing {count:,}/{n:,} Gaussians', flush=True)
        assert count > 0, f'No selected Gaussians: {method}'
        edited = subset(model, ~remove)
        if method == 'multiview':
            edited.save_ply(str(args.output_dir / 'point_cloud_multiview.ply'))
        metrics = {}
        for spec in evaluation:
            stem = Path(spec['img_name']).stem
            result = rgb(edited, spec)
            save_rgb(result, assets / f'{stem}_{method}.png')
            target, protected = masks[stem]
            metrics[stem] = region_metrics(references[stem], result, target, protected)
        metadata['per_view_metrics'][method] = metrics
        metadata['aggregate_metrics'][method] = {
            key: float(np.mean([v[key] for v in metrics.values()]))
            for key in ['protected_mae', 'protected_changed_fraction', 'target_change_mae']}
        del edited
        torch.cuda.empty_cache()
    (args.output_dir / 'results.json').write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata['aggregate_metrics'], indent=2), flush=True)


if __name__ == '__main__':
    main()
