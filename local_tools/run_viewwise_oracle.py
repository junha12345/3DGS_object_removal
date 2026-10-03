"""Diagnose view-specific Gaussian suppression using known masks of that view.

This is an oracle-mask diagnostic, not an evaluation of a learned predictor.
Colors, positions, rotations, scales and learned class logits are frozen.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gaussian_renderer import render
from scene.gaussian_model import GaussianModel
from validate_removal_multiview import PIPE, ATTRIBUTES, camera_from_json, save_rgb


def regions(mask_path, spec, labels, margin):
    mask = np.asarray(Image.open(mask_path))
    mask = cv2.resize(mask, (spec['width'], spec['height']), interpolation=cv2.INTER_NEAREST)
    target = np.isin(mask, labels).astype(np.uint8)
    kernel = np.ones((2 * margin + 1, 2 * margin + 1), dtype=np.uint8)
    protected = 1 - cv2.dilate(target, kernel)
    return mask, target, protected


def contributions(model, camera, target):
    colors = torch.zeros((len(model.get_xyz), 3), device='cuda', requires_grad=True)
    image = render(camera, model, PIPE, torch.zeros(3, device='cuda'),
                   override_color=colors, clamp_output=False)['render']
    weights = torch.tensor(np.stack([target, 1 - target, np.ones_like(target)]),
                           dtype=torch.float32, device='cuda')
    votes = torch.autograd.grad(image, colors, grad_outputs=weights)[0].detach()
    assert torch.isfinite(votes).all() and (votes >= -1e-6).all()
    assert torch.allclose(votes[:, 0] + votes[:, 1], votes[:, 2], atol=3e-3, rtol=1e-4)
    return votes


def image_rgb(model, camera):
    with torch.no_grad():
        image = render(camera, model, PIPE, torch.zeros(3, device='cuda'))['render']
    return image.permute(1, 2, 0).cpu().numpy()


def metrics(reference, edited, target, protected, mask):
    # Compare every method at the same exported 8-bit precision, including
    # cached baseline PNGs. Otherwise baseline quantization appears as damage.
    reference = (reference * 255).round().clip(0, 255) / 255
    edited = (edited * 255).round().clip(0, 255) / 255
    error = np.abs(reference - edited).mean(axis=2)
    target_region, protected_region = target.astype(bool), protected.astype(bool)
    result = dict(
        protected_mae=float(error[protected_region].mean()),
        protected_changed_fraction=float((error[protected_region] > .03).mean()),
        target_change_mae=float(error[target_region].mean()),
        target_changed_fraction=float((error[target_region] > .03).mean()),
        target_pixels=int(target_region.sum()), protected_pixels=int(protected_region.sum()),
    )
    for name, class_id in [('cup', 2), ('table', 4)]:
        region = (mask == class_id) & protected_region
        result[f'{name}_mae'] = float(error[region].mean()) if region.any() else None
    return result


def save_error(reference, edited, path):
    error = np.abs(reference - edited).mean(axis=2)
    scaled = (np.clip(error / .1, 0, 1) * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(scaled, cv2.COLORMAP_INFERNO)
    Image.fromarray(cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)).save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--aligned-mask-dir', type=Path, required=True)
    parser.add_argument('--legacy-mask-dir', type=Path, required=True)
    parser.add_argument('--multiview-ply', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--iteration', type=int, default=15000)
    parser.add_argument('--labels', nargs='+', type=int, default=[1, 3])
    parser.add_argument('--thresholds', nargs='+', type=float, default=[.5, .7, .8, .9, .95, .99])
    parser.add_argument('--margin', type=int, default=8)
    parser.add_argument('--min-contribution', type=float, default=.01)
    parser.add_argument('--frames', nargs='+', default=['frame_000107.jpg', 'frame_000146.jpg',
        'frame_000219.jpg', 'frame_000292.jpg', 'frame_000375.jpg', 'frame_000417.jpg'])
    args = parser.parse_args()
    assert torch.cuda.is_available(), 'CUDA required'
    assets = args.output_dir / 'assets'; assets.mkdir(parents=True, exist_ok=True)
    evidence = args.output_dir / 'evidence'; evidence.mkdir(exist_ok=True)
    specs = json.loads((args.model_dir / 'cameras.json').read_text())
    selected = [next(s for s in specs if s['img_name'] == f) for f in args.frames]
    original_path = args.model_dir / f'point_cloud/iteration_{args.iteration}/point_cloud.ply'
    baseline_paths = {
        'label_only': original_path.with_name('point_cloud_remove_1_3.ply'),
        'global_multiview': args.multiview_ply,
    }
    baseline_counts = {}
    for method, path in baseline_paths.items():
        baseline = GaussianModel(3); baseline.load_ply(str(path))
        for name in ATTRIBUTES:
            getattr(baseline, name).requires_grad_(False)
        baseline_counts[method] = len(baseline.get_xyz)
        for spec in selected:
            save_rgb(image_rgb(baseline, camera_from_json(spec)),
                     assets / f'{Path(spec["img_name"]).stem}_{method}.png')
        del baseline
        torch.cuda.empty_cache()
    model = GaussianModel(3); model.load_ply(str(original_path))
    for name in ATTRIBUTES:
        getattr(model, name).requires_grad_(False)
    original_opacity = model._opacity
    initial_opacity = model.get_opacity.detach()
    initial_xyz = model.get_xyz.detach().clone()
    learned_labels = model.get_object_logits.argmax(dim=1)
    label_candidate = torch.zeros(len(model.get_xyz), device='cuda', dtype=torch.bool)
    for label in args.labels:
        label_candidate |= learned_labels == label
    report = dict(
        experiment='Known per-view SAM2 masks; full-resolution Gaussian contribution and opacity suppression',
        diagnostic_only=True,
        metrics_precision='All methods compared as exported 8-bit RGB / 255',
        limitations=['Masks are SAM2 outputs, not manually verified object-free ground truth.',
                     'Each evaluated view supplies its own mask; this is not novel-view prediction.',
                     'Target image change measures editing strength, not removal accuracy.',
                     'Visibility is measured on the original scene; newly revealed layers are not repeatedly removed.'],
        parameters={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        original_gaussians=len(model.get_xyz), baseline_gaussians=baseline_counts,
        per_view={}, aggregate={},
    )
    for spec in selected:
        stem = Path(spec['img_name']).stem
        camera = camera_from_json(spec)
        reference = image_rgb(model, camera)
        save_rgb(reference, assets / f'{stem}_before.png')
        mask, target, protected = regions(args.aligned_mask_dir / f'{stem}.png', spec, args.labels, args.margin)
        _, legacy_target, _ = regions(args.legacy_mask_dir / f'{stem}.png', spec, args.labels, args.margin)
        Image.fromarray(target * 255).save(assets / f'{stem}_aligned_target.png')
        votes = contributions(model, camera, target)
        total = votes[:, 2]
        ratio = (votes[:, 0] / total.clamp_min(1e-8)).clamp(0, 1)
        eligible = (total >= args.min_contribution) & (votes[:, 0] >= args.min_contribution)
        legacy_votes = contributions(model, camera, legacy_target)
        legacy_ratio = (legacy_votes[:, 0] / legacy_votes[:, 2].clamp_min(1e-8)).clamp(0, 1)
        legacy_eligible = ((legacy_votes[:, 2] >= args.min_contribution) &
                           (legacy_votes[:, 0] >= args.min_contribution))
        soft = ratio * eligible.float()
        gates = {'oracle_soft': soft,
                 'oracle_soft_label_filtered': soft * label_candidate.float(),
                 'oracle_soft_legacy_resize': legacy_ratio * legacy_eligible.float()}
        for threshold in args.thresholds:
            name = 'oracle_hard_' + str(threshold).replace('.', '')
            gates[name] = (eligible & (ratio >= threshold)).float()
        gates['oracle_hard_08_label_filtered'] = (eligible & (ratio >= .8) & label_candidate).float()
        np.savez_compressed(evidence / f'{stem}.npz',
            contributions=votes.cpu().numpy(), target_fraction=ratio.cpu().numpy(),
            eligible=eligible.cpu().numpy(), learned_labels=learned_labels.cpu().numpy())
        row = {}
        for method in baseline_paths:
            result = np.asarray(Image.open(assets / f'{stem}_{method}.png'), dtype=np.float32) / 255
            row[method] = metrics(reference, result, target, protected, mask)
            save_error(reference, result, assets / f'{stem}_{method}_difference.png')
        try:
            for method, gate in gates.items():
                changed_opacity = (initial_opacity * (1 - gate[:, None])).clamp(1e-8, 1 - 1e-8)
                model._opacity = torch.nn.Parameter(torch.logit(changed_opacity), requires_grad=False)
                result = image_rgb(model, camera)
                save_rgb(result, assets / f'{stem}_{method}.png')
                save_error(reference, result, assets / f'{stem}_{method}_difference.png')
                row[method] = metrics(reference, result, target, protected, mask)
                row[method].update(suppressed_gaussians=int((gate > 0).sum()),
                    fully_suppressed_gaussians=int((gate >= 1 - 1e-6).sum()),
                    target_original_weight_remaining_estimate=float(
                        (votes[:, 0] * (1 - gate)).sum() / votes[:, 0].sum().clamp_min(1e-8)))
                print(stem, method,
                      f'protected MAE={row[method]["protected_mae"]:.6f}',
                      f'target change={row[method]["target_change_mae"]:.4f}', flush=True)
        finally:
            model._opacity = original_opacity
        mixed = eligible & (ratio > .1) & (ratio < .9)
        report['per_view'][stem] = dict(methods=row, evidence_summary={
            'visible_gaussians': int((total >= args.min_contribution).sum()),
            'eligible_gaussians': int(eligible.sum()),
            'mixed_gaussians': int(mixed.sum()),
            'target_weight_in_mixed_gaussians_fraction': float(
                votes[mixed, 0].sum() / votes[:, 0].sum().clamp_min(1e-8)),
            'aligned_target_pixels': int(target.sum()),
            'target_changed_pixels_vs_resize': int((target != legacy_target).sum()),
        })
        assert torch.equal(model._opacity, original_opacity)
        assert torch.equal(model.get_xyz, initial_xyz)
        del votes, legacy_votes, gates
        torch.cuda.empty_cache()
        (args.output_dir / 'results.partial.json').write_text(json.dumps(report, indent=2))
    methods = list(report['per_view'][Path(args.frames[0]).stem]['methods'])
    for method in methods:
        keys = ['protected_mae', 'protected_changed_fraction', 'target_change_mae',
                'target_changed_fraction', 'cup_mae', 'table_mae']
        report['aggregate'][method] = {
            key: float(np.mean([v['methods'][method][key] for v in report['per_view'].values()
                                if v['methods'][method][key] is not None]))
            for key in keys}
    (args.output_dir / 'results.json').write_text(json.dumps(report, indent=2))
    (args.output_dir / 'results.partial.json').unlink(missing_ok=True)
    print('Completed:', args.output_dir, flush=True)
    print(json.dumps(report['aggregate'], indent=2), flush=True)


if __name__ == '__main__':
    main()
