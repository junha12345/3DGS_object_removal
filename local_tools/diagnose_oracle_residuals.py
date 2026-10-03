"""Measure which surviving Gaussians contribute after oracle hard suppression."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from run_viewwise_oracle import contributions, regions
from validate_removal_multiview import ATTRIBUTES, camera_from_json
from scene.gaussian_model import GaussianModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment-dir', type=Path, required=True)
    args = parser.parse_args()
    root = args.experiment_dir
    report = json.loads((root / 'results.json').read_text())
    p = report['parameters']
    specs = {s['img_name']: s for s in json.loads((Path(p['model_dir']) / 'cameras.json').read_text())}
    model = GaussianModel(3)
    model.load_ply(str(Path(p['model_dir']) / f'point_cloud/iteration_{p["iteration"]}/point_cloud.ply'))
    for name in ATTRIBUTES:
        getattr(model, name).requires_grad_(False)
    original_opacity = model._opacity
    alpha = model.get_opacity.detach()
    results = {}
    for frame in report['per_view']:
        evidence = np.load(root / 'evidence' / f'{frame}.npz')
        old_votes = torch.from_numpy(evidence['contributions']).cuda()
        ratio = torch.from_numpy(evidence['target_fraction']).cuda()
        eligible = torch.from_numpy(evidence['eligible']).cuda()
        removed = eligible & (ratio >= .5)
        opacity = (alpha * (~removed)[:, None]).clamp(1e-8, 1 - 1e-8)
        model._opacity = torch.nn.Parameter(torch.logit(opacity), requires_grad=False)
        spec = specs[frame + '.jpg']
        _, target, _ = regions(Path(p['aligned_mask_dir']) / f'{frame}.png', spec, p['labels'], p['margin'])
        new_votes = contributions(model, camera_from_json(spec), target)
        initially_invisible = old_votes[:, 2] < p['min_contribution']
        originally_outside = (~initially_invisible) & (ratio < .1)
        originally_mixed = (~initially_invisible) & (ratio >= .1) & (ratio < .5)
        tiny_unselected = (~removed) & (~initially_invisible) & (ratio >= .5)
        denominator = new_votes[:, 0].sum().clamp_min(1e-8)
        labels = model.get_object_logits.argmax(dim=1)
        target_labels = torch.zeros(len(labels), dtype=torch.bool, device='cuda')
        for label in p['labels']:
            target_labels |= labels == label
        row = dict(
            newly_visible_target_weight_fraction=float(new_votes[initially_invisible, 0].sum() / denominator),
            originally_outside_target_weight_fraction=float(new_votes[originally_outside, 0].sum() / denominator),
            originally_mixed_target_weight_fraction=float(new_votes[originally_mixed, 0].sum() / denominator),
            tiny_unselected_target_weight_fraction=float(new_votes[tiny_unselected, 0].sum() / denominator),
            learned_target_label_weight_fraction=float(new_votes[target_labels, 0].sum() / denominator),
            target_total_compositing_weight_before=float(old_votes[:, 0].sum()),
            target_total_compositing_weight_after=float(new_votes[:, 0].sum()),
        )
        assert float(new_votes[removed, 0].sum()) < 1e-3
        results[frame] = row
        print(frame, row, flush=True)
        model._opacity = original_opacity
    output = dict(
        method='Oracle hard threshold .5; contribution remeasured after suppression',
        interpretation='These fractions describe all pixels within the old object mask, including newly exposed legitimate background. They are not object-residual percentages.',
        per_view=results,
        mean={key: float(np.mean([v[key] for v in results.values()])) for key in results[next(iter(results))]},
    )
    (root / 'residual_diagnosis.json').write_text(json.dumps(output, indent=2))
    print('Saved:', root / 'residual_diagnosis.json', flush=True)


if __name__ == '__main__':
    main()
