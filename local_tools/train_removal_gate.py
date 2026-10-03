"""Learn removal gates with one mask MSE, then a bounded angular correction.

The original Gaussian scene and class logits remain frozen. Mask projection
uses the ORIGINAL compositing weights, not weights after opacity suppression.
Edited RGB is an evaluation output, not a training target. Class 0 is ignored
by default. The target-complement control instead assumes SAM target masks are
complete and supervises their safe exterior as non-target (not a class label).
"""
import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gaussian_renderer import render
from scene.gaussian_model import GaussianModel
from validate_removal_multiview import ATTRIBUTES, PIPE, camera_from_json, save_rgb
from run_viewwise_oracle import metrics, regions, save_error

DEFAULT_TEST = [107, 146, 219, 292, 375, 417]


def file_hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    temp.replace(path)


def load_scene(path):
    model = GaussianModel(3)
    model.load_ply(str(path))
    for name in ATTRIBUTES:
        getattr(model, name).requires_grad_(False)
    return model


def gates(model, score, coefficients=None, camera=None, bound=1.):
    if coefficients is None:
        return score.sigmoid()
    direction = torch.nn.functional.normalize(camera.camera_center[None] - model.get_xyz, dim=1)
    correction = bound * torch.tanh((coefficients * direction).sum(dim=1))
    return (score + correction).sigmoid()


def score_projection(model, camera, q, coverage):
    colors = torch.stack([q, torch.zeros_like(q), torch.zeros_like(q)], dim=1)
    image = render(camera, model, PIPE, torch.zeros(3, device='cuda'),
                   override_color=colors, clamp_output=False)['render'][0]
    return image / coverage.clamp_min(1e-6)


def prepare_view(model, spec, args):
    camera = camera_from_json(spec, args.width)
    path = args.mask_dir / Path(spec['img_name']).with_suffix('.png').name
    mask = cv2.resize(np.asarray(Image.open(path)),
                      (camera.image_width, camera.image_height), interpolation=cv2.INTER_NEAREST)
    radius = max(1, round(args.margin * camera.image_width / spec['width']))
    kernel = np.ones((2 * radius + 1, 2 * radius + 1), np.uint8)
    target = np.isin(mask, args.labels).astype(np.uint8)
    retained = (1 - target) if args.negative_region == 'target-complement' else np.isin(mask, args.retain_labels).astype(np.uint8)
    inside = cv2.erode(target, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)
    outside = cv2.erode(retained, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)
    with torch.no_grad():
        coverage = render(camera, model, PIPE, torch.zeros(3, device='cuda'),
                          override_color=torch.ones((len(model.get_xyz), 3), device='cuda'),
                          clamp_output=False)['render'][0].detach()
    visible = coverage.cpu().numpy() > .05
    pos = np.flatnonzero((inside > 0) & visible)
    neg = np.flatnonzero((outside > 0) & visible)
    assert len(pos) > 0 and len(neg) > 0, (spec['img_name'], len(pos), len(neg))
    return dict(spec=spec, camera=camera, coverage=coverage,
                positive=torch.tensor(pos, device='cuda'), negative=torch.tensor(neg, device='cuda'))


def mask_loss(image, view, sample=0):
    flat = image.reshape(-1)
    pos, neg = view['positive'], view['negative']
    if sample:
        # Independent equal-sized samples from the two reliable regions.
        pos = pos[torch.randint(len(pos), (sample,), device='cuda')]
        neg = neg[torch.randint(len(neg), (sample,), device='cuda')]
    return .5 * ((flat[pos] - 1).square().mean() + flat[neg].square().mean())


@torch.no_grad()
def validate(model, views, score, coefficients, bound):
    rows = []
    for view in views:
        q = gates(model, score, coefficients, view['camera'], bound)
        image = score_projection(model, view['camera'], q, view['coverage'])
        flat = image.reshape(-1)
        p, n = flat[view['positive']], flat[view['negative']]
        rows.append(dict(frame=view['spec']['img_name'], loss=float(mask_loss(image, view)),
                         target_mean=float(p.mean()), retained_mean=float(n.mean()),
                         target_recall=float((p >= .5).float().mean()),
                         retained_false_positive=float((n >= .5).float().mean())))
    return dict(mean_loss=float(np.mean([r['loss'] for r in rows])),
                target_mean=float(np.mean([r['target_mean'] for r in rows])),
                retained_mean=float(np.mean([r['retained_mean'] for r in rows])),
                target_recall=float(np.mean([r['target_recall'] for r in rows])),
                retained_false_positive=float(np.mean([r['retained_false_positive'] for r in rows])),
                per_view=rows)


def train_stage(model, train_views, validation_views, score, coefficients, args, stage, history):
    is_direction = coefficients is not None
    parameter = coefficients if is_direction else score
    frozen_score = score.detach().clone() if is_direction else None
    epochs = args.direction_epochs if is_direction else args.common_epochs
    lr = args.direction_lr if is_direction else args.common_lr
    optimizer = torch.optim.Adam([parameter], lr=lr)
    initial = validate(model, validation_views, score, coefficients, args.bound)
    best_loss, best_epoch = initial['mean_loss'], 0
    best = parameter.detach().clone()
    history.append(dict(stage=stage, epoch=0, training_loss=None, validation=initial))
    print(f'{stage}: initial validation MSE {best_loss:.6f}', flush=True)
    for epoch in range(1, epochs + 1):
        order = np.random.permutation(len(train_views))
        losses = []
        started = time.monotonic()
        for index in order:
            view = train_views[index]
            optimizer.zero_grad(set_to_none=True)
            q = gates(model, score, coefficients, view['camera'], args.bound)
            image = score_projection(model, view['camera'], q, view['coverage'])
            loss = mask_loss(image, view, args.samples)
            assert torch.isfinite(loss), f'Nonfinite loss: {stage}/{epoch}'
            loss.backward()
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
            optimizer.step()
            losses.append(float(loss.detach()))
        row = dict(stage=stage, epoch=epoch, training_loss=float(np.mean(losses)),
                   seconds=time.monotonic() - started)
        if epoch % args.validate_every == 0 or epoch == epochs:
            result = validate(model, validation_views, score, coefficients, args.bound)
            row['validation'] = result
            if result['mean_loss'] < best_loss:
                best_loss, best_epoch = result['mean_loss'], epoch
                best = parameter.detach().clone()
            print(f'{stage} epoch {epoch}/{epochs}: train={row["training_loss"]:.6f}, '
                  f'val={result["mean_loss"]:.6f}, best={best_epoch}, '
                  f'{row["seconds"]:.1f}s/epoch', flush=True)
        else:
            print(f'{stage} epoch {epoch}/{epochs}: train={row["training_loss"]:.6f}, '
                  f'{row["seconds"]:.1f}s/epoch', flush=True)
        history.append(row)
        write_json(args.output_dir / 'training_history.json', history)
    with torch.no_grad():
        parameter.copy_(best)
    if is_direction:
        assert torch.equal(score, frozen_score), 'Common score changed in direction stage'
    return dict(best_epoch=best_epoch, best_validation_loss=best_loss,
                planned_epochs=epochs, learning_rate=lr)


@torch.no_grad()
def edited_rgb(model, camera, q=None):
    original = model._opacity
    try:
        if q is not None:
            opacity = (model.get_opacity * (1 - q[:, None])).clamp(1e-8, 1 - 1e-8)
            model._opacity = torch.nn.Parameter(torch.logit(opacity), requires_grad=False)
        image = render(camera, model, PIPE, torch.zeros(3, device='cuda'))['render']
        return image.permute(1, 2, 0).cpu().numpy()
    finally:
        model._opacity = original


def evaluate(model, specs, initial, common, coefficients, args, report):
    assets = args.output_dir / 'assets'
    assets.mkdir(exist_ok=True)
    per_view = {}
    for spec in specs:
        stem = Path(spec['img_name']).stem
        camera = camera_from_json(spec)
        reference = edited_rgb(model, camera)
        save_rgb(reference, assets / f'{stem}_before.png')
        mask, target, protected = regions(args.mask_dir / f'{stem}.png', spec, args.labels, args.margin)
        Image.fromarray(target * 255).save(assets / f'{stem}_target.png')
        row = {}
        variants = [('class_probability', initial, None), ('common', common, None),
                    ('direction', common, coefficients)]
        view = prepare_view(model, spec, args)
        for name, s, c in variants:
            with torch.no_grad():
                q = gates(model, s, c, camera, args.bound)
                image = edited_rgb(model, camera, q)
                projection_q = gates(model, s, c, view['camera'], args.bound)
                projection = score_projection(model, view['camera'], projection_q, view['coverage'])
                score_image = projection.cpu().numpy()
            save_rgb(image, assets / f'{stem}_{name}.png')
            Image.fromarray((score_image.clip(0, 1) * 255).round().astype(np.uint8)).save(
                assets / f'{stem}_{name}_score.png')
            save_error(reference, image, assets / f'{stem}_{name}_difference.png')
            row[name] = metrics(reference, image, target, protected, mask)
            row[name]['mask_mse'] = float(mask_loss(projection, view))
            row[name]['mean_gate'] = float(q.mean())
            row[name]['gaussians_gate_over_05'] = int((q > .5).sum())
            print(f'eval {stem}/{name}: mask={row[name]["mask_mse"]:.6f}, '
                  f'protected={row[name]["protected_mae"]:.6f}', flush=True)
        for name, old_method in [('label_only', 'label_only'), ('global_multiview', 'global_multiview')]:
            if args.oracle_dir is None:
                continue
            baseline = args.oracle_dir / 'assets' / f'{stem}_{old_method}.png'
            if baseline.exists():
                image = np.asarray(Image.open(baseline), dtype=np.float32) / 255
                save_rgb(image, assets / f'{stem}_{name}.png')
                row[name] = metrics(reference, image, target, protected, mask)
        per_view[stem] = row
    report['per_view'] = per_view
    report['aggregate'] = {}
    for name in next(iter(per_view.values())):
        report['aggregate'][name] = {}
        keys = list(next(iter(per_view.values()))[name])
        for key in keys:
            values = [r[name][key] for r in per_view.values() if r[name].get(key) is not None]
            report['aggregate'][name][key] = float(np.mean(values)) if values else None


def interpolate_spec(a, b, t, name):
    # Polar interpolation is smooth for these neighboring camera rotations.
    matrix = (1 - t) * np.asarray(a['rotation']) + t * np.asarray(b['rotation'])
    u, _, vt = np.linalg.svd(matrix)
    rotation = u @ np.diag([1., 1., np.linalg.det(u @ vt)]) @ vt
    spec = dict(a)
    spec.update(img_name=name, rotation=rotation.tolist(),
                position=((1 - t) * np.asarray(a['position']) + t * np.asarray(b['position'])).tolist())
    return spec


def render_novel_path(model, all_specs, common, coefficients, args, report):
    by_number = {int(Path(s['img_name']).stem.split('_')[-1]): s for s in all_specs}
    left, right = by_number[143], by_number[148]
    directory = args.output_dir / 'novel_views'
    directory.mkdir(exist_ok=True)
    frames, specs = [], []
    for index, t in enumerate(np.linspace(0, 1, 16)):
        spec = interpolate_spec(left, right, float(t), f'novel_{index:03d}')
        camera = camera_from_json(spec, 800)
        images = [edited_rgb(model, camera)]
        for coefficients_for_view in [None, coefficients]:
            q = gates(model, common, coefficients_for_view, camera, args.bound)
            images.append(edited_rgb(model, camera, q))
        canvas = Image.new('RGB', (camera.image_width * 3, camera.image_height + 30), '#ffffff')
        draw = ImageDraw.Draw(canvas)
        for column, (name, image) in enumerate(zip(['Original', 'Common gate', 'Angular gate'], images)):
            png = directory / f'novel_{index:03d}_{column}.png'
            save_rgb(image, png)
            draw.text((camera.image_width * column + 12, 8), name, fill='#24292f')
            canvas.paste(Image.open(png), (column * camera.image_width, 30))
        frames.append(canvas.resize((1440, round(canvas.height * 1440 / canvas.width))))
        specs.append(spec)
    playback = frames + frames[-2:0:-1]
    playback[0].save(args.output_dir / 'novel-view-comparison.gif', save_all=True,
                     append_images=playback[1:], duration=160, loop=0)
    frames[8].save(args.output_dir / 'novel-view-comparison.png')
    write_json(directory / 'cameras.json', specs)
    report['novel_views'] = dict(camera_count=16, camera_path='interpolation between frames 143 and 148',
                                  masks_used_at_inference=False, quantitative_ground_truth=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--mask-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--oracle-dir', type=Path,
                        help='Optional previous oracle experiment directory for baseline PNGs')
    parser.add_argument('--iteration', type=int, default=15000)
    parser.add_argument('--labels', nargs='+', type=int, default=[1, 3])
    parser.add_argument('--retain-labels', nargs='+', type=int, default=[2, 4])
    parser.add_argument('--negative-region', choices=['retained', 'target-complement'], default='retained')
    parser.add_argument('--train-views', type=int, default=96)
    parser.add_argument('--validation-views', type=int, default=12)
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--margin', type=int, default=8)
    parser.add_argument('--samples', type=int, default=4096)
    parser.add_argument('--common-epochs', type=int, default=40)
    parser.add_argument('--direction-epochs', type=int, default=30)
    parser.add_argument('--common-lr', type=float, default=.03)
    parser.add_argument('--direction-lr', type=float, default=.02)
    parser.add_argument('--bound', type=float, default=1.)
    parser.add_argument('--validate-every', type=int, default=4)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    assert torch.cuda.is_available(), 'CUDA required'
    torch.cuda.set_device(0)
    torch.set_num_threads(4)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source = args.model_dir / f'point_cloud/iteration_{args.iteration}/point_cloud.ply'
    checksum = file_hash(source)
    specs = sorted(json.loads((args.model_dir / 'cameras.json').read_text()), key=lambda s: s['img_name'])
    number = lambda s: int(Path(s['img_name']).stem.split('_')[-1])
    test = [s for s in specs if number(s) in DEFAULT_TEST]
    # Reserve neighboring cameras too, including the later novel camera path.
    pool = [s for s in specs if all(abs(number(s) - t) > 3 for t in DEFAULT_TEST)]
    vi = np.linspace(0, len(pool) - 1, args.validation_views, dtype=int)
    validation = [pool[i] for i in vi]
    val_names = {s['img_name'] for s in validation}
    candidates = [s for s in pool if s['img_name'] not in val_names]
    ti = np.linspace(0, len(candidates) - 1, args.train_views, dtype=int)
    train = [candidates[i] for i in ti]
    sets = [{s['img_name'] for s in subset} for subset in [train, validation, test]]
    assert not sets[0] & sets[1] and not sets[0] & sets[2] and not sets[1] & sets[2]
    assert len(sets[0]) == args.train_views and len(test) == 6
    split = dict(train=sorted(sets[0]), validation=sorted(sets[1]), test=sorted(sets[2]),
                 held_out_for='removal-gate fitting only; original 3DGS reconstruction used these views')
    write_json(args.output_dir / 'split.json', split)
    model = load_scene(source)
    original_values = {name: getattr(model, name).detach().clone() for name in ATTRIBUTES}
    with torch.no_grad():
        p = model.get_object_logits.softmax(dim=1)[:, args.labels].sum(dim=1)
        # Finite logits prevent saturated starting parameters from being immovable.
        initial = torch.logit(p.clamp(1e-4, 1 - 1e-4)).detach()
    common = torch.nn.Parameter(initial.clone())
    report = dict(experiment='One balanced mask MSE; frozen Gaussian scene; common gate then angular residual',
        parameters={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        source_ply_sha256=checksum, gaussian_count=len(model.get_xyz), split=split,
        ignored_labels=[0] if args.negative_region == 'retained' else [],
        negative_labels=args.retain_labels if args.negative_region == 'retained' else 'SAM target mask exterior',
        limitations=['Original reconstruction and class learning already saw the evaluation cameras.',
                     'Only the added removal-gate fitting holds evaluation masks out.',
                     'No edited-RGB loss, no object-free RGB ground truth.',
                     'Mask loss uses original visibility; newly exposed hidden Gaussians are not directly supervised.',
                     ('Class 0 and uncertain boundaries are ignored, not labeled as background.'
                      if args.negative_region == 'retained' else
                      'SAM target-mask completeness is assumed; missed target pixels receive incorrect non-target supervision.'),
                     'Unsupervised Gaussians retain their class-probability initialization.',
                     'Target RGB change is not removal accuracy.'])
    write_json(args.output_dir / 'experiment.json', report)
    prepared_train, prepared_validation = [], []
    for subset, output in [(train, prepared_train), (validation, prepared_validation)]:
        for i, spec in enumerate(subset):
            output.append(prepare_view(model, spec, args))
            if (i + 1) % 12 == 0:
                print(f'Prepared {i + 1}/{len(subset)} cameras', flush=True)
    if args.smoke:
        view = prepared_train[0]
        prediction = score_projection(model, view['camera'], common.sigmoid(), view['coverage'])
        loss = mask_loss(prediction, view, args.samples)
        loss.backward()
        assert common.grad is not None and float(common.grad.abs().sum()) > 0
        assert all(getattr(model, name).grad is None for name in ATTRIBUTES)
        print('Smoke check: connected gate gradients, frozen source attributes.', flush=True)
        return
    history = []
    report['stages'] = {}
    report['stages']['common'] = train_stage(model, prepared_train, prepared_validation,
                                            common, None, args, 'common', history)
    common.requires_grad_(False)
    common_checkpoint = common.detach().clone()
    coefficients = torch.nn.Parameter(torch.zeros((len(common), 3), device='cuda'))
    torch.save(dict(common_score=common.cpu(), initial_score=initial.cpu(),
                    source_ply_sha256=checksum, parameters=report['parameters']),
               args.output_dir / 'common_gate.pth')
    report['stages']['direction'] = train_stage(model, prepared_train, prepared_validation,
        common, coefficients, args, 'direction', history)
    assert torch.equal(common, common_checkpoint)
    torch.save(dict(common_score=common.cpu(), angular_coefficients=coefficients.detach().cpu(),
                    initial_score=initial.cpu(), source_ply_sha256=checksum,
                    bound=args.bound, parameters=report['parameters'], split=split),
               args.output_dir / 'removal_gate.pth')
    write_json(args.output_dir / 'training_summary.json', report['stages'])
    del prepared_train, prepared_validation
    torch.cuda.empty_cache()
    evaluate(model, test, initial, common, coefficients, args, report)
    with torch.no_grad():
        render_novel_path(model, specs, common, coefficients, args, report)
    for name, original in original_values.items():
        assert torch.equal(getattr(model, name), original), f'Source changed: {name}'
    assert file_hash(source) == checksum, 'Original PLY modified'
    report['source_unchanged_verified'] = True
    write_json(args.output_dir / 'results.json', report)
    print('Completed:', args.output_dir, flush=True)
    print(json.dumps(report['aggregate'], indent=2), flush=True)


if __name__ == '__main__':
    main()
