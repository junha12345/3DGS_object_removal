"""Render a fitted removal gate for any camera JSON; no SAM mask is needed."""
import argparse
import json
from pathlib import Path

import torch

from train_removal_gate import file_hash, load_scene, gates, edited_rgb
from validate_removal_multiview import camera_from_json, save_rgb


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-ply', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--cameras-json', type=Path, required=True)
    parser.add_argument('--frame', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--common-only', action='store_true')
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    assert file_hash(args.source_ply) == checkpoint['source_ply_sha256'], 'Checkpoint/PLY mismatch'
    model = load_scene(args.source_ply)
    specs = json.loads(args.cameras_json.read_text())
    matches = [s for s in specs if s['img_name'] == args.frame]
    assert len(matches) == 1, f'Expected one camera: {args.frame}'
    camera = camera_from_json(matches[0])
    score = checkpoint['common_score'].cuda()
    assert len(score) == len(model.get_xyz)
    coefficients = None if args.common_only else checkpoint['angular_coefficients'].cuda()
    bound = checkpoint.get('bound', checkpoint.get('parameters', {}).get('bound', 1.))
    with torch.no_grad():
        q = gates(model, score, coefficients, camera, bound)
        image = edited_rgb(model, camera, q)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_rgb(image, args.output)
    print('Saved:', args.output)


if __name__ == '__main__':
    main()
