"""Transfer class-index masks from input pixels to COLMAP-undistorted pixels."""
import argparse
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('standalone_colmap', ROOT / 'scene/colmap_loader.py')
colmap = importlib.util.module_from_spec(spec)
spec.loader.exec_module(colmap)


def intrinsics(camera):
    p = camera.params
    if camera.model in ('SIMPLE_PINHOLE', 'SIMPLE_RADIAL', 'RADIAL'):
        fx, cx, cy = p[:3]
        fy = fx
        distortion = np.zeros(4)
        if camera.model != 'SIMPLE_PINHOLE':
            distortion[0] = p[3]
        if camera.model == 'RADIAL':
            distortion[1] = p[4]
    elif camera.model in ('PINHOLE', 'OPENCV'):
        fx, fy, cx, cy = p[:4]
        distortion = np.zeros(4) if camera.model == 'PINHOLE' else p[4:8]
    else:
        raise ValueError(f'Unsupported camera model: {camera.model}')
    # COLMAP uses pixel centers at (x + .5, y + .5); OpenCV uses (x, y).
    matrix = np.array([[fx, 0, cx - .5], [0, fy, cy - .5], [0, 0, 1]], dtype=np.float64)
    return matrix, np.asarray(distortion, dtype=np.float64)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene-dir', type=Path, required=True)
    parser.add_argument('--mask-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--frames', nargs='+', default=['frame_000107.jpg', 'frame_000146.jpg',
        'frame_000219.jpg', 'frame_000292.jpg', 'frame_000375.jpg', 'frame_000417.jpg'])
    args = parser.parse_args()
    target_model = args.scene_dir / 'sparse/0'
    target_cameras = colmap.read_intrinsics_binary(target_model / 'cameras.bin')
    target_images = {x.name: x for x in colmap.read_extrinsics_binary(target_model / 'images.bin').values()}
    reconstructions = []
    for directory in (args.scene_dir / 'distorted/sparse').iterdir():
        if (directory / 'images.bin').is_file():
            images = {x.name: x for x in colmap.read_extrinsics_binary(directory / 'images.bin').values()}
            reconstructions.append((len(set(images) & set(target_images)), directory, images))
    overlap, source_model, source_images = max(reconstructions, key=lambda x: x[0])
    assert overlap == len(target_images), 'Distorted reconstruction does not cover target cameras'
    source_cameras = colmap.read_intrinsics_binary(source_model / 'cameras.bin')
    output_masks = args.output_dir / 'masks'
    assets = args.output_dir / 'assets'
    output_masks.mkdir(parents=True, exist_ok=True)
    assets.mkdir(exist_ok=True)
    maps = {}
    metadata = dict(source_reconstruction=str(source_model), target_reconstruction=str(target_model),
                    coordinate_convention='COLMAP half-pixel centers converted to OpenCV',
                    camera_pairs=[], rgb_alignment_checks={}, written_masks=0)
    for name, target_image in sorted(target_images.items()):
        source_image = source_images[name]
        assert np.allclose(source_image.qvec, target_image.qvec, atol=1e-8)
        assert np.allclose(source_image.tvec, target_image.tvec, atol=1e-8)
        source_camera = source_cameras[source_image.camera_id]
        target_camera = target_cameras[target_image.camera_id]
        assert target_camera.model in ('SIMPLE_PINHOLE', 'PINHOLE')
        key = (source_image.camera_id, target_image.camera_id)
        if key not in maps:
            source_matrix, distortion = intrinsics(source_camera)
            target_matrix, _ = intrinsics(target_camera)
            maps[key] = cv2.initUndistortRectifyMap(source_matrix, distortion, np.eye(3),
                target_matrix, (target_camera.width, target_camera.height), cv2.CV_32FC1)
            metadata['camera_pairs'].append(dict(source_model=source_camera.model,
                source_size=[source_camera.width, source_camera.height],
                source_parameters=source_camera.params.tolist(), target_model=target_camera.model,
                target_size=[target_camera.width, target_camera.height],
                target_parameters=target_camera.params.tolist()))
        mapx, mapy = maps[key]
        mask_path = args.mask_dir / Path(name).with_suffix('.png').name
        mask = np.asarray(Image.open(mask_path))
        assert mask.shape == (source_camera.height, source_camera.width), name
        aligned = cv2.remap(mask, mapx, mapy, cv2.INTER_NEAREST,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        Image.fromarray(aligned).save(output_masks / mask_path.name)
        metadata['written_masks'] += 1
        if name in args.frames:
            raw = np.asarray(Image.open(args.scene_dir / 'input' / name).convert('RGB'))
            target = np.asarray(Image.open(args.scene_dir / 'images' / name).convert('RGB'))
            reconstructed = cv2.remap(raw, mapx, mapy, cv2.INTER_LINEAR)
            error = np.abs(reconstructed.astype(float) - target.astype(float)).mean() / 255
            metadata['rgb_alignment_checks'][name] = dict(normalized_mae=float(error))
            assert error < 0.03, f'Undistortion mapping does not match COLMAP image: {name}, {error}'
            legacy = cv2.resize(mask, (target_camera.width, target_camera.height), interpolation=cv2.INTER_NEAREST)
            metadata['rgb_alignment_checks'][name]['changed_class_pixels_vs_resize'] = int((legacy != aligned).sum())
            colors = np.array([[0, 0, 0], [240, 70, 175], [40, 200, 90], [65, 110, 245], [245, 190, 50]])
            for method, labels in [('resize', legacy), ('aligned', aligned)]:
                overlay = target.astype(float).copy()
                selected = labels > 0
                overlay[selected] = .6 * overlay[selected] + .4 * colors[labels[selected]]
                Image.fromarray(overlay.clip(0, 255).astype(np.uint8)).save(assets / f'{Path(name).stem}_{method}_overlay.png')
    (args.output_dir / 'alignment.json').write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
