#!/usr/bin/env python3
"""
generate_dataset.py - Decode videos and produce masked images with COLMAP data.

Input structure:
    color_lut/
    ├── 0028.cube
    └── ...
    input/
    ├── manifest.json # Contains sequence metadata and frame IDs
    ├── AttitudePromenade/
    │   ├── AttitudePromenade_0028.mp4
    │   ├── AttitudePromenade_0103.mp4
    │   ├── colmap/ # camera calibration
    │   └── masks.tar.zst
    └── ...

Output structure:
    output/
    └── DanceNet3D/
        └── AttitudePromenade/
            ├── AttitudePromenade_res1/
            │   ├── 0000001/
            │   │   ├── images/ # RGBA with transparent background
            │   │   └── sparse/0/ # COLMAP + points3D.ply
            │   └── ...
            ├── AttitudePromenade_res2/
            │   ├── 0000001/
            │   │   ├── images/ # 1/2 resolution
            │   │   └── sparse/0/
            │   └── ...
            ├── AttitudePromenade_res4/
            └── AttitudePromenade_res8/
"""

import argparse
import json
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional
import os
import numpy as np


def format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{int(seconds // 60)}m {int(seconds % 60)}s"
    else:
        return f"{int(seconds // 3600)}h {int((seconds % 3600) // 60)}m"


def decode_camera_video(
    video_path: Path,
    output_base: Path,
    camera_id: str,
    frame_ids: list[str],
    content_type: str,
    image_extension: str,
    verbose: bool = False,
) -> tuple[bool, float]:

    start_time = time.time()

    # Create temp directory for extracted frames
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Extract all frames with ffmpeg
        ffmpeg_start = time.time()
        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-frames:v", str(len(frame_ids)),
            "-start_number", "0",
            str(temp_path / f"frame_%07d{image_extension}")
        ]

        if not verbose:
            cmd.insert(1, "-loglevel")
            cmd.insert(2, "warning")

        result = subprocess.run(cmd, capture_output=True, text=True)
        ffmpeg_time = time.time() - ffmpeg_start

        if result.returncode != 0:
            print(f"  Error decoding {camera_id}: {result.stderr}", file=sys.stderr)
            return False, time.time() - start_time

        # Move frames to correct locations with original frame IDs
        move_start = time.time()
        for i, frame_id in enumerate(frame_ids):
            # Source: extracted frame (0-indexed)
            src_frame = temp_path / f"frame_{i:07d}{image_extension}"

            # Destination: original structure
            dest_dir = output_base / frame_id / content_type
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest_frame = dest_dir / f"{camera_id}{image_extension}"

            if src_frame.exists():
                # Move original to images_no_lut/ (intermediate; removed after masking)
                shutil.move(src_frame, dest_frame)
            else:
                print(f"  Warning: Missing extracted frame {i} for {camera_id}", file=sys.stderr)

        move_time = time.time() - move_start

    total_time = time.time() - start_time
    if verbose:
        print(f"  Camera {camera_id}: ffmpeg {format_duration(ffmpeg_time)}, "
              f"move {format_duration(move_time)}, total {format_duration(total_time)}")

    return True, total_time


def copy_colmap_to_frames(
    colmap_dir: Path,
    output_base: Path,
    frame_ids: list[str],
    verbose: bool = False
) -> bool:
    """Copy shared colmap/ directory into each frame's sparse/0/ directory.

    The colmap data (cameras.txt, images.txt, etc.) is the same for all frames
    since cameras are static. Downstream scripts expect it at {frame}/sparse/0/.
    """
    if not colmap_dir.exists():
        print(f"  Warning: colmap directory not found: {colmap_dir}", file=sys.stderr)
        return False

    colmap_files = [f for f in colmap_dir.iterdir() if f.is_file()]
    if not colmap_files:
        print(f"  Warning: colmap directory is empty: {colmap_dir}", file=sys.stderr)
        return False

    for frame_id in frame_ids:
        sparse_0 = output_base / frame_id / "sparse" / "0"
        sparse_0.mkdir(parents=True, exist_ok=True)
        for src_file in colmap_files:
            shutil.copy2(src_file, sparse_0 / src_file.name)

    if verbose:
        print(f"  Copied colmap data ({len(colmap_files)} files) to {len(frame_ids)} frames")

    return True



def decompress_zst(src: Path, dst: Path, verbose: bool = False) -> bool:
    """Decompress a .zst file using the zstd CLI, falling back to Python zstandard."""
    cmd = ["zstd", "-d", str(src), "-o", str(dst), "--force"]
    if not verbose:
        cmd.append("-q")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            return True
        print(f"  Warning: zstd CLI failed: {result.stderr}", file=sys.stderr)
    except FileNotFoundError:
        if verbose:
            print("  zstd CLI not found, using Python zstandard", file=sys.stderr)

    try:
        import zstandard as zstd
    except ImportError:
        print("  Error: need zstd CLI or 'pip install zstandard' to extract masks", file=sys.stderr)
        return False

    with open(src, "rb") as f_in, open(dst, "wb") as f_out:
        zstd.ZstdDecompressor().copy_stream(f_in, f_out)
    return True


def masks_already_extracted(
    output_base: Path,
    frame_ids: list[str],
    cameras: list[str],
    image_extension: str,
) -> bool:
    for frame_id in frame_ids:
        mask_dir = output_base / frame_id / "masks"
        if not mask_dir.is_dir():
            return False
        for camera_id in cameras:
            if not (mask_dir / f"{camera_id}{image_extension}").exists():
                return False
    return True


def extract_masks_archive(
    tar_zst_path: Path,
    output_base: Path,
    frame_ids: Optional[list[str]] = None,
    cameras: Optional[list[str]] = None,
    image_extension: str = ".png",
    verbose: bool = False
) -> bool:
    """Decompress masks.tar.zst and extract to output directory.

    Preserves {frame_id}/masks/{camera_id}.png structure.
    """
    if not tar_zst_path.exists():
        print(f"  Warning: Masks archive not found: {tar_zst_path}", file=sys.stderr)
        return False

    if frame_ids and cameras and masks_already_extracted(
        output_base, frame_ids, cameras, image_extension
    ):
        if verbose:
            print("  Masks already extracted, skipping")
        return True

    tmp_tar = output_base / "masks.tar"

    try:
        if not decompress_zst(tar_zst_path, tmp_tar, verbose):
            return False

        output_base.mkdir(parents=True, exist_ok=True)
        frame_set = set(frame_ids) if frame_ids is not None else None
        with tarfile.open(tmp_tar, "r") as tar:
            for member in tar.getmembers():
                if frame_set is not None:
                    frame_id = member.name.split("/")[0]
                    if frame_id not in frame_set:
                        continue
                dest_path = output_base / member.name
                if member.isdir():
                    dest_path.mkdir(parents=True, exist_ok=True)
                else:
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    f = tar.extractfile(member)
                    if f:
                        dest_path.write_bytes(f.read())

        if verbose:
            print("  Extracted masks archive")

        return True

    finally:
        tmp_tar.unlink(missing_ok=True)


def cleanup_intermediate_outputs(
    output_base: Path,
    frame_ids: list[str],
    verbose: bool = False,
) -> None:
    """Remove decoded images and masks after masked images are created."""
    removed_dirs = 0
    for frame_id in frame_ids:
        frame_dir = output_base / frame_id
        for subdir in ("images_no_lut", "masks"):
            path = frame_dir / subdir
            if path.exists():
                shutil.rmtree(path)
                removed_dirs += 1

    if verbose and removed_dirs:
        print(f"  Removed {removed_dirs} intermediate directories")


def apply_masked_images(
    output_base: Path,
    frame_ids: list[str],
    cameras: list[str],
    image_extension: str,
    lut_map: Optional[dict] = None,
    verbose: bool = False,
) -> bool:
    """Composite images with foreground masks into RGBA PNGs in images/.

    Dancer pixels (mask >= 128) keep their color with alpha=255.
    Background pixels are set to RGB(0,0,0) with alpha=0.
    """
    import cv2

    count = 0
    missing = 0

    for frame_id in frame_ids:
        frame_dir = output_base / frame_id
        image_dir = frame_dir / "images_no_lut"
        mask_dir = frame_dir / "masks"
        out_dir = frame_dir / "images"
        out_dir.mkdir(parents=True, exist_ok=True)

        for camera_id in cameras:
            image_path = image_dir / f"{camera_id}{image_extension}"
            mask_path = mask_dir / f"{camera_id}{image_extension}"
            out_path = out_dir / f"{camera_id}{image_extension}"

            if not image_path.exists():
                missing += 1
                if verbose:
                    print(f"  Warning: Image not found: {image_path}", file=sys.stderr)
                continue
            if not mask_path.exists():
                missing += 1
                if verbose:
                    print(f"  Warning: Mask not found: {mask_path}", file=sys.stderr)
                continue

            img_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if img_bgr is None or mask is None:
                missing += 1
                continue

            lut_info = lut_map.get(camera_id) if lut_map else None
            if lut_info is not None:
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                img_lut = apply_3d_lut_rgb(img_rgb, lut_info)
                img_bgr = cv2.cvtColor(img_lut, cv2.COLOR_RGB2BGR)

            foreground = mask >= 128
            alpha = np.where(foreground, 255, 0).astype(np.uint8)
            bgra = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2BGRA)
            bgra[~foreground] = 0
            bgra[:, :, 3] = alpha
            cv2.imwrite(str(out_path), bgra)
            count += 1

    if count == 0:
        print("  Warning: No masked images created (missing images or masks)", file=sys.stderr)
        return False

    if verbose and missing:
        print(f"  Skipped {missing} image/mask pairs")

    return True


def apply_3d_lut_rgb(image_rgb, lut_info):
    """Apply a 3D color LUT to an RGB image using trilinear interpolation."""
    if lut_info is None:
        return image_rgb
    lut = lut_info['lut']
    size = lut_info['size']
    domain_min = lut_info['domain_min']
    domain_max = lut_info['domain_max']
    img = image_rgb.astype(np.float32) / 255.0
    dom_range = np.clip(domain_max - domain_min, 1e-6, None)
    img_norm = (img - domain_min) / dom_range
    img_norm = np.clip(img_norm, 0.0, 1.0)
    coord = img_norm * (size - 1)
    i0 = np.floor(coord).astype(np.int32)
    i1 = np.clip(i0 + 1, 0, size - 1)
    f = coord - i0
    fr = f[:, :, 0:1]
    fg = f[:, :, 1:2]
    fb = f[:, :, 2:2+1]
    r0, g0, b0 = i0[:, :, 0], i0[:, :, 1], i0[:, :, 2]
    r1, g1, b1 = i1[:, :, 0], i1[:, :, 1], i1[:, :, 2]
    c000 = lut[r0, g0, b0]
    c001 = lut[r0, g0, b1]
    c010 = lut[r0, g1, b0]
    c011 = lut[r0, g1, b1]
    c100 = lut[r1, g0, b0]
    c101 = lut[r1, g0, b1]
    c110 = lut[r1, g1, b0]
    c111 = lut[r1, g1, b1]
    c00 = c000 * (1 - fb) + c001 * fb
    c01 = c010 * (1 - fb) + c011 * fb
    c10 = c100 * (1 - fb) + c101 * fb
    c11 = c110 * (1 - fb) + c111 * fb
    c0 = c00 * (1 - fg) + c01 * fg
    c1 = c10 * (1 - fg) + c11 * fg
    out = c0 * (1 - fr) + c1 * fr
    out = np.clip(out * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return out


def load_cube_lut(cube_path, row_order: str = 'rgb'):
    row_order = row_order.lower()
    valid_orders = ('rgb', 'rbg', 'grb', 'gbr', 'brg', 'bgr')
    if row_order not in valid_orders:
        raise ValueError(f"Unsupported row_order: {row_order}")
    size = None
    domain_min = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    domain_max = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    table = []
    with open(cube_path, 'r', encoding='utf-8') as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith('#'):
                continue
            up = s.upper()
            if up.startswith('TITLE'):
                continue
            if up.startswith('DOMAIN_MIN'):
                parts = s.split()
                domain_min = np.array([float(parts[-3]), float(parts[-2]), float(parts[-1])], dtype=np.float32)
                continue
            if up.startswith('DOMAIN_MAX'):
                parts = s.split()
                domain_max = np.array([float(parts[-3]), float(parts[-2]), float(parts[-1])], dtype=np.float32)
                continue
            if up.startswith('LUT_3D_SIZE'):
                parts = s.split()
                size = int(parts[-1])
                continue
            parts = s.split()
            if len(parts) >= 3:
                table.append([float(parts[0]), float(parts[1]), float(parts[2])])
    if size is None:
        n = len(table)
        inferred = round(n ** (1.0 / 3.0))
        if inferred ** 3 != n:
            raise ValueError(f"Cannot infer LUT size from {n} rows in {cube_path}")
        size = inferred
    expected = size ** 3
    if len(table) < expected:
        raise ValueError(f"Incomplete LUT: expected {expected} rows, found {len(table)} in {cube_path}")
    data = np.asarray(table[:expected], dtype=np.float32)

    fastest_to_slowest = list(row_order)
    slowest_to_fastest = fastest_to_slowest[::-1]

    lut_raw = data.reshape((size, size, size, 3))

    shape_names = slowest_to_fastest
    transpose_order = [shape_names.index('r'), shape_names.index('g'), shape_names.index('b'), 3]
    lut = np.transpose(lut_raw, axes=transpose_order)
    return {
        'lut': lut,
        'size': size,
        'domain_min': domain_min,
        'domain_max': domain_max,
    }


def load_cube_luts_from_dir(lut_dir: str, row_order: str = 'rgb'):
    if not os.path.isdir(lut_dir):
        raise FileNotFoundError(f"LUT directory not found: {lut_dir}")
    lut_map = {}
    for fname in os.listdir(lut_dir):
        if not fname.lower().endswith('.cube'):
            continue
        stem = os.path.splitext(fname)[0]
        full_path = os.path.join(lut_dir, fname)
        try:
            lut_map[stem] = load_cube_lut(full_path, row_order=row_order)
        except Exception as e:
            print(f"Warning: failed to load LUT '{full_path}': {e}")
    return lut_map


def get_sequence_res_dir(output_path: Path, sequence_name: str, scale: int) -> Path:
    """Return {output}/DanceNet3D/{sequence_name}/{sequence_name}_res{scale}/."""
    return output_path / "DanceNet3D" / sequence_name / f"{sequence_name}_res{scale}"


def process_sequence(
    input_path: Path,
    output_path: Path,
    sequence_name: str,
    sequence_data: dict,
    cameras_filter: Optional[list[str]],
    parallel: int,
    verbose: bool,
    lut_map: Optional[dict] = None,
    max_frames: Optional[int] = None,
) -> bool:

    sequence_input = input_path / sequence_name
    # Output: {output}/DanceNet3D/{sequence_name}/{sequence_name}_res1/
    sequence_output = get_sequence_res_dir(output_path, sequence_name, 1)

    frame_ids = sequence_data["frame_ids"]
    if max_frames is not None:
        if max_frames <= 0:
            print(f"  Error: --max-frames must be positive, got {max_frames}", file=sys.stderr)
            return False
        frame_ids = frame_ids[:max_frames]
    content_type = sequence_data.get("content_type", "images_no_lut")
    image_extension = sequence_data.get("image_extension", ".png")
    cameras = sequence_data["cameras"]

    if cameras_filter:
        cameras = [c for c in cameras if c in cameras_filter]

    if not cameras:
        print(f"  No cameras to decode for {sequence_name}")
        return False

    lut_label = " + LUT on masked images" if lut_map else ""
    print(f"  Frames: {len(frame_ids)}, Cameras: {len(cameras)}{lut_label}")

    # Check for frame gaps
    expected_frames = set(range(int(frame_ids[0]), int(frame_ids[-1]) + 1))
    actual_frames = set(int(f) for f in frame_ids)
    missing_frames = expected_frames - actual_frames
    if missing_frames:
        missing_ids = sorted([f"{f:07d}" for f in missing_frames])
        print(f"Missing {len(missing_frames)} frames: {', '.join(missing_ids)}")

    # Decode videos
    decode_start = time.time()
    success_count = 0

    if parallel > 1:
        # Parallel decoding
        with ThreadPoolExecutor(max_workers=parallel) as executor:
            futures = {}
            for camera_id in cameras:
                video_path = sequence_input / f"{sequence_name}_{camera_id}.mp4"
                if not video_path.exists():
                    print(f"  Warning: Video not found: {video_path}", file=sys.stderr)
                    continue

                future = executor.submit(
                    decode_camera_video,
                    video_path, sequence_output, camera_id,
                    frame_ids, content_type, image_extension, verbose
                )
                futures[future] = camera_id

            for future in as_completed(futures):
                camera_id = futures[future]
                success, duration = future.result()
                if success:
                    success_count += 1
    else:
        # Sequential decoding with progress
        for i, camera_id in enumerate(cameras, 1):
            video_path = sequence_input / f"{sequence_name}_{camera_id}.mp4"
            if not video_path.exists():
                print(f"  Warning: Video not found: {video_path}", file=sys.stderr)
                continue

            if not verbose:
                print(f"  Decoding camera {i}/{len(cameras)}: {camera_id}...", end=" ")
                sys.stdout.flush()
            success, duration = decode_camera_video(
                video_path, sequence_output, camera_id,
                frame_ids, content_type, image_extension, verbose
            )
            if success:
                success_count += 1
                if not verbose:
                    print(f"{format_duration(duration)}")

    decode_time = time.time() - decode_start
    print(f"  Decoded {success_count}/{len(cameras)} cameras in {format_duration(decode_time)}")

    # Copy colmap directory to per-frame sparse/0/ if present
    if sequence_data.get("has_colmap", False):
        colmap_dir = sequence_input / "colmap"
        colmap_start = time.time()
        if copy_colmap_to_frames(colmap_dir, sequence_output, frame_ids, verbose):
            print(f"  Copied colmap to {len(frame_ids)} frames in {format_duration(time.time() - colmap_start)}")

    # Extract masks archive (required for masked image output)
    masks_path = sequence_input / "masks.tar.zst"
    masks_start = time.time()
    if not extract_masks_archive(
        masks_path, sequence_output, frame_ids, cameras, image_extension, verbose
    ):
        print("  Error: masks.tar.zst is required to create images", file=sys.stderr)
        return False
    print(f"  Extracted masks archive in {format_duration(time.time() - masks_start)}")

    mask_start = time.time()
    if not apply_masked_images(
        sequence_output, frame_ids, cameras, image_extension,
        lut_map=lut_map, verbose=verbose,
    ):
        return False
    print(f"  Created images for {len(frame_ids)} frames in "
          f"{format_duration(time.time() - mask_start)}")

    cleanup_intermediate_outputs(sequence_output, frame_ids, verbose)

    return success_count > 0


DOWN_SAMPLE_SCALES = (2, 4, 8)


def scale_colmap_cameras_txt(src: Path, dst: Path, scale: int) -> None:
    """Scale PINHOLE intrinsics in cameras.txt for downsampled images."""
    inv = 1.0 / scale
    lines_out = []
    with open(src, encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                lines_out.append(line if line.endswith("\n") else line + "\n")
                continue

            parts = stripped.split()
            if len(parts) >= 8 and parts[1] == "PINHOLE":
                cam_id, model = parts[0], parts[1]
                w, h = float(parts[2]), float(parts[3])
                fx, fy, cx, cy = map(float, parts[4:8])
                lines_out.append(
                    f"{cam_id} {model} {int(round(w * inv))} {int(round(h * inv))} "
                    f"{fx * inv} {fy * inv} {cx * inv} {cy * inv}\n"
                )
            else:
                lines_out.append(line if line.endswith("\n") else line + "\n")

    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "w", encoding="utf-8") as f:
        f.writelines(lines_out)


def copy_scaled_colmap_sparse(sparse_src: Path, sparse_dst: Path, scale: int) -> None:
    """Copy sparse/0/ data, scaling camera intrinsics for the downsampled resolution."""
    sparse_dst.mkdir(parents=True, exist_ok=True)
    for src_file in sparse_src.iterdir():
        if not src_file.is_file():
            continue
        if src_file.name == "cameras.txt":
            scale_colmap_cameras_txt(src_file, sparse_dst / src_file.name, scale)
        else:
            shutil.copy2(src_file, sparse_dst / src_file.name)


def downsample_frame_images(
    frame_src: Path,
    frame_dst: Path,
    scale: int,
    image_extension: str,
    verbose: bool = False,
) -> int:
    """Downsample {frame_idx}/images/ from res1 into {frame_idx}/images/ at 1/scale."""
    import cv2

    src_dir = frame_src / "images"
    dst_dir = frame_dst / "images"
    dst_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for image_path in sorted(src_dir.glob(f"*{image_extension}")):
        img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            if verbose:
                print(f"  Warning: Failed to read image: {image_path}", file=sys.stderr)
            continue

        height, width = img.shape[:2]
        new_size = (max(1, width // scale), max(1, height // scale))
        resized = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(dst_dir / image_path.name), resized)
        count += 1

    return count


def downsample_sequence(
    output_path: Path,
    sequence_name: str,
    scale: int,
    verbose: bool = False,
) -> bool:
    """Generate {sequence_name}_res{scale} from the corresponding res1 output."""
    res1_dir = get_sequence_res_dir(output_path, sequence_name, 1)
    res_dir = get_sequence_res_dir(output_path, sequence_name, scale)

    if not res1_dir.is_dir():
        print(f"  Warning: res1 output not found: {res1_dir}", file=sys.stderr)
        return False

    frame_dirs = sorted(
        p for p in res1_dir.iterdir()
        if p.is_dir() and (p / "images").is_dir()
    )
    if not frame_dirs:
        print(f"  Warning: No frames with images/ found in {res1_dir}", file=sys.stderr)
        return False

    image_files = list((frame_dirs[0] / "images").glob("*"))
    image_extension = image_files[0].suffix if image_files else ".png"
    total_images = 0

    for frame_src in frame_dirs:
        frame_id = frame_src.name
        frame_dst = res_dir / frame_id

        total_images += downsample_frame_images(
            frame_src, frame_dst, scale, image_extension, verbose
        )

        sparse_src = frame_src / "sparse" / "0"
        if sparse_src.is_dir():
            copy_scaled_colmap_sparse(sparse_src, frame_dst / "sparse" / "0", scale)

    if total_images == 0:
        print(f"  Warning: No images downsampled for {sequence_name}_res{scale}", file=sys.stderr)
        return False

    print(f"  {sequence_name}_res{scale}: {total_images} images across {len(frame_dirs)} frames")
    return True


def run_down_sampling(
    output_path: Path,
    sequence_names: list[str],
    verbose: bool = False,
) -> bool:
    """Downsample all res1 outputs to res2, res4, and res8 with scaled COLMAP data."""
    if not sequence_names:
        return True

    print(f"Downsampling {len(sequence_names)} sequences to res2/res4/res8...")
    start_time = time.time()
    success_count = 0
    total_tasks = len(sequence_names) * len(DOWN_SAMPLE_SCALES)

    for sequence_name in sequence_names:
        print(f"[{sequence_name} downsampling]")
        sequence_ok = True
        for scale in DOWN_SAMPLE_SCALES:
            if not downsample_sequence(output_path, sequence_name, scale, verbose):
                sequence_ok = False
        if sequence_ok:
            success_count += 1
        print()

    print(
        f"Downsampled {success_count}/{len(sequence_names)} sequences "
        f"({total_tasks} resolution variants) in {format_duration(time.time() - start_time)}"
    )
    return success_count == len(sequence_names)


POINT_CLOUD_FILENAME = "points3D.ply"


def get_pointcloud_archive_path(input_path: Path, sequence_name: str) -> Path:
    """Return pointcloud/{session}/{sequence_name}.tar.zst next to the input session dir."""
    session = input_path.name
    return input_path.parent / "pointcloud" / session / f"{sequence_name}.tar.zst"


def find_tar_ply_member(tar: tarfile.TarFile, frame_id: str) -> Optional[tarfile.TarInfo]:
    """Find a .ply member for the given frame ID inside a point cloud archive."""
    direct_names = (f"{frame_id}.ply", f"./{frame_id}.ply", f"{frame_id}/{frame_id}.ply")
    for name in direct_names:
        try:
            member = tar.getmember(name)
            if member.isfile():
                return member
        except KeyError:
            continue

    suffix = f"/{frame_id}.ply"
    for member in tar.getmembers():
        if member.isfile() and (member.name == f"{frame_id}.ply" or member.name.endswith(suffix)):
            return member
    return None


def load_point_cloud_from_bytes(ply_bytes: bytes):
    import open3d as o3d

    with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as tmp:
        tmp.write(ply_bytes)
        tmp_path = tmp.name

    try:
        point_cloud = o3d.io.read_point_cloud(tmp_path)
    finally:
        os.unlink(tmp_path)

    if point_cloud.is_empty():
        raise ValueError("PLY file contains no points")
    return point_cloud


def sample_point_cloud(point_cloud, sampling_num: int, rng: np.random.Generator):
    """Randomly subsample a point cloud down to sampling_num points."""
    num_points = len(point_cloud.points)
    if sampling_num <= 0:
        raise ValueError(f"sampling_num must be positive, got {sampling_num}")
    if num_points <= sampling_num:
        return point_cloud

    indices = rng.choice(num_points, sampling_num, replace=False)
    return point_cloud.select_by_index(indices)


def remove_shadow_points(
    point_cloud,
    foot_level: float = 0.05,
    shadow_bar: int = 32,
) -> tuple:
    """Remove dark shadow points from the highest *foot_level* fraction by Y.

    Feet lie toward +Y in DanceNet3D / COLMAP world space. Among points with Y
    at or above the ``(1 - foot_level)`` percentile, drops any point whose R, G,
    and B are all strictly below *shadow_bar* on a 0–255 scale.
    """
    if foot_level <= 0.0 or foot_level > 1.0:
        raise ValueError(f"foot_level must be in (0, 1], got {foot_level}")
    if shadow_bar < 0 or shadow_bar > 255:
        raise ValueError(f"shadow_bar must be in [0, 255], got {shadow_bar}")

    points = np.asarray(point_cloud.points)
    if len(points) == 0:
        return point_cloud, 0

    colors = np.asarray(point_cloud.colors)
    if colors.size == 0:
        return point_cloud, 0

    if colors.max() <= 1.0:
        colors_255 = colors * 255.0
    else:
        colors_255 = colors

    y = points[:, 1]
    y_cutoff = np.percentile(y, (1.0 - foot_level) * 100.0)
    in_foot_band = y >= y_cutoff
    is_dark = (
        (colors_255[:, 0] < shadow_bar)
        & (colors_255[:, 1] < shadow_bar)
        & (colors_255[:, 2] < shadow_bar)
    )
    remove_mask = in_foot_band & is_dark
    removed = int(remove_mask.sum())
    if removed == 0:
        return point_cloud, 0

    keep_indices = np.where(~remove_mask)[0]
    return point_cloud.select_by_index(keep_indices), removed


def save_point_cloud(point_cloud, dst_path: Path) -> None:
    import open3d as o3d

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    if not o3d.io.write_point_cloud(str(dst_path), point_cloud):
        raise RuntimeError(f"Failed to write point cloud: {dst_path}")


def place_point_clouds_for_sequence(
    input_path: Path,
    output_path: Path,
    sequence_name: str,
    frame_ids: list[str],
    sampling_num: Optional[int] = None,
    res_scales: tuple[int, ...] = (1,),
    no_shadow: bool = False,
    foot_level: float = 0.05,
    shadow_bar: int = 32,
    verbose: bool = False,
    rng: Optional[np.random.Generator] = None,
) -> str:
    """Extract per-frame point clouds and write points3D.ply into sparse/0/.

    Returns:
        "success" - at least one frame got a point cloud
        "skipped" - no archive for this session/sequence (images-only fallback)
        "failed"  - archive exists but nothing could be placed
    """
    session = input_path.name
    archive_path = get_pointcloud_archive_path(input_path, sequence_name)
    if not archive_path.exists():
        print(
            f"  No point cloud archive for {session}/{sequence_name}: {archive_path}\n"
            f"  Skipping point clouds for this sequence (images + COLMAP output unchanged)",
            file=sys.stderr,
        )
        return "skipped"

    if rng is None:
        rng = np.random.default_rng()

    tmp_tar = None
    placed = 0
    missing = 0
    start_time = time.time()

    try:
        with tempfile.NamedTemporaryFile(suffix=".tar", delete=False) as tmp:
            tmp_tar = Path(tmp.name)

        if not decompress_zst(archive_path, tmp_tar, verbose):
            print(
                f"  Failed to decompress point cloud archive: {archive_path}\n"
                f"  Skipping point clouds for this sequence (images + COLMAP output unchanged)",
                file=sys.stderr,
            )
            return "skipped"

        with tarfile.open(tmp_tar, "r") as tar:
            for frame_id in frame_ids:
                member = find_tar_ply_member(tar, frame_id)
                if member is None:
                    missing += 1
                    if verbose:
                        print(f"  Warning: Point cloud not found for frame {frame_id}", file=sys.stderr)
                    continue

                extracted = tar.extractfile(member)
                if extracted is None:
                    missing += 1
                    continue

                try:
                    point_cloud = load_point_cloud_from_bytes(extracted.read())
                    if no_shadow:
                        before_shadow = len(point_cloud.points)
                        point_cloud, removed_shadow = remove_shadow_points(
                            point_cloud, foot_level=foot_level, shadow_bar=shadow_bar
                        )
                        if verbose and removed_shadow:
                            print(
                                f"  Frame {frame_id}: removed {removed_shadow} shadow points "
                                f"from highest {foot_level * 100:.1f}% by Y (+Y = feet) "
                                f"({before_shadow} -> {len(point_cloud.points)})"
                            )
                    if sampling_num is not None:
                        original_count = len(point_cloud.points)
                        point_cloud = sample_point_cloud(point_cloud, sampling_num, rng)
                        if verbose:
                            print(
                                f"  Frame {frame_id}: sampled {len(point_cloud.points)} "
                                f"from {original_count} points"
                            )
                except ImportError:
                    print(
                        "  Error: open3d is required for point cloud processing "
                        "(pip install open3d)",
                        file=sys.stderr,
                    )
                    return "failed"
                except (OSError, ValueError, RuntimeError) as exc:
                    missing += 1
                    if verbose:
                        print(f"  Warning: Failed to process frame {frame_id}: {exc}", file=sys.stderr)
                    continue

                for scale in res_scales:
                    sparse_dir = get_sequence_res_dir(output_path, sequence_name, scale) / frame_id / "sparse" / "0"
                    save_point_cloud(point_cloud, sparse_dir / POINT_CLOUD_FILENAME)
                placed += 1

    finally:
        if tmp_tar is not None:
            tmp_tar.unlink(missing_ok=True)

    if placed == 0:
        print(
            f"  Warning: Point cloud archive exists but no frames were placed for {sequence_name}",
            file=sys.stderr,
        )
        return "failed"

    sampling_label = f", sampled to {sampling_num} points" if sampling_num is not None else ""
    print(
        f"  Placed point clouds for {placed}/{len(frame_ids)} frames"
        f"{sampling_label} in {format_duration(time.time() - start_time)}"
    )
    if missing and not verbose:
        print(f"  Warning: Missing point clouds for {missing} frames", file=sys.stderr)
    return "success"


def run_point_cloud_sampling(
    input_path: Path,
    output_path: Path,
    sequence_frames: dict[str, list[str]],
    sampling_num: Optional[int] = None,
    res_scales: tuple[int, ...] = (1,),
    no_shadow: bool = False,
    foot_level: float = 0.05,
    shadow_bar: int = 32,
    verbose: bool = False,
) -> None:
    """Extract point clouds for all sequences and place them under sparse/0/.

    Missing archives are skipped without stopping the pipeline.
    """
    if not sequence_frames:
        return

    print(f"Placing point clouds for {len(sequence_frames)} sequences...")
    if sampling_num is not None:
        print(f"Point cloud sampling: {sampling_num} points per frame")
    if no_shadow:
        print(
            f"Shadow removal: highest {foot_level * 100:.1f}% by Y (+Y = feet), "
            f"RGB < {shadow_bar} on 0–255 scale"
        )
    start_time = time.time()
    success_count = 0
    skipped_count = 0
    failed_count = 0
    rng = np.random.default_rng()

    for sequence_name, frame_ids in sequence_frames.items():
        print(f"[{sequence_name} point clouds]")
        result = place_point_clouds_for_sequence(
            input_path, output_path, sequence_name, frame_ids,
            sampling_num=sampling_num, res_scales=res_scales,
            no_shadow=no_shadow, foot_level=foot_level, shadow_bar=shadow_bar,
            verbose=verbose, rng=rng,
        )
        if result == "success":
            success_count += 1
        elif result == "skipped":
            skipped_count += 1
        else:
            failed_count += 1
        print()

    print(
        f"Point clouds: {success_count} placed, {skipped_count} skipped (no archive), "
        f"{failed_count} failed — continuing pipeline in "
        f"{format_duration(time.time() - start_time)}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Generate masked images and COLMAP data from videos")

    parser.add_argument("--input", "-i", type=Path, required=True,
                        help="Input path containing videos and manifest.json")
    parser.add_argument("--output", "-o", type=Path, required=True,
                        help="Output path for reconstructed images")
    parser.add_argument("--sequences", nargs="+", type=str,
                        help="Specific sequences to decode (default: all)")
    parser.add_argument("--cameras", nargs="+", type=str,
                        help="Specific cameras to decode (default: all)")
    parser.add_argument("--parallel", "-j", type=int, default=1,
                        help="Parallel camera decoding (default: 1)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose output")
    parser.add_argument("--apply-lut", nargs="?", const="auto", default=None,
                        metavar="LUT_DIR",
                        help="Apply color LUT when creating images/. "
                             "Optionally specify LUT directory (default: project color_lut/)")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Only decode and save the first N frames (also limits ffmpeg decoding)")
    parser.add_argument("--skip-downsample", action="store_true",
                        help="Skip generating res2/res4/res8 downsampled outputs")

    parser.add_argument("--skip-point-cloud-sampling", action="store_true",
                        help="Skip extracting point clouds into sparse/0/")
    parser.add_argument("--point-cloud-sampling-num", type=int, default=None,
                        help="Randomly subsample each point cloud down to this many points")
    parser.add_argument("--no-shadow", action="store_true",
                        help="Remove dark shadow points from the +Y foot band before sampling")
    parser.add_argument("--foot-level", type=float, default=0.05,
                        help="Highest fraction by Y (+Y = feet) to scan for shadows (default: 0.05 = 5%%)")
    parser.add_argument("--shadow-bar", type=int, default=32,
                        help="Drop points with R,G,B all below this value on 0–255 scale (default: 32)")

    args = parser.parse_args()

    if args.point_cloud_sampling_num is not None and args.point_cloud_sampling_num <= 0:
        print("Error: --point-cloud-sampling-num must be positive", file=sys.stderr)
        sys.exit(1)
    if args.foot_level <= 0.0 or args.foot_level > 1.0:
        print("Error: --foot-level must be in (0, 1]", file=sys.stderr)
        sys.exit(1)
    if args.shadow_bar < 0 or args.shadow_bar > 255:
        print("Error: --shadow-bar must be in [0, 255]", file=sys.stderr)
        sys.exit(1)

    # Load LUTs if requested
    lut_map = None
    if args.apply_lut is not None:
        if args.apply_lut == "auto":
            script_dir = Path(__file__).resolve().parent
            lut_dir = script_dir / "color_lut"
        else:
            lut_dir = Path(args.apply_lut)

        if not lut_dir.exists():
            print(f"Error: LUT directory not found: {lut_dir}", file=sys.stderr)
            sys.exit(1)

        lut_map = load_cube_luts_from_dir(str(lut_dir))
        print(f"Loaded {len(lut_map)} color LUTs from {lut_dir}")

    # Load manifest
    manifest_path = args.input / "manifest.json"
    if not manifest_path.exists():
        print(f"Error: Manifest not found: {manifest_path}", file=sys.stderr)
        sys.exit(1)

    with open(manifest_path) as f:
        manifest = json.load(f)

    content_type = manifest.get("content_type", "images_no_lut")

    # Add content_type to each sequence's data for convenience
    for seq_data in manifest["sequences"].values():
        if "content_type" not in seq_data:
            seq_data["content_type"] = content_type

    # Filter sequences
    if args.sequences:
        sequences = {k: v for k, v in manifest["sequences"].items() if k in args.sequences}
    else:
        sequences = manifest["sequences"]

    if not sequences:
        print("Error: No sequences to decode", file=sys.stderr)
        sys.exit(1)

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    print(f"Generating dataset for {len(sequences)} sequences")
    print("Output: DanceNet3D/{scene_name}/{scene_name}_res1/{frame_idx}/images/ + sparse/0/")
    if not args.skip_downsample:
        print("Then: DanceNet3D/{scene_name}/{scene_name}_res2|res4|res8/{frame_idx}/images/ + sparse/0/")
    if not args.skip_point_cloud_sampling:
        print("Point clouds: sparse/0/points3D.ply from pointcloud/{session}/{scene}.tar.zst")
        print("  (sequences without an archive are skipped; images/COLMAP still output)")
        if args.point_cloud_sampling_num is not None:
            print(f"Point cloud sampling: {args.point_cloud_sampling_num} points per frame")
        if args.no_shadow:
            print(
                f"Shadow removal: highest {args.foot_level * 100:.1f}% by Y (+Y = feet), "
                f"RGB < {args.shadow_bar} on 0–255 scale"
            )
    if args.max_frames is not None:
        print(f"Max frames: {args.max_frames}")
    print()

    # Process each sequence
    success_count = 0
    successful_sequences: list[str] = []
    for sequence_name, sequence_data in sequences.items():
        print(f"[{sequence_name}]")

        if process_sequence(
            args.input, args.output, sequence_name, sequence_data,
            args.cameras, args.parallel, args.verbose, lut_map,
            args.max_frames,
        ):
            success_count += 1
            successful_sequences.append(sequence_name)

        print()

    print(f"Generated {success_count}/{len(sequences)} sequences successfully")

    sequence_frames: dict[str, list[str]] = {}
    for sequence_name in successful_sequences:
        frame_ids = list(sequences[sequence_name]["frame_ids"])
        if args.max_frames is not None:
            frame_ids = frame_ids[:args.max_frames]
        sequence_frames[sequence_name] = frame_ids

    if not args.skip_point_cloud_sampling and sequence_frames:
        print()
        run_point_cloud_sampling(
            args.input, args.output, sequence_frames,
            sampling_num=args.point_cloud_sampling_num,
            res_scales=(1,),
            no_shadow=args.no_shadow,
            foot_level=args.foot_level,
            shadow_bar=args.shadow_bar,
            verbose=args.verbose,
        )

    if not args.skip_downsample and successful_sequences:
        print()
        run_down_sampling(args.output, successful_sequences, args.verbose)


if __name__ == "__main__":
    main()
