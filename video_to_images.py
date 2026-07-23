#!/usr/bin/env python3
"""
video_to_images.py - Decode videos back to per-frame structure.

Input structure:
    color_lut/ 
    ├── 0028.cube 
    └── ...
    input/
    ├── manifest.json # Contains sequence metadata and frame IDs
    ├── AttitudePromenade/
    │   ├── AttitudePromenade_0028.mp4
    │   ├── AttitudePromenade_0103.mp4
    │   └── sparse.tar.gz # camera calibration
    └── ...

Output structure:
    output/
    ├── AttitudePromenade/
    │   └── images_and_masks/
    │       ├── 0000001/
    │       │   └── images_no_lut/
    │       │       ├── 0028.png
    │       │       └── ...
    │       │   └── images/ # LUT applied images if enabled
    │       │   └── images_masked/ # RGBA with transparent background if --apply-masking
    │       ├── 0000002/
    │       └── ...
    └── ...
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
    lut_info: dict = None
) -> tuple[bool, float]:

    start_time = time.time()

    # Lazy import for LUT application
    if lut_info is not None:
        import cv2

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
        lut_count = 0
        for i, frame_id in enumerate(frame_ids):
            # Source: extracted frame (0-indexed)
            src_frame = temp_path / f"frame_{i:07d}{image_extension}"

            # Destination: original structure
            dest_dir = output_base / frame_id / content_type
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest_frame = dest_dir / f"{camera_id}{image_extension}"

            if src_frame.exists():
                if lut_info is not None:
                    # Read from temp, apply LUT, save to images/ — then move original
                    img_bgr = cv2.imread(str(src_frame))
                    if img_bgr is not None:
                        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                        img_lut = apply_3d_lut_rgb(img_rgb, lut_info)
                        img_out = cv2.cvtColor(img_lut, cv2.COLOR_RGB2BGR)
                        lut_dir = output_base / frame_id / "images"
                        lut_dir.mkdir(parents=True, exist_ok=True)
                        cv2.imwrite(str(lut_dir / f"{camera_id}{image_extension}"), img_out)
                        lut_count += 1

                # Move original to images_no_lut/ (shutil.move works cross-drive on Windows)
                shutil.move(src_frame, dest_frame)
            else:
                print(f"  Warning: Missing extracted frame {i} for {camera_id}", file=sys.stderr)

        move_time = time.time() - move_start

    total_time = time.time() - start_time
    lut_str = f", LUT: {lut_count} imgs" if lut_info else ""
    if verbose:
        print(f"  Camera {camera_id}: ffmpeg {format_duration(ffmpeg_time)}, "
              f"LUT {format_duration(move_time)}, total {format_duration(total_time)}{lut_str}")

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


def apply_masked_images(
    output_base: Path,
    frame_ids: list[str],
    cameras: list[str],
    image_extension: str,
    use_lut_images: bool,
    verbose: bool = False,
) -> bool:
    """Composite images with foreground masks into RGBA PNGs in images_masked/.

    Dancer pixels (mask >= 128) are opaque; background pixels are fully transparent.
    """
    import cv2

    image_dir_name = "images" if use_lut_images else "images_no_lut"
    count = 0
    missing = 0

    for frame_id in frame_ids:
        frame_dir = output_base / frame_id
        image_dir = frame_dir / image_dir_name
        mask_dir = frame_dir / "masks"
        out_dir = frame_dir / "images_masked"
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

            alpha = np.where(mask >= 128, 255, 0).astype(np.uint8)
            bgra = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2BGRA)
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
    apply_masking: bool = False,
) -> bool:

    sequence_input = input_path / sequence_name
    # Output: {output}/sequence_name/images_and_masks/
    sequence_output = output_path / sequence_name / "images_and_masks"

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

    lut_label = " + LUT" if lut_map else ""
    mask_label = " + masking" if apply_masking else ""
    print(f"  Frames: {len(frame_ids)}, Cameras: {len(cameras)}{lut_label}{mask_label}")

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

                camera_lut = lut_map.get(camera_id) if lut_map else None
                future = executor.submit(
                    decode_camera_video,
                    video_path, sequence_output, camera_id,
                    frame_ids, content_type, image_extension, verbose, camera_lut
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

            camera_lut = lut_map.get(camera_id) if lut_map else None
            if not verbose:
                print(f"  Decoding camera {i}/{len(cameras)}: {camera_id}...", end=" ")
                sys.stdout.flush()
            success, duration = decode_camera_video(
                video_path, sequence_output, camera_id,
                frame_ids, content_type, image_extension, verbose, camera_lut
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

    # Extract masks archive if present (also required for --apply-masking)
    has_masks = sequence_data.get("has_masks", False)
    if has_masks or apply_masking:
        masks_path = sequence_input / "masks.tar.zst"
        masks_start = time.time()
        if extract_masks_archive(
            masks_path, sequence_output, frame_ids, cameras, image_extension, verbose
        ):
            print(f"  Extracted masks archive in {format_duration(time.time() - masks_start)}")
        elif apply_masking:
            print(f"  Error: --apply-masking requires masks, but extraction failed", file=sys.stderr)
            return False

    if apply_masking:
        mask_start = time.time()
        if apply_masked_images(
            sequence_output, frame_ids, cameras, image_extension,
            use_lut_images=lut_map is not None, verbose=verbose,
        ):
            print(f"  Created images_masked for {len(frame_ids)} frames in "
                  f"{format_duration(time.time() - mask_start)}")
        else:
            return False

    return success_count > 0


def main():
    parser = argparse.ArgumentParser(
        description="Decode videos back to original frame structure")

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
                        help="Apply color LUT to images_no_lut and save to images/. "
                             "Optionally specify LUT directory (default: project color_lut/)")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Only decode and save the first N frames (also limits ffmpeg decoding)")
    parser.add_argument("--apply-masking", "--apply-mask", action="store_true",
                        help="Create images_masked/ RGBA PNGs with transparent background "
                             "(dancer opaque, background transparent)")

    args = parser.parse_args()

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

    print(f"Decoding {len(sequences)} sequences")
    print(f"Content type: {content_type}")
    if args.max_frames is not None:
        print(f"Max frames: {args.max_frames}")
    if args.apply_masking:
        print("Apply masking: images_masked/ (RGBA, transparent background)")
    print()

    # Process each sequence
    success_count = 0
    for sequence_name, sequence_data in sequences.items():
        print(f"[{sequence_name}]")

        if process_sequence(
            args.input, args.output, sequence_name, sequence_data,
            args.cameras, args.parallel, args.verbose, lut_map,
            args.max_frames, args.apply_masking,
        ):
            success_count += 1

        print()

    print(f"Decoded {success_count}/{len(sequences)} sequences successfully")


if __name__ == "__main__":
    main()
