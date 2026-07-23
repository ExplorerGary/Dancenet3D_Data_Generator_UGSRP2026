#!/usr/bin/env python3
"""Verify example_pcl_bin.bin against COLMAP official format."""

from __future__ import annotations

import importlib.util
import struct
import sys
import tempfile
import urllib.request
from pathlib import Path

import numpy as np

from generate_gsplat_dataset import read_colmap_points3d_binary


def load_official_colmap_reader():
    url = "https://raw.githubusercontent.com/colmap/colmap/3.11.1/scripts/python/read_write_model.py"
    script = Path(tempfile.gettempdir()) / "colmap_read_write_model.py"
    if not script.exists() or script.stat().st_size == 0:
        urllib.request.urlretrieve(url, script)
    spec = importlib.util.spec_from_file_location("colmap_rwm", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_binary_layout(bin_path: Path) -> None:
    size = bin_path.stat().st_size
    with open(bin_path, "rb") as fid:
        num_points = struct.unpack("<Q", fid.read(8))[0]
        expected_size = 8 + num_points * 51
        if size != expected_size:
            raise RuntimeError(
                f"Binary size mismatch: got {size}, expected {expected_size} "
                f"for {num_points} points with empty tracks"
            )
        for _ in range(num_points):
            fid.read(43)
            track_length = struct.unpack("<Q", fid.read(8))[0]
            if track_length:
                fid.read(8 * track_length)
        if fid.read():
            raise RuntimeError("Trailing bytes after last point record")


def main() -> int:
    bin_path = Path(sys.argv[1] if len(sys.argv) > 1 else "example_pcl_bin.bin")
    if not bin_path.exists():
        print(f"Error: file not found: {bin_path}", file=sys.stderr)
        return 1

    check_binary_layout(bin_path)
    print(f"[layout] OK: {bin_path.name} matches COLMAP points3D.bin record size")

    ours = read_colmap_points3d_binary(bin_path)
    print(f"[reader] OK: loaded {len(ours)} points with our reader")

    rwm = load_official_colmap_reader()
    official = rwm.read_points3D_binary(str(bin_path))
    print(f"[reader] OK: loaded {len(official)} points with COLMAP official reader")

    if len(official) != len(ours):
        raise RuntimeError(f"Point count mismatch: official={len(official)}, ours={len(ours)}")

    max_xyz = 0.0
    max_rgb = 0
    for point_id in official:
        o = official[point_id]
        u = ours[point_id]
        max_xyz = max(max_xyz, float(np.max(np.abs(o.xyz - u["xyz"]))))
        max_rgb = max(max_rgb, int(np.max(np.abs(o.rgb - u["rgb"]))))
        if len(o.image_ids) != 0 or len(o.point2D_idxs) != 0:
            raise RuntimeError(f"Point {point_id} has non-empty track in official reader")

    print(f"[compare] max xyz diff: {max_xyz}")
    print(f"[compare] max rgb diff: {max_rgb}")
    if max_xyz != 0.0 or max_rgb != 0:
        raise RuntimeError("Official reader and our reader disagree on point data")

    print(f"VALID: {bin_path} is a legal COLMAP points3D.bin file")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
