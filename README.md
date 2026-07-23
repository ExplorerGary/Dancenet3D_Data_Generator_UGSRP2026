---
license: cc-by-nc-4.0
language:
  - en
task_categories:
  - image-to-3d
tags:
  - multi-view
  - dance
  - motion-capture
  - 3d-reconstruction
  - novel-view-synthesis
  - colmap
  - synchronized-cameras
  - point-cloud
  - 3d
  - video
annotations_creators:
  - machine-generated
source_datasets:
  - original
pretty_name: DanceNet3D
viewer: false
size_categories:
  - 10K<n<100K
---

# DanceNet3D


A large-scale synchronized multi-view dance dataset captured with 28-29 calibrated cameras at 800x1280 resolution. The dataset contains 46 dance sequences across 3 recording sessions, totaling **36,319 frames** with per-frame camera calibration (COLMAP), foreground masks, optional color correction LUTs, and per-frame sparse point cloud reconstructions.

## Dataset Summary

| Session | Sequences | Cameras | Frames | Size |
|---------|-----------|---------|--------|------|
| s4 | 10 | 29 | 7,723 | ~35 GB |
| s5 | 23 | 29 | 17,928 | ~102 GB |
| s6 | 14 | 28 | 10,668 | ~63 GB |
| **Total** | **46** | | **36,319** | **~200 GB** |

## Data Format

Each session is stored as a directory (`s4/`, `s5/`, `s6/`) containing per-sequence subdirectories. Videos are encoded per-camera using **H.265 (libx265), CRF 18, yuv444p** at 1 fps. Each sequence includes:

- **Per-camera videos**: `{SequenceName}_{CameraID}.mp4` undistorted images without color LUT applied
- **COLMAP calibration**: `colmap/cameras.txt`, `colmap/images.txt` and binary formats
- **Foreground masks**: `masks.tar.zst` binary person segmentation masks generated with SAM3 and with manual quality review
- **Manifest**: `manifest.json` frame IDs, camera lists, and sequence metadata
- **Point clouds**: `pointcloud/{session}/{SequenceName}.tar.zst` per-frame sparse point clouds (`{FrameID}.ply`)

<details>
<summary><b>Directory Structure</b></summary>

```
DanceNet3D/
├── README.md
├── LICENSE
├── requirements.txt
├── video_to_images.py          # Extraction script
├── color_lut/                  # Per-camera color correction LUTs (.cube)
│   ├── 0028.cube
│   └── ...
├── s4/
│   ├── manifest.json
│   └── ...
├── s5/
│   ├── manifest.json
│   ├── AttitudePromenade/
│   │   ├── AttitudePromenade_0028.mp4
│   │   ├── AttitudePromenade_0103.mp4
│   │   ├── ...
│   │   ├── colmap/
│   │   │   ├── cameras.txt
│   │   │   ├── cameras.bin
│   │   │   ├── images.txt
│   │   │   └── images.bin
│   │   └── masks.tar.zst
│   └── ...
├── s6/
│   └── ...
└── pointcloud/
    ├── s4/
    │   └── HouseFootwork.tar.zst
    ├── s5/
    │   ├── BourreeTurns.tar.zst
    │   └── ...
    └── s6/
        ├── BiancaGolden_DropTurn.tar.zst
        └── ...
```

</details>

<details>
<summary><b>Extracted Frame Structure</b></summary>

After running `video_to_images.py`, the data is organized per-frame:

```
output/
└── AttitudePromenade/
    └── images_and_masks/
        ├── 0000001/
        │   ├── images_no_lut/       # Undistorted images (no color correction)
        │   │   ├── 0028.png
        │   │   ├── 0103.png
        │   │   └── ...
        │   ├── images/              # Color-corrected images (present if --apply-lut used)
        │   │   └── ...
        │   ├── masks/               # Binary foreground masks
        │   │   ├── 0028.png
        │   │   └── ...
        │   └── sparse/0/            # COLMAP calibration
        │       ├── cameras.txt
        │       ├── cameras.bin
        │       ├── images.txt
        │       └── images.bin
        ├── 0000002/
        └── ...
```

</details>

## Quick Start

### Download

```bash
# Install Hugging Face CLI
# macOS/Linux
curl -LsSf https://hf.co/cli/install.sh | bash
# Windows
powershell -ExecutionPolicy ByPass -c "irm https://hf.co/cli/install.ps1 | iex"

# Download entire dataset
hf download nyuvideolab/danceNet3D --repo-type dataset --local-dir DanceNet3D

# Download a specific session
hf download nyuvideolab/danceNet3D --repo-type dataset --local-dir DanceNet3D --include "s5/*"

# Download 3DGS only
hf download nyuvideolab/danceNet3D --repo-type dataset --local-dir DanceNet3D --include "3dgs/*"
```

### Extract Frames

**Prerequisites:** Python 3.8+, FFmpeg, zstd

```bash
pip install -r requirements.txt
```

```bash
# Extract a single session
python video_to_images.py --input s5 --output extracted/s5

# Extract specific sequences
python video_to_images.py --input s5 --output extracted/s5 --sequences AttitudePromenade Chacha

# Extract specific cameras only
python video_to_images.py --input s5 --output extracted/s5 --cameras 0028 1362

# Extract with color LUT correction applied
python video_to_images.py --input s5 --output extracted/s5 --apply-lut
```

### Extract Point Clouds

**Prerequisites:** zstd, tar

Per-frame sparse point clouds are available under the `pointcloud/` directory. Each `.tar.zst` archive contains one `.ply` file per frame, named by frame ID (e.g., `0000001.ply`).

```bash
# Extract a single sequence
mkdir -p BourreeTurns && zstd -d pointcloud/s5/BourreeTurns.tar.zst -o - | tar xf - -C BourreeTurns

# Or two-step
zstd -d pointcloud/s5/BourreeTurns.tar.zst -o BourreeTurns.tar
tar xf BourreeTurns.tar
```

### Using with COLMAP

The `colmap/` directory in each sequence contains pre-computed camera intrinsics and extrinsics in COLMAP format. Camera parameters correspond to the undistorted, rotated (portrait orientation) images.

## Sequences

<details>
<summary><b>Session 4 (s4) — 10 sequences, 29 cameras</b></summary>

| Sequence | Frames | Cameras | Status |
|----------|--------|---------|--------|
| 3PointStep | 920 | 29 | Available |
| BartSimpson | 471 | 29 | Available |
| BizMarkie | 703 | 29 | Available |
| HouseFootwork | 937 | 29 | Available |
| HouseFootworkAdvanced | 646 | 29 | Available |
| RoboCop | 920 | 29 | Available |
| RunningMan | 687 | 29 | Available |
| TheRooftop | 983 | 29 | Available |
| ToeTaps | 572 | 29 | Available |
| WuTang | 884 | 29 | Available |

</details>

<details>
<summary><b>Session 5 (s5) — 23 sequences, 29 cameras</b></summary>

| Sequence | Frames | Cameras | Status |
|----------|--------|---------|--------|
| AttitudePromenade | 814 | 29 | Available |
| BasicSuzieQ | 914 | 29 | Available |
| BigKicks | 750 | 29 | Available |
| BourreeTurns | 607 | 29 | Available |
| BourreeTurns2 | 688 | 29 | Available |
| Chacha | 942 | 29 | Available |
| ComboSeated | 903 | 29 | Available |
| DoubleSpiral | 769 | 29 | Available |
| Flair | 752 | 29 | Available |
| Jumping | 552 | 29 | Available |
| Pirouettes | 981 | 29 | Available |
| Portdebras | 765 | 29 | Available |
| PortdebrasSeated | 906 | 29 | Available |
| RonDeJambeAtere | 834 | 29 | Available |
| RonDeJambeAtere2 | 726 | 29 | Available |
| RonDeJambeInAir | 614 | 29 | Available |
| SalsaTurns | 729 | 29 | Available |
| Shoulders | 682 | 29 | Available |
| ShouldersSeated | 697 | 29 | Available |
| SonBasic | 791 | 29 | Available |
| SonBasicSeated | 924 | 29 | Available |
| Turns | 658 | 29 | Available |
| Twists | 930 | 29 | Available |

</details>

<details>
<summary><b>Session 6 (s6) — 14 sequences, 28 cameras</b></summary>

| Sequence | Frames | Cameras | Status |
|----------|--------|---------|--------|
| BiancaGolden_Breathing | 829 | 28 | Available |
| BiancaGolden_Chimee | 610 | 28 | Available |
| BiancaGolden_CircleTurns | 433 | 28 | Available |
| BiancaGolden_DropTurn | 611 | 28 | Available |
| BiancaGolden_GrandPlies | 1061 | 28 | Available |
| BiancaGolden_Ocho | 476 | 28 | Available |
| BiancaGolden_Portedbras | 940 | 28 | Available |
| BiancaGolden_ReleasetoFloor | 641 | 28 | Available |
| BiancaGolden_RollDown | 1,334 | 28 | Available |
| BiancaGolden_SalsaBasic | 450 | 28 | Available |
| BiancaGolden_StyleArms | 588 | 28 | Available |
| BiancaGolden_Swings | 1,025 | 28 | Available |
| BiancaGolden_SyncopatedGroove | 919 | 28 | Available |
| RobertRubama_RussiaCostume | 751 | 28 | Available |

</details>

## Technical Details

- **Resolution**: 800 x 1280
- **Cameras**: 28-29 synchronized Intel RealSense D455
- **Frame rate**: Captured at 30 fps
- **Image format**: PNG
- **Masks**: Binary foreground segmentation via SAM3, stored as PNG
- **Calibration**: COLMAP format
- **Color LUTs**: Per-camera 3D lookup tables for color correction

## Known Limitations

- Some sequences have small frame gaps, 1-2 frames in the middle of the video, due to capture dropouts
- Video encoding at CRF 18 introduces minor compression artifacts
- Color lut for camera 1000 and camera 1362 are generated by hand with Lightroom to get the visually cloest result. All other cameras were calibrated using a Macbeth chart and OpenCV.

## Authors

**NYU Video Lab**
- Shihang Wei
- Mingjian Li
- Ran Gong

**NYU Tandon @ The Yard**
- Reese Anspaugh
- Moira Zhang


## License

This dataset is owned by New York University (NYU) and released under the [Creative Commons Attribution-NonCommercial 4.0 International License (CC-BY-NC-4.0)](https://creativecommons.org/licenses/by-nc/4.0/) with additional supplementary terms. See the full [LICENSE](LICENSE) file for details.


<!-- ## Citation

```bibtex

``` -->
