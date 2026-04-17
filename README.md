# LiveND: Long-exposure tool based on stacking

I love minimalist landscape photography but don't want to bother bringing an ND1000 filter. OM System's LiveND feature only supports up to ND64 on my OM-3. While Photoshop can handle stacking, as a hobbyist, I find PS too bloated and the learning curve too steep. So I vibe-coded this project to do the stacking for me.

On land it works great as far as the camera is on a tripod.
Using it with my drone is another story. Although I expected the stacking to fully compensate for the drone's movement, it has its limits: it works well for continuous burst shooting (which has a limitation on the total number of pictures and thus total exposure time) but struggles with interval shooting (where you might want unlimited frames for a longer total exposure). Also, because my drone doesn't have a physical ND filter, the shutter speed is too fast. This makes the gap between sensor read-outs non-negligible, leading to obvious computational artifacts (similar to the staccato/"glitch" effect seen in videos shot without a proper ND filter).

Although I only has OM System mirrorless camera, by design this program will be useful for other raw files as well.


--AI-generated project. BELOW IS AI-GENERATED CONTENT---

## Overview

LiveND is a robust computational photography tool designed to simulate the effect of a physical Neutral Density (ND) filter by stacking multiple consecutive photos. By aligning and blending a series of frames (e.g., from a drone or handheld camera), LiveND produces a long-exposure effect—such as silky water or blurry clouds—while keeping static elements sharp.

## Features

- **Format Support:** Works with various compressed (JPG, PNG) and RAW formats using `rawpy` and `OpenCV`.
- **Video Support:** Feed it a single video file (MP4, MOV, etc.) and it will automatically extract and process its frames.
- **SIFT-based Alignment:** Automatic, robust alignment using SIFT and RANSAC, perfect for handheld or drone sequences with slight movement.
- **ROI Selection:** Interactive or automatic Region of Interest (ROI) selection to anchor alignment to a specific static area.
- **Exposure Matching:** Optional exposure normalization to handle changing lighting conditions between frames.
- **Blending Modes:**
  - `median`: Effectively removes moving subjects while creating the blur effect (default).
  - `mean` / `average`: Standard averaging for smooth motion blur.
  - `ema`: Exponential Moving Average.
- **Output Formats:** 16-bit PNG, 16-bit TIFF, or EXR (linear or sRGB).

## Requirements

The script requires Python 3.x and the following core dependencies:
- `numpy`
- `rawpy`
- `opencv-python` (`cv2`)
- `tqdm`

**Optional Dependencies (for specific output formats):**
- `tifffile` (for TIFF export)
- `imageio` (for EXR export)

Install all required dependencies effortlessly using the provided requirements file:
```bash
pip install -r requirements.txt
```

## Usage

```bash
python LiveND.py --glob "path/to/photos/*" --mode median --out result.png
```

### Basic Arguments

- `--glob`: **(Required)** The glob pattern to match input images. Example: `--glob "photos/*.DNG"`
- `--mode`: The blending mode (`median` [default], `mean`, `average`, `ema`).
- `--out`: Path to save the final stacked image as a 16-bit PNG.

### Advanced Arguments

- `--align`: Enabled by default. Uses SIFT features to align images before stacking.
- `--select-roi`: Opens a window to interactively select a static region for alignment. Press SPACE or ENTER after drawing the box.
- `--match-exposure`: Normalizes the exposure of frames to match the reference (first) frame.
- `--wb`: White balance for RAW files (`camera` [default], `auto`, `daylight`).
- `--ema-alpha`: Alpha value for EMA blending (default: 0.2).
- `--linear`: Output linear data instead of sRGB-encoded data for PNG.

### Output Options

- `--tiff`: Path to save a 16-bit TIFF.
- `--tiff-compress`: TIFF compression (`deflate`, `lzw`, `none`).
- `--tiff-srgb`: Apply sRGB encoding to TIFF output.
- `--exr`: Path to save an EXR file.

## Examples

**Example 1: Stack RAW files**
Stack a folder of RAW files using the median method and exposure matching, then output to a 16-bit TIFF:

```bash
python LiveND.py --glob "images/*.ORF" --mode median --match-exposure --tiff "output_stack.tiff"
```

**Example 2: Extract and Stack Video Frames**
Point the glob to a video file, it will automatically unpack the frames as high-quality JPGs and blend them:

```bash
python LiveND.py --glob "images/my_video.MP4" --out result.png
```
