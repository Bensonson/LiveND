# LiveND: Long-exposure tool based on stacking

I love minimalist landscapes but want to skip the filters, especially when I can just use computational photography to emulate instead. OM System's LiveND feature only supports up to ND64 on my OM-3. While Photoshop can handle stacking, as a hobbyist, I find it too bloated and the learning curve too steep. So I vibe-coded this project to do the stacking for me.

I've tested the program with a sequence of images shot from my DJI drone. Although I expected the stacking to fully compensate for the drone's movement, it has its limits. However, the resulting images are sharper than those produced by Affinity Photo 2, and the process is much more RAM-efficient. I plan to use it in the future for shorter drone bursts and with my mirrorless camera.

Although I only has OM System mirrorless camera, by design this program will be useful for other raw files as well.


--BELOW IS AI-GENERATED CONTENT---

## Overview

LiveND is a robust computational photography tool designed to simulate the effect of a physical Neutral Density (ND) filter by stacking multiple consecutive photos. By aligning and blending a series of frames (e.g., from a drone or handheld camera), LiveND produces a long-exposure effect—such as silky water or blurry clouds—while keeping static elements sharp.

## Features

- **Format Support:** Works with various compressed (JPG, PNG) and RAW formats using `rawpy` and `OpenCV`.
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

Install the core dependencies via pip:
```bash
pip install numpy rawpy opencv-python tqdm
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

## Example

Stack a folder of RAW files using the median method and exposure matching, then output to a 16-bit TIFF:

```bash
python LiveND.py --glob "images/*.ORF" --mode median --match-exposure --tiff "output_stack.tiff"
```
