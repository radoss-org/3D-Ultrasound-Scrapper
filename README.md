# 3D Ultrasound Scrapper

An interactive tool to inspect, tune, and export 3D ultrasound (and other RAW-like medical) volumes, plus a batch processor for large datasets. Use it to quickly discover working parameters (dimensions, strides, endianness, spacing, windowing) and export to NRRD for downstream processing.

![](https://files.mcaq.me/79hra.gif)

**Based on [RawImageGuess](https://github.com/acetylsalicyl/SlicerRawImageGuess) by acetylsalicyl**

> Caution: This program is for research and engineering use only and is not intended for clinical use.

---

## Why this tool?

When working with legacy or proprietary ultrasound exports (and other modalities saved as RAW-like files), the metadata you need (width, height, depth, bit depth, header/footer bytes, row/slice padding, endianness) is often unknown. 3D Ultrasound Scrapper provides:

- Fast, interactive parameter discovery with immediate visual feedback
- Realtime geometric correction via 8-corner warp and curve-based bending
- Windowing, brightness/contrast, and gamma for quick visibility improvements
- Cropping (top/bottom/left/right) during export/preview
- Header/footer handling, including header-end marker + optional header offset
- NRRD header generation and full volume export
- Saved configurations you can reapply to similar datasets
- A batch GUI to run an entire folder using a configuration template

### Finding Initial Parameters

For completely unknown formats, you may want to try **TomoVision's freeware viewer** first to get initial parameter estimates:
- **TomoVision freeware viewer**: [`https://www.tomovision.com/products/tomovision.html`](https://www.tomovision.com/products/tomovision.html) - Simple drag-and-drop viewer that can help identify basic format parameters
- **Supported Image Formats reference**: [`https://www.tomovision.com/products/format_image.html`](https://www.tomovision.com/products/format_image.html) - Comprehensive list of medical imaging formats and vendors to help identify your data type

Once you have initial guesses from TomoVision (or other sources), use 3D Ultrasound Scrapper for fine-tuning, geometric correction, and NRRD export.

---

## Features

We aim to have a much more comprehensive list of features compared to [RawImageGuess](https://github.com/acetylsalicyl/SlicerRawImageGuess):

- Interactive RAW volume loader with:
  - Pixel types: 8/16-bit (signed/unsigned), float, double, and 24-bit RGB
  - Endianness selection
  - Header/footer bytes, row stride/padding, slice padding, skip-slices
  - Dimensions: width, height, depth
- Image spacing (X/Y/Z), aspect-aware display
- Orientation ops: Flip X/Y/Z and ±90° rotations about X/Y/Z (with spacing swaps)
- Realtime 3D 8-corner warp with symmetry option and on-image corner overlays
- Realtime curve deformations along X/Y/Z halves for bending corrections
- Enhancement controls: brightness, contrast, gamma, and coarse windowing
- Cropping controls: T/B/L/R
- Header end marker + optional header offset for finding/skipping variable raw headers
- NRRD header generator (NHDR+RAW) and direct NRRD writer (with binary payload)
- Configuration save/load (JSON) including UI state, corner warp, curves, spacing
- Batch Processor GUI to process entire folders using a saved config (NRRD or MP4 output)

---

## Installation & Usage

### Option 1: Standalone Executables (Recommended)

No Python installation required! Download the pre-built executables:

- **Main Application**: `dist/3D_Ultrasound_Scrapper.exe`
- **Batch Processor**: `dist/3D_Ultrasound_Scrapper_Batch.exe`

Simply double-click to run, or from command line:

```bash
# Run the main application
dist/3D_Ultrasound_Scrapper.exe

# Run the batch processor
dist/3D_Ultrasound_Scrapper_Batch.exe

# Load with a specific configuration
dist/3D_Ultrasound_Scrapper.exe --config my_config.json
```

### Option 2: Python Scripts

Requirements:
- Python 3.8+
- PyQt5, NumPy, Matplotlib
- SciPy (optional, for higher-quality interpolation)

Install dependencies:

```bash
python -m pip install --upgrade pip
pip install PyQt5 numpy matplotlib scipy
```

Run the Python scripts directly:

```bash
# Main application
python scripts/test.py

# Batch processor
python scripts/batch_processor.py

# Load with configuration
python scripts/test.py --config my_config.json
```

### Building Executables

To build the executables yourself:

```bash
python scripts/build_exe.py
```

This creates standalone `.exe` files in the `dist/` folder.

---

## Quick Start

### Using Executables:
```bash
# Launch the main application
dist/3D_Ultrasound_Scrapper.exe
```

### Using Python:
```bash
# Launch the main application
python scripts/test.py
```

**Then:**
- Load a RAW-like file (Browse File → Load Image), then adjust:
  - Pixel type, endianness
  - Header/footer bytes, row stride/padding, slice stride + skip-slices
  - Width/height/depth
  - Spacing X/Y/Z
  - Optional: flips/rotations, corner warp, curve bending, cropping, enhancements

- When the volume looks correct, export:
  - Generate just a header: "Generate NRRD Header" (NHDR sidecar)
  - Save full volume: "Save as NRRD" (header + binary data)

---

## Configurations

You can save and reload all parameters and UI state as JSON.

- Save current configuration:
  - Click "Save Config" in the app
- Load a configuration at startup:

### Using Executables:
```bash
dist/3D_Ultrasound_Scrapper.exe --config my_config.json
# or
dist/3D_Ultrasound_Scrapper.exe -c my_config.json
```

### Using Python:
```bash
python scripts/test.py --config my_config.json
# or
python scripts/test.py -c my_config.json
```

Configuration captures:
- File params: pixel type, endianness, header/footer sizes
- Dimensions and strides: width, height, depth, row stride/padding, slice padding
- Spacing: X/Y/Z
- Enhancements: brightness, contrast, gamma, window min/max
- Deformation: 8-corner warp positions, symmetry flag, curve parameters
- Orientation history (flips/rotations) and relevant UI state

---

## Batch Processing

Use the Batch Processor GUI to apply a saved configuration to many files and export NRRDs (or MP4 videos):

### Using Executables:
```bash
dist/3D_Ultrasound_Scrapper_Batch.exe
```

### Using Python:
```bash
python scripts/batch_processor.py
```

**Steps:**
1. Input Settings: select an input folder, file pattern (e.g., `*.raw`), and your saved JSON config
2. Output Settings: choose an output folder (unless using "Save next to original file"), output format (NRRD or MP4), and options like "Preserve original filenames"
3. (Optional) File size filters: set min/max file size to limit which files are processed
4. Click "Start Processing," watch the progress bar and log

The batch runner:
- Loads each file with your config
- Applies orientation + optional warp/curves + optional cropping
- Writes NRRD or MP4 (depending on output selection)
- Continues past per-file errors

---

## Supported inputs and modalities

3D Ultrasound Scrapper is format-agnostic for RAW-like data (uncompressed pixel buffers with or without headers/footers and strides). It's frequently useful for reverse-engineering legacy ultrasound exports and other modalities where the structure is known or discoverable by inspection.

**Need help identifying your format?** TomoVision maintains an excellent reference of medical imaging formats and vendors across many modalities (CT, MR, Ultrasound, XA/X-ray, etc.):
- **Format identification reference**: [`https://www.tomovision.com/products/format_image.html`](https://www.tomovision.com/products/format_image.html)

This tool does not decode every proprietary format automatically. Instead, it provides powerful interactive tools to rapidly test hypotheses and fine-tune parameters once you have initial estimates, then export clean NRRD files.

## Example (red sphere RAW)

This repo includes a tiny generated RGB volume example at [example/red_sphere_300x300x300.raw](example/red_sphere_300x300x300.raw). You can use it to sanity-check your pipeline.

- Pixel type: `24 bit RGB`
- Dimensions: `300 x 300 x 300`
- Header/footer: `0 / 0`

You can generate an appropriate config by editing the settings in the app, or by loading one of the provided JSON configs (see `configs/red_sphere_test.json`) and exporting to NRRD.

---

## File Structure

```
project/
├── dist/                          # Standalone executables (ready to run)
│   ├── 3D_Ultrasound_Scrapper.exe          # Main application
│   ├── 3D_Ultrasound_Scrapper_Batch.exe    # Batch processor
├── scripts/                       # Python source files
│   ├── test.py                             # Main application source
│   ├── batch_processor.py                  # Batch processor source
│   └── build_exe.py                        # Build script for executables
├── example/                       # Generated example volumes
│   ├── make_example.py
│   └── red_sphere_300x300x300.raw
├── configs/                       # Configuration files
├── data/                          # (Optional) Additional sample data
└── README.md                     # This documentation
```

---

## Tips

- Test on one file first; then run batch
- Keep backups of original data
- Monitor the status/log areas for warnings and size/stride hints
- Ensure files in a batch share the same structure
- Large NRRDs require adequate disk space
- Use executables for easy deployment without Python installation requirements

---

## Disclaimer

This software is provided "as is," for research/engineering use only, with no guarantee of fitness for clinical applications.