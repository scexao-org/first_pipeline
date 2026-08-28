# first_pipeline

Pipeline to reduce the FIRST data (using the Visible Photonic Lantern) at SUBARU/SCEXAO.
The scripts are designed to run sequentially, each handling a specific stage of data reduction, calibration, and analysis. FITS file keywords are used to determine file roles and processing steps.

## ✅ Status: All Scripts Working

All `runPL_*` scripts have been updated and are fully functional after recent package restructuring. All entry points work correctly when installed with `pip install -e .`

## Project Structure

### Modern Modular Architecture with Interactive Development Support

The pipeline uses a modular structure where each script is organized as its own subpackage under a `src/` layout with **core algorithms separated from CLI interfaces**. This enables both command-line usage and interactive development in VS Code, Jupyter notebooks, or Python REPL.

```
first_pipeline/
├── README.md                     # This documentation
├── setup.py                      # Package setup and installation  
├── requirements.txt              # Python dependencies
├── runPL_dfits                   # FITS inspection shell script
└── src/                          # Source code directory (src layout)
    ├── first_pipeline_shared/    # Shared components across all modules
    │   ├── __init__.py
    │   ├── classes/              # Data structure classes
    │   │   ├── __init__.py
    │   │   ├── runPL_class_couplingMap.py
    │   │   ├── runPL_class_dataCube.py
    │   │   ├── runPL_class_pixelMap.py
    │   │   ├── runPL_class_flatMap.py
    │   │   ├── runPL_class_waveMap.py
    │   │   └── runPL_class_preproc.py
    │   └── libraries/            # Utility functions
    │       ├── __init__.py
    │       ├── runPL_library_io.py
    │       ├── runPL_library_linalg.py
    │       └── runPL_library_plots.py
    ├── changeKeyword/            # FITS keyword modification module
    │   ├── __init__.py
    │   ├── main.py              # CLI interface with argparse
    │   └── run_changeKeyword.py # Core algorithms & development defaults
    ├── createPixelMap/           # Pixel mapping module
    │   ├── __init__.py
    │   ├── main.py              # CLI interface with argparse
    │   └── run_createPixelMap.py # Core algorithms & development defaults
    ├── makePreproc/              # Data preprocessing module
    │   ├── __init__.py
    │   ├── main.py              # CLI interface with argparse
    │   └── run_makePreproc.py   # Core algorithms & development defaults
    ├── createFlatMap/            # Flat field mapping module
    │   ├── __init__.py
    │   ├── main.py              # CLI interface with argparse
    │   └── run_createFlatMap.py # Core algorithms & development defaults
    ├── createWaveMap/            # Wavelength mapping module
    │   ├── __init__.py
    │   ├── main.py              # CLI interface with argparse
    │   └── run_createWaveMap.py # Core algorithms & development defaults
    ├── createCouplingMap/        # Coupling efficiency mapping module
    │   ├── __init__.py
    │   ├── main.py              # CLI interface with argparse
    │   └── run_createCouplingMap.py # Core algorithms & development defaults
    ├── makeImage/                # Image reconstruction module
    │   ├── __init__.py
    │   ├── main.py              # CLI interface with argparse
    │   └── run_makeImage.py     # Core algorithms & development defaults
    ├── makeAstrometry/           # Astrometric processing module
        ├── __init__.py
        ├── main.py              # CLI interface with argparse
        └── run_makeAstrometry.py # Core algorithms & development defaults
    └── makeHalphaImaging/        # H-alpha continuum-subtracted imaging module
        ├── __init__.py
        ├── main.py              # CLI interface with argparse
        └── run_makeHalphaImaging.py # Core algorithms & development defaults
```

### Core/CLI Separation Benefits

1. **Interactive Development**: Core algorithms can be imported and used directly in VS Code, Jupyter notebooks, or Python REPL
2. **Development Defaults**: Automatic detection of development environment with user-specific file paths
3. **Clean Architecture**: CLI logic separated from scientific algorithms for better maintainability
4. **Flexible Usage**: Choose between command-line tools or programmatic access to algorithms
5. **Debugging Support**: Easy to test individual functions and step through code interactively
6. **Notebook Ready**: Perfect for exploratory data analysis and algorithm development

### Interactive Development Support

Each module provides both CLI and programmatic interfaces:

**Command Line Interface** (via `main.py`):
```bash
# Traditional CLI usage
runPL_make_preproc --object="HD 164461" /path/to/files*.fits
runPL_create_couplingMap --wavelength_smooth=7 *.fits
```

**Interactive Usage** (via `run_*.py` modules):
```python
# Import functions directly from run_* modules
from createCouplingMap.run_createCouplingMap import run_createCouplingMap

# Use with your own parameters
couplingMap, datalist = run_createCouplingMap(
    file_patterns=["/path/to/preproc/*.fits"],
    wavelength_smooth=1,
    wavelength_bin=1,
    Nsingular=114
)
```

### VS Code Interactive Mode (Direct `run_*.py` Execution)

Another supported workflow is to execute directly the Python module used by each `main.py` entry point.
For example, `runPL_create_couplingMap` calls `run_createCouplingMap`, which you can run directly in VS Code.

1. Open `src/createCouplingMap/run_createCouplingMap.py` in VS Code.
2. Select your pipeline interpreter (same one used for `pip install -e .`).
3. Run the file in the Interactive Window (`Run Current File in Interactive Window`) or execute `#%%` cells.
4. Edit the parameters in the `if __name__ == "__main__":` block, or call `run_createCouplingMap(...)` manually from a cell.

Example cell in VS Code Interactive Window:

```python
from createCouplingMap.run_createCouplingMap import run_createCouplingMap

couplingMap, datalist = run_createCouplingMap(
    file_patterns=["/Users/slacour/DATA/LANTERNE/20251230/preproc/*.fits"],
    object_name=None,
    wavelength_smooth=1,
    wavelength_bin=1,
    Nsingular=19*6,
    use_pyramids=False,
    center_data=False,
)

print(couplingMap.filename)
```

This interactive path is useful for debugging, parameter tuning, and quick algorithm checks without going through the CLI parser.

### Autonomous Development Defaults System

Each `run_*.py` module contains user-specific development defaults that are automatically applied to:
- **VS Code** with Python extension
- **Spyder** IDE
- **Jupyter** notebooks

The pipeline auto-detects your username and loads corresponding default paths:

```python
# Development defaults are automatically applied based on user:
# - slacour: Uses /Users/slacour/DATA/LANTERNE/... paths
# - jsarrazin: Uses /home/jsarrazin/Bureau/PLDATA/... paths  
# - ehuby: Uses /home/ehuby/WORK/DATA/FIRST-PL/... paths

# No parameters needed - functions work autonomously with defaults
from createPixelMap.run_createPixelMap import run_createPixelMap
result = run_createPixelMap()  # Automatically uses your user's default paths!

# You can still override any parameter as needed
result = run_createPixelMap(
    pixel_min=100, 
    pixel_max=1600,
    file_patterns=["/custom/path/*.fits"]
)

# Each script can also be run directly from the command line for development
# python -m first_pipeline.src.createPixelMap.run_createPixelMap
```

### Code Organization: Separation of Concerns

Each module folder contains:
- **`main.py`**: CLI interface for command-line usage
- **`run_*.py`**: Core processing logic with autonomous development defaults
  - Can be imported and used programmatically
  - Contains `if __name__ == "__main__":` block for direct execution
  - Auto-detects development environment and applies user defaults
  - Fully functional without requiring CLI

This separation allows using functions interactively in notebooks or IDEs without CLI overhead.

1. **Clear Separation of Concerns**: Each script is a self-contained module with its own namespace
2. **Interactive Development**: Direct access to core algorithms without CLI overhead
3. **Development Efficiency**: Automatic defaults for common development scenarios  
4. **Shared Code Reuse**: Common functionality in `first_pipeline_shared` accessible to all modules
5. **Better Maintainability**: Changes to one module don't affect others; easier to debug and extend
6. **Import Safety**: No naming conflicts between modules; each has its own scope
7. **Standard Python Structure**: Follows modern Python packaging conventions (src layout)
8. **CLI Compatibility**: All existing `runPL_*` commands continue to work exactly as before
9. **Notebook Ready**: Perfect for research and algorithm development workflows

### Key Components & Workflow
- **Modular scripts**: Each major step is now a separate subpackage module with CLI (`main.py`) and core algorithms (`run_*.py`)
- **Interactive development**: Core functions available for direct use with automatic development defaults
- **Shared components**: Common classes and libraries in `first_pipeline_shared`
- **Data flow**: Raw FITS files → Pixel Map → Preprocessing → Flat Map → Wavelength Map → Coupling Maps → Astrometry → Image Reconstruction
- **Flexible usage**: Choose between command-line tools or programmatic access to algorithms
- **Script chaining**: Output from one script is input for the next (unchanged workflow)
- **Modern CLI**: All scripts use `argparse` for professional command-line interfaces
- **Development efficiency**: Automatic user detection and path defaults for common development scenarios

## Installation

### Package Installation (Recommended)

1. **Clone the repository:**
   ```bash
   git clone https://github.com/scexao-org/first_pipeline.git
   cd first_pipeline
   ```

2. **Install the package in development mode:**
   ```bash
   pip install -e .
   ```
   
3. **Install ESO FITS Tools (required for `runPL_dfits`):**
   ```bash
   # On macOS with Homebrew:
   brew install cfitsio
   
   # Or install from source:
   # See: https://github.com/granttremblay/eso_fits_tools
   ```

4. **Verify installation:**
   ```bash
   runPL_changeKeyword --help
   runPL_create_pixelMap --help
   # All runPL_* commands should work from any directory
   ```

**Note**: Installed commands can be run from any directory as `runPL_changeKeyword`, `runPL_create_pixelMap`, etc.

## Interactive Development & Core Functions

The FIRST Pipeline provides both traditional command-line interfaces and modern interactive development capabilities. Each module's core algorithms can be imported and used directly in development environments.

### Development Environment Detection

The pipeline automatically detects when you're running in development environments (VS Code, Spyder, Jupyter) and provides user-specific default parameters:

```python
# No setup required - the pipeline detects your environment and user automatically!

# VS Code Interactive Window
from makePreproc.run_makePreproc import run_preprocess
results = run_preprocess()  # Uses your user's default paths automatically

# Jupyter Notebook
from createCouplingMap.run_createCouplingMap import run_createCouplingMap
coupling_map, datalist = run_createCouplingMap()  # Automatic user detection & defaults

# Python REPL/Script
from makeImage.run_makeImage import process_image_reconstruction_data
images = process_image_reconstruction_data()  # Works out of the box!
```

### User-Specific Development Defaults

The system provides different default paths for each user:

- **slacour**: `/Users/slacour/DATA/LANTERNE/...` 
- **jsarrazin**: `/home/jsarrazin/Bureau/PLDATA/...`
- **ehuby**: `/home/ehuby/WORK/DATA/FIRST-PL/...`

### Core Function Examples

Each module provides powerful core functions for interactive use:

**Pixel Map Creation:**
```python
from createPixelMap.run_createPixelMap import run_createPixelMap

# Use development defaults
raw_image, traces, header, x_found, y_found = run_createPixelMap()

# Or customize parameters
result = run_createPixelMap(
    folder="/custom/path",
    pixel_min=100,
    pixel_max=1600,
    pixel_wide=2
)
```

**Preprocessing:**
```python
from makePreproc.run_makePreproc import run_preprocess

# Automatic development defaults
processed_files = run_preprocess()

# Custom parameters  
processed_files = run_preprocess(
    file_patterns=["/path/to/raw/*.fits"],
    pixel_map="/path/to/pixel_map.fits",
    object_name="HD 164461"
)
```

**Wavelength Calibration:**
```python
from createWaveMap.run_createWaveMap import run_createWaveMap

# Development defaults
waveMap = run_createWaveMap()
print(f"Wavelength map saved: {waveMap.filename}")
```

**Coupling Map Generation:**
```python
from createCouplingMap.run_createCouplingMap import run_createCouplingMap

# Full SVD analysis with defaults
coupling_map, datalist = run_createCouplingMap()
print(coupling_map.filename)
```

**Image Reconstruction:**
```python
from makeImage.run_makeImage import process_image_reconstruction_data

# Reconstruct images with automatic setup
result = process_image_reconstruction_data()
print(f"Images reconstructed: {result['output_filename']}")

# Access reconstruction data
image_data = result['image_data']
figures = result['figures']  # Diagnostic plots
```

**Astrometric Analysis:**
```python
from makeAstrometry.run_makeAstrometry import process_astrometric_data

# High-precision astrometry
result = process_astrometric_data()
positions = result['xy_dev']  # Position measurements
quality = result['star_detected']  # Detection quality

# Multiple files processed
for file_result in result['results']:
    print(f"File: {file_result['output_filename']}")
    print(f"Detected stars: {file_result['star_detected'].sum()}")
```

### Interactive Workflow Benefits

1. **Instant Feedback**: See results immediately without command-line overhead
2. **Easy Debugging**: Step through algorithms, inspect intermediate results  
3. **Parameter Exploration**: Quickly test different settings and configurations
4. **Data Inspection**: Access all internal data structures and diagnostic information
5. **Custom Analysis**: Build on core functions for specialized research workflows
6. **Notebook Integration**: Perfect for research documentation and reproducible science

### Combining CLI and Interactive Usage

You can seamlessly combine command-line tools with interactive development:

```bash
# Use CLI for batch processing
runPL_make_preproc /large/dataset/*.fits

# Then interactively analyze results
```

```python
# Interactive analysis of CLI results
from createCouplingMap.run_createCouplingMap import run_createCouplingMap

# Process specific files interactively with custom parameters
coupling_map, datalist = run_createCouplingMap(
    file_patterns=["/large/dataset/preproc/specific_target*.fits"],
    wavelength_smooth=3,
    Nsingular=120
)
```

## Essential Scripts & Usage

### runPL_dfits
Shell script to quickly inspect the key parameters of a FIRST FITS file.  
**Requirements**: `dfits` from [ESO FITS Tools](https://github.com/granttremblay/eso_fits_tools)

**Usage:**
```bash
./runPL_dfits <path_to_fits_file>
```

---

### runPL_changeKeyword.py
Python script to modify FITS header keywords for FIRST Pipeline classification and processing control.  
Essential tool for classifying files and tracking their processing stages throughout the sequential pipeline workflow. Used for temporary keyword changes during data organization and to ensure proper file selection by downstream scripts.

**Usage:**
```bash
# After package installation:
runPL_changeKeyword [options] [files...]

# Examples (using installed commands):
runPL_changeKeyword --DATA-TYP=FLAT --X_FIRTYP=RAW *.fits
runPL_changeKeyword --OBJECT="HD 164461" --X_FIRTYP=PREPROC target_data/*.fits
runPL_changeKeyword --DATA-TYP=COMPARAISON --X_FIRTYP=RAW neon_calib.fits
runPL_changeKeyword --DATE=DEFAULT --X_FIRTYP=RAW recent_observations/*.fits
```

**Key Options:**
- `--DATA-TYP`: Data classification (FLAT=SuperK data, DARK=background, OBJECT=science targets, ACQUISITION=target acquisition, COMPARAISON=Neon calibration, TEST=validation)
- `--X_FIRTYP`: Processing stage (RAW=unprocessed, PREPROC=preprocessed, PIXELMAP/WAVEMAP/COULPLINGMAP=calibration products)
- `--OBJECT`: Target name for science observations (e.g., "HD 164461", "Beta Pic")
- `--X_FIRMID`: Modulation ID identifying specific modulation pattern
- `--X_FIRTRG`: Camera trigger mode (INT=internal, EXT=external synchronization)
- `--X_FIRWOL`: Wollaston prism status (IN=polarimetry mode, OUT=photometry mode)
- `--GAIN`: Camera gain setting value
- `--D_IMRRA`: Target right ascension in sexagesimal format
- `--D_IMRDEC`: Target declination in sexagesimal format
- `--DATE`: Observation date (use DEFAULT to extract from filename)

**Pipeline Integration:**
- Classify raw files before processing with `--DATA-TYP`
- Track processing stages with `--X_FIRTYP` as files move through pipeline steps
- Extract dates from filenames with `--DATE=DEFAULT` for automatic parsing
- Mark observing configurations with `--X_FIRWOL`, `--X_FIRTRG` for special modes

---

### runPL_create_pixelMap.py
Python script to generate pixel maps essential for FIRST Pipeline spectral trace alignment and calibration.  
Pixel maps detect and calibrate the positions of spectral traces across all fiber channels, enabling proper spectral extraction in downstream processing.

📝 **Interactive Development**: Also available as `run_createPixelMap()` function with automatic development defaults. Perfect for VS Code and Jupyter notebook usage.

**Usage:**
```bash
runPL_create_pixelMap [options] [file_patterns...]

# Examples:
runPL_create_pixelMap --pixel_min=100 --pixel_max=1600 --pixel_wide=2 *.fits
runPL_create_pixelMap --pixel_min=50 --pixel_max=1500 data/*.fits
runPL_create_pixelMap /data/raw/*.fits
```

**Key Options:**
- `--pixel_min`: Minimum pixel value along wavelength axis (default: 100)
- `--pixel_max`: Maximum pixel value along wavelength axis (default: 2100)  
- `--pixel_wide`: Window half width for peak detection (default: 2)

**Pipeline Integration:**
- Processes RAW files to create pixel alignment maps
- Essential first step before any spectral analysis
- Output maps used by runPL_make_preproc.py for spectral extraction
- Automatically handles different Wollaston configurations (38 vs 19 channels)

**Input**: Files with `X_FIRTYP=RAW`  
**Output**: Pixel map FITS file and PNG visualization in `pixelmaps/` directory

---

### runPL_make_preproc.py
Python script to preprocess raw FIRST Photonic Lantern data using pixel maps for spectral extraction and calibration.  
Transforms raw detector images into calibrated spectral data with quality assessment and diagnostic analysis.

📝 **Interactive Development**: Also available as `run_preprocess()` function with automatic development defaults. Ideal for interactive data exploration in development environments.

**Usage:**
```bash
runPL_make_preproc [options] [files...]

# Examples:
runPL_make_preproc --pixel_map=/path/to/pixel_map.fits /path/to/directory
runPL_make_preproc --object="HD 164461" /path/to/files*.fits
```

**Key Options:**
- `--pixel_map`: Specify pixel map file (auto-detected if not provided)
- `--object`: Process only files with specified OBJECT name

**Pipeline Integration:**
- Requires raw files (X_FIRTYP=RAW) and pixel maps (X_FIRTYP=PIXELMAP)
- Essential step before flat field, wavelength, and coupling map generation
- Quality metrics guide downstream data acceptance decisions

**Input**: Raw files with `X_FIRTYP=RAW` and pixel maps with `X_FIRTYP=PIXELMAP`  
**Output**: Preprocessed files with `X_FIRTYP=PREPROC` in `preproc/` directory plus diagnostic figures

---

### runPL_create_flatMap.py
Python script to generate flat field calibration maps from SuperK data for FIRST Pipeline photometric correction.  
Creates gain coefficients and quality metrics for pixel-to-pixel sensitivity correction using linear regression analysis.

**Usage:**
```bash
runPL_create_flatMap [options] [files...]

# Examples:
runPL_create_flatMap --wollaston IN --dark_files=dark*.fits flat_data/*.fits
runPL_create_flatMap --dark_files=/path/to/darks/*.fits *.fits
```

**Key Options:**
- `--wollaston`: Wollaston status (IN for polarimetry, OUT for photometry)
- `--dark_files`: Select specific dark file(s) for background subtraction

**Pipeline Integration:**
- Essential calibration step before coupling map generation
- Uses preprocessed flat field and dark files
- Output maps enable photometric correction in downstream analysis

**Input**: Preprocessed flat field files with `X_FIRTYP=PREPROC` and `DATA-TYP=FLAT`  
**Output**: Flat field maps with gain coefficients and quality metrics in `flatmaps/` directory

---

### runPL_create_waveMap.py
Python script to generate wavelength calibration maps from Neon emission line spectra for FIRST Pipeline spectral calibration.  
Detects emission lines, fits polynomial wavelength solutions, and generates 2D wavelength mapping with aberration correction.

**Usage:**
```bash
runPL_create_waveMap [options] [files...]

# Examples:
runPL_create_waveMap --wollaston IN --flatMap=/path/to/flat.fits *.fits
runPL_create_waveMap --Nexclude 3 --dark_files=dark*.fits neon_data/*.fits
```

**Key Options:**
- `--wollaston`: Wollaston status (IN for polarimetry, OUT for photometry)
- `--flatMap`: Select specific flat map file for enhanced calibration
- `--dark_files`: Select specific dark file(s) for background subtraction
- `--Nexclude`: Number of wavelength peaks to exclude from fit for outlier rejection (default: 4)

**Pipeline Integration:**
- Uses Neon calibration files and flat field maps for accurate calibration
- Essential for spectral analysis in downstream scripts
- Output maps enable precise wavelength calibration of science observations

**Input**: Neon calibration files with `X_FIRTYP=PREPROC` and `DATA-TYP=COMPARAISON`  
**Output**: Wavelength map with polynomial coefficients and aberration correction in `output/wave/` directory

---

### runPL_create_couplingMap.py
Python script to generate coupling efficiency maps from preprocessed FIRST Photonic Lantern data using SVD analysis.  
Analyzes coupling efficiency between telescope focal plane and photonic lantern channels, essential for image reconstruction.

📝 **Interactive Development**: Also available as `run_createCouplingMap()` function with automatic development defaults. Excellent for algorithm development and parameter optimization.

**Usage:**
```bash
runPL_create_couplingMap [options] [files...]

# Examples:
runPL_create_couplingMap --object_name="HD 164461" --wavelength_smooth=7 *.fits
runPL_create_couplingMap --modID=1 --modScale=2 --wollaston=IN data/*.fits
runPL_create_couplingMap --flatMap=/path/to/flat.fits --waveMap=/path/to/wave.fits *.fits
```

**Key Options:**
- `--object_name`: Select specific science target for processing
- `--wavelength_smooth`: Smoothing factor for wavelength processing (default: 7)
- `--wavelength_bin`: Binning factor for wavelength (default: 20)
- `--Nsingular`: Number of SVD singular values to retain (default: 19*6)
- `--modID/modScale`: Choose specific modulation patterns
- `--wollaston`: Wollaston status (IN for polarimetry, OUT for photometry)

**Pipeline Integration:**
- Requires preprocessed data, flat field maps, and wavelength calibration
- Critical step for converting fiber measurements to sky coordinates
- Output enables image reconstruction and astrometric analysis

**Input**: Preprocessed files with `X_FIRTYP=PREPROC`  
**Output**: Coupling maps with `X_FIRTYP=COUPLINGMAP` in `../couplingmaps/` directory plus PDF diagnostics

---

### runPL_make_image.py
Python script to reconstruct astronomical images from FIRST Photonic Lantern fiber measurements using coupling map inversion.  
Transforms fiber-based measurements into traditional astronomical images for conventional analysis techniques.

**Usage:**
```bash
runPL_make_image [options] [files...]

# Examples:
runPL_make_image --object_name="HD 164461" --wavelength_smooth=7 *.fits
runPL_make_image --coupling_map=/path/to/map.fits --modID=1 data/*.fits
runPL_make_image --save_individual_frames --save_individual_wavelength *.fits
```

**Key Options:**
- `--object_name`: Select specific target for reconstruction
- `--coupling_map`: Force selection of specific coupling map file
- `--wavelength_smooth`: Control spectral smoothing for noise reduction (default: 7)
- `--modID/modScale`: Choose optimal modulation patterns
- `--save_individual_frames`: Generate time-resolved image sequences (default: True)
- `--save_individual_wavelength`: Create spectral image cubes (default: False)
- `--wollaston`: Wollaston status (IN for polarimetry, OUT for photometry)

**Pipeline Integration:**
- Final step: converts fiber measurements to spatial images
- Enables conventional image analysis of photonic lantern data
- Results comparable with traditional imaging instruments

**Input**: Coupling maps and preprocessed data cubes  
**Output**: Reconstructed images, residuals, and diagnostic plots with optional frame sequences

---

### runPL_make_astrometry.py
Python script to perform high-precision astrometric measurements from FIRST Photonic Lantern data using coupling map analysis.  
Enables sub-milliarcsecond position measurements for binary stars, exoplanet detection, and precision astrometry applications.

📝 **Interactive Development**: Also available as `process_astrometric_data()` function with automatic development defaults. Perfect for research analysis and algorithm development.

**Usage:****
```bash
runPL_make_astrometry [options] [files...]

# Examples:
runPL_make_astrometry --wollaston IN --wavelength_smooth=2 *.fits
runPL_make_astrometry --coupling_map=/path/to/map.fits --pyramids target_data/*.fits
runPL_make_astrometry --save_individual_frames --save_individual_wavelength *.fits
```

**Key Options:**
- `--wollaston`: Wollaston status (IN for polarimetry, OUT for photometry)
- `--coupling_map`: Force selection of specific coupling map file
- `--dark_files`: Select specific dark file(s) for background subtraction
- `--wavelength_smooth`: Smoothing factor for position determination (default: 1)
- `--pyramids`: Enable pyramidal fitting for enhanced spatial resolution (default: False)
- `--save_individual_frames`: Generate time-resolved astrometric sequences (default: True)
- `--save_individual_wavelength`: Analyze chromatic astrometric effects (default: False)

**Pipeline Integration:**
- Final analysis step for precision position measurements
- Leverages photonic lantern spatial resolution enhancement
- Enables astrometry beyond conventional imaging limits

**Input**: Preprocessed FITS files with coupling maps  
**Output**: Astrometric measurements with sub-milliarcsecond precision, quality metrics, and uncertainty estimates

---

### runPL_make_halpha_imaging
Fits a polynomial continuum on the side bands around H-alpha, subtracts it at
each observed position and output, then measures the local Pearson correlation
between integrated H-alpha residual flux and continuum flux across neighbouring
sky positions.

```bash
runPL_make_halpha_imaging --line_center 656.28 --line_width 2.0 preproc/*_P.fits
```

**Input**: OBJECT preprocessed FITS files and a wavelength map; optional flat and dark calibrations.  
**Output**: FITS products in `../halpha_imaging` containing positions, continuum,
residual spectra, integrated H-alpha flux, and correlation values, plus a diagnostic PDF.

---

## Notes for Development

- **Modern CLI**: All scripts use `argparse` with help messages
- **Organized imports**: Classes in `classes/` directory, libraries in `libraries/`
- **Consistent patterns**: Follow existing argument and naming conventions
- **FITS compliance**: Maintain keyword conventions for pipeline compatibility

## Workflow Examples

### Command Line Interface (Traditional)

Using installed commands (after `pip install -e .`):
1. **Inspect FITS files**: `./runPL_dfits <file>`
2. **Classify raw data**: `runPL_changeKeyword --DATA-TYP=OBJECT --X_FIRTYP=RAW *.fits`
3. **Create pixel map**: `runPL_create_pixelMap *.fits`
4. **Preprocess data**: `runPL_make_preproc /data/directory`
5. **Mark preprocessed**: `runPL_changeKeyword --X_FIRTYP=PREPROC preproc/*.fits`
6. **Create flat field map**: `runPL_create_flatMap *.fits`
7. **Generate wavelength map**: `runPL_create_waveMap *.fits`
8. **Create coupling maps**: `runPL_create_couplingMap *.fits`
9. **Perform astrometry**: `runPL_make_astrometry *.fits`
10. **Reconstruct images**: `runPL_make_image *.fits`

### Interactive Development (New)

Using core functions directly for development and exploration:

```python
# Complete interactive workflow with automatic defaults
from createPixelMap.run_createPixelMap import run_createPixelMap  
from makePreproc.run_makePreproc import run_preprocess
from createFlatMap.run_createFlatMap import run_createFlatMap
from createWaveMap.run_createWaveMap import run_createWaveMap
from createCouplingMap.run_createCouplingMap import run_createCouplingMap
from makeAstrometry.run_makeAstrometry import process_astrometric_data
from makeImage.run_makeImage import process_image_reconstruction_data

# 1. Create pixel map (automatic user defaults)
raw_image, traces, header, x_found, y_found = run_createPixelMap()

# 2. Preprocess data  
processed_files = run_preprocess()

# 3. Create flat field map
flatMap = run_createFlatMap()

# 4. Generate wavelength map
waveMap = run_createWaveMap()
print(f"Wavelength map: {waveMap.filename}")

# 5. Create coupling maps
coupling_map, datalist = run_createCouplingMap()
print(coupling_map.filename)

# 6. Perform astrometry
astro_result = process_astrometric_data()
positions = astro_result['results'][0]['xy_dev']

# 7. Reconstruct images  
image_result = process_image_reconstruction_data()
reconstructed = image_result['image_data']
```

### Mixed Workflow (CLI + Interactive)

Combine the best of both approaches:

```bash
# Use CLI for batch processing of large datasets
runPL_make_preproc /large/dataset/*.fits
runPL_create_flatMap /large/dataset/preproc/*.fits
```

```python
# Then interactive analysis for specific targets
from createCouplingMap.run_createCouplingMap import run_createCouplingMap

# Custom coupling map for specific target with fine-tuned parameters
coupling_map, datalist = run_createCouplingMap(
    file_patterns=["/large/dataset/preproc/HD164461*.fits"],
    wavelength_smooth=3,
    Nsingular=150,
    use_pyramids=True
)

# Analyze results interactively
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.imshow(coupling_map.Q[:, :, 0])
plt.title('Q Matrix - First Channel')
plt.colorbar()
```

**Key Pipeline Notes:**
- **CLI**: Perfect for production, batch processing, and established workflows
- **Interactive**: Ideal for development, exploration, debugging, and research analysis  
- **Mixed approach**: Use CLI for heavy lifting, interactive for detailed analysis
- **Development defaults**: When in VS Code/Jupyter, all functions work without parameters
- **Flexibility**: All core functions accept custom parameters to override defaults
- Use `runPL_changeKeyword` to classify files at each stage for proper downstream processing
- Proper keyword classification ensures correct file selection by subsequent pipeline scripts
- Monitor processing stages with `X_FIRTYP` updates as data moves through the workflow
- With package installation, commands can be run from any directory

## Pipeline Classes & Core Functions

The FIRST Pipeline uses object-oriented design with specialized classes for different data products and provides both high-level processing functions and low-level class access. Each class provides consistent interfaces for loading, creating, and saving pipeline data.

### Processing Functions (Recommended)

Each module provides high-level processing functions that handle complete workflows:

```python
# High-level processing functions with development defaults
from makePreproc.run_makePreproc import run_preprocess  
from createWaveMap.run_createWaveMap import run_createWaveMap
from createCouplingMap.run_createCouplingMap import run_createCouplingMap
from makeImage.run_makeImage import process_image_reconstruction_data
from makeAstrometry.run_makeAstrometry import process_astrometric_data

# All functions support both development defaults and custom parameters
waveMap = run_createWaveMap()  # Uses defaults
waveMap = run_createWaveMap(file_patterns=["/custom/path/*.fits"])  # Custom
```

### Available Classes (Advanced Usage)

#### Preproc Class
**Purpose**: Handle preprocessed spectral data with quality metrics and modulation support  
**File**: `classes/runPL_class_preproc.py`

```python
# If using package installation:
from first_pipeline.classes.runPL_class_preproc import Preproc

# If running scripts directly, add to sys.path first:
# import sys
# sys.path.append('path/to/first_pipeline')
# from classes.runPL_class_preproc import Preproc

# Load existing preprocessed file
preproc = Preproc("path/to/preproc_file.fits")
print(f"Data shape: {preproc.data.shape}")
print(f"Has modulation: {preproc.has_modulation_data()}")

# Get quality metrics
quality = preproc.get_quality_summary()
print(quality['interpretations']['centroid_shift'])

# Create from raw data
preproc = Preproc()
output_file = preproc.create_from_raw(raw_file, pixel_map_file, output_dir)

# Save to new file
preproc.save("new_preproc_file.fits")
```

#### Other Data Product Classes
- **PixelMap**: Spectral trace positions and extraction parameters
- **FlatMap**: Flat field calibration data and normalization
- **WaveMap**: Wavelength calibration and spectral mapping
- **CouplingMap**: Fiber coupling efficiency measurements
- **DataCube**: Multi-dimensional spectral data containers

### Class Design Patterns

All pipeline classes follow consistent patterns:

```python
# Standard usage pattern for all classes
data_product = ClassName()           # Create empty
data_product = ClassName(file_path)  # Load existing from fits file
data_product.create_from_X(...)      # Create from source data
data_product.save(output_file)       # Save to FITS file
data_product.return_hdu_list()       # Get FITS HDU list
data_product.return_header()         # Get FITS header
```

**Key Benefits**:
- **High-level Functions**: Complete workflows with automatic defaults for development
- **Class Access**: Fine-grained control for advanced users
- **Encapsulation**: All related functionality in one place
- **Consistency**: Same interface across all data products  
- **Quality control**: Built-in validation and metrics
- **Modularity**: Easy to use individually or in pipelines
- **Interactive Ready**: Perfect for VS Code, Jupyter, and development environments

## Requirements

- **Python dependencies**: 
  - Core scientific stack: `numpy`, `scipy`, `matplotlib`
  - Astronomy libraries: `astropy`, `astroplan` 
  - Utility libraries: `tqdm` (progress bars)
- **External tools**: `dfits` from ESO FITS Tools for FITS inspection
- **FITS keywords**: Scripts rely on specific header keywords for file selection

## Getting Help

All scripts provide detailed help information:

```bash
# Using installed commands (from any directory):
runPL_changeKeyword --help
runPL_create_pixelMap --help
runPL_make_preproc --help
```

This displays:
- Complete usage syntax
- Detailed option descriptions  
- Input/output file requirements
- Practical examples
- Default values

## Summary of Recent Improvements

### 🔬 Interactive Development Architecture
- **Module/CLI Separation**: Each module now has `run_*.py` (algorithms with autonomy) and `main.py` (CLI interface)
- **Autonomous Functions**: Core functions can run standalone with auto-detected development defaults
- **Development Defaults**: Automatic user detection with customized default file paths in `run_*.py`
- **VS Code Ready**: Direct import and execution of core functions in interactive environments
- **Jupyter Compatible**: Perfect for research notebooks and algorithm development
- **Zero Configuration**: Functions work out-of-the-box in development environments

### 🛠 Enhanced Developer Experience  
- **Modular Architecture**: Clean separation of concerns with shared components
- **Flexible Usage**: Choose between command-line tools or programmatic access
- **Interactive Debugging**: Easy algorithm inspection and step-through debugging
- **Parameter Exploration**: Quick testing of different configurations and settings
- **Custom Workflows**: Build specialized analysis pipelines using core functions

### 🚀 Production Ready
- **CLI Compatibility**: All existing `runPL_*` commands work exactly as before
- **Package Installation**: Modern pip-installable package with entry points
- **Professional CLI**: Comprehensive argparse interfaces with detailed help
- **Quality Control**: Built-in validation, metrics, and diagnostic outputs
- **Pipeline Integration**: Seamless file flow between sequential processing steps

### 👥 Multi-User Support
- **User Detection**: Automatic recognition of `slacour`, `jsarrazin`, `ehuby` development environments
- **Custom Paths**: User-specific default file paths for streamlined development
- **Environment Aware**: Detects VS Code, Spyder, Jupyter environments automatically
- **Override Capable**: Custom parameters can override defaults when needed

The FIRST Pipeline now provides the best of both worlds: **powerful command-line tools for production** and **intuitive interactive functions for research and development**. Whether you're processing large datasets or exploring new algorithms, the pipeline adapts to your workflow.
