# first_pipeline

Pipeline to reduce the FIRST data (using the Visible Photonic Lantern) at SUBARU/SCEXAO.
The scripts are designed to run sequentially, each handling a specific stage of data reduction, calibration, and analysis. FITS file keywords are used to determine file roles and processing steps.

## Project Structure

### Directory Organization
```
first_pipeline/
├── __init__.py       # Package initialization
├── setup.py          # Package setup and installation
├── requirements.txt  # Python dependencies
├── classes/          # Data structure classes
│   ├── __init__.py
│   ├── runPL_class_couplingMap.py
│   ├── runPL_class_dataCube.py
│   ├── runPL_class_pixelMap.py
│   ├── runPL_class_flatMap.py
│   ├── runPL_class_waveMap.py
│   └── runPL_class_preproc.py
├── libraries/        # Utility functions
│   ├── __init__.py
│   ├── runPL_library_io.py
│   ├── runPL_library_linalg.py
│   └── runPL_library_plots.py
└── [main scripts]    # Primary pipeline scripts
```

### Key Components & Workflow
- **Shell and Python scripts**: Each major step is a separate script
- **Data flow**: Raw FITS files → Pixel Map → Preprocessing → Flat Map → Wavelength Map → Coupling Maps → Astrometry → Image Reconstruction
- **Script chaining**: Output from one script is often input for the next
- **Modern CLI**: All scripts use `argparse` for professional command-line interfaces

## Installation

There are two ways to install and use the FIRST Pipeline:

### Option 1: Development Installation (Recommended)

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

### Option 2: Using Scripts Directly

If you prefer to run scripts directly without package installation:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/scexao-org/first_pipeline.git
   cd first_pipeline
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run scripts from the first_pipeline directory:**
   ```bash
   # Run from the first_pipeline/ directory
   python runPL_changeKeyword.py [options] [files...]
   python runPL_create_pixelMap.py [options] [files...]
   # etc.
   ```

**Note**: 
- **Option 1** allows you to run commands from anywhere as `runPL_changeKeyword`, `runPL_create_pixelMap`, etc.
- **Option 2** requires you to be in the `first_pipeline/` directory and use `python script_name.py`

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
# After package installation (Option 1):
runPL_changeKeyword [options] [files...]

# Or running directly (Option 2):
python runPL_changeKeyword.py [options] [files...]

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

**Usage:**
```bash
runPL_create_pixelMap [options] [file_patterns...]

# Examples:
runPL_create_pixelMap --pixel_min=100 --pixel_max=1600 --pixel_wide=2 --filter_files *.fits
runPL_create_pixelMap --pixel_min=50 --pixel_max=1500 data/*.fits
runPL_create_pixelMap --filter_files /data/raw/*.fits
```

**Key Options:**
- `--pixel_min`: Minimum pixel value along wavelength axis (default: 100)
- `--pixel_max`: Maximum pixel value along wavelength axis (default: 2100)  
- `--pixel_wide`: Window half width for peak detection (default: 2)
- `--filter_files`: Quality control to exclude low-flux files for reliable detection

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

**Usage:**
```bash
runPL_make_preproc [options] [files...]

# Examples:
runPL_make_preproc --pixel_map=/path/to/pixel_map.fits /path/to/directory
runPL_make_preproc --object="HD 164461" /path/to/files*.fits
runPL_make_preproc --loop 30 /path/to/directory  # Monitor mode
```

**Key Options:**
- `--pixel_map`: Specify pixel map file (auto-detected if not provided)
- `--loop`: Monitor directory and process new files every X seconds (real-time mode)
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

### runPL_create_wavelengthMap.py
Python script to create a Wavelength Map from preprocessed data.  
Identifies emission lines and maps them to pixel positions for wavelength calibration.

**Usage:**
```bash
runPL_create_wavelengthMap [options] [files...]

# Examples:
runPL_create_wavelengthMap --wave_list="[753.6, 748.9, 743.9]" *.fits
runPL_create_wavelengthMap --poly_degree=3 data/*.fits
```

**Input**: Files with `X_FIRTYP=WAVE`  
**Output**: Wavelength map for spectral calibration

---

### runPL_create_couplingMap.py
Python script to generate coupling efficiency maps from preprocessed FIRST Photonic Lantern data using SVD analysis.  
Analyzes coupling efficiency between telescope focal plane and photonic lantern channels, essential for image reconstruction.

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

**Usage:**
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

## Notes for Development

- **Modern CLI**: All scripts use `argparse` with help messages
- **Organized imports**: Classes in `classes/` directory, libraries in `libraries/`
- **Consistent patterns**: Follow existing argument and naming conventions
- **FITS compliance**: Maintain keyword conventions for pipeline compatibility

## Workflow Example

Using installed commands (after `pip install -e .`):
1. **Inspect FITS files**: `./runPL_dfits <file>`
2. **Classify raw data**: `runPL_changeKeyword --DATA-TYP=OBJECT --X_FIRTYP=RAW *.fits`
3. **Create pixel map**: `runPL_create_pixelMap --filter_files *.fits`
4. **Preprocess data**: `runPL_make_preproc /data/directory`
5. **Mark preprocessed**: `runPL_changeKeyword --X_FIRTYP=PREPROC preproc/*.fits`
6. **Create flat field map**: `runPL_create_flatMap *.fits`
7. **Generate wavelength map**: `runPL_create_waveMap *.fits`
8. **Create coupling maps**: `runPL_create_couplingMap *.fits`
9. **Perform astrometry**: `runPL_make_astrometry *.fits`
10. **Reconstruct images**: `runPL_make_image *.fits`

Or using scripts directly (from `first_pipeline/` directory):
```bash
python runPL_changeKeyword.py --DATA-TYP=OBJECT --X_FIRTYP=RAW *.fits
python runPL_create_pixelMap.py --filter_files *.fits
python runPL_make_preproc.py /data/directory
# etc.
```

**Key Pipeline Notes:**
- Use `runPL_changeKeyword` to classify files at each stage for proper downstream processing
- Proper keyword classification ensures correct file selection by subsequent pipeline scripts
- Monitor processing stages with `X_FIRTYP` updates as data moves through the workflow
- With package installation, commands can be run from any directory

## Pipeline Classes

The FIRST Pipeline uses object-oriented design with specialized classes for different data products. Each class provides consistent interfaces for loading, creating, and saving pipeline data.

### Available Classes

#### Preproc Class
**Purpose**: Handle preprocessed spectral data with quality metrics and modulation support  
**File**: `classes/runPL_class_preproc.py`

```python
from classes.runPL_class_preproc import Preproc

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
- **Encapsulation**: All related functionality in one place
- **Consistency**: Same interface across all data products  
- **Quality control**: Built-in validation and metrics
- **Modularity**: Easy to use individually or in pipelines
- **Maintainability**: Clear separation of concerns

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
# Using installed commands:
runPL_changeKeyword --help
runPL_create_pixelMap --help
runPL_make_preproc --help

# Or using scripts directly:
python runPL_changeKeyword.py --help
python runPL_create_pixelMap.py --help
python runPL_make_preproc.py --help
```

This displays:
- Complete usage syntax
- Detailed option descriptions  
- Input/output file requirements
- Practical examples
- Default values
