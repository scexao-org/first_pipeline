# first_pipeline

Pipeline to reduce the FIRST data (using the Visible Photonic Lantern) at SUBARU/SCEXAO.
The scripts are designed to run sequentially, each handling a specific stage of data reduction, calibration, and analysis. FITS file keywords are used to determine file roles and processing steps.

## Project Structure

### Directory Organization
```
first_pipeline/
├── classes/          # Data structure classes
│   ├── runPL_class_couplingMap.py
│   ├── runPL_class_dataCube.py
│   └── runPL_class_pixelMap.py
├── libraries/        # Utility functions
│   ├── runPL_library_basic.py
│   ├── runPL_library_io.py
│   ├── runPL_library_linalg.py
│   └── runPL_library_plots.py
└── [main scripts]    # Primary pipeline scripts
```

### Key Components & Workflow
- **Shell and Python scripts**: Each major step is a separate script
- **Data flow**: Raw FITS files → Pixel Map → Preprocessing → Wavelength Map → Coupling Maps → Calibration → Image Reconstruction
- **Script chaining**: Output from one script is often input for the next
- **Modern CLI**: All scripts use `argparse` for professional command-line interfaces

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
Python script to update FITS header keywords for file classification.  
Useful for temporary keyword changes during data organization.

**Usage:**
```bash
python runPL_changeKeyword.py [options] [files...]

# Examples:
python runPL_changeKeyword.py --DATA-TYP=FLAT --X_FIRTYP=RAW *.fits
python runPL_changeKeyword.py --OBJECT="Target Name" --X_FIRTYP=SCIENCE data/*.fits
```

**Key Options:**
- `--DATA-TYP`: Data category (TEST, DARK, FLAT)
- `--X_FIRTYP`: Data product type (RAW, WAVE, FLAT, SCIENCE, PIXELS, SPECTRA)
- `--OBJECT`: Target name
- `--X_FIRWOL`: Wollaston status (IN/OUT)

---

### runPL_create_pixelMap.py
Python script to create a Pixel Map for preprocessing raw data.  
This map is essential for aligning and calibrating the data.

**Usage:**
```bash
python runPL_create_pixelMap.py [options] [file_patterns...]

# Examples:
python runPL_create_pixelMap.py --pixel_min=100 --pixel_max=1600 --pixel_wide=2 *.fits
python runPL_create_pixelMap.py --filter_files data/*.fits
```

**Key Options:**
- `--pixel_min`: Minimum pixel value along wavelength axis (default: 100)
- `--pixel_max`: Maximum pixel value along wavelength axis (default: 1600)  
- `--pixel_wide`: Window half width (default: 2)
- `--filter_files`: Filter out files with insufficient flux

**Input**: Files with `X_FIRTYP=RAW`  
**Output**: Pixel map FITS file and PNG visualization in `pixelmaps/` directory

---

### runPL_make_preproc.py
Python script to preprocess raw FIRST data using the pixel map.  
Applies pixel map alignment and performs initial data cleaning.

**Usage:**
```bash
python runPL_make_preproc.py [options] [files...]

# Examples:
python runPL_make_preproc.py --pixel_map=/path/to/pixel_map.fits /path/to/directory
python runPL_make_preproc.py --object="Target Name" /path/to/files*.fits
python runPL_make_preproc.py --loop 30 /path/to/directory  # Monitor mode
```

**Key Options:**
- `--pixel_map`: Specify pixel map file (auto-detected if not provided)
- `--loop`: Monitor directory and process new files every X seconds
- `--object`: Process only files with specified OBJECT name

**Input**: Files with `X_FIRTYP=RAW` and `X_FIRTYP=PIXELMAP`  
**Output**: Preprocessed files with `X_FIRTYP=PREPROC` in `preproc/` directory

---

### runPL_create_wavelengthMap.py
Python script to create a Wavelength Map from preprocessed data.  
Identifies emission lines and maps them to pixel positions for wavelength calibration.

**Usage:**
```bash
python runPL_create_wavelengthMap.py [options] [files...]

# Examples:
python runPL_create_wavelengthMap.py --wave_list="[753.6, 748.9, 743.9]" *.fits
python runPL_create_wavelengthMap.py --poly_degree=3 data/*.fits
```

**Input**: Files with `X_FIRTYP=WAVE`  
**Output**: Wavelength map for spectral calibration

---

### runPL_create_couplingMaps.py
Python script to create Coupling Maps from preprocessed data.  
Analyzes the coupling efficiency of the photonic lantern channels.

**Usage:**
```bash
python runPL_create_couplingMaps.py [options] [files...]

# Examples:
python runPL_create_couplingMaps.py --cmap_size=25 *.fits
python runPL_create_couplingMaps.py --output_dir=/custom/path data/*.fits
```

**Key Options:**
- `--cmap_size`: Size of coupling map grid
- `--output_dir`: Custom output directory

**Input**: Preprocessed files  
**Output**: Coupling efficiency maps

---

### runPL_make_image.py
Python script to reconstruct images from coupling maps.  
Processes data cubes and generates deconvolved images.

**Usage:**
```bash
python runPL_make_image.py [options] [files...]

# Examples:
python runPL_make_image.py --coupling_map=/path/to/map.fits *.fits
python runPL_make_image.py --deconvolution_method=richardson_lucy data/*.fits
```

**Input**: Coupling maps and data cubes  
**Output**: Reconstructed images and diagnostic plots

---

### runPL_make_astrometry.py
Python script for astrometric analysis from FIRST Photonic Lantern data.  
Performs precise position measurements and astrometric calibrations using coupling maps.

**Usage:**
```bash
python runPL_make_astrometry.py [options] [files...]

# Examples:
python runPL_make_astrometry.py --wollaston IN *.fits
python runPL_make_astrometry.py --coupling_map=/path/to/map.fits --save_individual_frames data/*.fits
python runPL_make_astrometry.py --wavelength_smooth 2 --pyramids *.fits
```

**Key Options:**
- `--wollaston`: Wollaston status (IN for internal, OUT for no wollaston)
- `--coupling_map`: Force selection of specific coupling map file
- `--dark_files`: Select specific dark file(s) to use
- `--wavelength_smooth`: Smoothing factor for wavelength processing (default: 1)
- `--save_individual_frames`: Save individual frames (default: True)
- `--save_individual_wavelength`: Save individual wavelength data (default: False)
- `--pyramids`: Use pyramids for data fitting by coupling map (default: False)

**Input**: Preprocessed FITS files with coupling maps  
**Output**: Astrometric measurements and calibrated position data

---

## Workflow Example

1. **Inspect FITS files**: `./runPL_dfits <file>`
2. **Update keywords**: `python runPL_changeKeyword.py --X_FIRTYP=RAW *.fits`
3. **Create pixel map**: `python runPL_create_pixelMap.py --filter_files *.fits`
4. **Preprocess data**: `python runPL_make_preproc.py /data/directory`
5. **Generate wavelength map**: `python runPL_create_wavelengthMap.py *.fits`
6. **Create coupling maps**: `python runPL_create_couplingMaps.py *.fits`
7. **Perform astrometry**: `python runPL_make_astrometry.py *.fits`
8. **Reconstruct images**: `python runPL_make_image.py *.fits`

## Requirements

- **Python dependencies**: 
  - Core scientific stack: `numpy`, `scipy`, `matplotlib`
  - Astronomy libraries: `astropy`, `astroplan` 
  - Utility libraries: `tqdm` (progress bars), `peakutils` (peak detection)
- **External tools**: `dfits` from ESO FITS Tools for FITS inspection
- **FITS keywords**: Scripts rely on specific header keywords for file selection

## Getting Help

All scripts provide detailed help information:
```bash
python [script_name] --help
```

This displays:
- Complete usage syntax
- Detailed option descriptions  
- Input/output file requirements
- Practical examples
- Default values

## Notes for Development

- **Modern CLI**: All scripts use `argparse` with professional help messages
- **Organized imports**: Classes in `classes/` directory, libraries in `libraries/`
- **Consistent patterns**: Follow existing argument and naming conventions
- **FITS compliance**: Maintain keyword conventions for pipeline compatibility
