# Copilot Instructions for FIRST Pipeline

## Project Overview
This pipeline processes data from the Visible Photonic Lantern at SUBARU/SCEXAO. Scripts are designed to run sequentially, each handling a specific stage of data reduction, calibration, and analysis. FITS file keywords are used to determine file roles and processing steps.

## Key Components & Workflow
- **Shell and Python scripts**: Each major step is a separate script (see below).
- **Data flow**: Raw FITS files → Pixel Map → Preprocessing → Wavelength Map → Coupling Maps → Calibration → Image Reconstruction.
- **Script chaining**: Output from one script is often input for the next. Maintain file naming and keyword conventions.

## Essential Scripts & Usage
- `runPL_dfits`: Shell script for FITS file inspection. Requires `dfits` from ESO FITS Tools.
- `runPL_changeKeyword.py`: Modifies FITS keywords for classification.
- `runPL_createPixelMap.py`: Generates pixel maps for alignment/calibration.
- `runPL_preprocess.py`: Applies pixel map, cleans/calibrates raw data.
- `runPL_createWavelengthMap.py`: Maps emission lines to pixels.
- `runPL_createCouplingMaps.py`: Analyzes coupling efficiency.
- `runPL_calibrateNeon.py`: Calibrates Neon spectrum peaks.
- `runPL_imageReconstruction.py`: Reconstructs images from coupling maps.

## Conventions & Patterns
- **FITS keyword usage**: Scripts rely on specific FITS header keywords for file selection and processing logic.
- **Command-line arguments**: Each script expects well-defined CLI arguments (see README for examples).
- **Temporary keyword changes**: Use `runPL_changeKeyword.py` for interim classification; revert when finalized.
- **Output file naming**: Follow naming conventions to ensure downstream compatibility.

## Integration & Dependencies
- **External tools**: `dfits` required for FITS inspection (install from ESO FITS Tools).
- **Python dependencies**: Standard scientific stack (numpy, astropy, etc.); check individual scripts for imports.

## Example Workflow
1. Inspect FITS files: `./runPL_dfits <file>`
2. Update keywords: `python runPL_changeKeyword.py ...`
3. Create pixel map: `python runPL_createPixelMap.py ...`
4. Preprocess data: `python runPL_preprocess.py ...`
5. Generate wavelength/coupling maps: `python runPL_createWavelengthMap.py ...`, `python runPL_createCouplingMaps.py ...`
6. Calibrate Neon: `python runPL_calibrateNeon.py ...`
7. Reconstruct images: `python runPL_imageReconstruction.py ...`

## Tips for AI Agents
- Always check FITS header keywords before processing.
- Maintain script output/input compatibility by following naming and argument conventions.
- Reference the README for up-to-date usage patterns and workflow order.
- When adding new scripts, document CLI usage and expected input/output formats.

---
*For more details, see `README.md` and individual script docstrings.*
