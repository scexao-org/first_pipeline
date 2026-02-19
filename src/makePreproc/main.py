#! /usr/bin/env python3
# -*- coding: iso-8859-15 -*-
"""
FIRST Pipeline - Data Preprocessing CLI Interface

Command-line interface for preprocessing raw FIRST Visible Photonic Lantern data 
using pixel maps. This script provides the CLI wrapper for the core preprocessing 
algorithms.

Created on Wed May 21 22:56:25 2025
@author: slacour
"""

import os
import argparse
import getpass
import matplotlib

if "VSCODE_PID" in os.environ:
    matplotlib.use('Qt5Agg')
else:
    matplotlib.use('Agg')

from astroplan import Observer
from astropy.time import Time

# Initialize Subaru observatory for day/night detection
subaru = Observer.at_site("Subaru")
now_time = Time.now()
if subaru.is_night(now_time):
    print("It's night at Subaru Observatory.")
else:
    print("It's day at Subaru Observatory.")


def main():
    """
    Main entry point for the preprocessing script.
    """
    parser = argparse.ArgumentParser(
        description="Preprocess raw FIRST Photonic Lantern data using pixel maps for spectral extraction and calibration.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
FIRST Pipeline Data Preprocessing Tool

This script transforms raw detector images into calibrated spectral data by applying
pixel maps for precise spectral extraction. It includes quality assessment and 
diagnostic analysis to ensure reliable data for downstream processing.

Examples:
    %(prog)s --pixel_map=/path/to/pixel_map.fits /path/to/directory
    %(prog)s --object="HD 164461" /path/to/files*.fits
    %(prog)s --loop 30 /path/to/directory  # Monitor mode
    %(prog)s /data/raw/*.fits

Pipeline Workflow Integration:
    1. Requires raw FITS files (X_FIRTYP=RAW) and pixel maps (X_FIRTYP=PIXELMAP)
    2. Applies spectral extraction using pixel map calibration
    3. Outputs preprocessed files (X_FIRTYP=PREPROC) for downstream analysis
    4. Essential step before flat field, wavelength, and coupling map generation

Input Files:
    - Raw FITS files: X_FIRTYP=RAW containing detector images
    - Pixel map files: X_FIRTYP=PIXELMAP from runPL_create_pixelMap.py
    - Automatic pixel map detection or manual selection with --pixel_map

Output Files:
    - Preprocessed FITS files: X_FIRTYP=PREPROC (preproc/ directory)
    - Diagnostic figures showing extraction quality and stability:
      * Pixel map overlay on raw images
      * Centroid shift analysis as function of time
      * Quality control metrics

Processing Features:
    - Spectral trace extraction using calibrated pixel positions
    - Quality control metrics for data validation
    - Centroid shift monitoring for instrument stability
    - Object-based selection for targeted processing
    - Monitor mode (--loop) for real-time processing during observations
    - Automatic handling of different Wollaston configurations

Monitor Mode:
    - Use --loop to continuously monitor directory for new raw files
    - Automatic processing when new files appear
    - Ideal for real-time data reduction during observations
    - Configurable polling interval in seconds

Quality Assessment:
    - Centroid shift tracking detects instrument drift
    - Extraction quality metrics identify problematic data
    - Diagnostic figures enable visual quality control
    - QC flags stored in FITS headers for downstream filtering

Technical Notes:
    - Pixel maps define exact spectral trace positions for extraction
    - Quality metrics guide data acceptance/rejection decisions
    - Compatible with all downstream pipeline scripts
    - Supports both polarimetry and photometry observing modes

Note: Monitor centroid shift plots to assess instrument stability.
Large shifts may indicate mechanical flexure or alignment issues requiring
attention before proceeding with scientific analysis.
        """
    )

    # needed to work in VSC:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--f", help=argparse.SUPPRESS)

    # Add positional argument for files/directories
    parser.add_argument('files', nargs='*', default=['*.fits'],
                       help='Directory or FITS files to process (default: *.fits)')

    # Add optional arguments
    parser.add_argument("--pixel_map", 
                       help="Specify which pixel map FITS file to use (default: auto-detect in directory)")
    parser.add_argument("--object", 
                       help="Specify the OBJECT name of data to reduced based on the FITS header")
    parser.add_argument("--only_with_modulation", action="store_true",
                       help="Also preprocess files that do not have a MODULATION extension in the FITS file.")
    parser.add_argument("--overwrite", action="store_true",
                       help="Overwrite existing preprocessed files if they exist.")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Import core functions
    
    # Extract arguments
    file_patterns = args.files if args.files else ['*.fits']
    pixel_map = args.pixel_map
    object_name = args.object
    only_with_modulation = args.only_with_modulation
    overwrite = args.overwrite

    from .core import run_preprocess

    # Process files
    processed_files = run_preprocess(
        file_patterns=file_patterns,
        pixel_map=pixel_map,
        object_name=object_name,
        only_with_modulation=only_with_modulation,
        overwrite=overwrite
    )
    

if __name__ == "__main__":
    main()