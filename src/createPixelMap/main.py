# ! /usr/bin/env python3
# -*- coding: iso-8859-15 -*-
"""
FIRST Pipeline - Pixel Map Generation CLI Interface

Command-line interface for creating pixel maps essential for preprocessing raw 
FIRST Visible Photonic Lantern data. This script provides the CLI wrapper for 
the core pixel map generation algorithms.

Created on Wed May 21 22:56:25 2025
@author: slacour
"""
#%%
import os
import sys
# Add src directory to path for imports to work in both interactive and package contexts
if os.path.join(os.path.dirname(__file__), '..') not in sys.path:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from astropy.io import fits
import argparse
import numpy as np
import getpass
import matplotlib

if "VSCODE_PID" in os.environ:
    matplotlib.use('Qt5Agg')
else:
    matplotlib.use('Agg')

from first_pipeline_shared.classes.runPL_class_fileList import FileList


def main():
    """
    Main entry point for the pixel map creation script.
    """
    parser = argparse.ArgumentParser(
        description="Generate pixel maps for FIRST Pipeline spectral trace alignment and calibration.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
FIRST Pipeline Pixel Map Generation Tool

This script creates pixel maps that are essential for preprocessing raw FIRST 
Visible Photonic Lantern data. Pixel maps detect and calibrate the positions of 
spectral traces across all fiber channels, enabling proper spectral extraction 
in downstream processing.

Examples:
    %(prog)s --pixel_min=100 --pixel_max=1600 --pixel_wide=2 --filter_files *.fits
    %(prog)s --pixel_min=50 --pixel_max=1500 data/*.fits
    %(prog)s --filter_files /data/raw/*.fits

Pipeline Workflow Integration:
    1. This script processes RAW files to create pixel alignment maps
    2. Output pixel maps are used by runPL_make_preproc.py for spectral extraction
    3. Essential first step before any spectral analysis can be performed

Input Files:
    - Raw FITS files with X_FIRTYP=RAW
    - Automatically separates files by Wollaston status (IN/OUT)
    - Requires sufficient flux for reliable peak detection

Output Files:
    - FITS file with pixel map calibration data (pixelmaps/ directory)
    - PNG visualization of detected spectral traces
    - Diagnostic plots showing peak detection quality

Processing Details:
    - Detects spectral trace peaks across wavelength axis (pixel_min to pixel_max)
    - Uses peak detection with configurable window width (pixel_wide)
    - Automatically determines output channels based on Wollaston status:
      * 38 channels for Wollaston IN (polarimetry mode)
      * 19 channels for Wollaston OUT (photometry mode)
    - Optional flux filtering to ensure reliable peak detection
    - Processes files separately by Wollaston configuration

Technical Parameters:
    - pixel_min/max: Define wavelength axis range for peak detection
    - pixel_wide: Half-width of detection window (full width = 2*pixel_wide+1)
    - filter_files: Quality control to exclude low-flux files

Note: Proper pixel maps are critical for accurate spectral extraction. 
Recommend using --filter_files for reliable results, especially with 
varying observation conditions.
        """
    )

    # needed to work in VSC:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--f", help=argparse.SUPPRESS)

    # Add positional argument for file patterns
    parser.add_argument('file_patterns', nargs='*', default=['*.fits'],
                       help='One or more glob patterns for FITS files (default: *.fits)')

    # Add optional arguments with better defaults and validation
    parser.add_argument("--pixel_min", type=int, default=100,
                       help="Minimum pixel value along wavelength axis (default: %(default)s)")
    parser.add_argument("--pixel_max", type=int, default=2100,
                       help="Maximum pixel value along wavelength axis (default: %(default)s)")
    parser.add_argument("--pixel_wide", type=int, default=2,
                       help="Window half width (default: %(default)s) (full width = 2*pixel_wide+1)")
    # Parse arguments
    args = parser.parse_args()
    
    # Import core functions
    from .run_createPixelMap import run_createPixelMap
    
    # Extract the parsed arguments
    pixel_min = args.pixel_min
    pixel_max = args.pixel_max
    pixel_wide = args.pixel_wide
    file_patterns = args.file_patterns

    run_createPixelMap(pixel_min=pixel_min, pixel_max=pixel_max, pixel_wide=pixel_wide, file_patterns=file_patterns)

if __name__ == "__main__":
    main()
# %%
