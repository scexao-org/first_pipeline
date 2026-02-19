#! /usr/bin/env python3
# -*- coding: iso-8859-15 -*-
"""
FIRST Pipeline - Flat Field Map Generation CLI Interface

Command-line interface for creating flat field maps from SuperK calibration data.
This script provides the CLI wrapper for the core flat field generation algorithms.

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

# Suppress figure warnings for batch processing
import matplotlib as mpl
mpl.rcParams['figure.max_open_warning'] = 0


def main():
    """
    Main entry point for the flat field map creation script.
    """
    parser = argparse.ArgumentParser(
        description="Generate flat field maps from SuperK data for FIRST Pipeline photometric correction.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
FIRST Pipeline Flat Field Map Generation Tool

This script creates flat field maps essential for photometric calibration from 
SuperK illumination data. Flat field maps correct for pixel-to-pixel sensitivity 
variations and provide gain coefficients for accurate photometric measurements.

Examples:
    %(prog)s --wollaston IN --dark_files=dark*.fits flat_data/*.fits
    %(prog)s --dark_files=/path/to/darks/*.fits *.fits
    %(prog)s --override-flat-keyword --Nflat_smooth=15 *.fits

Pipeline Workflow Integration:
    1. Essential calibration step before coupling map generation
    2. Processes preprocessed flat field and dark files
    3. Output maps enable photometric correction in downstream analysis

Input Files:
    - Flat field data: X_FIRTYP=PREPROC and DATA-TYP=FLAT (SuperK illumination)
    - Corresponding dark frames: X_FIRTYP=PREPROC and DATA-TYP=DARK
    - Files automatically grouped by Wollaston status (IN/OUT)
    - Use --override-flat-keyword to process files without DATA-TYP=FLAT requirement

Output Files:
    - Flat field map FITS files (flatmaps/ directory)
    - Gain coefficient matrices for each spectral channel
    - Quality assessment metrics and fit residuals
    - Diagnostic plots showing calibration quality

Processing Details:
    - Linear regression for gain correction per pixel
    - Dark subtraction for accurate flat field measurement
    - Quality metrics assess calibration reliability
    - Separate processing for polarimetry (Wollaston IN) and photometry (OUT) modes
    - Handles variable illumination patterns from SuperK source

Calibration Quality:
    - Statistical analysis of gain coefficients
    - Residual mapping to identify problematic pixels
    - Quality flags for reliable vs uncertain calibrations
    - Diagnostic plots for visual inspection

Technical Parameters:
    - Nflat_smooth: Smoothing parameter for flat field computation
    - Wollaston status: IN (polarimetry) vs OUT (photometry) modes
    - Dark subtraction: Essential for accurate flat field measurement

Note: Proper flat field calibration is essential for reliable photometry.
Quality assessment plots help identify systematic calibration issues.
        """
    )

    # needed to work in VSC:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--f", help=argparse.SUPPRESS)

    # Add positional argument for files
    parser.add_argument('files', nargs='*', default=['*.fits', './preproc/*.fits'],
                       help='FITS files to process (supports wildcards)')

    # Add optional arguments
    parser.add_argument("--dark_files", 
                       help="Select one or more specific dark files to use")
    parser.add_argument("--wollaston", 
                       help="Wollaston status. Use IN for internal or OUT for no wollaston (default: first in the list of files)")
    parser.add_argument("--Nflat_smooth", default=25, type=int,
                       help="Smoothing parameter for flat field computation [default: 25]")
    parser.add_argument("--override-flat-keyword", action="store_true",
                       help="Override the requirement for DATA-TYP=FLAT keyword in input files")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Import core functions
    from .core import process_flat_field_data
    
    # Extract arguments
    file_patterns = args.files if args.files else ['*.fits','./preproc/*.fits']
    dark_patterns = args.dark_files
    wollaston = args.wollaston
    Nflat_smooth = args.Nflat_smooth
    override_flat_keyword = args.override_flat_keyword

    # Development/interactive mode handling
    if ("VSCODE_PID" in os.environ or os.environ.get('TERM_PROGRAM') == 'vscode' or 
        os.environ.get('SPYDER_DEBUG_FILE')):
        print("Running in compiler")
        if getpass.getuser() == "slacour":
            file_patterns = "/Users/slacour/DATA/LANTERNE/20251125/preproc"
            file_patterns = "/Users/slacour/DATA/LANTERNE/20251231/preproc/firstpl_2025-12-31T00?3*fits"
            file_patterns = "/Users/slacour/DATA/LANTERNE/test_flat/preproc"
            file_patterns = "/Users/slacour/DATA/LANTERNE/raw/20260114/preproc"
            file_patterns = "/Users/slacour/DATA/LANTERNE/raw/20260114/preproc_noedge"
            override_flat_keyword = True
            
            print(f"Development override - file_patterns: {file_patterns}")
            print(f"Development override - override_flat_keyword: {override_flat_keyword}")

        elif getpass.getuser() == "jsarrazin":
            file_patterns = "/home/jsarrazin/Bureau/PLDATA/moreTest/2024-11-21_13-48-32_science_copie/preproc"
            file_patterns = "/home/jsarrazin/Bureau/PLDATA/novembre/les_preproc"
            
        elif getpass.getuser() == "ehuby":
            file_patterns = "/home/ehuby/WORK/DATA/FIRST-PL/2025-05-10/preproc/"

        # Ensure file_patterns is a list
        file_patterns = [file_patterns] if isinstance(file_patterns, str) else file_patterns

    # Process flat field data
    try:
        output_filename = process_flat_field_data(
            file_patterns=file_patterns,
            dark_patterns=dark_patterns,
            wollaston=wollaston,
            Nflat_smooth=Nflat_smooth,
            override_flat_keyword=override_flat_keyword
        )
        
        print(f"Successfully created flat field map: {output_filename}")
        
    except Exception as e:
        print(f"Error processing flat field data: {e}")
        raise


if __name__ == "__main__":
    main()