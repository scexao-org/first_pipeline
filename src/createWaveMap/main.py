#! /usr/bin/env python3
# -*- coding: iso-8859-15 -*-
"""
FIRST Pipeline - Wavelength Map Generation CLI

Command-line interface for wavelength map generation from Neon calibration spectra.
This module provides the CLI wrapper for interactive execution of wavelength
calibration algorithms.

Created on Wed May 21 22:56:25 2025
@author: slacour
"""

import argparse
from first_pipeline_shared.libraries.runPL_library_cli import check_file_options
import getpass
import os
from .run_createWaveMap import run_createWaveMap, find_wollaston_modes


def main():
    """
    Main entry point for wavelength map generation CLI.
    """
    parser = argparse.ArgumentParser(
        description="Generate wavelength calibration maps from Neon emission line spectra for FIRST Pipeline spectral calibration.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
FIRST Pipeline Wavelength Map Generation Tool

This script creates wavelength calibration maps from Neon comparison lamp spectra.
It detects emission lines, fits polynomial wavelength solutions, and generates 
2D wavelength mapping with aberration correction for precise spectral calibration.

Examples:
    %(prog)s --wollaston IN --flatMap=/path/to/flat.fits *.fits
    %(prog)s --Nexclude 3 neon_data/*.fits --dark_files dark*.fits
    %(prog)s /data/comparison/*.fits

Pipeline Workflow Integration:
    1. Requires preprocessed Neon calibration files (X_FIRTYP=PREPROC, DATA-TYP=COMPARISON)
    2. Uses flat field maps for proper calibration
    3. Output wavelength maps enable spectral analysis in downstream scripts
    4. Essential for accurate wavelength calibration of science observations

Input Files:
    - Neon calibration spectra: X_FIRTYP=PREPROC and DATA-TYP=COMPARISON
    - Corresponding dark frames: X_FIRTYP=PREPROC and DATA-TYP=DARK
    - Flat field maps (optional): for enhanced calibration accuracy
    - Files automatically grouped by Wollaston status (IN/OUT)

Output Files:
    - FITS file with wavelength calibration map (output/wave/ directory)
    - Diagnostic plots showing line detection and polynomial fits
    - Quality assessment figures for calibration validation

Processing Details:
    - Detects Neon emission peaks using advanced peak finding algorithms
    - Fits polynomial wavelength solutions with configurable degree
    - Applies aberration correction for spatial variations
    - Excludes problematic peaks with --Nexclude parameter
    - Handles both polarimetry (Wollaston IN) and photometry (OUT) modes
    - Dark subtraction for accurate line measurement

Calibration Quality:
    - Automatic outlier rejection for robust fitting
    - Residual analysis to assess calibration accuracy  
    - Spatial mapping to handle optical aberrations
    - Quality metrics saved with wavelength maps

Technical Notes:
    - Nexclude: Number of peaks to exclude from fitting (handles outliers)
    - Wollaston status affects channel configuration and processing
    - Polynomial degree optimized for FIRST optical system

Note: Quality wavelength calibration is essential for accurate spectroscopy.
Review diagnostic plots to ensure proper line detection and fitting.
        """
    )

    # VSCode compatibility
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--f", help=argparse.SUPPRESS)

    # Add positional argument for files
    parser.add_argument('files', nargs='*', default=[],
                       help='FITS files to process (supports wildcards)')

    # Add optional arguments
    parser.add_argument("--dark_files", nargs='+', help="Select one or more specific dark(s) files to use")
    parser.add_argument("--flatMap", 
                       help="Select a specific flat Map to use")
    parser.add_argument("--wollaston", 
                       help="Wollaston status: IN or OUT. Default: one wavelength map is made for EACH mode present among the Neon files")
    parser.add_argument('--Nexclude', type=int, default=4,
                       help="Number of wavelength peak to exclude from the fit (default: 4)")
    parser.add_argument('--vacuum', action='store_true',
                       help="Produce the wavelength scale in vacuum instead of standard air (default: air, "
                            "the usual optical convention; H-alpha = 656.28 nm in air, 656.46 nm in vacuum)")
    parser.add_argument('--interpolation', choices=['lanczos3', 'linear'], default='lanczos3',
                       help="Resampling kernel onto the common wavelength grid (default: lanczos3, which "
                            "preserves the line profile; linear broadens it differently for each output)")
    parser.add_argument('--fiberOffsets',
                       help="Reference wavelength map (FITS) from which to take fixed per-fiber x offsets "
                            "(V-groove misalignment). Default: fit them on the Neon data")
    
    # Parse the arguments
    args = parser.parse_args()
    check_file_options(parser, args.files, dark_files=args.dark_files)
    file_patterns = args.files if args.files else ['*.fits','./preproc/*.fits']

    # Extract the parsed arguments
    dark_patterns = args.dark_files
    flat_patterns = args.flatMap
    wollaston = args.wollaston
    Nexclude = args.Nexclude

    # Note: Development environment detection and default paths
    # are handled autonomously in run_createWaveMap()

    # One wavelength map per Wollaston mode (IN and OUT have different numbers of outputs)
    if wollaston is not None:
        modes = [wollaston]
    elif args.fiberOffsets:
        # fixed fibre offsets only make sense for the mode of the reference map
        from astropy.io import fits
        modes = [fits.getheader(args.fiberOffsets).get('X_FIRWOL')]
        print(f"Wollaston mode taken from the --fiberOffsets reference map: {modes[0]}")
    else:
        modes = find_wollaston_modes(file_patterns)
        if len(modes) > 1:
            print(f"Neon files found for Wollaston modes {modes}: one wavelength map per mode.")

    for mode in modes:
        print("\n" + "#" * 72 + f"\n# Wollaston {mode}\n" + "#" * 72)
        waveMap, datalist, residual_rms_nm = run_createWaveMap(
            file_patterns=file_patterns,
            dark_patterns=dark_patterns,
            flat_patterns=flat_patterns,
            wollaston=mode,
            Nexclude=Nexclude,
            fiber_offsets_file=args.fiberOffsets,
            vacuum=args.vacuum,
            interpolation=args.interpolation
        )
        del datalist
        import matplotlib.pyplot as plt
        plt.close('all')   # otherwise the next mode's PDF also contains these figures
        report_result(waveMap, residual_rms_nm, mode)


def report_result(waveMap, residual_rms_nm, mode):
    """Print the outcome of one wavelength map."""
    print(f"Wavelength map (Wollaston {mode}) created successfully: {waveMap.filename}")
    print(f"Wavelength solution residuals (RMS): {residual_rms_nm:.4f} nm")
    if residual_rms_nm > 0.1:
        print("\n" + "!" * 72)
        print("!!! WARNING: FIT RESIDUAL IS TOO LARGE !!!")
        print(f"!!! Residual RMS = {residual_rms_nm:.4f} nm (> 0.1 nm) !!!")
        print("!!! Increase or decrease --Nexclude and run the fit again. !!!")
        print("!" * 72 + "\n")


if __name__ == "__main__":
    main()