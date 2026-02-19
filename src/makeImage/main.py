#! /usr/bin/env python3
# -*- coding: iso-8859-15 -*-
"""
FIRST Pipeline - Image Reconstruction CLI

Command-line interface for image reconstruction from preprocessed FIRST data.
This module provides the CLI wrapper for interactive execution of image
reconstruction algorithms using coupling maps.

Created on Wed May 21 22:56:25 2025
@author: slacour
"""

import argparse
import getpass
import os
import numpy as np
from .run_makeImage import process_image_reconstruction_data


def main():
    """
    Main entry point for image reconstruction CLI.
    """
    parser = argparse.ArgumentParser(
        description="Reconstruct astronomical images from FIRST Photonic Lantern fiber measurements using coupling map inversion.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
FIRST Pipeline Image Reconstruction Tool

This script transforms fiber-based photonic lantern measurements back into 
traditional astronomical images using advanced coupling map inversion techniques.
It enables conventional image analysis of photonic lantern observations.

Examples:
    %(prog)s --object_name="HD 164461" --wavelength_smooth=7 *.fits
    %(prog)s --coupling_map=/path/to/map.fits --modID=1 data/*.fits
    %(prog)s --save_individual_frames --save_individual_wavelength *.fits
    %(prog)s --wollaston=IN target_data/*.fits

Pipeline Workflow Integration:
    1. Requires preprocessed data files (X_FIRTYP=PREPROC) and coupling maps
    2. Final step in pipeline: converts fiber measurements to spatial images
    3. Enables traditional image analysis techniques on photonic lantern data
    4. Results can be compared with conventional imaging instruments

Input Files:
    - Preprocessed FITS files: X_FIRTYP=PREPROC containing spectral measurements
    - Coupling map files: X_FIRTYP=COUPLINGMAP from runPL_create_couplingMap.py
    - Dark frames for background subtraction
    - Automatic coupling map detection or manual selection

Output Files:
    - Reconstructed image FITS files with spatial information restored
    - Summed images combining all wavelength channels
    - Residual maps showing reconstruction quality
    - Optional individual frame sequences for time-resolved analysis
    - Optional wavelength-resolved image cubes for spectral analysis

Reconstruction Features:
    - Advanced coupling map inversion algorithms
    - Wavelength smoothing for enhanced signal-to-noise
    - Modulation pattern selection for optimal reconstruction
    - Support for both polarimetry (Wollaston IN) and photometry (OUT) modes
    - Quality assessment through residual analysis

Advanced Options:
    - object_name: Select specific target for reconstruction
    - modID/modScale: Choose optimal modulation patterns
    - wavelength_smooth: Control spectral smoothing for noise reduction
    - save_individual_frames: Generate time-resolved image sequences
    - save_individual_wavelength: Create spectral image cubes

Reconstruction Quality:
    - Residual analysis quantifies reconstruction fidelity
    - Signal-to-noise optimization through parameter tuning
    - Comparison with direct imaging when available
    - Quality metrics guide parameter selection

Technical Notes:
    - Coupling maps define fiber-to-spatial transformation
    - Inversion algorithms handle noise and incomplete sampling
    - Wavelength smoothing balances resolution vs sensitivity
    - Compatible with standard astronomical image analysis tools

Scientific Applications:
    - Exoplanet detection and characterization
    - Binary star observations with enhanced resolution
    - Extended object imaging (circumstellar disks, nebulae)
    - Comparison studies with direct imaging instruments

Note: Review residual maps to assess reconstruction quality.
Optimize smoothing and modulation parameters for best results with your data.
        """
    )

    # VSCode compatibility
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--f", help=argparse.SUPPRESS)

    # Add positional argument for files
    parser.add_argument('files', nargs='*', default=['*.fits'],
                       help='FITS files to process (supports wildcards)')

    # Add optional arguments
    parser.add_argument("--object_name", 
                       help="Selection of the data by the Object name (default: first target in the list)")
    parser.add_argument("--dark_files", 
                       help="Select one or more specific dark(s) files to use")
    parser.add_argument("--coupling_map", 
                       help="Force to select which coupling map file to use (default: the one in the directory)")
    parser.add_argument("--wavelength_smooth", type=int, default=1,
                       help="Smoothing factor for wavelength (default: %(default)s)")
    parser.add_argument("--modID", type=int, 
                       help="Selection of the modulation pattern by user (default: first in the list)")
    parser.add_argument("--modScale", type=int, 
                       help="Selection of the modulation scale by user (default: first in the list)")
    parser.add_argument("--wollaston", 
                       help="Wollaston status. Use IN for internal or OUT for no wollaston (default: first in the list)")
    parser.add_argument("--save_individual_frames", action="store_true", default=True,
                       help="Save individual frames (default: True)")
    parser.add_argument("--save_individual_wavelength", action="store_true",
                       help="Save individual wavelength slices (default: False)")
    parser.add_argument("--Npixels", type=int, default=75,
                       help="Number of pixels for reconstructed images (default: %(default)s)")

    # Parse the arguments
    args = parser.parse_args()
    file_patterns = args.files if args.files else ['*.fits','./preproc/*.fits']

    # Extract the parsed arguments
    object_name = args.object_name
    dark_patterns = args.dark_files
    coupling_map = args.coupling_map
    wavelength_smooth = args.wavelength_smooth
    modID = args.modID
    modScale = args.modScale
    wollaston = args.wollaston
    save_individual_frames = args.save_individual_frames
    save_individual_wavelength = args.save_individual_wavelength
    Npixels = args.Npixels

    # Note: Development environment detection and default paths
    # are handled autonomously in run_makeImage()

    # Process image reconstruction data
    result = process_image_reconstruction_data(
        file_patterns=file_patterns,
        object_name=object_name,
        dark_patterns=dark_patterns,
        coupling_map=coupling_map,
        wavelength_smooth=wavelength_smooth,
        modID=modID,
        modScale=modScale,
        wollaston=wollaston,
        save_individual_frames=save_individual_frames,
        save_individual_wavelength=save_individual_wavelength,
        Npixels=Npixels
    )

    print(f"Image reconstruction completed successfully!")
    print(f"Number of processed files: {len(result['results'])}")
    for i, res in enumerate(result['results']):
        print(f"  File {i+1}: {res['output_filename']}")
        print(f"    Stars detected: {np.sum(res['star_detected'])}/{len(res['star_detected'])} frames")


if __name__ == "__main__":
    main()