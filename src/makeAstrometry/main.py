#!/usr/bin/env python3
# -*- coding: iso-8859-15 -*-
"""
FIRST Pipeline - Astrometric Analysis CLI

Command-line interface for performing high-precision astrometric measurements
from preprocessed FIRST Visible Photonic Lantern data using coupling maps.

Created on Wed May 21 22:56:25 2025
@author: slacour
"""

import argparse
import getpass
import os

from .core import process_astrometric_data, check_observatory_status


def main():
    """
    Main entry point for the astrometric analysis script.
    """
    parser = argparse.ArgumentParser(
        description="Perform high-precision astrometric measurements from FIRST Photonic Lantern data using coupling map analysis.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
FIRST Pipeline Astrometric Analysis Tool

This script performs precision astrometric measurements using photonic lantern
data and coupling maps. It enables position measurements with enhanced precision
for binary stars, exoplanet detection, and fundamental astrometry applications.

Examples:
    %(prog)s --wollaston IN --wavelength_smooth=2 *.fits
    %(prog)s --coupling_map=/path/to/map.fits --pyramids target_data/*.fits
    %(prog)s --save_individual_frames --save_individual_wavelength *.fits
    %(prog)s --dark_files=dark*.fits binary_star_data/*.fits

Pipeline Workflow Integration:
    1. Requires preprocessed data files (X_FIRTYP=PREPROC) and coupling maps
    2. Final analysis step for precision position measurements
    3. Leverages photonic lantern spatial resolution enhancement
    4. Outputs calibrated astrometric measurements for scientific analysis

Input Files:
    - Preprocessed FITS files: X_FIRTYP=PREPROC containing spectral measurements
    - Coupling map files: X_FIRTYP=COUPLINGMAP from runPL_create_couplingMap.py
    - Dark frames for accurate background subtraction
    - Automatic coupling map detection or manual selection

Output Files:
    - Astrometric measurement FITS files with position data
    - Time-resolved position measurements for orbital motion
    - Quality assessment metrics and uncertainty estimates
    - Optional individual frame analysis for high-cadence observations
    - Optional wavelength-resolved astrometry for chromatic effects

Astrometric Features:
    - High-precision position measurements using coupling map analysis
    - Wavelength smoothing for enhanced signal-to-noise in position determination
    - Support for both polarimetry (Wollaston IN) and photometry (OUT) modes
    - Pyramidal fitting options for enhanced spatial resolution
    - Quality assessment and uncertainty quantification

Scientific Applications:
    - Binary star orbit determination with enhanced precision
    - Exoplanet astrometric detection and characterization
    - Proper motion measurements for stellar kinematics
    - Reference frame calibration and maintenance
    - Fundamental astrometry for parallax determination

Note: Astrometric precision depends critically on coupling map quality and
instrument stability. Review quality metrics and systematic effects carefully
for high-precision applications.
        """
    )

    # Add positional argument for files
    parser.add_argument('files', nargs='*', default=['*.fits'],
                       help='FITS files to process (supports wildcards)')

    # Add optional arguments
    parser.add_argument("--object_name",
                       help="Selection of the data by the Object name")
    parser.add_argument("--wollaston", 
                       help="Wollaston status. Use IN for internal or OUT for no wollaston (default: first in the list)")
    parser.add_argument("--coupling_map", 
                       help="Force to select which coupling map file to use (default: the one in the directory)")
    parser.add_argument("--dark_files", 
                       help="Select one or more specific dark(s) files to use")
    parser.add_argument("--wavelength_smooth", type=int, default=1,
                       help="Smoothing factor for wavelength (default: %(default)s)")
    parser.add_argument("--save_individual_frames", action="store_true", default=True,
                       help="Save individual frames (default: %(default)s)")
    parser.add_argument("--save_individual_wavelength", action="store_true", default=False,
                       help="Save individual wavelength (default: %(default)s)")
    parser.add_argument("--pyramids", action="store_true", default=False,
                       help="Use pyramids for data fitting by coupling map (default: %(default)s)")

    # Check for interactive development environment
    if (("VSCODE_PID" in os.environ or os.environ.get('TERM_PROGRAM') == 'vscode') 
        or os.environ.get('SPYDER_DEBUG_FILE')):
        print("Running in development environment")
        
        # Use core defaults
        from .core import get_development_defaults
        defaults = get_development_defaults()
        file_patterns = defaults['file_patterns']
        dark_patterns = [defaults['dark_patterns']] if defaults['dark_patterns'] else None
        coupling_map = defaults['coupling_map']
        object_name = defaults['object_name']
        wollaston = defaults['wollaston']
        wavelength_smooth = defaults['wavelength_smooth']
        save_individual_frames = defaults['save_individual_frames']
        save_individual_wavelength = defaults['save_individual_wavelength']
        pyramids = defaults['pyramids']
    else:
        # Parse command line arguments
        args = parser.parse_args()
        file_patterns = args.files if args.files else ['*.fits']
        object_name = args.object_name
        wavelength_smooth = args.wavelength_smooth
        dark_patterns = [args.dark_files] if args.dark_files else None
        wollaston = args.wollaston
        save_individual_frames = args.save_individual_frames
        save_individual_wavelength = args.save_individual_wavelength
        pyramids = args.pyramids
        coupling_map = args.coupling_map

    try:
        # Check observatory status
        status = check_observatory_status()
        print(status)

        print(f"Processing astrometric data with patterns: {file_patterns}")
        
        # Run the astrometric analysis
        results = process_astrometric_data(
            file_patterns=file_patterns,
            object_name=object_name,
            dark_patterns=dark_patterns,
            coupling_map=coupling_map,
            wavelength_smooth=wavelength_smooth,
            wollaston=wollaston,
            save_individual_frames=save_individual_frames,
            save_individual_wavelength=save_individual_wavelength,
            pyramids=pyramids
        )
        
        print(f"Astrometric analysis completed successfully!")
        print(f"Processed {len(results['results'])} file(s)")
        
        for i, result in enumerate(results['results']):
            print(f"- File {i+1}: {result['output_filename']}")
            print(f"  Stars detected: {result['star_detected'].sum() if hasattr(result['star_detected'], 'sum') else len(result['star_detected'])} frames")
            
    except Exception as e:
        print(f"Error in astrometric analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()