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

from .run_makeAstrometry import process_astrometric_data, check_observatory_status


def main():
    """
    Main entry point for the astrometric analysis script.
    """
    parser = argparse.ArgumentParser(
        description="Measure the wavelength-dependent photocenter shift (spectro-astrometry) from preprocessed FIRST Photonic Lantern data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
FIRST Pipeline Spectro-Astrometry Tool

This script recovers the small, wavelength-dependent astrometric shift of a
source across a spectral line (e.g. H-alpha). The known modulation dither is
used to build, around each interior dither point, a local response Jacobian
relating output-flux changes to sky-position changes. A separable (variable-
projection) least-squares solve then eliminates the per-output flat gains
analytically and returns the RA/DEC photocenter shift versus wavelength. The
continuum is estimated both by notched-Hanning smoothing (over a range of
window sizes) and by a low-order polynomial fit on the line's side windows.

Examples:
    %(prog)s preproc/*_HD163296_P.fits
    %(prog)s --wollaston IN --object_name HD142527 preproc/*.fits
    %(prog)s --line_center 656.28 --line_width 1.5 preproc/*.fits
    %(prog)s --wave_files wavemaps/ --dark_files dark*.fits preproc/*.fits

Input Files:
    - Preprocessed FITS files: X_FIRTYP=PREPROC (selected as OBJECT data)
    - Wavelength maps (X_FIRTYP=WAVEMAP) and flat maps (X_FIRTYP=FLATMAP),
      auto-discovered in sibling ../wavemaps and ../flatmaps folders or set
      explicitly with --wave_files / --flat_files
    - Dark frames for background subtraction (default: the input files)

Output Files (written to a sibling ../astrometry folder):
    - ASTROMETRY FITS file (X_FIRTYP=ASTROMETRY) with HDUs:
      WAVE, FLUX_SCALED, ASTROMETRY_SHIFT (per Hanning window), ASTROMETRY_XY
      (over the line), and X_HANNING
    - Multi-page PDF with the RA/DEC astrometry vs wavelength (full band and
      zoomed on the line) and the RA/DEC scatter colored by Doppler velocity

Note: the source must be dithered across several modulation positions to break
the degeneracy between the astrometric signal and the per-output flat gains.
        """
    )

    # Add positional argument for files
    parser.add_argument('files', nargs='*', default=['*.fits'],
                       help='FITS files to process (supports wildcards)')

    # Add optional arguments (mirror process_astrometric_data parameters)
    parser.add_argument("--object_name",
                       help="Selection of the data by the Object name")
    parser.add_argument("--wollaston", 
                       help="Wollaston status. Use IN for internal or OUT for no wollaston (default: first in the list)")
    parser.add_argument("--dark_files", 
                       help="Select one or more specific dark(s) files to use")
    parser.add_argument("--flat_files",
                       help="Force to select which flat map file(s) to use (default: the one in the directory)")
    parser.add_argument("--wave_files",
                       help="Force to select which wavelength map file(s) to use (default: the one in the directory)")
    parser.add_argument("--modID", type=int,
                       help="Modulation pattern ID to select (default: all)")
    parser.add_argument("--modScale", type=int,
                       help="Modulation scale to select (default: any)")
    parser.add_argument("--Nsingular", type=int, default=19*6,
                       help="Number of singular values kept in the SVD filtering (default: %(default)s)")
    parser.add_argument("--line_center", type=float, default=656.28,
                       help="Central wavelength of the spectral line in nm (default: %(default)s)")
    parser.add_argument("--line_width", type=float, default=2.0,
                       help="Width of the spectral line in nm (default: %(default)s)")
    parser.add_argument("--PA", type=float, default=-45.0,
                       help="Reference position angle in degrees drawn on the scatter plot (for plotting only, does not affect the results; default: %(default)s)")

    # Parse command line arguments
    # Development environment defaults are handled autonomously in run_makeAstrometry()
    args = parser.parse_args()
    file_patterns = args.files if args.files else ['*.fits']
    object_name = args.object_name
    dark_patterns = [args.dark_files] if args.dark_files else None
    flat_patterns = [args.flat_files] if args.flat_files else None
    wave_patterns = [args.wave_files] if args.wave_files else None
    wollaston = args.wollaston
    modID = args.modID
    modScale = args.modScale
    Nsingular = args.Nsingular
    line_center = args.line_center
    line_width = args.line_width
    PA = args.PA

    try:
        # Check observatory status
        status = check_observatory_status()
        print(status)

        print(f"Processing astrometric data with patterns: {file_patterns}")
        
        # Run the astrometric analysis
        process_astrometric_data(
            file_patterns=file_patterns,
            object_name=object_name,
            dark_patterns=dark_patterns,
            flat_patterns=flat_patterns,
            wave_patterns=wave_patterns,
            modID=modID,
            modScale=modScale,
            wollaston=wollaston,
            Nsingular=Nsingular,
            line_center=line_center,
            line_width=line_width,
            PA=PA,
        )
        
        print(f"Astrometric analysis completed successfully!")
            
    except Exception as e:
        print(f"Error in astrometric analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()