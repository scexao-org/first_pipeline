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
import sys
import traceback

from first_pipeline_shared.libraries.runPL_library_cli import check_file_options
from .run_makeAstrometry import process_astrometric_data, check_observatory_status


def parse_positive_odd_int(value):
    """Parse a positive odd integer for an argparse option."""
    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(
            "Ncube_average must be a positive odd integer") from error
    if parsed < 1 or parsed % 2 == 0 or str(parsed) != str(value).strip():
        raise argparse.ArgumentTypeError(
            "Ncube_average must be a positive odd integer")
    return parsed


def parse_nonnegative_int(value):
    """Parse a non-negative integer for an argparse option."""
    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(
            "jac_poly_deg must be a non-negative integer") from error
    if parsed < 0 or str(parsed) != str(value).strip():
        raise argparse.ArgumentTypeError(
            "jac_poly_deg must be a non-negative integer")
    return parsed


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
continuum is estimated by low-order polynomial fits on the line's side windows.

Caveat: PSF jitter and deformation between the poses of a Jacobian block
attenuate the amplitude of the recovered shift by an achromatic factor
(kappa < 1) that must be calibrated by simulation (see astrometry_scale.py);
the wavelength structure and position angle are unbiased.

Examples:
    %(prog)s preproc/*_HD163296_P.fits
    %(prog)s --wollaston IN --object_name HD142527 preproc/*.fits
    %(prog)s --line_center 656.28 --line_width 1.5 preproc/*.fits
    %(prog)s preproc/*.fits --wave_files wavemaps/ --dark_files dark*.fits
    %(prog)s --dark_files 'dark*.fits' preproc/*.fits

Input Files:
    - Preprocessed FITS files: X_FIRTYP=PREPROC (selected as OBJECT data)
    - Wavelength maps (X_FIRTYP=WAVEMAP) and flat maps (X_FIRTYP=FLATMAP),
      auto-discovered in sibling ../wavemaps and ../flatmaps folders or set
      explicitly with --wave_files / --flat_files
    - Dark frames for background subtraction (default: the input files)
    --dark_files, --flat_files and --wave_files accept several files or a
    quoted wildcard. Unquoted wildcards must come after the input files.

Output Files (written to a sibling ../astrometry folder):
    - ASTROMETRY FITS file (X_FIRTYP=ASTROMETRY) with HDUs:
      WAVE (working window), FLUX_SCALED, ASTROMETRY_XY and ASTROMETRY_COV
      (one track per continuum polynomial degree), POLY_DEG, LINE_MASK
    - Multi-page PDF with the RA/DEC astrometry vs wavelength (full band and
      zoomed on the line) and the RA/DEC scatter colored by Doppler velocity

Note: the source must be dithered across several modulation positions to break
the degeneracy between the astrometric signal and the per-output flat gains.
        """
    )

    # Add positional argument for files
    parser.add_argument('files', nargs='*', default=[],
                       help='FITS files to process (supports wildcards)')

    # Add optional arguments (mirror process_astrometric_data parameters)
    parser.add_argument("--object_name",
                       help="Selection of the data by the Object name")
    parser.add_argument("--wollaston", 
                       help="Wollaston status. Use IN for internal or OUT for no wollaston (default: first in the list)")
    parser.add_argument("--dark_files", nargs='+',
                       help="Select one or more specific dark(s) files to use")
    parser.add_argument("--flat_files", nargs='+',
                       help="Force to select which flat map file(s) to use (default: the one in the directory)")
    parser.add_argument("--wave_files", nargs='+',
                       help="Force to select which wavelength map file(s) to use (default: the one in the directory)")
    parser.add_argument("--modID", type=int,
                       help="Modulation pattern ID to select (default: all)")
    parser.add_argument("--modScale", type=int,
                       help="Modulation scale to select (default: any)")
    parser.add_argument("--X_FIROBX", type=float,
                       help="Object X offset (X_FIROBX keyword) to select (default: any)")
    parser.add_argument("--X_FIROBY", type=float,
                       help="Object Y offset (X_FIROBY keyword) to select (default: any)")
    parser.add_argument("--line_center", type=float, default=656.28,
                       help="Central wavelength of the spectral line in nm (default: %(default)s)")
    parser.add_argument("--line_width", type=float, default=2.0,
                       help="Width of the spectral line in nm (default: %(default)s)")
    parser.add_argument("--PA", type=float, default=-45.0,
                       help="Reference position angle in degrees drawn on the scatter plot (for plotting only, does not affect the results; default: %(default)s)")
    parser.add_argument("--Ncube_average", type=parse_positive_odd_int, default=3,
                       help="Number of nearest cubes to average for the Jacobian; must be a positive odd integer (default: %(default)s)")
    parser.add_argument("--jac_half_window", type=int, default=1,
                       help="Half window of the local Jacobian: 2*h+1 poses per block (default: %(default)s, i.e. 3 poses)")
    parser.add_argument("--jac_fit_order", type=int, choices=(1, 2), default=1,
                       help="Order of the local Jacobian fit: 1 = gradient, 2 = gradient + curvature (needs jac_half_window >= 3) (default: %(default)s)")
    parser.add_argument("--jac_poly_deg", type=parse_nonnegative_int, default=1,
                       help="Polynomial degree used to smooth the Jacobian over wavelength (default: %(default)s)")
    parser.add_argument("--jac_fit_region", choices=("all", "continuum", "line"), default="all",
                       help="Wavelength channels on which the Jacobian polynomial (degree jac_poly_deg) is fitted: "
                            "the whole working window, the continuum only (window minus the line) or the line only "
                            "(default: %(default)s)")
    parser.add_argument("--calibrate_scale", dest="calibrate_scale", action="store_true",
                       default=True,
                       help="Measure the PSF jitter/deformation on the data and calibrate by simulation the "
                            "attenuation factor kappa of the fitted amplitude (default: on, adds a few seconds)")
    parser.add_argument("--no_calibrate_scale", dest="calibrate_scale", action="store_false",
                       help="Skip the kappa calibration (step 2)")
    parser.add_argument("--save_npz",
                       help="Save the working arrays (datacube, variance, dither, wavelength) to this .npz file for offline tests")
    # Parse command line arguments
    # Development environment defaults are handled autonomously in run_makeAstrometry()
    args = parser.parse_args()
    check_file_options(parser, args.files, dark_files=args.dark_files,
                       flat_files=args.flat_files, wave_files=args.wave_files)
    file_patterns = args.files if args.files else ['*.fits']
    object_name = args.object_name
    dark_patterns = args.dark_files
    flat_patterns = args.flat_files
    wave_patterns = args.wave_files
    wollaston = args.wollaston
    modID = args.modID
    modScale = args.modScale
    firObX = args.X_FIROBX
    firObY = args.X_FIROBY
    line_center = args.line_center
    line_width = args.line_width
    PA = args.PA
    Ncube_average = args.Ncube_average

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
            firObX=firObX,
            firObY=firObY,
            wollaston=wollaston,
            line_center=line_center,
            line_width=line_width,
            PA=PA,
            Ncube_average=Ncube_average,
            jac_half_window=args.jac_half_window,
            jac_fit_order=args.jac_fit_order,
            jac_poly_deg=args.jac_poly_deg,
            jac_fit_region=args.jac_fit_region,
            save_npz=args.save_npz,
            calibrate_scale=args.calibrate_scale,
        )
        
        print("Astrometric analysis completed successfully!")

    except Exception as e:
        traceback.print_exc()
        print(f"Error in astrometric analysis: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()