#! /usr/bin/env python3
# -*- coding: iso-8859-15 -*-
"""
FIRST Pipeline - Coupling Map Generation CLI

Command-line interface for coupling map generation from preprocessed FIRST data.
This module provides the CLI wrapper for interactive execution of coupling
efficiency analysis algorithms.

Created on Wed May 21 22:56:25 2025
@author: slacour
"""

import argparse
import getpass
import os
from .run_createCouplingMap import run_createCouplingMap


def main():
    """
    Main entry point for coupling map generation CLI.
    """
    parser = argparse.ArgumentParser(
        description="Generate coupling efficiency maps from preprocessed FIRST Photonic Lantern data using SVD analysis.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
FIRST Pipeline Coupling Map Generation Tool

This script analyzes the coupling efficiency between the telescope focal plane 
and photonic lantern channels using advanced SVD-based decomposition. Coupling 
maps are essential for accurate image reconstruction and astrometric measurements.

Examples:
    %(prog)s --object_name="HD 164461" --wavelength_smooth=7 *.fits
    %(prog)s --modID=1 --modScale=2 --wollaston=IN data/*.fits
    %(prog)s --flatMap=/path/to/flat.fits --waveMap=/path/to/wave.fits *.fits

Pipeline Workflow Integration:
    1. Processes preprocessed data files (X_FIRTYP=PREPROC)
    2. Uses flat field and wavelength calibration maps
    3. Output coupling maps enable image reconstruction and astrometry
    4. Critical step for converting fiber measurements to sky coordinates

Input Files:
    - Preprocessed FITS files: X_FIRTYP=PREPROC
    - Flat field maps (automatic detection or manual selection)
    - Wavelength calibration maps (automatic detection or manual selection)
    - Dark frames for background subtraction
    - Files grouped by object name, modulation pattern, and Wollaston status

Output Files:
    - Coupling map FITS files: X_FIRTYP=COUPLINGMAP (../couplingmaps/ directory)
    - PDF diagnostic report with SVD analysis and quality plots
    - Triangular and pyramidal coupling coefficient matrices
    - Quality assessment metrics and validation plots

Processing Details:
    - SVD-based decomposition to extract coupling patterns
    - Wavelength smoothing and binning for noise reduction
    - Modulation pattern analysis for enhanced sensitivity
    - Automatic selection of singular values (configurable with --Nsingular)
    - Support for both polarimetry (Wollaston IN) and photometry (OUT) modes

Advanced Options:
    - object_name: Select specific science target for processing
    - modID/modScale: Choose specific modulation patterns
    - wavelength_smooth/bin: Control spectral processing parameters
    - Nsingular: Number of SVD modes to retain (affects map quality vs noise)

Technical Notes:
    - SVD analysis identifies dominant coupling modes
    - Coupling maps quantify spatial response of each fiber channel
    - Quality metrics assess map reliability and completeness
    - Results enable precise astrometric and photometric measurements

Note: Quality coupling maps are critical for accurate image reconstruction.
Review PDF diagnostics to ensure proper SVD convergence and coupling patterns.
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
    parser.add_argument("--flatMap", 
                       help="Select a specific flat Map to use (default: most recent in the flatmaps folder)")
    parser.add_argument("--waveMap", 
                       help="Select a specific wave Map to use (default: most recent in the wavemaps folder)")
    parser.add_argument("--wavelength_smooth", type=int, default=1,
                       help="Smoothing factor for wavelength (default: %(default)s)")
    parser.add_argument("--wavelength_bin", type=int, default=1,
                       help="Binning factor for wavelength (default: %(default)s)")
    parser.add_argument("--Nsingular", type=int, default=19*6,
                       help="Number of singular values to use (default: %(default)s)")
    parser.add_argument("--modID", type=int, 
                       help="Selection of the modulation pattern by user (default: first in the list)")
    parser.add_argument("--modScale", type=int, 
                       help="Selection of the modulation scale by user (default: first in the list)")
    parser.add_argument("--wollaston", 
                       help="Wollaston status. Use IN for internal or OUT for no wollaston (default: first in the list)")
    parser.add_argument("--use_pyramids", action="store_true",
                       help="Use pyramids instead of triangles for coupling map analysis (default: use triangles)")

    # Parse the arguments
    args = parser.parse_args()
    file_patterns = args.files if args.files else ['*.fits','./preproc/*.fits']

    # Extract the parsed arguments
    modID = args.modID
    modScale = args.modScale
    object_name = args.object_name
    wollaston = args.wollaston
    Nsingular = args.Nsingular
    wavelength_smooth = args.wavelength_smooth
    wavelength_bin = args.wavelength_bin
    dark_patterns = args.dark_files
    flat_patterns = args.flatMap
    wave_patterns = args.waveMap
    use_pyramids = args.use_pyramids
    
    # Note: Development environment detection and default paths
    # are handled autonomously in run_createCouplingMap()

    # Set default patterns if not specified
    if dark_patterns is None:
        dark_patterns = file_patterns
    if flat_patterns is None and file_patterns:
        folder = os.path.dirname(file_patterns[0])
        flat_patterns = file_patterns + [os.path.join(folder,"../flatmaps")] + [os.path.join(folder,"flatmaps")]
    if wave_patterns is None and file_patterns:
        folder = os.path.dirname(file_patterns[0])
        wave_patterns = file_patterns + [os.path.join(folder,"../wavemaps")] + [os.path.join(folder,"wavemaps")]

    # Process coupling map data
    couplingMap,_ = run_createCouplingMap(
        file_patterns=file_patterns,
        object_name=object_name,
        dark_patterns=dark_patterns,
        flat_patterns=flat_patterns,
        wave_patterns=wave_patterns,
        wavelength_smooth=wavelength_smooth,
        wavelength_bin=wavelength_bin,
        Nsingular=Nsingular,
        modID=modID,
        modScale=modScale,
        wollaston=wollaston,
        use_pyramids=use_pyramids
    )

    print(f"Coupling map created successfully:\n{couplingMap.filename}")


if __name__ == "__main__":
    main()