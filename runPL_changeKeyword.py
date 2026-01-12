#! /usr/bin/env python3
# -*- coding: iso-8859-15 -*-
"""
FIRST Pipeline - FITS Header Keyword Management

This script modifies FITS header keywords for the FIRST Visible Photonic Lantern 
pipeline at SUBARU/SCEXAO. It's used to classify and tag FITS files for proper 
processing by downstream pipeline scripts.

The script updates essential keywords that determine how files are processed in 
the sequential pipeline workflow: Raw FITS → Pixel Map → Preprocessing → 
Wavelength Map → Coupling Maps → Calibration → Image Reconstruction.

Created on Wed May 21 22:56:25 2025
@author: slacour
"""

import os
import sys
from astropy.io import fits
from glob import glob
import argparse
import libraries.runPL_library_io as runlib

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Modify FITS header keywords for FIRST Pipeline classification and processing control.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
FIRST Pipeline Keyword Management Tool

This script is essential for the FIRST Visible Photonic Lantern data processing 
pipeline. It modifies FITS header keywords that control how files are classified 
and processed by subsequent pipeline scripts.

Examples:
    %(prog)s --DATA-TYP=FLAT --X_FIRTYP=RAW *.fits
    %(prog)s --OBJECT="HD 164461" --X_FIRTYP=PREPROC target_data/*.fits
    %(prog)s --DATA-TYP=COMPARAISON --X_FIRTYP=RAW neon_calib.fits
    %(prog)s --DATE=DEFAULT --X_FIRTYP=RAW recent_observations/*.fits

Pipeline Workflow Integration:
    1. Use this script to classify raw FITS files by DATA-TYP
    2. Mark processing stages with X_FIRTYP as files move through pipeline
    3. Temporary keyword changes for interim classification (revert when finalized)

Critical Keywords:
    DATA-TYP (Data Classification):
        FLAT         - SuperK flat field data for pixel mapping
        DARK         - Dark frames for background subtraction  
        OBJECT       - Science target observations
        ACQUISITION  - Target acquisition data
        COMPARAISON  - Neon calibration spectra for wavelength mapping
        TEST         - Test/validation data
        
    X_FIRTYP (Processing Stage):
        RAW          - Raw, unprocessed data
        PREPROC      - Pre-processed (pixel map applied, cleaned)
        PIXELMAP     - Pixel mapping calibration files
        WAVEMAP      - Wavelength mapping calibration files
        COULPLINGMAP - Coupling efficiency mapping files

    X_FIRMID (Modulation ID):
        Identifier for specific modulation pattern used during observation
        
    X_FIRTRG (Camera Trigger):
        INT          - Internal camera trigger
        EXT          - External trigger synchronization
        
    X_FIRWOL (Wollaston Status):  
        IN           - Wollaston prism inserted (polarimetry mode)
        OUT          - Wollaston prism removed (photometry mode)

Usage in Pipeline:
    - Classify raw files before processing: --DATA-TYP to set observation type
    - Track processing stages: --X_FIRTYP as files move through pipeline steps  
    - Extract dates from filenames: --DATE=DEFAULT for automatic date parsing
    - Mark special configurations: --X_FIRWOL, --X_FIRTRG for observing modes

Note: Proper keyword classification ensures correct file selection and processing 
logic in downstream pipeline scripts (createPixelMap, preprocess, wavelengthMap, etc.).
        """
    )

    # Add positional argument for files
    parser.add_argument('files', nargs='*', 
                       help='FITS files to process (supports wildcards). If none specified, processes all .fits files in current directory.')
    
    # Add optional arguments for header keywords
    parser.add_argument("-c", "--DATA-TYP", 
                       choices=["OBJECT", "TEST", "ACQUISITION", "DARK", "FLAT", "COMPARAISON"],
                       help="Classify data type for pipeline processing: FLAT (SuperK data), DARK (background), OBJECT (science targets), ACQUISITION (target acquisition), COMPARAISON (Neon calibration), TEST (validation)")
    parser.add_argument("-o", "--OBJECT", 
                       help="Target name for science observations (e.g., 'HD 164461', 'Beta Pic')")
    parser.add_argument("-t", "--X_FIRTYP", 
                       choices=["RAW", "PREPROC", "COULPLINGMAP", "PIXELMAP", "WAVEMAP"],
                       help="Processing stage identifier: RAW (unprocessed), PREPROC (preprocessed), PIXELMAP/WAVEMAP/COULPLINGMAP (calibration products)")
    parser.add_argument("-i", "--X_FIRMID", 
                       help="Modulation ID identifying the specific modulation pattern used during observation")
    parser.add_argument("-r", "--X_FIRTRG", 
                       help="Camera trigger mode: INT (internal) or EXT (external synchronization)")
    parser.add_argument("-w", "--X_FIRWOL", 
                       choices=["IN", "OUT"],
                       help="Wollaston prism status: IN (polarimetry mode) or OUT (photometry mode)")
    parser.add_argument("-g", "--GAIN", 
                       help="Camera gain setting value")
    parser.add_argument("-d", "--DATE", 
                       help="Observation date (use DEFAULT to automatically extract from filename)")

    args = parser.parse_args()

    filelist = []
    # If the user specifies file names or wildcards
    if len(args.files) > 0:
        for f in args.files:
            filelist += [file for file in glob(f) if file.endswith(".fits")]
    # Processing of the full current directory
    else:
        for file in os.listdir("."):
            if file.endswith(".fits"):
                filelist.append(file)

    filelist.sort()  # process the files in alphabetical order

    # Update FITS headers based on provided options
    header_updates = {
        'OBJECT': args.OBJECT,
        'DATA-TYP': getattr(args, 'DATA_TYP'),
        'X_FIRTYP': args.X_FIRTYP,
        'X_FIRMID': args.X_FIRMID,
        'X_FIRTRG': args.X_FIRTRG,
        'X_FIRWOL': args.X_FIRWOL,
        'GAIN': args.GAIN,
        'DATE': args.DATE if args.DATE != "DEFAULT" else None,
    }

    if any(v is not None for v in header_updates.values()) or args.DATE == "DEFAULT":
        for filename in filelist:
            updates = header_updates.copy()
            if args.DATE == "DEFAULT":
                updates['DATE'] = runlib.get_date_from_filename(filename)
            string_print = filename + "   ----->"
            with fits.open(filename, mode='update') as filehandle:
                for key, value in updates.items():
                    if value is not None:
                        filehandle[0].header[key] = value
                        string_print += f'   {key}={value}'
            print(string_print)
