#! /usr/bin/env python3
# -*- coding: iso-8859-15 -*-
"""
Created on Sun May 24 22:56:25 2015

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
        description="Update the FIRST_PL FITS header keywords.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    %(prog)s --DATA-TYP=FLAT --X_FIRTYP=RAW *.fits
    %(prog)s --OBJECT="Target Name" --X_FIRTYP=SCIENCE data/*.fits

Keyword meanings:
    DATA-TYP:
        TEST   - Test data
        DARK   - Dark data
        FLAT   - Flat field data
        
    X_FIRTYP:
        RAW      - Raw data from the camera
        WAVE     - Neon source data
        FLAT     - Data from SuperK
        SCIENCE  - Night time observation data
        PIXELS   - Pixel map on the detector (for REDUCED data)
        SPECTRA  - Extracted spectra (for REDUCED data)

This script updates the specified FITS header keywords for all matching files.
        """
    )

    # Add positional argument for files
    parser.add_argument('files', nargs='*', 
                       help='FITS files to process (supports wildcards). If none specified, processes all .fits files in current directory.')
    
    # Add optional arguments for header keywords
    parser.add_argument("-c", "--DATA-TYP", 
                       help="DATA-TYP gives the category of data")
    parser.add_argument("-o", "--OBJECT", 
                       help="OBJECT gives the name of the observed target")
    parser.add_argument("-t", "--X_FIRTYP", 
                       help="X_FIRTYP gives the type of dataproduct")
    parser.add_argument("-i", "--X_FIRMID", 
                       help="X_FIRMID gives the modulation ID of the data")
    parser.add_argument("-r", "--X_FIRTRG", 
                       help="Trigger of camera. Use INT for internal or EXT for external trigger")
    parser.add_argument("-w", "--X_FIRWOL", 
                       help="Wollaston status. Use IN for internal or OUT for no wollaston")
    parser.add_argument("-g", "--GAIN", 
                       help="Gain value")
    parser.add_argument("-d", "--DATE", 
                       help="Date value (use DEFAULT to extract from filename)")

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
