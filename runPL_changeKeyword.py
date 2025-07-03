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
from optparse import OptionParser
import runPL_library_io as runlib

# Add options
usage = """
    usage:  %prog files.fits *files.fits

    Goal: update the FIRST_PL Data keywords

    example:
    runPL_changeKeyword.py --DATA_CAT=FLAT --X_FIRTYP=RAW

    Fits header Keywords:

    DATA_CAT = RAW , PREPROC, REDUCED
    X_FIRTYP =  WAVE, FLAT, SCIENCE, PIXELS, SPECTRA

    X_FIRTYP gives the type of dataproduct:
    RAW means raw data from the camera
    PREPROC means the data has been cut and compressed
    REDUCED means that the data has been reduced

    DATA_CAT gives the category of data
    WAVE is Neon source data
    FLAT is data from SuperK
    SCIENCE is the night time observation data
    But REDUCED data can also have special types:
    PIXELS is the pixel map on the detector
    SPECTRA is the extracted spectra

    Update the keywords.
"""

parser = OptionParser(usage)
parser.add_option("-c","--DATA-TYP", action="store",
                  help="DATA-TYP gives the category of data")
parser.add_option("-t","--X_FIRTYP", action="store", 
                  help="X_FIRTYP gives the type of dataproduct")
parser.add_option("-i","--X_FIRMID", action="store",
                  help="X_FIRMID gives the modulation ID of the data")
parser.add_option("-d","--X_FIRTRG", action="store", 
                  help="Trigger of camera. Use INT for internal or EXT for external trigger")
parser.add_option("-g","--GAIN", action="store", 
                  help="")
parser.add_option("-d","--DATE", action="store", 
                  help="Use DEFAULT to get the date from the filename")

(argoptions, args) = parser.parse_args()


filelist=[]
## If the user specifies a file name or wild cards ("*_0001.fits")
if len(args) > 0 :
    for f in args:
        filelist += [file for file in glob(f) if file.endswith(".fits")]
## Processing of the full current directory
else :
    for file in os.listdir("."):
        if file.endswith(".fits"):
            filelist.append(file)

filelist.sort() # process the files in alphabetical order

    
# Update FITS headers based on provided options
header_updates = {
    'DATA-TYP': argoptions.DATA_TYP,
    'X_FIRTYP': argoptions.X_FIRTYP,
    'X_FIRMID': argoptions.X_FIRMID,
    'X_FIRTRG': argoptions.X_FIRTRG,
    'GAIN': argoptions.GAIN,
    'DATE': argoptions.DATE if argoptions.DATE != "DEFAULT" else None,
}

if any(v is not None for v in header_updates.values()) or argoptions.DATE == "DEFAULT":
    for filename in filelist:
        updates = header_updates.copy()
        if argoptions.DATE == "DEFAULT":
            updates['DATE'] = runlib.get_date_from_filename(filename)
        string_print = filename + "   ----->"
        with fits.open(filename, mode='update') as filehandle:
            for key, value in updates.items():
                if value is not None:
                    filehandle[0].header[key] = value
                    string_print += f'   {key}={value}'
        print(string_print)
