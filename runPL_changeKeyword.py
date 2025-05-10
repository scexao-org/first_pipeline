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
    runPL_changeKeyword.py --DATA_CAT=RAW --DATA_TYP=FLAT

    Fits header Keywords:

    DATA_CAT = RAW , PREPROC, REDUCED
    DATA_TYP =  WAVE, FLAT, SCIENCE, PIXELS, SPECTRA

    DATA_CAT gives the level of reduction:
    RAW means raw data from the camera
    PREPROC means the data has been cut and compressed
    REDUCED means that the data has been reduced

    DATA_TYP gives the type of data
    WAVE is Neon source data
    FLAT is data from SuperK
    SCIENCE is the night time observation data
    But REDUCED data can also have special types:
    PIXELS is the pixel map on the detector
    SPECTRA is the extracted spectra

    Update the keywords.
"""

parser = OptionParser(usage)
parser.add_option("-c","--X_FIRTYP", action="store",
                  help="X_FIRTYP gives the level of reduction")
parser.add_option("-t","--X_FIRTYP", action="store", 
                  help="X_FIRTYP gives the type of data")
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

    
if (argoptions.X_FIRTYP!=None)|(argoptions.DATA_CAT!=None)|(argoptions.GAIN!=None)|(argoptions.DATE=="DEFAULT"):
    for filename in filelist:
        string_print=filename+"   ----->"
        with fits.open(filename, mode='update') as filehandle:
            if argoptions.DATA_CAT:
                filehandle[0].header['X_FIRTYP'] = argoptions.DATA_CAT
                string_print+='   X_FIRTYP='+argoptions.DATA_CAT
            if argoptions.X_FIRTYP:
                filehandle[0].header['X_FIRTYP'] = argoptions.X_FIRTYP
                string_print+='   X_FIRTYP='+argoptions.X_FIRTYP
            if argoptions.GAIN:
                filehandle[0].header['GAIN'] = argoptions.GAIN
                string_print+='   GAIN='+argoptions.GAIN
            if argoptions.DATE:
                argoptions.DATE = runlib.get_date_from_filename(filename)
                filehandle[0].header['DATE'] = argoptions.DATE
                string_print+='   DATE='+argoptions.DATE
        print(string_print)

