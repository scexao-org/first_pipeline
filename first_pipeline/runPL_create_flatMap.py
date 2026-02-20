#! /usr/bin/env python3
# -*- coding: iso-8859-15 -*-
#%%
"""
FIRST Pipeline - Flat Field Map Generation

This script creates flat field maps from SuperK calibration data for the FIRST
Visible Photonic Lantern at SUBARU/SCEXAO. Flat field maps correct for pixel-to-pixel
sensitivity variations and provide gain coefficients essential for accurate 
photometric calibration.

Flat field calibration is crucial for consistent photometric measurements
across all spectral channels and wavelengths.

Created on Wed May 21 22:56:25 2025
@author: slacour
"""

import os
import sys
from astropy.io import fits
from glob import glob
import argparse
import numpy as np
from typing import List
from scipy.signal import find_peaks
import itertools

import getpass
import matplotlib
if "VSCODE_PID" in os.environ:
    matplotlib.use('Qt5Agg')
else:
    matplotlib.use('Agg')
     
import matplotlib.pyplot as plt
from matplotlib.pyplot import axis, plot,hist,clf,figure,legend,imshow, xlim
from datetime import datetime
from tqdm import tqdm
import matplotlib as mpl
mpl.rcParams['figure.max_open_warning'] = 0

from .libraries import runPL_library_io as runlib_io
from .libraries import runPL_library_plots as runlib_plots
# Import FIRST pipeline classes
from .classes.runPL_class_flatMap import FlatMap
from .classes.runPL_class_fileList import FileList
from .classes.runPL_class_dataCube import DataCube 


#plt.ion()
# Add options
usage = """
Usage: %prog [options] files

Goal: Create a flat field map from the provided FITS files.

Summary:
- Searches for FITS files with X_FIRTYP=PREPROC and DATA-TYP=FLAT keywords.
- Finds corresponding dark files (X_FIRTYP=PREPROC, DATA-TYP=DARK).
- Reads flat field files, subtracts the median of the dark files.
- Performs linear regression to compute gain correction for each pixel.
- Saves the flat field map as a FITS file in the output directory.
- Generates and saves figures for visualization and quality assessment.
- Output files are stored in a "flatmaps" directory.

Options:
    --dark_files            Select one or more specific dark files to use
    --wollaston             Wollaston status. Use IN for internal or OUT for no wollaston (default: first in the list of files)
    --override-flat-keyword Override the requirement for DATA-TYP=FLAT keyword in input files

Examples:
    runPL_create_flatMap.py *.fits --dark_files="dark*.fits" --wollaston=IN
    runPL_create_flatMap.py *.fits --override-flat-keyword --dark_files="dark*.fits"
"""

def get_filelist_wave(flat_patterns, dark_patterns, wollaston):

        fits_keywords = {'X_FIRTYP': ['PREPROC'],
                        'DATA-TYP': ['FLAT'],
                        }    
        
        # Adding other constraints if asked by user
        if wollaston is not None:
            fits_keywords['X_FIRWOL'] = [wollaston]
        
        print(flat_patterns)
        filelist = runlib_io.get_filelist(flat_patterns, fits_keywords)

        # Adding new constraints if not asked by user
        hd=fits.getheader(filelist[0])
        wollaston = hd.get('X_FIRWOL', None)
        if wollaston is not None:
            fits_keywords['X_FIRWOL'] = [wollaston]

        print("----------------")
        print(f"Selected wollaston={wollaston}")

        filelist = runlib_io.get_filelist(flat_patterns, fits_keywords)

        print(f"Found {len(filelist)} files matching criteria.")
        print("----------------")

        # finding darks files
        fits_keywords['DATA-TYP'] = ['DARK']

        try:
            filelist_dark = runlib_io.get_filelist(dark_patterns, fits_keywords,  name_search="dark")
        except FileNotFoundError as e:
            print(f"WARNING!!! {e}")
            filelist_dark = []

        files_with_dark = runlib_io.associate_dark(filelist, filelist_dark)

        return files_with_dark


def compute_flat(datalist, Nflat_smooth = 25):

    flats=np.array([np.nansum(d.data,axis=(0,1)) for d in datalist])    
    valid_mask = ~np.isnan(flats[:,0,0])
    flats = flats[valid_mask]

    # variance=np.array([np.nansum(d.variance,axis=(0,1)) for d in datalist])
    # variance = variance[valid_mask]

    flats_smooth = np.zeros_like(flats)
    window = np.hanning(Nflat_smooth)
    window[Nflat_smooth//2] = 0.0
    window /= window.sum()
    conv_ref = np.convolve(np.ones(len(flats[0,0])), window, mode='same')
    
    for f in range(flats_smooth.shape[0]):
        for o in range(flats_smooth.shape[1]):
            flats_smooth[f, o, :] =  np.convolve(flats[f, o, :], window, mode='same') / conv_ref

    flat_individual = flats/flats_smooth
    flat_full=flats.sum(axis=0)/flats_smooth.sum(axis=0)


    return flat_full, flat_individual


if __name__ == "__main__":
    '''
    run for neon only, call functions for star

    to change the parameters to skip wavelenght or consider more peaks, change value in function
    findPeaks directly, in the instance of "run_trials_for_all_combination_of_waves"
    '''
    parser = argparse.ArgumentParser(
        description="Generate flat field calibration maps from SuperK data for FIRST Pipeline photometric correction.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
FIRST Pipeline Flat Field Map Generation Tool

This script creates flat field maps from SuperK calibration data to correct for
pixel-to-pixel sensitivity variations. Linear regression analysis generates gain
coefficients and quality metrics essential for accurate photometric calibration.

Examples:
    %(prog)s --wollaston IN --dark_files=dark*.fits flat_data/*.fits
    %(prog)s --dark_files=/path/to/darks/*.fits *.fits
    %(prog)s /data/flats/*.fits
    %(prog)s --override-flat-keyword --dark_files=dark*.fits non_flat_data/*.fits

Pipeline Workflow Integration:
    1. Processes preprocessed flat field files (X_FIRTYP=PREPROC, DATA-TYP=FLAT)
    2. Uses corresponding dark frames for background subtraction
    3. Output flat maps enable photometric correction in downstream analysis
    4. Essential calibration step before coupling map generation

Input Files:
    - Flat field data: X_FIRTYP=PREPROC and DATA-TYP=FLAT (SuperK illumination)
    - Corresponding dark frames: X_FIRTYP=PREPROC and DATA-TYP=DARK
    - Files automatically grouped by Wollaston status (IN/OUT)
    - Use --override-flat-keyword to process files without DATA-TYP=FLAT requirement

Output Files:
    - Flat field map FITS files (flatmaps/ directory)
    - Gain coefficient matrices for each spectral channel
    - Quality assessment metrics and fit residuals
    - Diagnostic plots showing calibration quality

Processing Details:
    - Linear regression for gain correction per pixel
    - Dark subtraction for accurate flat field measurement
    - Quality metrics assess calibration reliability
    - Separate processing for polarimetry (Wollaston IN) and photometry (OUT) modes
    - Handles variable illumination patterns from SuperK source

Calibration Quality:
    - Statistical analysis of gain coefficients
    - Residual mapping to identify problematic pixels
    - Quality flags for reliable vs uncertain calibrations
    - Diagnostic plots for visual inspection

Technical Notes:
    - Wollaston status affects channel configuration and processing
    - Gain maps normalize pixel response variations
    - Quality metrics guide downstream processing decisions
    - Compatible with coupling map and image reconstruction scripts

Note: Quality flat field calibration is essential for accurate photometry.
Review diagnostic plots to ensure proper gain correction and identify 
any systematic issues with SuperK illumination or detector response.
        """
    )


    # needed to work in VSC:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--f", help=argparse.SUPPRESS)

    # Add positional argument for files
    parser.add_argument('files', nargs='*', default=['*.fits'],
                       help='FITS files to process (supports wildcards)')
    # Add optional arguments
    parser.add_argument("--dark_files", 
                       help="Select one or more specific dark(s) files to use")
    parser.add_argument("--wollaston", 
                       help="Wollaston status. Use IN for internal or OUT for no wollaston (default: first in the list of files)")
    parser.add_argument("--Nflat_smooth", default = 25,
                       help="Smoothing parameter for flat field computation [default: 25]", type=int)
    parser.add_argument("--override-flat-keyword", action="store_true",
                       help="Override the requirement for DATA-TYP=FLAT keyword in input files")
    
    # Parse the arguments
    args = parser.parse_args()
    file_patterns = args.files if args.files else ['*.fits','./preproc/*.fits']

    # Extract the parsed arguments
    dark_patterns = args.dark_files
    wollaston = args.wollaston
    Nflat_smooth =args.Nflat_smooth
    override_flat_keyword = args.override_flat_keyword

    if ("VSCODE_PID" in os.environ or os.environ.get('TERM_PROGRAM') == 'vscode' or os.environ.get('SPYDER_DEBUG_FILE')):
        print("Running in compiler")
        if getpass.getuser() == "slacour":
            file_patterns = "/Users/slacour/DATA/LANTERNE/20251125/preproc"
            file_patterns = "/Users/slacour/DATA/LANTERNE/20251231/preproc/firstpl_2025-12-31T00?3*fits"
            # dark_patterns = "/Users/slacour/DATA/LANTERNE/20251231/preproc/firstpl_*fits"
            file_patterns = "/Users/slacour/DATA/LANTERNE/test_flat/preproc"
            file_patterns = "/Users/slacour/DATA/LANTERNE/raw/20260114/preproc"
            # file_patterns = "/Users/slacour/DATA/LANTERNE/raw/20260114/preproc_noedge"
            override_flat_keyword = True

        if getpass.getuser() == "jsarrazin":
            file_patterns = "/home/jsarrazin/Bureau/PLDATA/moreTest/2024-11-21_13-48-32_science_copie/preproc"
            file_patterns = "/home/jsarrazin/Bureau/PLDATA/novembre/les_preproc"
        if getpass.getuser() == "ehuby":
            file_patterns = "/home/ehuby/WORK/DATA/FIRST-PL/2025-05-10/preproc/"
            file_patterns = "/home/ehuby/WORK/DATA/FIRST-PL/2025-05-10/preproc/"
        

def main():
    """
    Main entry point for the flat field map creation script.
    """
    # needed to work in VSC:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--f", help=argparse.SUPPRESS)

    # Add positional argument for files
    parser.add_argument('files', nargs='*', default=['*.fits'],
                       help='FITS files to process (supports wildcards)')
    # Add optional arguments
    parser.add_argument("--dark_files", 
                       help="Select one or more specific dark(s) files to use")
    parser.add_argument("--wollaston", 
                       help="Wollaston status. Use IN for internal or OUT for no wollaston (default: first in the list of files)")
    parser.add_argument("--Nflat_smooth", default = 25,
                       help="Smoothing parameter for flat field computation [default: 25]", type=int)
    parser.add_argument("--override-flat-keyword", action="store_true",
                       help="Override the requirement for DATA-TYP=FLAT keyword in input files")
    
    # Parse the arguments
    args = parser.parse_args()
    file_patterns = args.files if args.files else ['*.fits','./preproc/*.fits']

    # Extract the parsed arguments
    dark_patterns = args.dark_files
    wollaston = args.wollaston
    Nflat_smooth =args.Nflat_smooth
    override_flat_keyword = args.override_flat_keyword

    if ("VSCODE_PID" in os.environ or os.environ.get('TERM_PROGRAM') == 'vscode' or os.environ.get('SPYDER_DEBUG_FILE')):
        print("Running in compiler")
        if getpass.getuser() == "slacour":
            file_patterns = "/Users/slacour/DATA/LANTERNE/20251125/preproc"
            file_patterns = "/Users/slacour/DATA/LANTERNE/20251231/preproc/firstpl_2025-12-31T00?3*fits"
            # dark_patterns = "/Users/slacour/DATA/LANTERNE/20251231/preproc/firstpl_*fits"
            file_patterns = "/Users/slacour/DATA/LANTERNE/test_flat/preproc"
            file_patterns = "/Users/slacour/DATA/LANTERNE/raw/20260114/preproc"
            # file_patterns = "/Users/slacour/DATA/LANTERNE/raw/20260114/preproc_noedge"
            override_flat_keyword = True

        if getpass.getuser() == "jsarrazin":
            file_patterns = "/home/jsarrazin/Bureau/PLDATA/moreTest/2024-11-21_13-48-32_science_copie/preproc"
            file_patterns = "/home/jsarrazin/Bureau/PLDATA/novembre/les_preproc"
        if getpass.getuser() == "ehuby":
            file_patterns = "/home/ehuby/WORK/DATA/FIRST-PL/2025-05-10/preproc/"
            file_patterns = "/home/ehuby/WORK/DATA/FIRST-PL/2025-05-10/preproc/"
        

    # If the user specify a dark, use it. Otherwise, use the science file pattern
    if dark_patterns is None:
        dark_patterns = file_patterns

    # Set data_type based on override flag
    data_type = None if override_flat_keyword else 'FLAT'
    if override_flat_keyword:
        print("WARNING: Overriding FLAT keyword requirement. Processing files without DATA-TYP=FLAT constraint.")
    fileList = FileList(file_patterns, data_type=data_type, first_type='PREPROC', wollaston=wollaston)
    fileList.make_association(dark_patterns=dark_patterns)

    datalist : List[DataCube] = fileList.extract_data_from_list(center = False)

    flat_full, flat_individual = compute_flat(datalist, Nflat_smooth)

# #%%

# filenumber = 6
# basename = datalist[filenumber].basename

# flats=np.array([np.nansum(d.data,axis=(0)) for d in datalist[6:7]])    
# valid_mask = ~np.isnan(flats[:,0,0,0])
# flats = flats[valid_mask]

# # variance=np.array([np.nansum(d.variance,axis=(0,1)) for d in datalist])
# # variance = variance[valid_mask]

# flats_smooth = np.zeros_like(flats)
# window = np.hanning(Nflat_smooth)
# window[Nflat_smooth//2] = 0.0
# window /= window.sum()
# conv_ref = np.convolve(np.ones(len(flats[0,0,0])), window, mode='same')

# for f in range(flats_smooth.shape[0]):
#     for o in range(flats_smooth.shape[1]):
#         for k in range(flats_smooth.shape[2]):
#             flats_smooth[f, o, k, :] =  np.convolve(flats[f, o, k, :], window, mode='same') / conv_ref

# flat_i = flats/flats_smooth
# flat_f=flats.sum(axis=(0,1))/flats_smooth.sum(axis=(0,1))

# # Create figure with flats[2,:,10] and flat_individual
# fig, (ax1, ax2) = plt.subplots(2, 1, num=10, figsize=(12, 8), sharex=True, sharey=True,clear=True)

# fig.suptitle(f'Flat Field Analysis for files {datalist[6].basename}')
# # Upper plot: imshow of flats[2,:,10]
# im1 = ax1.imshow(flats[0,:,6], origin='lower', aspect='auto', interpolation='none')
# ax1.set_title('Output 10 - Raw flat field data')
# ax1.set_xlabel('X pixel')
# ax1.set_ylabel('Modulation Step (over 10mas)')
# plt.colorbar(im1, ax=ax1)

# # Lower plot: imshow of flat_individual
# im2 = ax2.imshow(flat_i[0,:,6], origin='lower', aspect='auto', interpolation='none',vmax=1.1, vmin=0.9)
# ax2.set_title('Output 10 - Normalized flat field')
# ax2.set_xlabel('X pixel') 
# ax2.set_ylabel('Modulation Step (over 10mas)')
# plt.colorbar(im2, ax=ax2)

# plt.tight_layout()
# flat_f=flats.sum(axis=(0,1))/flats_smooth.sum(axis=(0,1))
# fig.savefig(f'flat_analysis_{basename}_output10_zoom.pdf')





# %%


    fig=plt.figure("Flat Field Computation",clear=True,figsize=(18,6))
    plt.plot(flat_full.T+np.arange(flat_full.shape[0])*0.1)
    plt.xlim((0,len(flat_full.T)))
    plt.ylim((0.85,1.15+len(flat_full)*0.1))
    plt.title(fig.get_label())
    plt.xlabel("x pixel")
    plt.ylabel("gain _ output * 10%")
    plt.tight_layout()
    Noutput = len(flat_full)
    for o in range(Noutput):
        fig=plt.figure("Flat values for output %i"%o,clear=True,figsize=(18,6))
        flat_output=flat_individual[:,o]/flat_full[o]
        plt.plot(flat_output.T+np.arange(flat_output.shape[0])*0.05)
        plt.xlim((0,len(flat_output.T)))
        plt.ylim((0.85,1.15+len(flat_output)*0.05))
        plt.title(fig.get_label())
        plt.xlabel("x pixel")
        plt.ylabel("gain -<mean gain> _ file number * 5%")
        plt.tight_layout()
        Noutput = len(flat_output)

    ############### Save results ####################
    # Create FlatMap object and save using the new save method
    
    flatMap = FlatMap()
    flatMap.create_from_data(1/flat_full)

    header = datalist[-1].header
    # Définir le chemin complet du sous-dossier "output/couplingmaps"
    folder = fileList.get_most_common_dir()
    output_dir = os.path.join(folder,"../flatmaps")

    filenames = [d.filename for d in datalist]
    for i, filename in enumerate(filenames):
        header['Q_FM_F%i' % i] = (filename, 'filename of the extracted flux')

    header['Q_FMNAME'] = (runlib_io.create_basename(header), 'name of the flatwave map file')

    # Créer les dossiers "output" et "pixel" s'ils n'existent pas déjà
    os.makedirs(output_dir, exist_ok=True)

    output_filename = os.path.join(output_dir, header['Q_FMNAME'])

    # Save using the FlatMap save method
    flatMap.save(output_filename, header)

    ############### checking flat results ####################
    # checking all the flats for individual files
    # Load the flat map from the saved file  
    flatMap_loaded = FlatMap(output_filename)

    datalist_1 : List[DataCube] = fileList.extract_data_from_list(flatMap = None, center = False)
    datalist_2 : List[DataCube] = fileList.extract_data_from_list(flatMap = flatMap_loaded, center = False)

    data_noflat=np.array([np.nanmean(d.data,axis=(0,1)) for d in datalist_1])
    data_withflat=np.array([np.nanmean(d.data,axis=(0,1)) for d in datalist_2])

    for i in range(len(data_noflat)):
        d=datalist_1[0]
        # runlib_plots.plot_flux_map(np.nanmean(d.data,axis=(0,2,3)), d.xmod[0], d.ymod[0], desc = f'Flux map for file {d.basename}')
        fig = plt.figure(figsize=(18,10),clear=True)
        plt.subplot(2, 2, 1)
        vmin,vmax=np.percentile(data_noflat[i], (5,95))
        im0=plt.imshow(data_noflat[i], origin='lower', aspect='auto', vmin=vmin, vmax=vmax, interpolation='none', rasterized=True)
        plt.title(f'Without flat - File {i}')
        plt.colorbar(im0)
        plt.subplot(2, 2, 2)
        im1=plt.imshow(data_withflat[i], origin='lower', aspect='auto', vmin=vmin, vmax=vmax, interpolation='none', rasterized=True)
        plt.title(f'With flat - File {i}')
        plt.colorbar(im1)
        plt.subplot(2, 1, 2)
        plt.plot(data_withflat[i].T,'k')
        plt.plot(data_noflat[i].T)
        plt.suptitle(f'Flat correction comparison for file {d.basename}')
        plt.tight_layout()

    runlib_plots.save_pdf_in_file(output_filename)


if __name__ == "__main__":
    main()

# %%
