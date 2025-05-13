#! /usr/bin/env python3
# -*- coding: iso-8859-15 -*-
#%%
"""
Created on Sun May 24 22:56:25 2015

@author: slacour
"""

import os
import sys
from astropy.io import fits
from glob import glob
from optparse import OptionParser
import numpy as np
import peakutils

import getpass
import matplotlib
if "VSCODE_PID" in os.environ:
    matplotlib.use('Qt5Agg')
else:
    matplotlib.use('Agg')
     
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot,hist,clf,figure,legend,imshow
from datetime import datetime
from tqdm import tqdm
import runPL_library_io as runlib
import runPL_library_basic as basic
import runPL_library_imaging as runlib_i
import shutil
from collections import defaultdict
import time

plt.ion()

# Add options
usage = """
    usage:  %prog [options] [directory | files.fits]

    Goal: Preprocess the data using the pixel map.

    Output: files of type DPR_CATG=PREPROC in the preproc directory.
    Also, a figure of the pixel is saved in the preproc directory.
    Also, a figure of the centroid of the data in the pixel map as a function of time.
    This last figure is useful to check if the position of the pixels changed.
    This information (pixel shift) is also stored in the header ('PIX_SHIF').

    Example:
    runPL_preprocess.py --pixel_map=/path/to/pixel_map.fits /path/to/directory

    Options:
    --pixel_map: Force to select which pixel map file to use (default: the one in the directory)
"""


def filter_filelist(filelist , filelist_pixelmap):

    # Keys to keep only the RAW files
    fits_keywords = {'X_FIRTYP': ['RAW']}
        
    # Use the function to clean the filelist
    filelist_rawdata = runlib.clean_filelist(fits_keywords, filelist)
    print("runPL filelist : ", filelist_rawdata)

    # raise an error if filelist_cleaned is empty
    if len(filelist_rawdata) == 0:
        raise FileNotFoundError("No good file to pre-process")

    fits_keywords = {'X_FIRTYP': ['PIXELMAP']}
        
    # Use the function to clean the filelist
    filelist_pixelmap = runlib.clean_filelist(fits_keywords, filelist_pixelmap)
    print("Pixel map file ==>> ",filelist_pixelmap)

    # raise an error if filelist_cleaned is empty
    if len(filelist_pixelmap) == 0:
        raise FileNotFoundError("No pixel map to pre-process")

    # raise an error if filelist_cleaned is more than one
    if len(filelist_pixelmap) > 1:
        raise ValueError("Two many pixel maps to use! I can only use one.\n Please specify which one to use with the option --pixel_map")

    files_by_dir = defaultdict(list)
    for file in filelist_rawdata:
        dir_path = os.path.dirname(os.path.realpath(file))
        files_by_dir[dir_path].append(file)

    return filelist_pixelmap,files_by_dir


def preprocess(filelist_pixelmap,files_by_dir):
    """
    Preprocesses the data files using the provided pixel map and organizes them by directory.
    This function handles the preprocessing of raw data files, applying the pixel map to extract
    relevant pixel data, and saves the processed data along with diagnostic figures.
    Args:
        filelist_pixelmap (list): A list containing the pixel map file(s).
        files_by_dir (dict): A dictionary where keys are directory paths and values are lists of
                             raw data files in those directories.
    """

    pixelMap=basic.PixelMap(filelist_pixelmap[-1])
    pixel_min = pixelMap.pixel_min
    pixel_max = pixelMap.pixel_max
    pixel_wide = pixelMap.pixel_wide
    output_channels = pixelMap.output_channels
    traces_loc = pixelMap.traces_loc
    pm_check = pixelMap.pm_check

    # Process each directory separately 
    for dir_path, files in files_by_dir.items():
        raw_image = None
        center_image = None
        files_out = []

        # create a directory named preproc if it does not exist
        preproc_dir_path = os.path.join(dir_path, "preproc")
        if not os.path.exists(preproc_dir_path):
            os.makedirs(preproc_dir_path)

        
        for file in tqdm(files[:], desc=f"Pre-processing of files in {dir_path}"):
            # first read the header of the file
            header = fits.getheader(file)


            header['X_FIRTYP'] = "PREPROC"
            header['ORG_NAME'] = os.path.basename(file)
            header['PIX_MIN'] = pixel_min
            header['PIX_MAX'] = pixel_max
            header['PIX_WIDE'] = pixel_wide
            header['OUT_CHAN'] = output_channels
            header['PIXELS'] = filelist_pixelmap[-1]
            header['PM_CHECK'] = pm_check
            header['X_FIRMID'] = int(header['X_FIRMID']) # for old data reduction

            date = header.get('DATE', 'NODATE')
            date_preproc = datetime.fromtimestamp(os.path.getctime(file)).strftime('%Y-%m-%dT%H:%M:%S')
            if date == 'NODATE':
                header['DATE'] = date_preproc
                date = date_preproc

            output_filename = runlib.create_output_filename(header)
            output_filename_full = os.path.join(preproc_dir_path, output_filename)

            # Check if the file already exists
            if os.path.exists(output_filename_full):
                existing_header = fits.getheader(output_filename_full)
                if existing_header.get('PM_CHECK') == pm_check:
                    continue

            # now reading the data
            data = fits.getdata(file)

            if len(data.shape) == 2:
                data = data[None]

            if raw_image is None:
                raw_image = np.zeros_like(data.sum(axis=0), dtype=np.double)

            raw_image += data.sum(axis=0)
            
            data_cut_pixels, data_dark_pixels = basic.preprocess_cutData(data, pixelMap, True)

            perc_background=np.percentile(data_dark_pixels.ravel(),[50-34.1,50,50+34.1],axis=0)
            data_mean= np.percentile(np.mean(data_cut_pixels,axis=(1,2)),90,axis=0)
            data_cut = np.sum(data_cut_pixels,axis=-1,dtype='uint32')
            flux_mean = np.mean(data_cut,axis=(0,1,2))-perc_background[1]*(pixel_wide*2+1)

            if center_image is None:
                center_image = data_mean[:,None]
            else:
                center_image = np.concatenate((center_image,data_mean[:,None]),axis=1)

            centered=data_mean.argmax()-pixel_wide

            comp_hdu = fits.PrimaryHDU(data_cut, header=header)

            # Add quality control values to header with the values read in the header above
            comp_hdu.header['QC_SHIFT'] = centered
            comp_hdu.header['QC_BACK'] = perc_background[1]
            comp_hdu.header['QC_BACKR'] = (perc_background[2]-perc_background[0])/2*np.sqrt(2)
            comp_hdu.header['QC_FLUX'] = flux_mean


            # Add the MODULATION extension from the original file to the new FITS file
            if 'MODULATION' in fits.open(file):
                modulation_hdu = fits.open(file)['MODULATION']
                comp_hdu.header['MOD_LEN'] = modulation_hdu.header['NAXIS2']
                comp_hdu = fits.HDUList([comp_hdu, modulation_hdu])

                #make coupling map
                xmod = fits.getdata(file,'MODULATION')['XMOD']
                ymod = fits.getdata(file,'MODULATION')['YMOD']
                fluxes = data_cut_pixels.mean(axis=(1,2,3))
                fig= runlib_i.plot_couplinng_map(fluxes, xmod, ymod)
                fig.savefig(output_filename_full[:-5]+".png", dpi=300)

            files_out += [output_filename]
            comp_hdu.writeto(output_filename_full, overwrite=True, output_verify='fix', checksum=True)
            
        if len(files_out) == 0:
            print(f"No files to process in {dir_path}.")
            continue

        # copy filelist_pixelmap[-1] to the preproc directory
        shutil.copy(filelist_pixelmap[-1], preproc_dir_path)

        # Generate and save the figure for the directory
        fig,ax = runlib.make_figure_of_trace(raw_image, traces_loc, pixel_wide, pixel_min, pixel_max)
        fig.savefig(os.path.join(preproc_dir_path, f"firstpl_"+date_preproc+"_PREPROC.png"), dpi=300)

        # print("file saved as: " + os.path.join(preproc_dir_path, f"firstpl_PIXELS_{os.path.basename(dir_path)}.png"))

        fig = figure("Vertical offset of the dispersed outputs with respect to extracted windows", clear=True, figsize=(5+len(files_out)*0.1, 6))
        imshow(np.log(center_image), aspect='auto', interpolation='none', extent=(-0.5, - 0.5 + len(center_image[0]), +pixel_wide + 0.5, - pixel_wide - 0.5))
        plt.title(f"{fig.get_label()}")
        plt.plot([-0.5, center_image.shape[1] - 0.5], [0, 0], ':', color='k')
        plt.plot(center_image.argmax(axis=0)-pixel_wide, 'o-', color='r')
        plt.xticks(ticks=np.arange(len(files_out)), labels=files_out, rotation=90)
        plt.ylabel("File number")
        plt.ylabel("Pixel shift")
        plt.tight_layout()
        filename_out = os.path.join(preproc_dir_path, f"firstpl_"+date_preproc+"_PREPROCSHIFT.png")
        fig.savefig(filename_out, dpi=300)
        print("PNG saved as: "+filename_out)
    

def run_preprocess(folder = ".",pixel_map_file = None):
    # Default values
    filelist = runlib.get_filelist(folder)
    if pixel_map_file==None :
        pixel_map_file = folder + "pixelmaps"
    
    filelist_pixelmap,files_by_dir = filter_filelist(filelist, pixel_map_file)
    preprocess(filelist_pixelmap,files_by_dir, output_channels_nb=1)#38)


if __name__ == "__main__":
    debug = False

    parser = OptionParser(usage)
    # Default values
    default_folder ="."
    loop = 0
    pixel_map = None

    # Add options for these values
    parser.add_option("--pixel_map", type="string", default=pixel_map,
                    help="Force to select which pixel map file to use")
    parser.add_option("--loop", type="int", default=1,
                    help="loop for X seconds (default: %default)")


    if ("VSCODE_PID" in os.environ or os.environ.get('TERM_PROGRAM') == 'vscode' or 
        os.environ.get('SPYDER_DEBUG_FILE')):
        print("Running in compiler^")
        if getpass.getuser() == "slacour":
            file_patterns = "/Users/slacour/DATA/LANTERNE/2025-05-13/firstpl/firstpl_19:51:23.573368642.fits"
            pixel_map = "/Users/slacour/DATA/LANTERNE/2025-05-13/preproc/firstpl_2025-05-13T09:10:06_PIXELMAP.fits"
        if getpass.getuser() == "jsarrazin":
            file_patterns = "/home/jsarrazin/Bureau/PLDATA/moreTest/2024-11-21_13-48-32_science_copie/preproc"
            pixel_map = "/home/jsarrazin/Bureau/PLDATA/novembre/les_preproc"
        if getpass.getuser() == "ehuby":
            file_patterns = "/home/ehuby/WORK/DATA/FIRST-PL/2025-05-10/preproc/"
    else:
        (options, args) = parser.parse_args()
        file_patterns=args if args else ['*.fits','./pixelmaps/*.fits']

        # If the user specifies a pixel map use it, otherwise look into the arguments
        pixel_map = options.pixel_map
        loop = options.loop

    if pixel_map is None:
        pixel_map = file_patterns

    time_start = time.time()
    time_wait = 30 # in seconds

    filelist=runlib.get_filelist( file_patterns )
    filelist_pixelmap=runlib.get_filelist( pixel_map )
    filelist_pixelmap,files_by_dir = filter_filelist(filelist , filelist_pixelmap)
    preprocess(filelist_pixelmap,files_by_dir)
    
    while time.time()+time_wait < loop+time_start:
        time.sleep(time_wait)
        filelist_new=runlib.get_filelist( file_patterns )
        # Check for new files in filelist
        new_files = [file for file in filelist_new if file not in filelist]
        if new_files:
            print(f"New files detected: {new_files}")
            filelist.extend(new_files)
            filelist_pixelmap,files_by_dir = filter_filelist(new_files , filelist_pixelmap)
            preprocess(filelist_pixelmap,files_by_dir)
        else:
            print("Waiting for new files...", end="\r")


# %%
