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
import shutil
from collections import defaultdict

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
    fits_keywords = {'DATA-CAT': ['RAW']}
        
    # Use the function to clean the filelist
    filelist_rawdata = runlib.clean_filelist(fits_keywords, filelist)
    print("runPL filelist : ", filelist_rawdata)

    # raise an error if filelist_cleaned is empty
    if len(filelist_rawdata) == 0:
        raise ValueError("No good file to pre-process")

    fits_keywords = {'DATA-CAT': ['PIXELMAP']}
        
    # Use the function to clean the filelist
    filelist_pixelmap = runlib.clean_filelist(fits_keywords, filelist_pixelmap)
    print("Pixel map file ==>> ",filelist_pixelmap)

    # raise an error if filelist_cleaned is empty
    if len(filelist_pixelmap) == 0:
        raise ValueError("No pixel map to pre-process")

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


    # Process each directory separately 
    for dir_path, files in files_by_dir.items():
        raw_image = None
        center_image = None
        files_out = []
        
        for file in tqdm(files[:], desc=f"Pre-processing of files in {dir_path}"):
            data = fits.getdata(file)
            header = fits.getheader(file)
            object = header.get('OBJECT', "NONAME")
            date = header.get('DATE', 'NODATE')
            type = header.get('DATA-TYP',None)
            date_preproc = datetime.fromtimestamp(os.path.getctime(file)).strftime('%Y-%m-%dT%H:%M:%S')

            header['GAIN'] = 1

            if date == 'NODATE':
                header['DATE'] = date_preproc
                date = date_preproc

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

            # Update the header with the values read in the headers above
            comp_hdu.header['DATA-CAT'] = "PREPROC"
            comp_hdu.header['ORG_NAME'] = os.path.basename(file)
            comp_hdu.header['PIX_MIN'] = pixel_min
            comp_hdu.header['PIX_MAX'] = pixel_max
            comp_hdu.header['PIX_WIDE'] = pixel_wide
            comp_hdu.header['OUT_CHAN'] = output_channels
            comp_hdu.header['PIXELS'] = filelist_pixelmap[-1]
            comp_hdu.header['QC_SHIFT'] = centered
            comp_hdu.header['QC_BACK'] = perc_background[1]
            comp_hdu.header['QC_BACKR'] = (perc_background[2]-perc_background[0])/2*np.sqrt(2)
            comp_hdu.header['QC_FLUX'] = flux_mean

            # Add the MODULATION extension from the original file to the new FITS file
            if 'MODULATION' in fits.open(file):
                modulation_hdu = fits.open(file)['MODULATION']
                comp_hdu = fits.HDUList([comp_hdu, modulation_hdu])


            # create a directory named preproc if it does not exist
            preproc_dir_path = os.path.join(dir_path, "preproc")
            if not os.path.exists(preproc_dir_path):
                os.makedirs(preproc_dir_path)
            
            output_filename = runlib.create_output_filename(header)
            files_out += [output_filename]
            comp_hdu.writeto(os.path.join(preproc_dir_path, output_filename), overwrite=True, output_verify='fix', checksum=True)
            

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

    # Add options for these values
    parser.add_option("--pixel_map", type="string", default=None,
                    help="Force to select which pixel map file to use (default: the one in the directory)")

    (options, args) = parser.parse_args()
    file_patterns=args if args else ['*.fits']

    # If the user specifies a pixel map use it, otherwise look into the arguments
    pixel_map = options.pixel_map
    if pixel_map is None:
        pixel_map = ['./pixelmaps/*.fits']

    filelist=runlib.get_filelist( file_patterns )
    filelist_pixelmap=runlib.get_filelist( pixel_map )
    filelist_pixelmap,files_by_dir = filter_filelist(filelist , filelist_pixelmap)

    preprocess(filelist_pixelmap,files_by_dir)

# %%
