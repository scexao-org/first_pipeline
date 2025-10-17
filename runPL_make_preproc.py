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
import runPL_library_io as runlib_io
import runPL_library_plots as runlib_plots
import shutil
from collections import defaultdict
import time
from astroplan import Observer
from astropy.time import Time
from runPL_class_pixelMap import PixelMap

subaru = Observer.at_site("Subaru")
now_time = Time.now()
if subaru.is_night(now_time):
    print("It's night at Subaru Observatory.")
else:
    print("It's day at Subaru Observatory.")

plt.ion()

# Add options
usage = """
Usage: %prog [options] [directory | files.fits]

Goal:
    Preprocess the data using the pixel map.

Input: 
    - Files of type X_FIRTYP=RAW in the directory.
    - Files of type X_FIRTYP=PIXELMAP in the directory.
    - The pixel map file is used to extract the data from the raw files.
    - The pixel map file is used to create a new FITS file with the data extracted from the raw files. 

Output:
    - Files of type X_FIRTYP=PREPROC in the preproc directory.
    - Diagnostic figures saved in the preproc directory:
        * Pixel map overlay on raw images.
        * Centroid shift of the data in the pixel map as a function of time.
    - Pixel shift information is also stored in the FITS header ('QC_SHIFT').

Options:
    --pixel_map=FILE   Specify which pixel map FITS file to use (default: auto-detect in directory)
    --loop=SECONDS     Loop and check for new files every X seconds (default: 0, i.e., run once)

Examples:
    runPL_preprocess.py --pixel_map=/path/to/pixel_map.fits /path/to/directory
    runPL_preprocess.py /path/to/files*.fits

Notes:
    The centroid shift figure is useful to check if the position of the pixels changed over time.
"""



def preprocess(files_with_pixelmap, plot_sum =False):
    """
    Preprocesses the data files using the provided pixel map and organizes them by directory.
    This function handles the preprocessing of raw data files, applying the pixel map to extract
    relevant pixel data, and saves the processed data along with diagnostic figures.
    Args:
        filelist_pixelmap (list): A list containing the pixel map file(s).
        files_by_dir (dict): A dictionary where keys are directory paths and values are lists of
                             raw data files in those directories.
    """

    center_image = None
    files_out = []
    print(files_with_pixelmap)
    dir_path_0 = os.path.dirname(list(files_with_pixelmap.keys())[0])
    if len(dir_path_0) == 0:
        dir_path_0 = '.'

    # Process each directory separately 
    for file, pixelmap in tqdm(files_with_pixelmap.items(), desc=f"Pre-processing of files in {dir_path_0}"):

        pixelMap=PixelMap(pixelmap)
        pixel_min = pixelMap.pixel_min
        pixel_max = pixelMap.pixel_max
        pixel_wide = pixelMap.pixel_wide
        output_channels = pixelMap.output_channels
        traces_loc = pixelMap.traces_loc
        pm_check = pixelMap.pm_check

        dir_path = os.path.dirname(file)
        # Create a directory named preproc if it does not exist
        preproc_dir_path = os.path.join(dir_path, "../preproc")
        if not os.path.exists(preproc_dir_path):
            os.makedirs(preproc_dir_path)
        
        
        # first read the header of the file
        header = fits.getheader(file)


        header['X_FIRTYP'] = "PREPROC"
        header['X_FIRWOL'] = fits.getheader(file).get('X_FIRWOL', 'IN')
        header['X_FIRMID'] = int(header['X_FIRMID']) # for old data reduction

        # Add date and time to the header
        current_time = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
        header['DATE-PRO'] = current_time

        header['ORG_NAME'] = os.path.basename(file)
        header['PIX_MIN'] = pixel_min
        header['PIX_MAX'] = pixel_max
        header['PIX_WIDE'] = pixel_wide
        header['OUT_CHAN'] = output_channels
        header['PM_FILE'] = pixelmap
        header['PM_CHECK'] = pm_check

        date_preproc = datetime.fromtimestamp(os.path.getctime(file)).strftime('%Y-%m-%dT%H:%M:%S')
        date = header.get('DATE', None)
        if date is None:
            header['DATE'] = date_preproc
        else:
             date = "2025-07-14T11:20:30"
             obs_time = Time(date)
             #if data taken during daytime (calibration source, for exemaple) overide the OBJECT keyword
             if not subaru.is_night(obs_time):
                 header['OBJECT'] = "DAY"


        output_filename = runlib_io.create_output_filename(header)
        output_filename_full = os.path.join(preproc_dir_path, output_filename)

        # Check if the file already exists
        if os.path.exists(output_filename_full):
            existing_header = fits.getheader(output_filename_full)
            if existing_header.get('PM_CHECK') == pm_check:
                if 'MODULATION' in fits.open(file):
                    # Check if the MODULATION extension is present in the original file
                    if 'MODULATION' in fits.open(output_filename_full):
                        # If it is, we can skip this file
                        continue
                else:
                    continue

        # now reading the data
        data = fits.getdata(file)

        if len(data.shape) == 2:
            data = data[None]

        raw_image = data.sum(axis=0)

        # Generate and save the figure for the directory
        fig,ax = runlib_io.make_figure_of_trace(raw_image, traces_loc, pixel_wide, pixel_min, pixel_max)
        fig.savefig(output_filename_full[:-5]+"_1.png", dpi=300)
        
        data_cut_pixels, data_dark_pixels = pixelMap.preprocess_cutData(data, True)

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
        comp_hdu.header['P_P_CENT'] = centered
        comp_hdu.header['P_P_BACK'] = perc_background[1]
        comp_hdu.header['P_P_BACN'] = (perc_background[2]-perc_background[0])/2*np.sqrt(2)
        comp_hdu.header['P_P_FLUX'] = flux_mean
        comp_hdu.header['P_P_NAME'] = output_filename

        pmhd= pixelMap.header
        #add all headers keywords from pmhd that starts with P_PM to comp_hdu.header
        for key in pmhd.keys():
            if key.startswith('P_PM'):
                comp_hdu.header[key] = pmhd[key]

        # Add the MODULATION extension from the original file to the new FITS file
        if comp_hdu.header.get('X_FIRMID', 0) > 1:
            if 'MODULATION' in fits.open(file):
                modulation_hdu = fits.open(file)['MODULATION']
                comp_hdu.header['MOD_LEN'] = modulation_hdu.header['NAXIS2']
                hd=comp_hdu.header
                comp_hdu = fits.HDUList([comp_hdu, modulation_hdu])

                #make coupling map
                xmod = fits.getdata(file,'MODULATION')['XMOD']
                ymod = fits.getdata(file,'MODULATION')['YMOD']
                if len(xmod) > 9:
                    fluxes = data_cut_pixels.mean(axis=(1,2,3))
                    fig= runlib_plots.plot_flux_map(fluxes, xmod, ymod)
                    fig.suptitle(hd['OBJECT']+" - "+hd['DATA-TYP']+" - "+str(hd['EXPTIME'])+'s')
                    fig.savefig(output_filename_full[:-5]+"_2.png", dpi=300)

        files_out += [output_filename]
        comp_hdu.writeto(output_filename_full, overwrite=True, output_verify='fix', checksum=True)

        # copy the pixelmap to the preproc directory
        # dest_pixelmap = os.path.join(preproc_dir_path, os.path.basename(pixelmap))
        # if not os.path.exists(dest_pixelmap):
        #     shutil.copy(pixelmap, preproc_dir_path)
            
    if len(files_out) == 0:
        print(f"No files to process in {dir_path}.")
        return

    if plot_sum == True:
        # copy filelist_pixelmap[-1] to the preproc directory
        filename_out = files_out[-1]
        filename_out = "_".join(filename_out.split("_")[:-2])
        filename_out_full = os.path.join(preproc_dir_path, filename_out)

        try:
            fig = figure("Vertical offset of the dispersed outputs with respect to extracted windows", clear=True, figsize=(5+len(files_out)*0.1, 6))
            imshow(np.log(center_image), aspect='auto', interpolation='none', extent=(-0.5, - 0.5 + len(center_image[0]), +pixel_wide + 0.5, - pixel_wide - 0.5))
            plt.title(f"{fig.get_label()}")
            plt.plot([-0.5, center_image.shape[1] - 0.5], [0, 0], ':', color='k')
            plt.plot(center_image.argmax(axis=0)-pixel_wide, 'o-', color='r')
            plt.xticks(ticks=np.arange(len(files_out)), labels=files_out, rotation=90)
            plt.ylabel("File number")
            plt.ylabel("Pixel shift")
            plt.tight_layout()
            fig.savefig(filename_out_full+"_PREPROCSHIFT.png", dpi=300)
            print("PNG saved as: "+filename_out_full+"_PREPROCSHIFT.png")
        except:
            print("Error while plotting the vertical offset of the dispersed outputs with respect to extracted windows")
    

def run_preprocess(folder = ".",pixel_map_file = None):
    # Default values
    filelist = runlib_io.get_filelist(folder, {'X_FIRTYP': ['RAW']})
    if pixel_map_file==None :
        pixel_map_file = folder + "pixelmaps"

    pixel_map_file = runlib_io.get_filelist(pixel_map_file, {'X_FIRTYP': ['PIXELMAP']})

    files_with_pixelmap = runlib_io.associate_pixelmap(filelist, pixel_map_file)
    preprocess(files_with_pixelmap)


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
    parser.add_option("--loop", type="int", default=loop,
                    help="loop for X seconds (default: %default)")


    if ("VSCODE_PID" in os.environ or os.environ.get('TERM_PROGRAM') == 'vscode' or 
        os.environ.get('SPYDER_DEBUG_FILE')):
        print("Running in compiler^")
        if getpass.getuser() == "slacour":
            dir_files="/Users/slacour/DATA/LANTERNE/20250808/preproc/"
            file_patterns = dir_files+"firstpl_2025-08-08T07:17:??_HIP84212_P.fits"
            pixel_map = dir_files + "../pixelmaps"
        if getpass.getuser() == "jsarrazin":
            file_patterns = "/home/jsarrazin/Bureau/PLDATA/moreTest/2024-11-21_13-48-32_science_copie/preproc"
            pixel_map = "/home/jsarrazin/Bureau/PLDATA/novembre/les_preproc"
        if getpass.getuser() == "ehuby":
            file_patterns = "/home/ehuby/WORK/DATA/FIRST-PL/2025-05-10/preproc/"
    else:
        (options, args) = parser.parse_args()
        file_patterns=args if args else ['*.fits']

        # If the user specifies a pixel map use it, otherwise look into the arguments
        pixel_map = options.pixel_map
        loop = options.loop

    if pixel_map is None:
        pixel_map = file_patterns + ['../pixelmaps/*.fits']
    if loop > 0:
        plot_sum = False
    else:
        plot_sum = True


    time_start = time.time()
    time_wait = 30 # in seconds

    filelist = runlib_io.get_filelist( file_patterns , {'X_FIRTYP': ['RAW']})
    print(f"Found {len(filelist)} files to process in {file_patterns}")
    # print(filelist)
    filelist_pixelmap = runlib_io.get_filelist( pixel_map , {'X_FIRTYP': ['PIXELMAP']},  name_search="pixel map")
    print(f"Found {len(filelist_pixelmap)} pixel map files in {pixel_map}")
    # print(filelist_pixelmap)
    files_with_pixelmap = runlib_io.associate_pixelmap(filelist , filelist_pixelmap)
    preprocess(files_with_pixelmap, plot_sum = plot_sum)
    
    while time.time()+time_wait < loop+time_start:
        time.sleep(time_wait)
        filelist_new=runlib_io.get_filelist( file_patterns , {'X_FIRTYP': ['RAW']})
        # Check for new files in filelist
        new_files = [file for file in filelist_new if file not in filelist]
        if new_files:
            print(f"New files detected: {new_files}")
            filelist.extend(new_files)
            filelist_pixelmap,files_by_dir = runlib_io.associate_pixelmap(new_files , filelist_pixelmap, plot_sum= False)
            preprocess(files_with_pixelmap)
        else:
            print("Waiting for new files for the next %i seconds"%(int(loop+time_start-time.time())), end="\r")


# %%
