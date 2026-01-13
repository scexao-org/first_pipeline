#! /usr/bin/env python3
# -*- coding: iso-8859-15 -*-
#%%
"""
FIRST Pipeline - Data Preprocessing

This script preprocesses raw FIRST Visible Photonic Lantern data at SUBARU/SCEXAO
using pixel maps to extract spectral traces. It applies pixel map alignment, 
cleans/calibrates raw data, and generates quality metrics essential for 
downstream processing.

Preprocessing is a critical step that transforms raw detector images into
calibrated spectral data suitable for scientific analysis.

Created on Wed May 21 22:56:25 2025
@author: slacour
"""

import os
import sys
from astropy.io import fits
from glob import glob
import argparse
import numpy as np

import getpass
import matplotlib
# Import FIRST pipeline classes
from classes.runPL_class_fileList import FileList

if "VSCODE_PID" in os.environ:
    matplotlib.use('Qt5Agg')
else:
    matplotlib.use('Agg')
     
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot,hist,clf,figure,legend,imshow
from datetime import datetime
from tqdm import tqdm
import libraries.runPL_library_io as runlib_io
import libraries.runPL_library_plots as runlib_plots
import shutil
from collections import defaultdict
import time
from astroplan import Observer
from astropy.time import Time
from classes.runPL_class_dataCube import DataCube
from classes.runPL_class_pixelMap import PixelMap

subaru = Observer.at_site("Subaru")
now_time = Time.now()
if subaru.is_night(now_time):
    print("It's night at Subaru Observatory.")
else:
    print("It's day at Subaru Observatory.")

# plt.ion()
# Add options
usage = """
Usage: python runPL_make_preproc.py [options] [directory | files.fits]

Goal:
    Preprocess raw FITS files using pixel maps to extract spectral traces.
    This script applies pixel maps to raw data, extracts traces, computes quality metrics,
    and outputs preprocessed FITS files with diagnostic figures.

Input: 
    - Raw FITS files with X_FIRTYP='RAW' in the specified directory
    - Pixel map FITS files with X_FIRTYP='PIXELMAP' for trace extraction
    - Optional MODULATION extension in raw files for coupling analysis

Output:
    - Preprocessed FITS files with X_FIRTYP='PREPROC' in ../preproc/ directory
    - Diagnostic PNG figures:
        * Trace overlay visualization (*_1.png)
        * Flux coupling maps (*_2.png, when modulation data available)
        * Summary centroid shift plot (*_PREPROCSHIFT.png)
    - Updated FITS headers with quality control metrics (Q_P_* keywords)

Options:
    --pixel_map=PATH   Specify pixel map file or directory (default: auto-detect from ../pixelmaps)
    --object=NAME      Process only files matching OBJECT header keyword
    
Examples:
    python runPL_make_preproc.py /path/to/raw/data/
    python runPL_make_preproc.py --pixel_map=/path/to/pixelmaps/ /path/to/raw/
    python runPL_make_preproc.py --object=HIP84212 /data/science/

Notes:
    - Files are processed only if no matching preprocessed file exists with same PM_CHECK
    - Centroid shift diagnostics help monitor trace stability over time
    - Supports both single files and directory processing with glob patterns
    - Compatible with FIRST Visible Photonic Lantern pipeline workflow
"""



def preprocess(fileList, plot_sum =False):
    """""
    Preprocesses raw FITS files using provided pixel map(s), extracts and aggregates spectral
    traces, computes basic quality-control metrics, and writes preprocessed FITS files and
    diagnostic PNG figures into a per-directory "preproc" folder.
    This function is designed to be run on a mapping of individual raw file paths to their
    corresponding pixel map files. Each raw file is processed independently (but results are
    aggregated for optional summary plotting). The function will skip processing for files
    that already have a preprocessed output with a matching pixel-map check value (PM_CHECK),
    unless the presence/absence of a MODULATION extension implies further processing is required.
    Parameters
    ----------
    files_with_pixelmap : dict
        Mapping of raw FITS file path -> pixel map file path. Each key is a path to a raw
        FITS file and the corresponding value is the pixel map filename used to extract traces.
    plot_sum : bool, optional
        If True, a summary PNG showing the vertical offset of extracted windows across all
        processed files will be produced and saved into the last processed directory's
        preproc folder. Default is False.
    Returns
    -------
    list
        A list of output filenames (basename of created preprocessed FITS files) that were
        created during this call. If no files were processed, an empty list is returned.
    Side effects
    ------------
    - Creates a "preproc" subdirectory next to each input file's directory if it does not
      already exist.
    - Writes one preprocessed FITS file per input file into that "preproc" directory.
    - Writes diagnostic PNG(s) for each file (trace figure and optional coupling/flux map)
      and an optional summary PNG when plot_sum is True.
    - Modifies/creates FITS headers inside the output files to record provenance and QC
      metrics.
    - May read a MODULATION HDU from the original file and append it to the output HDUList.
    FITS header keys added or modified in the output primary HDU
    ------------------------------------------------------------
    Provenance / processing info:
    - X_FIRTYP   : "PREPROC"
    - X_FIRWOL   : copied from original header (fallback 'IN')
    - X_FIRMID   : original X_FIRMID cast to int (used to decide on MODULATION handling)
    - DATE-PRO   : processing timestamp (YYYY-MM-DDThh:mm:ss)
    - ORG_NAME   : basename of the original raw file
    - PM_FILE    : path to the pixel map file used
    - PM_CHECK   : pixel map check value (from PixelMap.pm_check)
    Pixel-map / extraction parameters:
    - PIX_MIN, PIX_MAX, PIX_WIDE : integers describing extraction window limits and width
    - OUT_CHAN   : number of output channels / traces
    Quality-control (QC) metrics (stored as Q_P_* keys):
    - Q_P_CENT   : integer pixel index of the extracted window center (relative to window)
    - Q_P_BACK   : background level (median estimate)
    - Q_P_BACN   : background noise estimate (approx 1-σ equivalent)
    - Q_P_FLUX   : mean extracted flux (background-subtracted) used for diagnostics
    - Q_P_NAME   : output filename (basename) for cross-reference
    Modulation-related:
    - MOD_LEN    : length (NAXIS2) of the MODULATION HDU if present; the MODULATION HDU
                   is copied into the output file when appropriate.
    Additional:
    - Any header keywords from the pixel map header starting with 'P_PM' are copied into
      the output header to preserve pixel-map metadata.
    Note about FITS header comments
    -------------------------------
    - A human-readable comment should be inserted for important header keywords to aid users
      inspecting the file (for example, describing the meaning of PM_CHECK or Q_P_NAME).
      Example TODO: add a FITS header comment for 'PM_CHECK' explaining that it is the
      pixel-map checksum/version used to decide whether an existing preprocessed file
      matches the pixel map. (Insert the comment using the fits header comment parameter
      when setting the header keyword.)
    Behavioral details and heuristics
    --------------------------------
    - If the raw FITS file data is 2D, it will be treated as a single frame (added a leading
      frame axis). If already multi-frame (3D), it is processed as-is.
    - The function computes a summed "raw_image" to produce a trace-location diagnostic
      figure via runlib_io.make_figure_of_trace and saves it alongside the preprocessed file.
    - PixelMap.preprocess_cutData is used to extract the per-trace pixel arrays and dark
      pixels; background percentiles and per-channel flux statistics are computed from
      these arrays for QC.
    - If X_FIRMID indicates newer-format data and the original file contains a MODULATION
      extension, that extension is attached to the output file and a coupling map figure
      may be produced (requires sufficient modulation points).
    - Files are skipped if an existing output file is present with the same PM_CHECK and
      (when relevant) matching presence/absence of the MODULATION extension; this avoids
      redundant re-processing.
    Errors and exceptions
    ---------------------
    - The function relies on valid FITS files and a valid PixelMap implementation. IO/reading
      errors will propagate from astropy.io.fits and from filesystem operations (os.path,
      os.makedirs). The caller may want to catch and handle these exceptions (e.g. FileNotFoundError,
      OSError, astropy.io.fits-related errors) depending on usage.
    Notes
    -----
    - The function writes FITS outputs with overwrite=True and output_verify='fix' and
      checksum=True to provide consistent checksums in produced files.
    - The pixelmap copy-to-preproc step is present but commented out in the implementation;
      re-enable if a local copy of the pixelmap inside preproc is desired.
    """


    center_image = None
    dir_path_0 = fileList.get_most_common_dir()
    files_out = []
    # print(list(files_with_pixelmap.keys()))
    # dir_path_0 = os.path.dirname(list(files_with_pixelmap.keys())[0])
    # if len(dir_path_0) == 0:
    #     dir_path_0 = '.'

    # Process each directory separately 
    for file_withpixelmap in tqdm(fileList.files_with_associated_files, desc=f"Pre-processing of files in {dir_path_0}"):

        file = file_withpixelmap['file']
        pixelmap = file_withpixelmap['pixelMap']

        if pixelmap is None:
            print(f"No pixel map associated with {file}, skipping.")
            continue

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
        try:
            data = fits.getdata(file)
        except Exception as e:
            print(f"Error reading {file}: {e}")
            continue

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
        comp_hdu.header['Q_P_CENT'] = (centered, 'center of extracted window (pixel index)')
        comp_hdu.header['Q_P_BACK'] = (perc_background[1],'average background detected')
        comp_hdu.header['Q_P_BACN'] = ((perc_background[2]-perc_background[0])/2*np.sqrt(2), 'background noise estimate')
        # Quality control: mean extracted flux (background-subtracted) stored for diagnostics
        comp_hdu.header['Q_P_FLUX'] = (flux_mean, 'mean extracted flux per pixel (background-subtracted)')    
        comp_hdu.header['Q_P_NAME'] = (output_filename, 'output filename of preprocessed data')

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
                    string_title = hd['OBJECT']+" - "+hd['DATA-TYP']+" - "+str(hd['EXPTIME'])+'s \n X_FIROBX = '+str(hd.get('X_FIROBX', 'N/A'))+", X_FIROBY = "+str(hd.get('X_FIROBY', 'N/A'))
                    fig.suptitle(string_title)
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

    parser = argparse.ArgumentParser(
        description="Preprocess raw FIRST Photonic Lantern data using pixel maps for spectral extraction and calibration.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
FIRST Pipeline Data Preprocessing Tool

This script transforms raw detector images into calibrated spectral data by applying
pixel maps for precise spectral extraction. It includes quality assessment and 
diagnostic analysis to ensure reliable data for downstream processing.

Examples:
    %(prog)s --pixel_map=/path/to/pixel_map.fits /path/to/directory
    %(prog)s --object="HD 164461" /path/to/files*.fits
    %(prog)s --loop 30 /path/to/directory  # Monitor mode
    %(prog)s /data/raw/*.fits

Pipeline Workflow Integration:
    1. Requires raw FITS files (X_FIRTYP=RAW) and pixel maps (X_FIRTYP=PIXELMAP)
    2. Applies spectral extraction using pixel map calibration
    3. Outputs preprocessed files (X_FIRTYP=PREPROC) for downstream analysis
    4. Essential step before flat field, wavelength, and coupling map generation

Input Files:
    - Raw FITS files: X_FIRTYP=RAW containing detector images
    - Pixel map files: X_FIRTYP=PIXELMAP from runPL_create_pixelMap.py
    - Automatic pixel map detection or manual selection with --pixel_map

Output Files:
    - Preprocessed FITS files: X_FIRTYP=PREPROC (preproc/ directory)
    - Diagnostic figures showing extraction quality and stability:
      * Pixel map overlay on raw images
      * Centroid shift analysis as function of time
      * Quality control metrics
    - Quality metrics in FITS headers (QC_SHIFT for position stability)

Processing Features:
    - Spectral trace extraction using calibrated pixel positions
    - Quality control metrics for data validation
    - Centroid shift monitoring for instrument stability
    - Object-based selection for targeted processing
    - Monitor mode (--loop) for real-time processing during observations
    - Automatic handling of different Wollaston configurations

Monitor Mode:
    - Use --loop to continuously monitor directory for new raw files
    - Automatic processing when new files appear
    - Ideal for real-time data reduction during observations
    - Configurable polling interval in seconds

Quality Assessment:
    - Centroid shift tracking detects instrument drift
    - Extraction quality metrics identify problematic data
    - Diagnostic figures enable visual quality control
    - QC flags stored in FITS headers for downstream filtering

Technical Notes:
    - Pixel maps define exact spectral trace positions for extraction
    - Quality metrics guide data acceptance/rejection decisions
    - Compatible with all downstream pipeline scripts
    - Supports both polarimetry and photometry observing modes

Note: Monitor centroid shift plots to assess instrument stability.
Large shifts may indicate mechanical flexure or alignment issues requiring
attention before proceeding with scientific analysis.
        """
    )

    # needed to work in VSC:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--f", help=argparse.SUPPRESS)

    # Add positional argument for files/directories
    parser.add_argument('files', nargs='*', default=['*.fits'],
                       help='Directory or FITS files to process (default: *.fits)')

    # Add optional arguments
    parser.add_argument("--pixel_map", 
                       help="Specify which pixel map FITS file to use (default: auto-detect in directory)")
    parser.add_argument("--object", 
                       help="Specify the OBJECT name of data to reduced based on the FITS header")
    parser.add_argument("--only_with_modulation", action="store_true",
                       help="Only preprocess files that have a MODULATION extension in the FITS file.")
    
    # Initialize default values
    args = parser.parse_args()
    file_patterns = args.files if args.files else ['*.fits']
    pixel_map = args.pixel_map
    object = args.object
    only_with_modulation = args.only_with_modulation

    if ("VSCODE_PID" in os.environ or os.environ.get('TERM_PROGRAM') == 'vscode' or 
        os.environ.get('SPYDER_DEBUG_FILE')):
        print("Running in compiler^")
        if getpass.getuser() == "slacour":
            dir_files="/Users/slacour/DATA/LANTERNE/20250808/preproc/"
            file_patterns = dir_files+"firstpl_2025-08-08T07:17:??_HIP84212_P.fits"
            dir_files = "/Users/slacour/DATA/LANTERNE/tmp/"
            file_patterns = dir_files + "*.fits"
            file_patterns=["/Users/slacour/DATA/LANTERNE/raw/20251119/firstpl"]
        
        if getpass.getuser() == "jsarrazin":
            file_patterns = "/home/jsarrazin/Bureau/PLDATA/moreTest/2024-11-21_13-48-32_science_copie/preproc"
            pixel_map = "/home/jsarrazin/Bureau/PLDATA/novembre/les_preproc"
        if getpass.getuser() == "ehuby":
            file_patterns = "/home/ehuby/WORK/DATA/FIRST-PL/2025-05-10/preproc/"

    if pixel_map is None:
        folder = os.path.dirname(file_patterns[0])
        print("Using pixel map folder: ",folder)
        pixel_map = file_patterns + [os.path.join(folder,"../pixelmaps")]

    fileList = FileList(file_patterns, first_type='RAW', object_name=object, modID=[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20] if args.only_with_modulation else None)
    
    fileList.make_association(pixelMap=pixel_map)


    print(f"Found {len(fileList.filelist)} files to process in {file_patterns}")
    
    preprocess(fileList)
    

# %%
