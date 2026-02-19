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

if "VSCODE_PID" in os.environ:
    matplotlib.use('Qt5Agg')
else:
    matplotlib.use('Agg')
     

import matplotlib.pyplot as plt
from matplotlib.pyplot import plot,hist,clf,figure,legend,imshow
from datetime import datetime
from tqdm import tqdm
from .libraries import runPL_library_io as runlib_io
from .libraries import runPL_library_plots as runlib_plots
import shutil
from collections import defaultdict
import time
from astroplan import Observer
from astropy.time import Time
from .classes.runPL_class_fileList import FileList
from .classes.runPL_class_dataCube import DataCube
from .classes.runPL_class_pixelMap import PixelMap
from .classes.runPL_class_preproc import Preproc

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



def preprocess(fileList, overwrite , plot_sum=False):
    """
    Preprocesses raw FITS files using provided pixel map(s), extracts and aggregates spectral
    traces, computes basic quality-control metrics, and writes preprocessed FITS files and
    diagnostic PNG figures into a per-directory "preproc" folder.
    
    This function now uses the Preproc class to handle individual file processing.
    
    Parameters
    ----------
    fileList : FileList
        FileList object containing raw files and their associated pixel maps
    plot_sum : bool, optional
        If True, a summary PNG showing the vertical offset of extracted windows across all
        processed files will be produced and saved. Default is False.
        
    Returns
    -------
    list
        A list of output filenames (basename of created preprocessed FITS files) that were
        created during this call. If no files were processed, an empty list is returned.
    """

    center_image = None
    dir_path_0 = fileList.get_most_common_dir()
    files_out = []
    centroid_data = []  # Store centroid data for summary plot
    
    # Process each file using the Preproc class
    for file_withpixelmap in tqdm(fileList.files_with_associated_files, desc=f"Pre-processing of files in {dir_path_0}"):
        
        file = file_withpixelmap['file']
        pixelmap_file = file_withpixelmap['pixelMap']
        output_dir = os.path.join(os.path.dirname(file), "../preproc")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        if pixelmap_file is None:
            print(f"No pixel map associated with {file}, skipping.")
            continue

        pixelmap = PixelMap(pixelmap_file)
            
        # try:
        if True:
            # Create Preproc instance and process the file
            preproc = Preproc()
            
            preproc_created = preproc.create_from_raw(file, pixelmap, output_dir, check_if_exist=not overwrite)
            
            if preproc_created:

                preproc.generate_diagnostic_figures(pixelmap)

                # Collect centroid data for summary plot
                if preproc.quality_metrics:
                    centroid_data.append(preproc.quality_metrics.get('Q_P_CENT', 0))
                    
                # Save the preprocessed file
                preproc.save()

        # except Exception as e:
        #     print(f"Error processing {file}: {e}")
        #     continue
    
    if len(files_out) == 0:
        print(f"No files to process in {dir_path_0}.")
        return []

    if plot_sum and len(centroid_data) > 0:
        # Create summary plot for centroid shifts
        preproc_dir_path = os.path.join(dir_path_0, "../preproc")
        filename_out = files_out[-1] if files_out else "summary"
        filename_out = "_".join(filename_out.split("_")[:-2]) if "_" in filename_out else filename_out
        filename_out_full = os.path.join(preproc_dir_path, filename_out)
        
        try:
            fig = figure("Centroid shift summary", clear=True, figsize=(max(8, len(files_out)*0.3), 6))
            plt.plot(range(len(centroid_data)), centroid_data, 'o-', color='red', markersize=4)
            plt.axhline(y=0, color='black', linestyle=':', alpha=0.7)
            plt.title("Vertical offset of extracted windows (centroid shift)")
            plt.xlabel("File index")
            plt.ylabel("Pixel shift")
            plt.grid(True, alpha=0.3)
            
            # Set x-axis labels if not too many files
            if len(files_out) <= 20:
                plt.xticks(range(len(files_out)), files_out, rotation=90)
            else:
                # For many files, show only some labels
                step = max(1, len(files_out) // 10)
                indices = range(0, len(files_out), step)
                labels = [files_out[i] for i in indices]
                plt.xticks(indices, labels, rotation=90)
                
            plt.tight_layout()
            fig.savefig(filename_out_full + "_PREPROCSHIFT.png", dpi=300)
            print("PNG saved as: " + filename_out_full + "_PREPROCSHIFT.png")
        except Exception as e:
            print(f"Error while plotting centroid shift summary: {e}")
            
    return files_out
    

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

def main():
    """
    Main entry point for the preprocessing script.
    """
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
                       help="Also preprocess files that do not have a MODULATION extension in the FITS file.")
    parser.add_argument("--overwrite", action="store_true",
                       help="Overwrite existing preprocessed files if they exist.")
    
    # Initialize default values
    args = parser.parse_args()
    file_patterns = args.files if args.files else ['*.fits']
    pixel_map = args.pixel_map
    object = args.object
    only_with_modulation = args.only_with_modulation
    overwrite = args.overwrite

    if ("VSCODE_PID" in os.environ or os.environ.get('TERM_PROGRAM') == 'vscode' or 
        os.environ.get('SPYDER_DEBUG_FILE')):
        print("Running in compiler^")
        if getpass.getuser() == "slacour":
            dir_files="/Users/slacour/DATA/LANTERNE/20250808/preproc/"
            file_patterns = dir_files+"firstpl_2025-08-08T07:17:??_HIP84212_P.fits"
            dir_files = "/Users/slacour/DATA/LANTERNE/tmp/"
            file_patterns = dir_files + "*.fits"
            file_patterns=["/Users/slacour/DATA/LANTERNE/raw/20251119/firstpl"]
            # file_patterns = ["/Users/slacour/DATA/LANTERNE/raw/20251118/firstpl/"]
        
        if getpass.getuser() == "jsarrazin":
            file_patterns = "/home/jsarrazin/Bureau/PLDATA/moreTest/2024-11-21_13-48-32_science_copie/preproc"
            pixel_map = "/home/jsarrazin/Bureau/PLDATA/novembre/les_preproc"
        if getpass.getuser() == "ehuby":
            file_patterns = "/home/ehuby/WORK/DATA/FIRST-PL/2025-05-10/preproc/"

    if pixel_map is None:
        folder = os.path.dirname(file_patterns[0])
        print("Using pixel map folder: ",folder)
        pixel_map = file_patterns + [os.path.join(folder,"../pixelmaps")]

    fileList = FileList(file_patterns, first_type='RAW', object_name=object, modID=[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20] if only_with_modulation else None)
    
    fileList.make_association(pixelMap=pixel_map)

    print(f"Found {len(fileList.filelist)} files to process in {file_patterns}")
    
    print ( "Overwrite existing already preprocessed files: ", overwrite)
    preprocess(fileList, overwrite=overwrite, plot_sum=True)


if __name__ == "__main__":
    main()

# %%
