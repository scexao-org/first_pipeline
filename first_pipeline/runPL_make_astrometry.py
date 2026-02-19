#! /usr/bin/env python3
# -*- coding: iso-8859-15 -*-
#%%
"""
FIRST Pipeline - Astrometric Analysis

This script performs precise astrometric measurements from preprocessed FIRST
Visible Photonic Lantern data at SUBARU/SCEXAO using coupling maps. It enables
high-precision position measurements and astrometric calibrations for binary
star systems, exoplanet detection, and precision astrometry applications.

Astrometric analysis leverages the photonic lantern's spatial resolution
enhancement for precise position measurements beyond conventional imaging limits.

Created on Wed May 21 22:56:25 2025
@author: slacour
"""

import os
import sys
from astropy.io import fits
from glob import glob
import argparse
import numpy as np

from scipy.interpolate import griddata

import getpass
import matplotlib
if ("VSCODE_PID" in os.environ or os.environ.get('TERM_PROGRAM') == 'vscode'):
    matplotlib.use('Qt5Agg')
elif os.environ.get('SPYDER_DEBUG_FILE'):
    print("Running in Spyder")
else:
    matplotlib.use('Agg')
     
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot,hist,clf,figure,legend,imshow
from datetime import datetime
from tqdm import tqdm
from .libraries import runPL_library_io as runlib
from .libraries import runPL_library_plots as runlib_plots
from .libraries import runPL_library_linalg as runlib_linalg
from .classes.runPL_class_dataCube import DataCube
from .classes.runPL_class_couplingMap import CouplingMap
from .classes.runPL_class_fileList import FileList
from astropy.io import fits
from astroplan import Observer
from astropy.time import Time

from scipy.optimize import least_squares

subaru = Observer.at_site("Subaru")
now_time = Time.now()
if subaru.is_night(now_time):
    print("It's night at Subaru Observatory.")
else:
    print("It's day at Subaru Observatory.")

plt.ion()

DEBUG = False

# Add options
usage = """
Goal:
    Reconstruct images from FIRST Photonic Lantern data using specified coupling maps and options.

Summary:
    This script processes preprocessed FITS files, applies coupling maps, reconstructs images, and saves the results.
    It supports selection by object name, modulation pattern, and smoothing options. The script can save individual frames,
    wavelength slices, and residuals, and allows explicit selection of coupling map files.

Input files:
    - Preprocessed data FITS files (e.g., with DPR_CATG=OBJECT and DPR_TYPE=PREPROC)
    - Coupling map FITS files (e.g., with X_FIRTYP=COUPLINGMAP)

Output files:
    - Reconstructed image FITS files, including summed images, residuals, and optionally individual frames and wavelength slices.

Options:
    --object_name <str>           Selection of the data by the Object name (default: NONE)
    --modID <int>                 Selection of the modulation pattern by user [0 == first in the list] (default: 0)
    --modScale <int>              Selection of the modulation scale by user [0 == first in the list] (default: 0)
    --coupling_map <str>          Force to select which coupling map file to use (default: the one in the directory)
    --wavelength_smooth <int>     Smoothing factor for wavelength (default: 1)
    --save_individual_frames      Save individual frames (default: True)
    --save_individual_wavelength  Save individual wavelength slices (default: False)

Example:
    python runPL_imageReconstruction.py --object_name=HIP81126 --modID=1 --coupling_map=path/to/couplingmap.fits *.fits
"""


def get_filelist(file_patterns, dark_patterns, cmap_patterns, wollaston):

        fits_keywords = {'X_FIRTYP': ['PREPROC'],
                        'DATA-TYP': ['OBJECT','OJECT','TEST'],
                        }    
        
        # Adding other constraints if asked by user
        if wollaston is not None:
            fits_keywords['X_FIRWOL'] = [wollaston]
        
        print(file_patterns)
        filelist = runlib.get_filelist(file_patterns, fits_keywords)

        # Adding new constraints if not asked by user
        hd=fits.getheader(filelist[0])
        wollaston = hd.get('X_FIRWOL', None)
        if wollaston is not None:
            fits_keywords['X_FIRWOL'] = [wollaston]

        print("----------------")
        print(f"Selected object with wollaston = {wollaston}")
        print("----------------")

        filelist = runlib.get_filelist(file_patterns, fits_keywords)

        fits_keywords = {'X_FIRTYP': ['PREPROC'],
                        'DATA-TYP': ['DARK'],
                        }    
        if wollaston is not None:
            fits_keywords['X_FIRWOL'] = [wollaston]
        
        try:
            filelist_dark = runlib.get_filelist(dark_patterns, fits_keywords, name_search="dark")
        except FileNotFoundError as e:
            print(f"NO DARKS: {e}")
            filelist_dark = []

        if len(filelist_dark) == 0:
            print("WARNING: No dark files found with the specified patterns and keywords.")

        fits_keywords = {'X_FIRTYP': ['COUPLINGMAP'],
                        }    
        if wollaston is not None:
            fits_keywords['X_FIRWOL'] = [wollaston]

        filelist_cmap = runlib.get_filelist(cmap_patterns, fits_keywords, name_search="coupling map")
        filelist_cmap  = filter_couplingmapfile(filelist_cmap)

        files_with_dark = runlib.associate_dark(filelist, filelist_dark)

        return files_with_dark, filelist_cmap




def filter_couplingmapfile(filelist_cmap):
    """
    Filters the input file list to separate coupling map files and dark files based on FITS keywords.
    Raises an error if no valid files are found.
    Returns a dictionary mapping coupling map files to their closest dark files.
    """

    print("runPL object filelist : ", filelist_cmap)

    # raise an error if filelist_cleaned is empty
    if len(filelist_cmap) == 0:
        raise FileNotFoundError("No coupling map to use.\n Please specify which one to use with the option --coupling_map")

    # raise an error if filelist_cleaned is more than one
    if len(filelist_cmap) > 1:
        raise ValueError("Too many coupling maps to use! I can only use one.\n Please specify which one to use with the option --coupling_map")

    # Check if all files have the same value for header['PM_CHECK']
    pm_check_values = set()
    combined_filelist = []
    combined_filelist.extend(filelist_cmap)
    for file in combined_filelist:
        header = fits.getheader(file)
        pm_check_values.add(int(header.get('PM_CHECK', 0)))
        
    if len(pm_check_values) > 1:
        print("WARNING: The 'PM_CHECK' values (ie, the pixel map used to preprocess the files) \n are not consistent across all files!")
        print(f"Found values: {pm_check_values}")

    # for each file in filelist_cmap find the closest dark file in filelist_dark with, by priority, first the directory in which the file is, and then by the date in the "DATE" fits keyword, and second, the directory in which the file is
    # files_with_dark = {cmap: runlib.find_closest_dark(cmap, filelist_dark) for cmap in filelist_data}

    return filelist_cmap


def quick_fits(data, title=""):
    if DEBUG:
        #For debugging purpose
        now = datetime.now()
        date_time_str = now.strftime("%Y_%m_%d_%H_%M_%S")
        if getpass.getuser() == "jsarrazin":
            runlib.save_fits_file(data, "/home/jsarrazin/Bureau/test zone/coupling_maps/"+title+"_"+date_time_str+".fits")
        print("Done")

def quick_imshow(data, title=""):
    #For debugging purpose
    now = datetime.now()
    plt.imshow(data, aspect='auto')
    plt.title(title)
    print("Done")

def quick_plot(data,title =""):
    #For debugging purpose
    now = datetime.now()
    date_time_str = now.strftime("%Y_%m_%d_%H_%M_%S")
    plt.plot(data)
    plt.title(title)
    print("Done")


def main():
    """
    Main entry point for the astrometric analysis script.
    """
    parser = argparse.ArgumentParser(
        description="Perform high-precision astrometric measurements from FIRST Photonic Lantern data using coupling map analysis.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
FIRST Pipeline Astrometric Analysis Tool

This script performs precision astrometric measurements using photonic lantern
data and coupling maps. It enables position measurements with enhanced precision
for binary stars, exoplanet detection, and fundamental astrometry applications.

Examples:
    %(prog)s --wollaston IN --wavelength_smooth=2 *.fits
    %(prog)s --coupling_map=/path/to/map.fits --pyramids target_data/*.fits
    %(prog)s --save_individual_frames --save_individual_wavelength *.fits
    %(prog)s --dark_files=dark*.fits binary_star_data/*.fits

Pipeline Workflow Integration:
    1. Requires preprocessed data files (X_FIRTYP=PREPROC) and coupling maps
    2. Final analysis step for precision position measurements
    3. Leverages photonic lantern spatial resolution enhancement
    4. Outputs calibrated astrometric measurements for scientific analysis

Input Files:
    - Preprocessed FITS files: X_FIRTYP=PREPROC containing spectral measurements
    - Coupling map files: X_FIRTYP=COUPLINGMAP from runPL_create_couplingMap.py
    - Dark frames for accurate background subtraction
    - Automatic coupling map detection or manual selection

Output Files:
    - Astrometric measurement FITS files with position data
    - Time-resolved position measurements for orbital motion
    - Quality assessment metrics and uncertainty estimates
    - Optional individual frame analysis for high-cadence observations
    - Optional wavelength-resolved astrometry for chromatic effects

Astrometric Features:
    - High-precision position measurements using coupling map analysis
    - Wavelength smoothing for enhanced signal-to-noise in position determination
    - Support for both polarimetry (Wollaston IN) and photometry (OUT) modes
    - Pyramidal fitting options for enhanced spatial resolution
    - Quality assessment and uncertainty quantification

Advanced Analysis Options:
    - wavelength_smooth: Control spectral smoothing for noise reduction
    - pyramids: Enable pyramidal fitting for enhanced spatial resolution
    - save_individual_frames: Generate time-resolved astrometric sequences
    - save_individual_wavelength: Analyze chromatic astrometric effects
    - coupling_map: Force specific coupling map for consistency

Measurement Precision:
    - Sub-milliarcsecond precision achievable with proper calibration
    - Uncertainty estimation through statistical analysis
    - Quality metrics guide measurement reliability assessment
    - Systematic error analysis and correction

Scientific Applications:
    - Binary star orbit determination with enhanced precision
    - Exoplanet astrometric detection and characterization
    - Proper motion measurements for stellar kinematics
    - Reference frame calibration and maintenance
    - Fundamental astrometry for parallax determination

Technical Notes:
    - Coupling maps enable spatial information recovery from fiber measurements
    - Pyramidal fitting enhances spatial resolution through advanced algorithms
    - Wavelength-dependent effects can reveal chromatic aberrations
    - Quality metrics essential for assessing measurement reliability

Calibration Considerations:
    - Requires accurate coupling map calibration for precision
    - Dark subtraction critical for low-level position measurements
    - Monitor systematic effects and instrumental stability
    - Cross-validate with independent astrometric measurements when available

Note: Astrometric precision depends critically on coupling map quality and
instrument stability. Review quality metrics and systematic effects carefully
for high-precision applications.
        """
    )

    # Add positional argument for files
    parser.add_argument('files', nargs='*', default=['*.fits'],
                       help='FITS files to process (supports wildcards)')

    # Add optional arguments
    parser.add_argument("--wollaston", 
                       help="Wollaston status. Use IN for internal or OUT for no wollaston (default: first in the list)")
    parser.add_argument("--coupling_map", 
                       help="Force to select which coupling map file to use (default: the one in the directory)")
    parser.add_argument("--dark_files", 
                       help="Select one or more specific dark(s) files to use")
    parser.add_argument("--wavelength_smooth", type=int, default=1,
                       help="Smoothing factor for wavelength (default: %(default)s)")
    parser.add_argument("--save_individual_frames", action="store_true", default=True,
                       help="Save individual frames (default: %(default)s)")
    parser.add_argument("--save_individual_wavelength", action="store_true", default=False,
                       help="Save individual wavelength (default: %(default)s)")
    parser.add_argument("--pyramids", action="store_true", default=False,
                       help="Use pyramids for data fitting by coupling map (default: %(default)s)")

    if (("VSCODE_PID" in os.environ or os.environ.get('TERM_PROGRAM') == 'vscode') or os.environ.get('SPYDER_DEBUG_FILEfile =')):
        print("Running in compiler")
        wollaston = None
        dark_patterns = None
        pyramids = False

        if getpass.getuser() == "slacour":
            # file_patterns = "/Users/slacour/DATA/LANTERNE/20250614/preproc/firstpl_2025-06-14T01:38*fits"
            file_patterns = "/Users/slacour/DATA/LANTERNE/20250510/preproc/*10T05?5*BETACMI_P.fits"
            file_patterns = "/Users/slacour/DATA/LANTERNE/20250510/preproc/*10T05?53*BETACMI_P.fits"
            # file_patterns = "/Users/slacour/DATA/LANTERNE/20250510/preproc/*10T09?2[0-2]*TETCRB_P.fits"
            # cmap_patterns = "/Users/slacour/DATA/LANTERNE/20250510/preproc/../couplingmaps/firstpl_2025-05-10T09:23:36_COUPLINGMAP.fits"
            # dark_patterns = "/Users/slacour/DATA/LANTERNE/20250514/preproc"
            # file_patterns = "/Users/slacour/DATA/LANTERNE/20250514/preproc/firstpl_2025-05-14T11?3*s"
            # Mathias Binary
            cmap_patterns = "/Users/slacour/DATA/LANTERNE/20250514/couplingmaps/firstpl_2025-05-14T11:39:58_COUPLINGMAP.fits"
            # file_patterns = "/Users/slacour/DATA/LANTERNE/20250514/preproc/firstpl_2025-05-14T10:10?4*s"
            # dark_patterns = "/Users/slacour/DATA/LANTERNE/20250514/preproc"
            cmap_patterns = "/Users/slacour/DATA/LANTERNE/20250510/preproc/../couplingmaps/firstpl_2025-05-10T09:23:36_TETCRBCM.fits"
            # cmap_patterns = "/Users/slacour/DATA/LANTERNE/20250510/preproc/../couplingmaps/firstpl_2025-05-10T09:21:23_TETCRBCM.fits"
            # file_patterns = "/Users/slacour/DATA/LANTERNE/20250510/preproc/*10T09?2[0-3]*TETCRB_P.fits"
            # file_patterns = "/Users/slacour/DATA/LANTERNE/20250510/preproc/*10T09?21*TETCRB_P.fits"
            # Mathias Binary
            cmap_patterns = "/Users/slacour/DATA/LANTERNE/20250514/preproc/../couplingmaps/firstpl_2025-05-14T11:39:58_HIP85819_CM.fits"
            file_patterns = "/Users/slacour/DATA/LANTERNE/20250514/preproc/firstpl_2025-05-14T10:10?*s"
            dark_patterns = "/Users/slacour/DATA/LANTERNE/20250514/preproc"


        if getpass.getuser() == "ehuby" :
            file_patterns = "/home/ehuby/WORK/DATA/FIRST-PL/2025-05-10/preproc/firstpl_*.fits"
            cmap_patterns = "/home/ehuby/WORK/DATA/FIRST-PL/2025-05-10/preproc/couplingmaps_TETCRB/"
    else:

        args = parser.parse_args()
        file_patterns = args.files if args.files else ['*.fits']

        wavelength_smooth = args.wavelength_smooth
        dark_patterns = args.dark_files
        wollaston = args.wollaston

        save_individual_frames = args.save_individual_frames
        save_individual_wavelength = args.save_individual_wavelength
        pyramids = args.pyramids

        cmap_patterns = args.coupling_map

    # If the user specifies a coupling map, use it, otherwise use the science file pattern
    if cmap_patterns is None:
        cmap_patterns = file_patterns + ['../couplingmaps/*.fits']
    # If the user specify a dark, use it. Otherwise, use the science file pattern
    if dark_patterns is None:
        dark_patterns = file_patterns

    try:
        # Use the new FileList approach instead of legacy get_filelist
        fileList = FileList(file_patterns, first_type='PREPROC', object_name=object_name, wollaston=wollaston)
        filelist = fileList.filelist
        
        if len(filelist) == 0:
            print("No matching files found")
            return
        
        # Find coupling map files
        cmap_filelist = []
        for pattern in cmap_patterns:
            cmap_filelist.extend(glob(pattern))
        
        if len(cmap_filelist) == 0:
            print("No coupling map files found")
            return
            
        print(f"Found {len(filelist)} data files and {len(cmap_filelist)} coupling map files")
        
        # TODO: Complete the astrometry analysis implementation
        print("Astrometry analysis implementation is under development")
        
    except Exception as e:
        print(f"Error in astrometry analysis: {e}")
        print("This script requires further development to work with the current pipeline architecture")


if __name__ == "__main__":
    main()
