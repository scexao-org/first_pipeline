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
import libraries.runPL_library_io as runlib
import libraries.runPL_library_plots as runlib_plots
import libraries.runPL_library_basic as runlib_basic
import libraries.runPL_library_linalg as runlib_linalg
from classes.runPL_class_dataCube import DataCube, extract_datalist
from classes.runPL_class_couplingMap import CouplingMap
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Perform astrometric analysis from FIRST Photonic Lantern data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Summary:
    This script processes preprocessed FITS files for astrometric measurements using coupling maps.
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


    files_with_dark, filelist_cmap = get_filelist(file_patterns, dark_patterns, cmap_patterns, wollaston)

    couplingMap = CouplingMap(filelist_cmap[0], pyramids = pyramids)
    Npos = couplingMap.Npositions
        

    #Input preproc
    #clean and sum all data

    datalist : list[DataCube]=extract_datalist(files_with_dark,Nsmooth=wavelength_smooth,Nbin=couplingMap.wavelength_bin,flat = couplingMap.flat)

    for i,d in enumerate(datalist):
    #     pass

    # if True:

        flux = d.flux
        datacube= d.data 
        datacube_var= d.variance 
        ra_dec = d.compute_xy_sky(couplingMap) 
        # xmod=datalist[0].xmod
        # ymod=datalist[0].ymod
        Ncube = ra_dec.shape[0]  # number of cubes
        Nmod = ra_dec.shape[1]  # number of modulation positions
        Npos = ra_dec.shape[2]  # Positions on sky
        Nwave = datacube.shape[3]  # number of wavelength channels
        Noutput = datacube.shape[2]  # number of outputs
        Nimages = Ncube * Nmod

        filename = d.filename
        print( f"---->  Filename : {filename}")

        flux_goodData,flux_threshold = runlib_basic.flux_filtering(flux)
        print(f"* Percentage of good data: {np.sum(flux_goodData)/Nimages*100:.1f} % (flux threshold)")
        data_svdfiltered,fit_goodData,errors = runlib_basic.svd_filtering(datacube,flux_goodData)
        goodData = flux_goodData & fit_goodData
        print(f"* Percentage of good data: {np.sum(goodData)/Nimages*100:.1f} % (flux and svd threshold)")

        #select data based on filtering data
        datacube_T=datacube[goodData].transpose((2,1,0))
        ra_dec=ra_dec[goodData]
        flux=flux[goodData]

        star_detected, star_index, star_radec, _ = couplingMap.chi2_filtering(datacube_T, ra_dec)
        print(f"* Percentage of good data: {np.sum(star_detected)/Nimages*100:.1f} % (flux, svd and chi2 threshold)")


        #select data based on wheter the star is detected on our data or not
        datacube_T = datacube_T[:,:,star_detected]
        ra_dec = ra_dec[star_detected]
        flux = flux[star_detected]
        spectra = flux.mean(axis=0)


        wmin = len(spectra) // 4
        wmax = 3 * len(spectra) // 4
        QT_broadband, R_broadband = couplingMap.compute_broadband_QR(wmin, wmax, spectra)

        QTdata = couplingMap.QT_dot_data(star_index, datacube_T)

        Nimages = QTdata.shape[2]
        Nqr = couplingMap.Nqr
        QTdata_star_removed = np.zeros_like(QTdata)
        R = couplingMap.R * spectra[None,:,None,None]
        R_dxy = np.zeros((Nwave, Nqr, Nimages, 2))


        for i in tqdm(range(Nimages), desc="Computing XY positions"):
            t = star_index[i]
            center = couplingMap.position[t]

            QTdata_broadband = QT_broadband[t] @ QTdata[wmin:wmax,:,i].ravel()
            
            if Nqr == 6:
                x_hat_broadband, y_hat_broadband, k_hat_broadband, chi2_broadband, _ = runlib_linalg.fit_QR_6(QTdata_broadband, R_broadband[t])
            else:
                x_hat_broadband, y_hat_broadband, k_hat_broadband, chi2_broadband, _ = runlib_linalg.solve_QR_3(QTdata_broadband, R_broadband[t])

            if Nqr == 6:
                v = np.array([1.0, x_hat_broadband, y_hat_broadband, x_hat_broadband*y_hat_broadband, x_hat_broadband**2, y_hat_broadband**2])
                dv_dx = np.array([0.0, 1.0, 0.0, y_hat_broadband, 2.0*x_hat_broadband, 0.0])
                dv_dy = np.array([0.0, 0.0, 1.0, x_hat_broadband, 0.0, 2.0*y_hat_broadband])
            else:
                v = np.array([1.0, x_hat_broadband, y_hat_broadband])
                dv_dx = np.array([0.0, 1.0, 0.0])
                dv_dy = np.array([0.0, 0.0, 1.0])

            r = R[t] @ v
            Kernel_v = np.identity(len(v)) - (r[:,:,None] @ r[:,None]) / (r[:,None] @ r[:,:,None])
            QTdata_star_removed[:,:,i] = (Kernel_v @ QTdata[:,:,i,None])[...,0]

            dev_phi = np.array((dv_dx,dv_dy)).T
            R_dxy[:,:,i] = Kernel_v @ (R[t] @ dev_phi)


        data_star_removed = couplingMap.Q_dot_QTdata(star_index, QTdata_star_removed)

        xy_dev = np.linalg.pinv(R_dxy.reshape((Nwave,-1,2))) @ QTdata_star_removed.reshape((Nwave,-1,1))
        xy_dev = xy_dev[...,0]

        header = d.header
        header['X_FIRTYP'] = 'ASTROMETRY'

        list_of_hdus = []
        # Create a primary HDU with the data
        hdu_primary = fits.PrimaryHDU(xy_dev)
        hdu_residual = fits.ImageHDU(xy_dev, name="RESIDUAL")
        list_of_hdus += [hdu_primary, hdu_residual]


        # Add date and time to the header
        current_time = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
        header['DATE-PRO'] = current_time

        # Add input parameters to the header
        header['WLSMOOTH'] = wavelength_smooth  # Add wavelength smoothing factor

        # Définir le chemin complet du sous-dossier "astrometry"
        output_dir = os.path.join(d.dirname,"../astrometry")

        # Créer les dossiers "output" et "pixel" s'ils n'existent pas déjà
        os.makedirs(output_dir, exist_ok=True)

        hdu_primary.header.extend(header, strip=True)

        # Combine all HDUs into an HDUList
        hdul = fits.HDUList(list_of_hdus)

        output_filename = os.path.join(output_dir, runlib.create_output_filename(header))

        # Write to a FITS file
        hdul.writeto(output_filename, overwrite=True)
        print(f"AAstrometry saved to {output_filename}")

# %%
