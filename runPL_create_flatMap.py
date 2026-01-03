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
from matplotlib.pyplot import plot,hist,clf,figure,legend,imshow
from datetime import datetime
from tqdm import tqdm

import libraries.runPL_library_io as runlib_io
import libraries.runPL_library_plots as runlib_plots
from classes.runPL_class_flatMap import FlatMap
from classes.runPL_class_fileList import FileList
from classes.runPL_class_dataCube import DataCube 

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
    --dark_files  Select one or more specific dark files to use
    --wollaston   Wollaston status. Use IN for internal or OUT for no wollaston (default: first in the list of files)

Example:
    runPL_create_flatMap.py *.fits --dark_files="dark*.fits" --wollaston=IN
"""

neon_wavelengths = np.array([
   576.44188, 585.24878, 588.1895, 594.4834, 597.55343,
    602.99968, 607.43376, 609.6163, 614.30627, 616.35937, 621.72812,
    626.65085, 633.44304, 638.29917, 640.2231, 650.65279, 653.28825,
    659.89543, 667.82752, 671.70456, 692.94672, 703.24128, 717.3938,
    724.51665, 743.88981, 748.88712, 753.57739, 772.46233
])

neon_intensity = np.array([
    0.00734, 1.0, 0.10191, 0.10492, 0.03624,
    0.0376, 0.15792, 0.20348, 0.34517, 0.11925, 0.05824,
    0.02077, 0.08928, 0.27498, 0.0414, 0.15098, 0.02809,
    0.03268, 0.33628, 0.02421, 0.19251, 0.5563, 0.02255,
    0.19111, 0.04466, 0.03884, 0.04444, 0.03404
])

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


def compute_flat(datalist, intercept_at_zero = False):

    flats=np.concatenate([np.concatenate(d.data) for d in datalist])
    variance=np.concatenate([np.concatenate(d.variance) for d in datalist])
    valid_mask = ~np.isnan(flats[:,0,0])
    flats = flats[valid_mask]
    variance = variance[valid_mask]
    
    Nflat_smooth = 100
    flats_smooth = np.zeros_like(flats)
    window = np.hanning(Nflat_smooth)
    window /= window.sum()
    conv_ref = np.convolve(np.ones(len(flats[0,0])), window, mode='same')
    
    for f in range(flats_smooth.shape[0]):
        for o in range(flats_smooth.shape[1]):
            flats_smooth[f, o, :] =  np.convolve(flats[f, o, :], window, mode='same') / conv_ref

    # Fit first order polynomial for each pixel position
    poly_coeffs = np.zeros((flats.shape[1], flats.shape[2], 2))  # Store coefficients for each pixel
    fit_quality = np.zeros((flats.shape[1], flats.shape[2], 3))  # Store quality metrics: [chi2_reduced, r_squared, weighted_rmse]
    
    for o in tqdm(range(flats.shape[1]),desc="Linear fit of gain for each pixels"):  # For each output
        for p in range(flats.shape[2]):  # For each pixel
            # Extract the data for this pixel across all flat frames
            y_data = flats[:, o, p]
            x_data = flats_smooth[:, o, p]
            w_data = 1/variance[:, o, p]
            
            # Fit linear function: y = ax + b using weighted least squares
            # Calculate using normal equations for weighted linear regression
            sum_w = np.sum(w_data)
            sum_wx = np.sum(w_data * x_data)
            sum_wy = np.sum(w_data * y_data)
            sum_wxx = np.sum(w_data * x_data * x_data)
            sum_wxy = np.sum(w_data * x_data * y_data)

            # Solve normal equations: [sum_wxx sum_wx; sum_wx sum_w] * [a; b] = [sum_wxy; sum_wy]
            det = sum_wxx * sum_w - sum_wx * sum_wx
            
            if abs(det) > 1e-12:  # Check for numerical stability
                a = (sum_w * sum_wxy - sum_wx * sum_wy) / det
                b = (sum_wxx * sum_wy - sum_wx * sum_wxy) / det
            else:
                # Fallback to weighted least squares through origin if matrix is singular
                a = sum_wxy / sum_wxx if sum_wxx != 0 else 0
                b = 0

            if intercept_at_zero:
                a = sum_wxy / sum_wxx if sum_wxx != 0 else 0
                b = 0

            poly_coeffs[o, p, 0] = a  # slope
            poly_coeffs[o, p, 1] = b  # intercept
            
            # Calculate fit quality metrics
            y_fit = a * x_data + b
            residuals = y_data - y_fit
            
            # 1. Reduced Chi-squared (should be ~1 for good fit)
            chi2 = np.sum(w_data * residuals**2)
            dof = len(y_data) - 2  # degrees of freedom (n_points - n_parameters)
            chi2_reduced = chi2 / dof if dof > 0 else np.inf
            
            # 2. Weighted R-squared coefficient of determination
            y_mean = np.sum(w_data * y_data) / sum_w
            ss_tot = np.sum(w_data * (y_data - y_mean)**2)
            ss_res = np.sum(w_data * residuals**2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            # 3. Weighted RMSE (normalized by typical signal level)
            weighted_rmse = np.sqrt(ss_res / sum_w) / np.abs(y_mean) if y_mean != 0 else np.inf
            
            fit_quality[o, p, 0] = chi2_reduced
            fit_quality[o, p, 1] = r_squared
            fit_quality[o, p, 2] = weighted_rmse

    return poly_coeffs, fit_quality


if __name__ == "__main__":
    '''
    run for neon only, call functions for star

    to change the parameters to skip wavelenght or consider more peaks, change value in function
    findPeaks directly, in the instance of "run_trials_for_all_combination_of_waves"
    '''
    parser = argparse.ArgumentParser(
        description="Create a wavelength map from the provided FITS files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
    Summary:
    - Searches for FITS files with X_FIRTYP=PREPROC and DATA-TYP=WAVE keywords.
    - Finds corresponding dark files (X_FIRTYP=PREPROC, DATA-TYP=DARK).
    - Reads wave files, subtracts the median of the dark files.
    - Detects emission peaks and fits a polynomial to create a wavelength map.
    - N (number of peaks) is determined by the number of wavelengths in --wave_list.
    - Saves the wavelength map as a FITS file in the output directory.
    - Generates and saves figures for visualization.
    - Output files are stored in an "output/wave" directory.

    Example:
        %(prog)s --wave_list="[748.9, 743.9, 724.5, 717.4, 703.2, 693, 671.7, 667.8, 659.9, 653.3, 650.7, 640.2, 638.2, 633.4, 630.5, 626.7, 621.7, 616.4]"
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
    
    # Parse the arguments
    args = parser.parse_args()
    file_patterns = args.files if args.files else ['*.fits','./preproc/*.fits']

    # Extract the parsed arguments
    dark_patterns = args.dark_files
    wollaston = args.wollaston

    if ("VSCODE_PID" in os.environ or os.environ.get('TERM_PROGRAM') == 'vscode' or os.environ.get('SPYDER_DEBUG_FILE')):
        print("Running in compiler")
        if getpass.getuser() == "slacour":
            file_patterns = "/Users/slacour/DATA/LANTERNE/20251125/preproc"
            file_patterns = "/Users/slacour/DATA/LANTERNE/20251231/preproc/firstpl_2025-12-31T00?3*fits"
            dark_patterns = "/Users/slacour/DATA/LANTERNE/20251231/preproc/firstpl_*fits"
        if getpass.getuser() == "jsarrazin":
            file_patterns = "/home/jsarrazin/Bureau/PLDATA/moreTest/2024-11-21_13-48-32_science_copie/preproc"
            file_patterns = "/home/jsarrazin/Bureau/PLDATA/novembre/les_preproc"
        if getpass.getuser() == "ehuby":
            file_patterns = "/home/ehuby/WORK/DATA/FIRST-PL/2025-05-10/preproc/"
            file_patterns = "/home/ehuby/WORK/DATA/FIRST-PL/2025-05-10/preproc/"
        

    # If the user specify a dark, use it. Otherwise, use the science file pattern
    if dark_patterns is None:
        dark_patterns = file_patterns

    fileList = FileList(file_patterns, data_type='FLAT', first_type='PREPROC', wollaston=wollaston)
    fileList.make_association(darks_pattern=dark_patterns)

    datalist : List[DataCube] = fileList.extract_data_from_list(center = False)

    poly_coeffs, fit_quality = compute_flat(datalist, intercept_at_zero = True)

    # making pictures
    fig_flat=runlib_plots.plot_flat_fit_quality(poly_coeffs, fit_quality)

    ############### Save results ####################
    # Save arrays into a FITS file

    # Create a primary HDU with no data, just the header
    hdu_primary = fits.PrimaryHDU()

    # Create HDUs for each array
    hdu = [fits.ImageHDU(data=1/poly_coeffs[:,:,0], name='FLAT')]

    header = datalist[-1].header
    # Définir le chemin complet du sous-dossier "output/couplingmaps"
    folder = fileList.get_most_common_dir()
    output_dir = os.path.join(folder,"../flatmaps")

    header['X_FIRTYP'] = 'FLATMAP'

    # Add date and time to the header
    current_time = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
    header['DATE-PRO'] = current_time

    filenames = [d.filename for d in datalist]
    for i, filename in enumerate(filenames):
        header['Q_FM_F%i' % i] = (filename, 'filename of the extracted flux')

    header['Q_FMNAME'] = (runlib_io.create_output_filename(header), 'name of the flatwave map file')

    # Créer les dossiers "output" et "pixel" s'ils n'existent pas déjà
    os.makedirs(output_dir, exist_ok=True)

    hdu_primary.header.extend(header, strip=True)

    # Combine all HDUs into an HDUList
    hdul = fits.HDUList([hdu_primary, *hdu])

    output_filename = os.path.join(output_dir, header['Q_FMNAME'])

    # Write to a FITS file
    print(f"Saving data to {output_filename}")
    hdul.writeto(output_filename, overwrite=True)


    # checking all the flats for individual files

    flatMap =  FlatMap(output_filename)

    datalist_1 : List[DataCube] = fileList.extract_data_from_list(flatMap = None, center = False)
    datalist_2 : List[DataCube] = fileList.extract_data_from_list(flatMap = flatMap, center = False)

    data_noflat=np.array([np.nanmean(d.data,axis=(0,1)) for d in datalist_1])
    data_withflat=np.array([np.nanmean(d.data,axis=(0,1)) for d in datalist_2])

    for i in range(len(data_noflat)):
        d=datalist_1[0]
        runlib_plots.plot_flux_map(np.nanmean(d.data,axis=(0,2,3)), d.xmod[0], d.ymod[0], desc = f'Flux map for file {d.basename}')
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



# %%
