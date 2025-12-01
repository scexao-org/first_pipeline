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
from classes.runPL_class_flatMap import FlatWaveMap
from classes.runPL_class_dataCube import DataCube, extract_datalist 


#plt.ion()

# Add options
usage = """
Usage: %prog [options]

Goal: Create a wavelength map from the provided FITS files.

Summary:
- Searches for FITS files with X_FIRTYP=PREPROC and DATA-TYP=WAVE keywords.
- Finds corresponding dark files (X_FIRTYP=PREPROC, DATA-TYP=DARK).
- Reads wave files, subtracts the median of the dark files.
- Detects emission peaks and fits a polynomial to create a wavelength map.
- N (number of peaks) is determined by the number of wavelengths in --wave_list.
- Saves the wavelength map as a FITS file in the output directory.
- Generates and saves figures for visualization.
- Output files are stored in an "output/wave" directory.

Options:
    --wave_list   Comma-separated list of emission lines (default: [748.9, 743.9, 724.5, 717.4, 703.2, 693, 671.7, 667.8, 659.9, 653.3, 650.7, 640.2, 638.2, 633.4, 630.5, 626.7, 621.7, 616.4])
    --filelist    Folder containing the preprocessed FITS files (default: .)

Example:
    runPL_create_waveMap.py --wave_list="[748.9, 743.9, 724.5, 717.4, 703.2, 693, 671.7, 667.8, 659.9, 653.3, 650.7, 640.2, 638.2, 633.4, 630.5, 626.7, 621.7, 616.4]"
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

def get_filelist_wave(wave_patterns, dark_patterns, flat_patterns, wollaston):

        fits_keywords = {'X_FIRTYP': ['PREPROC'],
                        'DATA-TYP': ['FLAT'],
                        }    
        
        # Adding other constraints if asked by user
        if wollaston is not None:
            fits_keywords['X_FIRWOL'] = [wollaston]
        
        print(wave_patterns)
        filelist = runlib_io.get_filelist(wave_patterns, fits_keywords)

        # Adding new constraints if not asked by user
        hd=fits.getheader(filelist[0])
        wollaston = hd.get('X_FIRWOL', None)
        if wollaston is not None:
            fits_keywords['X_FIRWOL'] = [wollaston]

        print("----------------")
        print(f"Selected wollaston={wollaston}")

        filelist = runlib_io.get_filelist(wave_patterns, fits_keywords)

        print(f"Found {len(filelist)} files matching criteria.")
        print("----------------")

        # finding darks files
        fits_keywords['DATA-TYP'] = ['DARK']

        try:
            filelist_dark = runlib_io.get_filelist(dark_patterns, fits_keywords,  name_search="dark")
        except FileNotFoundError as e:
            print(f"WARNING!!! {e}")
            filelist_dark = []

        # finding flats files
        fits_keywords['DATA-TYP'] = ['COMPARAISON']

        try:
            filelist_neon = runlib_io.get_filelist(flat_patterns, fits_keywords,  name_search="flat")
        except FileNotFoundError as e:
            print(f"WARNING!!! {e}")
            filelist_neon =[]

        files_with_dark = runlib_io.associate_dark(filelist, filelist_dark)
        if len(filelist_neon)>0:
            neons_with_dark = runlib_io.associate_dark(filelist_neon, filelist_dark)
        else:
            neons_with_dark = []

        return files_with_dark, neons_with_dark


def compute_flat(datalist, intercept_at_zero = False):

    flats=np.concatenate([np.concatenate(d.data) for d in datalist])
    variance=np.concatenate([np.concatenate(d.variance) for d in datalist])
    valid_mask = ~np.isnan(flats[:,0,0])
    flats = flats[valid_mask]
    variance = variance[valid_mask]
    
    Nflat_smooth = 25
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


def calculate_pixel_peaks_and_aberations(neon):

    def find_N_peaks(spectrum,N=1000):

        min_peak_separation = 6
        prominence_threshold = 0.01 * (np.max(spectrum) - np.median(spectrum))
        
        peaks, properties = find_peaks(
            spectrum,
            prominence=prominence_threshold,  # tune this
            distance=min_peak_separation      # in pixels
        )
        peak_prominence = properties["prominences"]
        idx_peak = np.argsort(peak_prominence)[-N:]
        return peaks[np.sort(idx_peak)]


    def subpixel_parabolic(spectrum, peaks):
        """
        spectrum: 1D array of intensities
        peaks: integer pixel indices from scipy.signal.find_peaks
        returns: array of subpixel peak positions
        """
        subpixels = []

        for i in peaks:
            # avoid edges
            if i <= 0 or i >= len(spectrum) - 1:
                subpixels.append(float(i))
                continue

            y1 = spectrum[i-1]
            y2 = spectrum[i]
            y3 = spectrum[i+1]

            denom = (y1 - 2*y2 + y3)
            if denom == 0:
                subpixels.append(float(i))
                continue

            delta = 0.5 * (y1 - y3) / denom
            subpixels.append(i + delta)

        return np.array(subpixels)
    
    spectrum_0=neon.sum(axis=0)
    peaks_0=find_N_peaks(spectrum_0,10)

    peaks_all=[]
    for spectrum in neon:
        peaks=find_N_peaks(spectrum)
        idx_peak=[]
        for p0 in peaks_0:
            idx_peak+=[np.argmin(np.abs(peaks-p0))]
        peaks_all+=[peaks[idx_peak]]


    peaks_all = np.array(peaks_all)
    roll_index = np.median((peaks_all-peaks_0),axis=1)
    roll_index = roll_index.astype(int)


    # Roll each spectrum in the neon array by its corresponding roll_index
    neon_rolled = np.array([np.roll(spectrum, -roll) for spectrum, roll in zip(neon, roll_index)])
    spectrum_0=neon_rolled.sum(axis=0)
    peaks_0=find_N_peaks(spectrum_0,20)
    peaks_all=[]
    peaks_all_sub=[]
    for spectrum in neon_rolled:
        peaks=find_N_peaks(spectrum)
        idx_peak=[]
        for p0 in peaks_0:
            idx_peak+=[np.argmin(np.abs(peaks-p0))]
        peaks_all+=[peaks[idx_peak]]

        peaks_sub = subpixel_parabolic(spectrum, peaks[idx_peak])
        peaks_all_sub+=[peaks_sub]

    peaks_all = np.array(peaks_all)
    peaks_all_sub = np.array(peaks_all_sub)+roll_index[:,None]

    peaks_diff=np.diff(peaks_all_sub,axis=0)
    peaks_diff_offset=peaks_diff-np.median(peaks_diff,axis=1)[:,None]
    N_lines_good = np.max(np.abs(peaks_diff_offset),axis=0) < 1.5

    # N_lines_good &= False

    peaks_all_sub_good = peaks_all_sub[:,N_lines_good]


    line_pixel_ref = np.mean(peaks_all_sub_good, axis=0)
    aberations = peaks_all_sub_good - line_pixel_ref[None, :]

    # Fit 2D aberrations: first order in x (pixel position), second order in y (PL output)
    # Create coordinate arrays
    x_coords = line_pixel_ref  # pixel positions (first axis)
    y_coords = np.arange(len(aberations))  # PL output indices (second axis)
    X, Y = np.meshgrid(x_coords, y_coords)

    # Flatten arrays for fitting
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    aberations_flat = aberations.flatten()

    # Create design matrix: [1, x, y, y^2] for each point
    A = np.column_stack([
        np.ones(len(X_flat)),  # constant term
        X_flat,                # first order in x (pixel position)
        Y_flat,                # first order in y (PL output)
        X_flat*Y_flat,         # cross term x*y
        Y_flat**2              # second order in y (PL output)
    ])

    # Solve least squares: A * coeffs = aberations_flat
    coeffs = np.linalg.lstsq(A, aberations_flat, rcond=None)[0]

    # Generate fitted aberrations
    aberations_fit = (coeffs[0] + 
                    coeffs[1] * X + 
                    coeffs[2] * Y + 
                    coeffs[3] * X * Y +
                    coeffs[4] * Y**2)
    
    fig = runlib_plots.plot_wavefit_coeffs(peaks_all_sub, peaks_all_sub_good, aberations, aberations_fit)

    ref_pixels_lines = (peaks_all_sub_good - aberations_fit).mean(axis=0)


    X, Y = np.meshgrid(np.arange(neon.shape[1]), np.arange(neon.shape[0]))
    aberated_image = (coeffs[0] + 
                    coeffs[1] * X + 
                    coeffs[2] * Y + 
                    coeffs[3] * X * Y +
                    coeffs[4] * Y**2)

    return ref_pixels_lines, aberated_image, coeffs, fig


def calculate_the_pixel_to_wavelength_mapping(ref_pixels_lines, neon_wavelengths, Nexclude):

    def fit_and_score(peak_ref, line_ref, ref_pixels_lines, neon_wavelengths,Nexclude):
        # Fit linear

        peak_pixels = ref_pixels_lines[peak_ref]
        ref_lambda = neon_wavelengths[line_ref]

        a = (ref_lambda[1] - ref_lambda[0]) / (peak_pixels[1] - peak_pixels[0])
        b = ref_lambda[0] - a * peak_pixels[0]

        # Predict wavelength for *all* peaks
        lambda_pred = a*ref_pixels_lines + b

        # For each predicted λ, find nearest reference λ
        idx = []
        for lp in lambda_pred:
            idx += [np.argmin(np.abs(neon_wavelengths - lp))]

        # Find duplicate indices
        unique_idx, counts = np.unique(idx, return_counts=True)
        duplicate_mask = np.isin(idx, unique_idx[counts > 1])
        duplicate_mask[0] = True  # Always ignore the first point
        duplicate_mask[-1] = True  # Always ignore the last point

        if len(np.unique(idx)) < 12:
            rms = np.inf
            bad_idx = rms > 0
        else:
            x = neon_wavelengths[idx]
            y = ref_pixels_lines
            coeffs_poly = np.polyfit(x[~duplicate_mask], y[~duplicate_mask], 2)
            p2w = np.poly1d(coeffs_poly)
            residuals = y - p2w(x)
            bad_idx = residuals >= np.sort(residuals)[-Nexclude]
            rms = np.std(residuals[~bad_idx])
    
        return rms, idx, bad_idx


    N=len(ref_pixels_lines)
    pairs_lines_index = []
    for i in range(0,np.min((4,N))):
        for j in range(np.max((0,N-4)),N):
            pairs_lines_index +=[(j,i)]

    best_rms = np.inf
    rms_table = []
    for line_ref in tqdm(itertools.combinations(np.arange(len(neon_wavelengths)), 2)):
        for peak_ref in pairs_lines_index:
            # enforce same ordering (monotonic mapping)
            rms, idx, bad_idx = fit_and_score(
                np.array(peak_ref),
                np.array(line_ref),
                ref_pixels_lines,
                neon_wavelengths,
                Nexclude
            )
            rms_table += [rms]
            if rms < best_rms:
                best_rms = rms
                best_idx = idx
                best_valid_idx = np.logical_not(bad_idx)

    y = neon_wavelengths[best_idx][best_valid_idx]
    x = ref_pixels_lines[best_valid_idx]

    # Fit a second order polynomial to the data
    coeffs_poly = np.polyfit(x, y, 2)

    spectrum=neon.sum(axis=0)

    fig=runlib_plots.plot_results_of_line_identification(spectrum, ref_pixels_lines, neon_wavelengths, best_idx, best_valid_idx, coeffs_poly, Nexclude)

    p2w = np.poly1d(coeffs_poly)
    wave_1D_mapping = p2w(np.arange(neon.shape[1]))

    return wave_1D_mapping, coeffs_poly, fig
    

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

    # Add positional argument for files
    parser.add_argument('files', nargs='*', default=['*.fits'],
                       help='FITS files to process (supports wildcards)')

    # Add optional arguments
    parser.add_argument('-f',"--flat_files", 
                       help="Select a specific flat file to use (default: use what is in argument)")
    parser.add_argument("-n","--neon_files", 
                       help="Select a specific neon file to use (default: use what is in argument)")
    parser.add_argument("-d","--dark_files", 
                       help="Select one or more specific dark(s) files to use")
    parser.add_argument('-w',"--wollaston", 
                       help="Wollaston status. Use IN for internal or OUT for no wollaston (default: first in the list of files)")
    parser.add_argument('--Nexclude', type=int, default=4,
                       help="Number of wavelength peak to exclude from the fit (default: 4)")
    
    # Parse the arguments
    args = parser.parse_args()
    file_patterns = args.files if args.files else ['*.fits','./preproc/*.fits']

    # Extract the parsed arguments
    wollaston = args.wollaston
    neon_files = args.neon_files
    flat_files = args.flat_files
    dark_patterns = args.dark_files
    Nexclude = args.Nexclude

    if ("VSCODE_PID" in os.environ or os.environ.get('TERM_PROGRAM') == 'vscode' or os.environ.get('SPYDER_DEBUG_FILE')):
        print("Running in compiler")
        if getpass.getuser() == "slacour":
            file_patterns = "/Users/slacour/DATA/LANTERNE/20251125/preproc"
        if getpass.getuser() == "jsarrazin":
            file_patterns = "/home/jsarrazin/Bureau/PLDATA/moreTest/2024-11-21_13-48-32_science_copie/preproc"
            file_patterns = "/home/jsarrazin/Bureau/PLDATA/novembre/les_preproc"
        if getpass.getuser() == "ehuby":
            file_patterns = "/home/ehuby/WORK/DATA/FIRST-PL/2025-05-10/preproc/"
            file_patterns = "/home/ehuby/WORK/DATA/FIRST-PL/2025-05-10/preproc/"
        

    # If the user specifies a coupling map, use it, otherwise look into the arguments
    if neon_files is None:
        neon_files = file_patterns
    # If the user specify a dark, use it. Otherwise, use the science file pattern
    if dark_patterns is None:
        dark_patterns = file_patterns

    flats_with_dark, neon_with_dark = get_filelist_wave(file_patterns, dark_patterns, neon_files, wollaston)


    # calculate the flat by fitting a linear function on each pixel
    datalist_flat : List[DataCube] = extract_datalist(flats_with_dark, center = False)
    poly_coeffs, fit_quality = compute_flat(datalist_flat)
    flat = poly_coeffs[:,:,0] #slope

    # making pictures
    fig_flat=runlib_plots.plot_flat_fit_quality(poly_coeffs, fit_quality)

    datalist : List[DataCube] = extract_datalist(neon_with_dark, center = False, flat=flat) 

    # calculating optical aberration (deformation) of the wavelength on the detector
    neon=np.array([np.nanmean(d.data, axis=(0,1)) for d in datalist]).sum(axis=0)
    ref_pixels_lines, aberated_image, coef_2d, fig_aberations = calculate_pixel_peaks_and_aberations(neon)

    # calculating the wavelength on each pixel on the image without aberration (1D adjustement)
    wave_1D_mapping, coef_1d, fig_1d_mapping = calculate_the_pixel_to_wavelength_mapping(ref_pixels_lines, neon_wavelengths, Nexclude)

    # computing final 2D wavelength map
    wave_2D_mapping = wave_1D_mapping + aberated_image
    wave_axis = wave_1D_mapping[wave_1D_mapping > wave_2D_mapping[:,-1].max()]
    wave_axis = wave_axis[wave_axis < wave_2D_mapping[:,0].min()]

    # computing the pixel index for each wavelength in the range, and the weights for interpolation
    index = np.zeros((len(wave_axis),len(wave_2D_mapping),2), dtype=int)
    weights = np.zeros((len(wave_axis),len(wave_2D_mapping),2))
    for i,lambda_0 in tqdm(enumerate(wave_axis), desc = "Calculating weights for wavelength interpolation"):
        idx=np.abs(wave_2D_mapping-lambda_0).argsort(axis=1)[:,:2]
        for o in range(len(idx)):
            lambda_1=wave_2D_mapping[o,idx[o,0]]
            lambda_2=wave_2D_mapping[o,idx[o,1]]
            denom = 1 / (lambda_2 - lambda_1)
            w1 = (lambda_2 - lambda_0) * denom
            w2 = (lambda_0 - lambda_1) * denom
            weights[i,o] = (w1,w2)
        index[i] = idx

    ############### Save results ####################
    # Save arrays into a FITS file

    # Create a primary HDU with no data, just the header
    hdu_primary = fits.PrimaryHDU()

    # Create HDUs for each array
    hdu = [fits.ImageHDU(data=flat, name='FLAT')]
    hdu += [fits.ImageHDU(data=wave_axis, name='WAVELENGTH')]
    hdu += [fits.ImageHDU(data=index, name='INDEX')]
    hdu += [fits.ImageHDU(data=weights, name='WEIGHT')]

    header = datalist_flat[-1].header
    # Définir le chemin complet du sous-dossier "output/couplingmaps"
    folder = datalist_flat[-1].dirname
    output_dir = os.path.join(folder,"../flatwavemaps")

    header['X_FIRTYP'] = 'FLATWAVEMAP'

    # Add date and time to the header
    current_time = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
    header['DATE-PRO'] = current_time

    # Add input parameters to the header
    header['Q_WM1D'] = (coef_1d[2],  'wavelength 2nd order poly')
    header['Q_WM1DX'] = (coef_1d[1],  'wavelength 2nd order poly')
    header['Q_WM1DX2'] = (coef_1d[0],  'wavelength 2nd order poly')
    header['Q_WM2D'] = (coef_2d[0],  'Aberrations constant')
    header['Q_WM2DX'] = (coef_2d[1],  'Aberrations X')
    header['Q_WM2DY'] = (coef_2d[2],  'Aberrations Y')
    header['Q_WM2DXY'] = (coef_2d[3],  'Aberrations XY')
    header['Q_WM2DY2'] = (coef_2d[4],  'Aberrations Y2')

    # for i, filename in enumerate(filenames):
    #     header['Q_WM_F%i' % i] = (filename, 'filename of the extracted flux')

    header['Q_WMNAME'] = (runlib_io.create_output_filename(header), 'name of the flatwave map file')

    # Créer les dossiers "output" et "pixel" s'ils n'existent pas déjà
    os.makedirs(output_dir, exist_ok=True)

    hdu_primary.header.extend(header, strip=True)

    # Combine all HDUs into an HDUList
    hdul = fits.HDUList([hdu_primary, *hdu])

    output_filename = os.path.join(output_dir, header['Q_WMNAME'])

    # Write to a FITS file
    print(f"Saving data to {output_filename}")
    hdul.writeto(output_filename, overwrite=True)


# %%
