#! /usr/bin/env python3
# -*- coding: iso-8859-15 -*-
#%%
"""
FIRST Pipeline - Wavelength Map Generation

This script creates wavelength maps from Neon calibration spectra for the FIRST
Visible Photonic Lantern at SUBARU/SCEXAO. Wavelength maps enable precise 
spectral calibration by detecting emission lines and fitting polynomial 
wavelength solutions with aberration correction.

The wavelength mapping is critical for accurate spectral analysis, providing
the wavelength-to-pixel relationship needed for scientific observations.

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
import warnings

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
# Import FIRST pipeline classes
from classes.runPL_class_flatMap import FlatMap
from classes.runPL_class_fileList import FileList
from classes.runPL_class_dataCube import DataCube
from classes.runPL_class_waveMap import WaveMap

#plt.ion()
# Add options
usage = """
Usage: %prog [options] [files]

Goal: Create a wavelength map from the provided FITS files.

Summary:
- Searches for FITS files with X_FIRTYP=PREPROC and DATA-TYP=COMPARISON keywords.
- Finds corresponding dark files (X_FIRTYP=PREPROC, DATA-TYP=DARK).
- Reads neon comparison files, subtracts the median of the dark files.
- Detects emission peaks and fits a polynomial to create a wavelength map.
- Generates 2D wavelength mapping with aberration correction.
- Saves the wavelength map as a FITS file in the output directory.
- Generates and saves figures for visualization.
- Output files are stored in a "wavemaps" directory.

Options:
    files             FITS files to process (supports wildcards, default: *.fits)
    --dark_files      Select one or more specific dark files to use
    --flatMap         Select a specific flat map to use
    --wollaston       Wollaston status: IN for internal or OUT for no wollaston
    --Nexclude        Number of wavelength peaks to exclude from fit (default: 4)

Example:
    runPL_create_waveMap.py *.fits --Nexclude=3
    runPL_create_waveMap.py /path/to/files/*.fits --dark_files="dark*.fits" --flatMap="flatmap.fits"
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


def calculate_pixel_peaks_and_aberations(neon):
    """
    Calculate pixel positions of spectral line peaks and fit optical aberrations in a neon calibration dataset.
    This routine processes a 2D neon spectrum array to identify spectral line peaks across multiple 
    photonic lantern outputs and fits a polynomial model to characterize optical aberrations as a 
    function of pixel position and output fiber number.
    The algorithm performs the following steps:
    1. Identifies prominent peaks in the collapsed spectrum to establish reference lines
    2. Applies coarse integer pixel shifts to align spectra across different outputs
    3. Refines peak positions using subpixel parabolic interpolation
    4. Filters out poorly behaved spectral lines based on consistency criteria
    5. Fits a 2D polynomial model (1st order in pixel position, 2nd order in output number)
       to characterize systematic aberrations
    Parameters
    ----------
    neon : numpy.ndarray
        2D array of neon calibration spectra with shape (n_outputs, n_pixels),
        where each row represents the spectrum from one photonic lantern output
    Returns
    -------
    ref_pixels_lines : numpy.ndarray
        1D array of reference pixel positions for each good spectral line after
        aberration correction
    aberated_image : numpy.ndarray
        2D array with same shape as input, containing the fitted aberration map
        in pixel units across the entire detector
    coeffs : numpy.ndarray
        1D array of polynomial coefficients [c0, c1, c2, c3, c4] for the aberration
        fit: c0 + c1*x + c2*y + c3*x*y + c4*y^2
    fig : matplotlib.figure.Figure
        Diagnostic plot showing the peak fitting results and aberration model
    """


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
    peaks_0=find_N_peaks(spectrum_0,15)

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
    peaks_0=find_N_peaks(spectrum_0,25)
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
    """
    Calculate the pixel-to-wavelength mapping for spectroscopic data using reference lines.
    This function performs wavelength calibration by finding the best polynomial mapping
    between pixel positions and known wavelengths from reference spectral lines (e.g., Neon).
    It uses an iterative approach to test different combinations of reference lines and
    fits a second-order polynomial to establish the pixel-to-wavelength relationship.
    Parameters
    ----------
    ref_pixels_lines : array_like
        Array of pixel positions where reference spectral lines are detected.
    neon_wavelengths : array_like
        Array of known wavelengths (in appropriate units) corresponding to reference
        spectral lines from a calibration source (e.g., Neon lamp).
    Nexclude : int
        Number of worst outlier points to exclude when calculating the RMS error
        during the fitting process.
    Returns
    -------
    wave_1D_mapping : numpy.ndarray
        1D array containing the wavelength value for each pixel position across
        the entire detector/spectrum.
    coeffs_poly : numpy.ndarray
        Coefficients of the second-order polynomial used for the pixel-to-wavelength
        mapping, in descending order of powers.
    fig : matplotlib.figure.Figure
        Figure object containing plots showing the results of line identification
        and wavelength calibration quality.
    Notes
    -----
    The function performs the following steps:
    1. Tests multiple combinations of reference line pairs to establish initial linear fits
    2. For each combination, predicts wavelengths for all detected lines
    3. Matches predicted wavelengths to known reference wavelengths
    4. Fits a second-order polynomial while excluding outliers
    5. Selects the best fit based on minimum RMS error
    6. Generates diagnostic plots showing calibration quality
    The algorithm handles duplicate line assignments and enforces monotonic mapping
    to ensure physical consistency of the wavelength solution.
    """



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
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', np.exceptions.RankWarning)
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
        description="Generate wavelength calibration maps from Neon emission line spectra for FIRST Pipeline spectral calibration.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
FIRST Pipeline Wavelength Map Generation Tool

This script creates wavelength calibration maps from Neon comparison lamp spectra.
It detects emission lines, fits polynomial wavelength solutions, and generates 
2D wavelength mapping with aberration correction for precise spectral calibration.

Examples:
    %(prog)s --wollaston IN --flatMap=/path/to/flat.fits *.fits
    %(prog)s --Nexclude 3 --dark_files=dark*.fits neon_data/*.fits
    %(prog)s /data/comparison/*.fits

Pipeline Workflow Integration:
    1. Requires preprocessed Neon calibration files (X_FIRTYP=PREPROC, DATA-TYP=COMPARAISON)
    2. Uses flat field maps for proper calibration
    3. Output wavelength maps enable spectral analysis in downstream scripts
    4. Essential for accurate wavelength calibration of science observations

Input Files:
    - Neon calibration spectra: X_FIRTYP=PREPROC and DATA-TYP=COMPARAISON
    - Corresponding dark frames: X_FIRTYP=PREPROC and DATA-TYP=DARK
    - Flat field maps (optional): for enhanced calibration accuracy
    - Files automatically grouped by Wollaston status (IN/OUT)

Output Files:
    - FITS file with wavelength calibration map (output/wave/ directory)
    - Diagnostic plots showing line detection and polynomial fits
    - Quality assessment figures for calibration validation

Processing Details:
    - Detects Neon emission peaks using advanced peak finding algorithms
    - Fits polynomial wavelength solutions with configurable degree
    - Applies aberration correction for spatial variations
    - Excludes problematic peaks with --Nexclude parameter
    - Handles both polarimetry (Wollaston IN) and photometry (OUT) modes
    - Dark subtraction for accurate line measurement

Calibration Quality:
    - Automatic outlier rejection for robust fitting
    - Residual analysis to assess calibration accuracy  
    - Spatial mapping to handle optical aberrations
    - Quality metrics saved with wavelength maps

Technical Notes:
    - Nexclude: Number of peaks to exclude from fitting (handles outliers)
    - Wollaston status affects channel configuration and processing
    - Polynomial degree optimized for FIRST optical system
    - Output maps compatible with subsequent pipeline scripts

Note: Quality wavelength calibration is essential for accurate spectroscopy.
Review diagnostic plots to ensure proper line detection and fitting.
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
    parser.add_argument("--flatMap", 
                       help="Select a specific flat Map to use")
    parser.add_argument("--wollaston", 
                       help="Wollaston status. Use IN for internal or OUT for no wollaston (default: first in the list of files)")
    parser.add_argument('--Nexclude', type=int, default=4,
                       help="Number of wavelength peak to exclude from the fit (default: 4)")
    
    # Parse the arguments
    args = parser.parse_args()
    file_patterns = args.files if args.files else ['*.fits','./preproc/*.fits']

    # Extract the parsed arguments
    dark_patterns = args.dark_files
    flat_patterns = args.flatMap
    wollaston = args.wollaston
    Nexclude = args.Nexclude

    if ("VSCODE_PID" in os.environ or os.environ.get('TERM_PROGRAM') == 'vscode' or os.environ.get('SPYDER_DEBUG_FILE')):
        print("Running in compiler")
        if getpass.getuser() == "slacour":
            file_patterns = ["/Users/slacour/DATA/LANTERNE/20251125/preproc"]
        if getpass.getuser() == "jsarrazin":
            file_patterns = ["/home/jsarrazin/Bureau/PLDATA/moreTest/2024-11-21_13-48-32_science_copie/preproc"]
            file_patterns = ["/home/jsarrazin/Bureau/PLDATA/novembre/les_preproc"]
        if getpass.getuser() == "ehuby":
            file_patterns = ["/home/ehuby/WORK/DATA/FIRST-PL/2025-05-10/preproc/"]
        file_patterns = [file_patterns] if isinstance(file_patterns, str) else file_patterns


    # If the user specify a dark, use it. Otherwise, use the science file pattern
    if dark_patterns is None:
        dark_patterns = file_patterns
    # If the user specifies a coupling map, use it, otherwise look into the arguments
    if flat_patterns is None:
        folder = os.path.dirname(file_patterns[0])
        flat_patterns = file_patterns + [os.path.join(folder,"../flatmaps")]

    fileList = FileList(file_patterns, data_type='COMPARISON', first_type='PREPROC', wollaston=wollaston)
    fileList.make_association(darks_pattern=dark_patterns)
    file_flat = fileList.get_flatmap_file(flat_patterns)

    flatMap =  FlatMap(file_flat) if file_flat is not None else None

    datalist : List[DataCube] = fileList.extract_data_from_list(flatMap = flatMap, center = False)

    # calculating optical aberration (deformation) of the wavelength on the detector
    neon=np.array([np.nanmean(d.data, axis=(0,1)) for d in datalist]).sum(axis=0)
    ref_pixels_lines, aberated_image, coef_2d, fig_aberations = calculate_pixel_peaks_and_aberations(neon)

    # calculating the wavelength on each pixel on the image without aberration (1D adjustement)
    wave_1D_mapping, coef_1d, fig_1d_mapping = calculate_the_pixel_to_wavelength_mapping(ref_pixels_lines, neon_wavelengths, Nexclude)

    # computing final 2D wavelength map

    index_pixel_2d_float=np.arange(neon.shape[1])+aberated_image

    # Get integer indices below and above ref_pixel_2d for interpolation
    index_pixel_2d_floor = np.floor(index_pixel_2d_float).astype(int)
    index_pixel_2d_ceil = np.ceil(index_pixel_2d_float).astype(int)
    # Calculate interpolation weights
    weights_floor = index_pixel_2d_ceil - index_pixel_2d_float
    weights_ceil = index_pixel_2d_float - index_pixel_2d_floor

    weights = np.array((weights_floor, weights_ceil))
    index = np.array((index_pixel_2d_floor, index_pixel_2d_ceil))
    good_index = (index.min(axis=(0,1)) >= 0) & (index.max(axis=(0,1)) < neon.shape[1])
    index = index[:,:,good_index]
    weights = weights[:,:,good_index]
    wave = wave_1D_mapping[good_index]

    ############### Save results ####################
    # Create WaveMap object and save using the new save method
    
    waveMap = WaveMap()
    waveMap.create_from_data(wave, index, weights)

    header = datalist[-1].header
    # Définir le chemin complet du sous-dossier "output/couplingmaps"
    folder = fileList.get_most_common_dir()
    output_dir = os.path.join(folder,"../wavemaps")

    header['X_FIRTYP'] = 'WAVEMAP'

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

    output_filename = os.path.join(output_dir, header['Q_WMNAME'])

    # Save using the WaveMap save method
    waveMap.save(output_filename, header)

    ############### Save figures ####################
    # Plot interpolation coefficients


    # Create a figure to display the coefficients
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    ax.axis('off')  # Turn off axis

    # Create text strings for the coefficients
    coef_1d_text = f"""1D Wavelength Mapping Coefficients:
    lambda(x pixel) = {coef_1d[0]:.6e} * x^2 + {coef_1d[1]:.6e} * x + {coef_1d[2]:.6e}

    a2 = {coef_1d[0]:.6e}
    a1 = {coef_1d[1]:.6e} 
    a0 = {coef_1d[2]:.6e}"""

    coef_2d_text = f"""2D Aberration Coefficients:
    Dpixel(x,y) = {coef_2d[0]:.6e} + {coef_2d[1]:.6e} * x + {coef_2d[2]:.6e} * y + {coef_2d[3]:.6e} * xy + {coef_2d[4]:.6e} * y^2

    c0 = {coef_2d[0]:.6e}
    c1 = {coef_2d[1]:.6e}
    c2 = {coef_2d[2]:.6e}
    c3 = {coef_2d[3]:.6e}
    c4 = {coef_2d[4]:.6e}"""

    # Display the text
    ax.text(0.02, 0.98, coef_1d_text, transform=ax.transAxes, fontsize=10, 
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))

    ax.text(0.02, 0.48, coef_2d_text, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8))

    ax.set_title('Wavelength Mapping Coefficients', fontsize=14, fontweight='bold')
    plt.tight_layout()


    runlib_plots.save_pdf_in_file(output_filename)

# %%
