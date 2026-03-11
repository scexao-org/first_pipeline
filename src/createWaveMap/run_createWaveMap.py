#%%

"""
FIRST Pipeline - Wavelength Map Generation Core Algorithms

Core functions for creating wavelength maps from Neon calibration spectra.
Separated from CLI interface to enable interactive use in VS Code and notebooks.

Created on Wed May 21 22:56:25 2025
@author: slacour
"""

import os
import getpass
import matplotlib
if "VSCODE_PID" in os.environ:
    matplotlib.use('Qt5Agg')
else:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
from typing import List, Tuple
from scipy.signal import find_peaks
import itertools
import warnings

try:
    RankWarning = np.exceptions.RankWarning  # numpy >=2
except AttributeError:
    RankWarning = np.RankWarning            # numpy <2
     

from tqdm import tqdm
from first_pipeline_shared.libraries import runPL_library_io as runlib_io
from first_pipeline_shared.libraries import runPL_library_plots as runlib_plots
from first_pipeline_shared.classes.runPL_class_flatMap import FlatMap
from first_pipeline_shared.classes.runPL_class_fileList import FileList
from first_pipeline_shared.classes.runPL_class_dataCube import DataCube
from first_pipeline_shared.classes.runPL_class_waveMap import WaveMap

# Reference wavelengths for Neon calibration lines (in nm)
neon_wavelengths = np.array([
   576.44188, 585.24878, 588.1895, 594.4834, 597.55343,
    602.99968, 607.43376, 609.6163, 614.30627, 616.35937, 621.72812,
    626.65085, 633.44304, 638.29917, 640.2231, 650.65279, 653.28825,
    659.89543, 667.82752, 671.70456, 692.94672, 703.24128, 717.3938,
    724.51665, 743.88981, 748.88712, 753.57739, 772.46233
])
neon_wavelengths = np.array([
   576.44188, 585.24878, 588.1895, 594.4834, 597.55343,
    602.99968, 607.43376, 609.6163, 614.30627, 616.35937, 621.72812,
    626.65085, 633.44304, 638.29917, 640.2231, 650.65279, 653.28825,
    659.89543, 667.82752, 671.70456, 692.94672, 703.24128, 717.3938,
    724.51665, 743.88981, 753.57739, 772.46233
])




def find_N_peaks(spectrum, N=1000):
    """
    Find the most prominent spectral peaks in a 1D spectrum.
    
    Parameters
    ----------
    spectrum : array_like
        1D array of spectral intensities
    N : int, optional
        Maximum number of peaks to return (default: 1000)
        
    Returns
    -------
    numpy.ndarray
        Array of peak pixel indices in sorted order
    """
    min_peak_separation = 6
    prominence_threshold = 0.01 * (np.max(spectrum) - np.median(spectrum))
    
    peaks, properties = find_peaks(
        spectrum,
        prominence=prominence_threshold,
        distance=min_peak_separation
    )
    peak_prominence = properties["prominences"]
    idx_peak = np.argsort(peak_prominence)[-N:]
    return peaks[np.sort(idx_peak)]


def subpixel_parabolic(spectrum, peaks):
    """
    Refine peak positions using parabolic interpolation for subpixel accuracy.
    
    Parameters
    ----------
    spectrum : array_like
        1D array of intensities
    peaks : array_like
        Integer pixel indices from peak detection
        
    Returns
    -------
    numpy.ndarray
        Array of refined subpixel peak positions
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


def calculate_pixel_peaks_and_aberations(neon):
    """
    Detect spectral line peaks and fit optical aberration model across detector.
    
    This function identifies consistent spectral lines across all fiber outputs and
    fits a 2D polynomial model to characterize optical aberrations as a 
    function of pixel position and output fiber number.
    
    Parameters
    ----------
    neon : numpy.ndarray
        2D array of neon calibration spectra with shape (n_outputs, n_pixels)
    
    Returns
    -------
    ref_pixels_lines : numpy.ndarray
        1D array of reference pixel positions for each good spectral line
    aberated_image : numpy.ndarray  
        2D array containing the fitted aberration map in pixel units
    coeffs : numpy.ndarray
        1D array of polynomial coefficients [c0, c1, c2, c3, c4] for aberration fit
    fig : matplotlib.figure.Figure
        Diagnostic plot showing the peak fitting results and aberration model
    """
    spectrum_0 = neon.sum(axis=0)
    peaks_0 = find_N_peaks(spectrum_0, 15)

    # Find corresponding peaks in each fiber output
    peaks_all = []
    for spectrum in neon:
        peaks = find_N_peaks(spectrum)
        idx_peak = []
        for p0 in peaks_0:
            idx_peak += [np.argmin(np.abs(peaks-p0))]
        peaks_all += [peaks[idx_peak]]

    peaks_all = np.array(peaks_all)
    roll_index = np.median((peaks_all-peaks_0), axis=1)
    roll_index = roll_index.astype(int)

    # Align spectra by rolling to common wavelength grid
    neon_rolled = np.array([np.roll(spectrum, -roll) for spectrum, roll in zip(neon, roll_index)])
    spectrum_0 = neon_rolled.sum(axis=0)
    peaks_0 = find_N_peaks(spectrum_0, 25)
    
    peaks_all = []
    peaks_all_sub = []
    for spectrum in neon_rolled:
        peaks = find_N_peaks(spectrum)
        idx_peak = []
        for p0 in peaks_0:
            idx_peak += [np.argmin(np.abs(peaks-p0))]
        peaks_all += [peaks[idx_peak]]

        peaks_sub = subpixel_parabolic(spectrum, peaks[idx_peak])
        peaks_all_sub += [peaks_sub]

    peaks_all = np.array(peaks_all)
    peaks_all_sub = np.array(peaks_all_sub) + roll_index[:, None]

    # Filter out inconsistent lines 
    peaks_diff = np.diff(peaks_all_sub, axis=0)
    peaks_diff_offset = peaks_diff - np.median(peaks_diff, axis=1)[:, None]
    N_lines_good = np.max(np.abs(peaks_diff_offset), axis=0) < 1.5

    peaks_all_sub_good = peaks_all_sub[:, N_lines_good]

    # Calculate reference line positions and aberrations
    line_pixel_ref = np.mean(peaks_all_sub_good, axis=0)
    aberations = peaks_all_sub_good - line_pixel_ref[None, :]

    # Fit 2D aberration model: c0 + c1*x + c2*y + c3*x*y + c4*y^2
    x_coords = line_pixel_ref
    y_coords = np.arange(len(aberations))
    X, Y = np.meshgrid(x_coords, y_coords)

    X_flat = X.flatten()
    Y_flat = Y.flatten()
    aberations_flat = aberations.flatten()

    A = np.column_stack([
        np.ones(len(X_flat)),
        X_flat,
        Y_flat,
        X_flat*Y_flat,
        Y_flat**2
    ])

    coeffs = np.linalg.lstsq(A, aberations_flat, rcond=None)[0]

    aberations_fit = (coeffs[0] + 
                    coeffs[1] * X + 
                    coeffs[2] * Y + 
                    coeffs[3] * X * Y +
                    coeffs[4] * Y**2)
    
    fig = runlib_plots.plot_wavefit_coeffs(peaks_all_sub, peaks_all_sub_good, aberations, aberations_fit)

    ref_pixels_lines = (peaks_all_sub_good - aberations_fit).mean(axis=0)

    # Apply aberration model to full detector
    X_full, Y_full = np.meshgrid(np.arange(neon.shape[1]), np.arange(neon.shape[0]))
    aberated_image = (coeffs[0] + 
                    coeffs[1] * X_full + 
                    coeffs[2] * Y_full + 
                    coeffs[3] * X_full * Y_full +
                    coeffs[4] * Y_full**2)

    return ref_pixels_lines, aberated_image, coeffs, fig


def fit_and_score(peak_ref, line_ref, ref_pixels_lines, neon_wavelengths, Nexclude):
    """
    Fit linear mapping between two reference lines and score the quality.
    
    Parameters
    ----------
    peak_ref : array_like
        Two pixel positions for reference lines
    line_ref : array_like  
        Two reference wavelengths
    ref_pixels_lines : array_like
        All detected pixel positions
    neon_wavelengths : array_like
        All reference wavelengths
    Nexclude : int
        Number of outliers to exclude from RMS calculation
        
    Returns
    -------
    rms : float
        RMS error of polynomial fit
    idx : array_like
        Indices of matched reference lines
    bad_idx : array_like
        Boolean array marking outliers
    """
    peak_pixels = ref_pixels_lines[peak_ref]
    ref_lambda = neon_wavelengths[line_ref]

    # Linear fit between reference points
    a = (ref_lambda[1] - ref_lambda[0]) / (peak_pixels[1] - peak_pixels[0])
    b = ref_lambda[0] - a * peak_pixels[0]

    # Predict wavelengths for all detected peaks
    lambda_pred = a*ref_pixels_lines + b

    # Match predicted wavelengths to reference catalog
    idx = []
    for lp in lambda_pred:
        idx += [np.argmin(np.abs(neon_wavelengths - lp))]

    # Handle duplicate assignments
    unique_idx, counts = np.unique(idx, return_counts=True)
    duplicate_mask = np.isin(idx, unique_idx[counts > 1])
    duplicate_mask[0] = True   # Always ignore first point
    duplicate_mask[-1] = True  # Always ignore last point

    if len(np.unique(idx)) < 12:
        rms = np.inf
        bad_idx = np.ones_like(idx, dtype=bool)
    else:
        # Fit second-order polynomial
        x = neon_wavelengths[idx]
        y = ref_pixels_lines
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RankWarning)
            coeffs_poly = np.polyfit(x[~duplicate_mask], y[~duplicate_mask], 2)
        p2w = np.poly1d(coeffs_poly)
        residuals = y - p2w(x)
        bad_idx = residuals >= np.sort(residuals)[-Nexclude]
        rms = np.std(residuals[~bad_idx])
    
    return rms, idx, bad_idx


def calculate_the_pixel_to_wavelength_mapping(ref_pixels_lines, neon_wavelengths, Nexclude, neon_spectrum=None):
    """
    Calculate pixel-to-wavelength mapping using iterative fitting approach.
    
    This function finds the best polynomial mapping between pixel positions and 
    known wavelengths by testing multiple combinations of reference lines.
    
    Parameters
    ----------
    ref_pixels_lines : array_like
        Array of pixel positions where reference spectral lines are detected
    neon_wavelengths : array_like
        Array of known wavelengths for reference spectral lines
    Nexclude : int
        Number of worst outlier points to exclude from RMS calculation
    neon_spectrum : array_like, optional
        2D neon spectrum array for determining spectrum width (for diagnostic plots)
        
    Returns
    -------
    wave_1D_mapping : numpy.ndarray
        1D array containing wavelength value for each pixel position
    coeffs_poly : numpy.ndarray
        Coefficients of second-order polynomial fit
    fig : matplotlib.figure.Figure
        Diagnostic plot showing line identification and calibration quality
    """
    N = len(ref_pixels_lines)
    pairs_lines_index = []
    for i in range(0, np.min((4, N))):
        for j in range(np.max((0, N-4)), N):
            pairs_lines_index += [(j, i)]

    best_rms = np.inf
    rms_table = []
    
    for line_ref in tqdm(itertools.combinations(np.arange(len(neon_wavelengths)), 2)):
        for peak_ref in pairs_lines_index:
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

    # Final polynomial fit with best parameters
    y = neon_wavelengths[best_idx][best_valid_idx]
    x = ref_pixels_lines[best_valid_idx]
    
    coeffs_poly = np.polyfit(x, y, 2)
    
    # Generate spectrum for plotting
    if neon_spectrum is not None:
        spectrum = neon_spectrum.sum(axis=0)
        n_pixels = neon_spectrum.shape[1]
    else:
        # Create dummy spectrum for plotting if not provided
        spectrum = np.zeros(len(ref_pixels_lines) * 10)  # Reasonable default size
        n_pixels = len(spectrum)
    
    fig = runlib_plots.plot_results_of_line_identification(
        spectrum, ref_pixels_lines, neon_wavelengths, best_idx, 
        best_valid_idx, coeffs_poly, Nexclude
    )

    p2w = np.poly1d(coeffs_poly)
    wave_1D_mapping = p2w(np.arange(n_pixels))

    return wave_1D_mapping, coeffs_poly, fig


def get_filelist_wave(file_patterns, dark_patterns, flat_patterns, wollaston):
    """
    Create file list for wavelength calibration data with appropriate associations.
    
    Parameters
    ----------
    file_patterns : list
        List of file patterns to search for COMPARISON data
    dark_patterns : list or None
        List of patterns for dark files, uses file_patterns if None
    flat_patterns : list or None  
        List of patterns for flat field files
    wollaston : str or None
        Wollaston polarizer status ('IN' or 'OUT')
        
    Returns
    -------
    fileList : FileList
        Configured file list object with dark associations
    flatMap : FlatMap or None
        Flat field map object if available
    """
    fileList = FileList(file_patterns, data_type='COMPARISON', first_type='PREPROC', wollaston=wollaston)
    fileList.make_association(dark_patterns=dark_patterns)
    file_flat = fileList.get_flatmap_file(flat_patterns)
    flatMap = FlatMap(file_flat) if file_flat is not None else None
    
    return fileList, flatMap


def save_wavelength_map(waveMap, header, coef_1d, coef_2d, output_dir):
    """
    Save wavelength map with calibration coefficients to FITS file.
    
    Parameters
    ----------
    waveMap : WaveMap
        Wavelength map object containing wave, index, and weights data
    header : astropy.io.fits.Header
        FITS header to be updated with calibration parameters
    coef_1d : array_like
        1D wavelength mapping polynomial coefficients
    coef_2d : array_like
        2D aberration correction coefficients
    output_dir : str
        Output directory path for saving files
        
    Returns
    -------
    str
        Full path to saved wavelength map file
    """
    # Add calibration parameters to header
    header['Q_WM1D'] = (coef_1d[2], 'wavelength 2nd order poly constant')
    header['Q_WM1DX'] = (coef_1d[1], 'wavelength 2nd order poly linear') 
    header['Q_WM1DX2'] = (coef_1d[0], 'wavelength 2nd order poly quadratic')
    header['Q_WM2D'] = (coef_2d[0], 'Aberrations constant')
    header['Q_WM2DX'] = (coef_2d[1], 'Aberrations X')
    header['Q_WM2DY'] = (coef_2d[2], 'Aberrations Y')
    header['Q_WM2DXY'] = (coef_2d[3], 'Aberrations XY')
    header['Q_WM2DY2'] = (coef_2d[4], 'Aberrations Y2')
    header['Q_WMNAME'] = (runlib_io.create_basename(header), 'name of the wave map file')

    # Create output directory and save
    os.makedirs(output_dir, exist_ok=True)
    output_filename = os.path.join(output_dir, header['Q_WMNAME'])
    waveMap.save(output_filename, header)
    
    return output_filename


def create_coefficients_plot(coef_1d, coef_2d):
    """
    Create diagnostic plot showing wavelength mapping coefficients.
    
    Parameters
    ----------
    coef_1d : array_like
        1D wavelength polynomial coefficients [a2, a1, a0]
    coef_2d : array_like
        2D aberration coefficients [c0, c1, c2, c3, c4]
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure showing the calibration coefficients
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    ax.axis('off')

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

    ax.text(0.02, 0.98, coef_1d_text, transform=ax.transAxes, fontsize=10, 
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))

    ax.text(0.02, 0.48, coef_2d_text, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8))

    ax.set_title('Wavelength Mapping Coefficients', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig


def run_createWaveMap(file_patterns=None, dark_patterns=None, flat_patterns=None,
                               wollaston=None, Nexclude=None):
    """
    Complete workflow for wavelength map generation from Neon calibration data.
    
    This is the main processing function that orchestrates the entire wavelength
    calibration workflow from file loading through final map generation.
    
    Parameters
    ----------
    file_patterns : list
        List of file patterns to search for COMPARISON data files
    dark_patterns : list, optional
        List of patterns for dark files, uses file_patterns if None
    flat_patterns : list, optional  
        List of patterns for flat field files
    wollaston : str, optional
        Wollaston polarizer status ('IN' or 'OUT'), auto-detected if None
    Nexclude : int, optional
        Number of wavelength peaks to exclude from fit (default: 4)
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'output_filename': path to saved wavelength map
        - 'waveMap': WaveMap object
        - 'coef_1d': 1D wavelength polynomial coefficients  
        - 'coef_2d': 2D aberration coefficients
        - 'figures': list of diagnostic figures
    """

    # Set up file patterns
    if dark_patterns is None:
        dark_patterns = file_patterns
    if flat_patterns is None and file_patterns:
        folder = os.path.dirname(file_patterns[0])
        flat_patterns = file_patterns + [os.path.join(folder, "../flatmaps")]

    # Get file list and flat map
    fileList, flatMap = get_filelist_wave(file_patterns, dark_patterns, flat_patterns, wollaston)
    
    # Extract data
    datalist: List[DataCube] = fileList.extract_data_from_list(flatMap=flatMap, center=False)

    # Calculate optical aberration mapping
    neon = np.array([np.nanmean(d.data, axis=(0,1)) for d in datalist]).sum(axis=0)
    ref_pixels_lines, aberated_image, coef_2d, fig_aberations = calculate_pixel_peaks_and_aberations(neon)

    # Calculate 1D wavelength mapping
    wave_1D_mapping, coef_1d, fig_1d_mapping = calculate_the_pixel_to_wavelength_mapping(
        ref_pixels_lines, neon_wavelengths, Nexclude, neon)

    # Compute final 2D wavelength map
    index_pixel_2d_float = np.arange(neon.shape[1]) + aberated_image

    # Get integer indices for interpolation
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

    # Create WaveMap object
    waveMap = WaveMap()
    waveMap.create_from_data(wave, index, weights)

    # Set up output directory
    header = datalist[-1].header
    folder = fileList.get_most_common_dir()
    output_dir = os.path.join(folder, "../wavemaps")

    # Save wavelength map
    output_filename = save_wavelength_map(waveMap, header, coef_1d, coef_2d, output_dir)

    # Create coefficient plot
    fig_coeffs = create_coefficients_plot(coef_1d, coef_2d)
    
    # Save all plots
    runlib_plots.save_pdf_in_file(output_filename)

    return waveMap, datalist


if __name__ == "__main__":
    """
    Run wavelength map creation with development defaults.
    Perfect for testing and direct execution of core functionality.
    """
    print("Running createWaveMap core with development defaults...")
    

    if getpass.getuser() == "slacour":
        dark_patterns = None
        flat_patterns = None
        wollaston = None
        Nexclude = 5
        file_patterns = ["/Users/slacour/DATA/LANTERNE/raw/20260114/preproc/"]
        file_patterns = ["/Users/slacour/DATA/LANTERNE/20260307/preproc/"]
        
        print(f"Development override: dark_patterns={dark_patterns}, flat_patterns={flat_patterns}, wollaston={wollaston}, Nexclude={Nexclude}")
        print(f"Development file patterns: {file_patterns}")

    # Process wavelength map data
    waveMap, datalist = run_createWaveMap(
        file_patterns=file_patterns,
        dark_patterns=dark_patterns,
        flat_patterns=flat_patterns,
        wollaston=wollaston,
        Nexclude=Nexclude
    )

    waveMap2 = WaveMap(waveMap.filename)


    dataset = datalist[0]


    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Plot before wavelength calibration
    ax1.plot(dataset.wave,dataset.data.mean(axis=(0,1,2)))
    ax1.set_xlabel(f'{dataset.wave_label}')
    ax1.set_ylabel('Flux (summed over fibers and exposures)')
    ax1.set_title('Before Wavelength Calibration')


    waveMap2.interpolate_data(dataset)

    # Plot after wavelength calibration
    ax2.plot(dataset.wave,dataset.data.mean(axis=(0,1,2)))
    ax2.set_xlabel(f'{dataset.wave_label}')
    ax2.set_ylabel('Flux (summed over fibers and exposures)')
    ax2.set_title('After Wavelength Calibration')

    plt.tight_layout()
    plt.show()


# %%
