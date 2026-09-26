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
    matplotlib.use('macosx')
else:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
from typing import List, Tuple
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from scipy.special import erf
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

# Reference wavelengths of the Neon (Ne I) calibration lines, in nm, in STANDARD AIR
# (15 C, 101325 Pa, dry air), from the NIST Atomic Spectra Database (observed
# wavelengths). Air is the usual convention for optical spectroscopy, and matches
# the air rest wavelengths used downstream (e.g. H-alpha = 656.28 nm).
# Use `air_to_vacuum` (or --vacuum in the CLI) to calibrate in vacuum instead.
# Note: 748.88712 nm (bright) is deliberately not included.
neon_wavelengths_air = np.array([
    576.44188, 585.24878, 588.18950, 594.48340,
    602.99968, 607.43376, 609.61630, 614.30627, 616.35937, 621.72812,
    626.64952, 633.44276, 638.29914, 640.22480, 650.65277, 653.28824,
    659.89528, 667.82766, 671.70430, 692.94672, 703.24128, 717.39380,
    724.51665, 743.88981, 753.57739
])
neon_wavelengths = neon_wavelengths_air  # backward compatibility


def air_to_vacuum(wavelength_air_nm):
    """
    Convert standard-air wavelengths to vacuum wavelengths.

    Uses the inverse of the Morton (2000) / Ciddor (1996) refractive index of
    standard air, as derived by N. Piskunov for VALD. Valid from 200 nm to the
    near-infrared; accuracy much better than 1e-4 nm in the visible.
    Example: H-alpha 656.279 nm (air) -> 656.460 nm (vacuum).

    Parameters
    ----------
    wavelength_air_nm : array_like
        Wavelengths in standard air, in nm

    Returns
    -------
    numpy.ndarray
        Wavelengths in vacuum, in nm
    """
    wavelength_air_nm = np.asarray(wavelength_air_nm, dtype=float)
    s2 = (1e3 / wavelength_air_nm) ** 2  # wavenumber squared, in micron^-2
    n = (1 + 0.00008336624212083
         + 0.02408926869968 / (130.1065924522 - s2)
         + 0.0001599740894897 / (38.92568793293 - s2))
    return wavelength_air_nm * n


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


def _pixel_integrated_gaussian(x, amplitude, center, sigma, offset):
    """Gaussian line profile integrated over each pixel (x = pixel centres), plus a constant."""
    k = np.sqrt(2) * np.abs(sigma)
    return amplitude * 0.5 * (erf((x + 0.5 - center) / k) - erf((x - 0.5 - center) / k)) + offset


def subpixel_gaussian(spectrum, peaks, half_width=3):
    """
    Refine peak positions by least-squares fitting a pixel-integrated Gaussian
    plus a constant background over +/- half_width pixels around each peak.

    The 3-point parabola of `subpixel_parabolic` is biased towards pixel centres
    ("pixel locking") when lines are close to critically sampled (FWHM ~2 px, as
    for the FIRST neon lines), by up to ~0.05 px. The Gaussian fit is unbiased
    for Gaussian-like lines and insensitive to a local background level.

    Falls back to the parabolic estimate when the fit fails, when the window is
    truncated by the edge, or when the fitted centre moves more than 1 pixel.

    Parameters
    ----------
    spectrum : array_like
        1D array of intensities
    peaks : array_like
        Integer pixel indices from peak detection
    half_width : int, optional
        Half-size of the fitting window in pixels (default: 3)

    Returns
    -------
    numpy.ndarray
        Array of refined subpixel peak positions
    """
    spectrum = np.asarray(spectrum, dtype=float)
    first_guess = subpixel_parabolic(spectrum, peaks)
    subpixels = []

    for i, guess in zip(peaks, first_guess):
        if i - half_width < 0 or i + half_width >= len(spectrum):
            subpixels.append(guess)
            continue

        x = np.arange(i - half_width, i + half_width + 1)
        y = spectrum[x]
        if not np.all(np.isfinite(y)):
            subpixels.append(guess)
            continue

        offset0 = np.min(y)
        p0 = [(spectrum[i] - offset0) * 2.5, guess, 1.0, offset0]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                popt, _ = curve_fit(_pixel_integrated_gaussian, x, y, p0=p0, maxfev=2000)
        except (RuntimeError, ValueError):
            subpixels.append(guess)
            continue

        center = popt[1]
        if not np.isfinite(center) or abs(center - i) > 1 or popt[0] <= 0:
            subpixels.append(guess)
        else:
            subpixels.append(center)

    return np.array(subpixels)


def calculate_pixel_peaks_and_aberations(neon, fiber_offsets=None):
    """
    Detect spectral line peaks and fit optical aberration model across detector.
    
    This function identifies consistent spectral lines across all fiber outputs and
    fits a model of the line position shift (in pixels) as a function of pixel
    position x and output fiber number y:

        Dpixel(x, y) = c0 + c1*x + c2*y + c3*x*y + c4*y^2 + c5*x*y^2 + dx[y]

    The polynomial describes the smooth optical aberrations of the spectrograph.
    dx[y] is a per-fiber offset along x, caused by the single-mode fibers not
    being perfectly aligned in the V-groove (typically ~0.1 pixel). The offsets
    are defined orthogonal to (1, y, y^2), i.e. they only contain the part that
    the smooth polynomial cannot describe.
    
    Parameters
    ----------
    neon : numpy.ndarray
        2D array of neon calibration spectra with shape (n_outputs, n_pixels)
    fiber_offsets : array_like or None, optional
        If None (default), the per-fiber offsets dx[y] are fitted together with
        the polynomial. If an array of length n_outputs is given, those offsets
        are held fixed (e.g. read from a reference wavelength map with
        `read_fiber_offsets`) and only the polynomial is fitted.
    
    Returns
    -------
    ref_pixels_lines : numpy.ndarray
        1D array of reference pixel positions for each good spectral line
    aberated_image : numpy.ndarray  
        2D array containing the fitted aberration map in pixel units
    coeffs : numpy.ndarray
        1D array of polynomial coefficients [c0, c1, c2, c3, c4, c5]
    fiber_offsets : numpy.ndarray
        1D array (n_outputs) of per-fiber x offsets dx[y], in pixels
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

        peaks_sub = subpixel_gaussian(spectrum, peaks[idx_peak])
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

    # Fit 2D aberration model:
    #   c0 + c1*x + c2*y + c3*x*y + c4*y^2 + c5*x*y^2 + dx[y]
    n_outputs = aberations.shape[0]
    x_coords = line_pixel_ref
    y_coords = np.arange(n_outputs)
    X, Y = np.meshgrid(x_coords, y_coords)

    def smooth_model(c, x, y):
        return c[0] + c[1]*x + c[2]*y + c[3]*x*y + c[4]*y**2 + c[5]*x*y**2

    if fiber_offsets is None:
        # One free constant per fiber + the x-dependent terms. The constants
        # absorb every function of y alone (c0, c2, c4 and dx), which are then
        # separated by fitting a quadratic in y to them.
        onehot = [(Y == k).astype(float).ravel() for k in range(n_outputs)]
        A = np.column_stack(onehot + [X.ravel(), (X*Y).ravel(), (X*Y**2).ravel()])
        sol = np.linalg.lstsq(A, aberations.ravel(), rcond=None)[0]
        per_fiber_constant = sol[:n_outputs]
        c4, c2, c0 = np.polyfit(y_coords, per_fiber_constant, 2)
        c1, c3, c5 = sol[n_outputs:]
        coeffs = np.array([c0, c1, c2, c3, c4, c5])
        fiber_offsets = per_fiber_constant - np.polyval([c4, c2, c0], y_coords)
    else:
        fiber_offsets = np.asarray(fiber_offsets, dtype=float)
        if fiber_offsets.shape != (n_outputs,):
            raise ValueError(f"fiber_offsets has shape {fiber_offsets.shape}, "
                             f"expected ({n_outputs},) (one value per output)")
        target = aberations - fiber_offsets[:, None]
        A = np.column_stack([np.ones(X.size), X.ravel(), Y.ravel(), (X*Y).ravel(),
                             (Y**2).ravel(), (X*Y**2).ravel()])
        coeffs = np.linalg.lstsq(A, target.ravel(), rcond=None)[0]

    aberations_fit = smooth_model(coeffs, X, Y) + fiber_offsets[:, None]
    
    fig = runlib_plots.plot_wavefit_coeffs(peaks_all_sub, peaks_all_sub_good, aberations, aberations_fit)

    fig_offsets, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.bar(y_coords, fiber_offsets, color='C0')
    ax.axhline(0, color='k', lw=0.8)
    ax.set_xlabel('PL output')
    ax.set_ylabel('Fiber offset dx (pixels)')
    residual_rms = np.std(aberations - aberations_fit)
    ax.set_title(f'Per-fiber x offsets (V-groove misalignment), '
                 f'residual rms of aberration fit = {residual_rms:.3f} px')
    fig_offsets.tight_layout()

    ref_pixels_lines = (peaks_all_sub_good - aberations_fit).mean(axis=0)

    # Apply aberration model to full detector
    X_full, Y_full = np.meshgrid(np.arange(neon.shape[1]), np.arange(neon.shape[0]))
    aberated_image = smooth_model(coeffs, X_full, Y_full) + fiber_offsets[:, None]

    return ref_pixels_lines, aberated_image, coeffs, fiber_offsets, fig


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
        Boolean array marking outliers (largest absolute residuals)
    duplicate_mask : array_like
        Boolean array marking peaks excluded from the fit (ambiguous catalog
        assignments, plus first and last peaks)
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
        duplicate_mask = np.ones_like(idx, dtype=bool)
    else:
        # Fit second-order polynomial
        x = neon_wavelengths[idx]
        y = ref_pixels_lines
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RankWarning)
            coeffs_poly = np.polyfit(x[~duplicate_mask], y[~duplicate_mask], 2)
        p2w = np.poly1d(coeffs_poly)
        residuals = y - p2w(x)
        # Exclude the Nexclude largest |residuals| (both signs); Nexclude=0 keeps all
        bad_idx = np.zeros(len(residuals), dtype=bool)
        if Nexclude > 0:
            bad_idx[np.argsort(np.abs(residuals))[-Nexclude:]] = True
        rms = np.std(residuals[~bad_idx])
    
    return rms, idx, bad_idx, duplicate_mask


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
        Number of lines with the largest absolute residuals to exclude from
        the fit and from the RMS
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
    rms_nm : float
        RMS of the final fit residuals (catalog minus fitted wavelength), in nm
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
            rms, idx, bad_idx, duplicate_mask = fit_and_score(
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
                best_valid_idx = ~bad_idx & ~duplicate_mask

    # Final polynomial fit with best parameters
    best_idx = np.asarray(best_idx)
    y = neon_wavelengths[best_idx][best_valid_idx]
    x = ref_pixels_lines[best_valid_idx]
    
    coeffs_poly = np.polyfit(x, y, 2)

    # Residuals of the wavelength solution, in nanometers
    residuals_nm = y - np.poly1d(coeffs_poly)(x)
    rms_nm = np.std(residuals_nm)

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

    return wave_1D_mapping, coeffs_poly, fig, rms_nm


def compute_interpolation_kernel(position, interpolation='lanczos3'):
    """
    Interpolation indices and weights to sample data at fractional pixel positions.

    Parameters
    ----------
    position : numpy.ndarray
        Fractional raw-pixel positions to sample, shape (n_outputs, n_samples)
    interpolation : {'lanczos3', 'linear'}
        'linear' : 2 taps, weights (1-f, f). Broadens the line profile by
                   f(1-f) px^2, which depends on the fractional shift f.
        'lanczos3' : 6 taps, windowed sinc sinc(d)*sinc(d/3), normalised to a
                   sum of 1. Preserves the line profile for (near) Nyquist-
                   sampled spectra, at the cost of small negative lobes.
        At integer positions both reduce to a single tap of weight 1 (all
        other weights are exactly 0).

    Returns
    -------
    index : numpy.ndarray of int, shape (n_taps, n_outputs, n_samples)
    weights : numpy.ndarray of float, shape (n_taps, n_outputs, n_samples)
    """
    floor = np.floor(position).astype(int)
    if interpolation == 'linear':
        frac = position - floor
        index = np.array((floor, floor + 1))
        weights = np.array((1.0 - frac, frac))
    elif interpolation == 'lanczos3':
        a = 3
        offsets = np.arange(-a + 1, a + 1)                        # -2 .. 3
        index = floor[None] + offsets[:, None, None]
        d = position[None] - index                                 # in ]-3, 3[
        weights = np.sinc(d) * np.sinc(d / a)
        weights /= weights.sum(axis=0, keepdims=True)
    else:
        raise ValueError(f"Unknown interpolation {interpolation!r} (use 'lanczos3' or 'linear')")
    return index, weights


def find_wollaston_modes(file_patterns):
    """
    Wollaston modes (X_FIRWOL values, e.g. ['IN', 'OUT']) present among the
    preprocessed Neon (COMPARISON) files matching `file_patterns`.
    """
    from first_pipeline_shared.classes.runPL_class_fileList import get_filelist
    from astropy.io import fits
    files = get_filelist(file_patterns, {'DATA-TYP': ['COMPARISON'], 'X_FIRTYP': ['PREPROC']},
                         name_search="neon")
    return sorted({fits.getheader(f).get('X_FIRWOL', 'UNKNOWN') for f in files})


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


def read_fiber_offsets(wavemap_file):
    """
    Read the per-fiber x offsets (Q_WMFnnn keywords) from a wavelength map file.

    Parameters
    ----------
    wavemap_file : str
        Path to a wavelength map FITS file created by createWaveMap

    Returns
    -------
    numpy.ndarray
        1D array of per-fiber offsets in pixels
    """
    from astropy.io import fits
    header = fits.getheader(wavemap_file)
    n_outputs = header.get('Q_WMNFO')
    if n_outputs is None:
        raise KeyError(f"No fiber offsets (Q_WMNFO keyword) in {wavemap_file}")
    return np.array([header['Q_WMF%03d' % k] for k in range(n_outputs)])


def save_wavelength_map(waveMap, header, coef_1d, coef_2d, output_dir,
                        fiber_offsets=None, fiber_offsets_source=None):
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
        2D aberration correction coefficients [c0, c1, c2, c3, c4, c5]
    output_dir : str
        Output directory path for saving files
    fiber_offsets : array_like, optional
        Per-fiber x offsets in pixels, saved as Q_WMFnnn keywords
    fiber_offsets_source : str, optional
        Reference file the offsets were taken from (None if they were fitted)
        
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
    header['Q_WM2XY2'] = (coef_2d[5], 'Aberrations X*Y2')
    if fiber_offsets is not None:
        header['Q_WMNFO'] = (len(fiber_offsets), 'number of per-fiber x offsets')
        header['Q_WMFOFI'] = (fiber_offsets_source is None, 'fiber offsets fitted (T) or fixed (F)')
        if fiber_offsets_source is not None:
            header['Q_WMFOSR'] = (os.path.basename(fiber_offsets_source), 'fiber offsets reference file')
        for k, dx in enumerate(fiber_offsets):
            header['Q_WMF%03d' % k] = (float(dx), 'x offset of fiber %d (pixel)' % k)
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
        2D aberration coefficients [c0, c1, c2, c3, c4, c5]
        
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
    Dpixel(x,y) = c0 + c1*x + c2*y + c3*xy + c4*y^2 + c5*xy^2 + dx[y]   (dx: per-fiber offsets)

    c0 = {coef_2d[0]:.6e}
    c1 = {coef_2d[1]:.6e}
    c2 = {coef_2d[2]:.6e}
    c3 = {coef_2d[3]:.6e}
    c4 = {coef_2d[4]:.6e}
    c5 = {coef_2d[5]:.6e}"""

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
                               wollaston=None, Nexclude=None, fiber_offsets_file=None,
                               vacuum=False, interpolation='lanczos3'):
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
        Number of detected lines with the largest absolute residuals to exclude
        from the fit (the CLI default is 4)
    fiber_offsets_file : str, optional
        Reference wavelength map whose per-fiber x offsets are used as fixed
        values. If None (default), the offsets are fitted on the Neon data.
    vacuum : bool, optional
        If False (default), the wavelength scale is in standard air (like the
        Neon catalogue). If True, the catalogue is converted to vacuum first,
        so the wavelength map is in vacuum.
    interpolation : {'lanczos3', 'linear'}, optional
        Resampling kernel stored in the map (default 'lanczos3'). Linear
        interpolation broadens the line profile by f(1-f) px^2 (f = fractional
        shift), i.e. differently for each output; Lanczos-3 preserves it.
        
    Returns
    -------
    waveMap : WaveMap
        Wavelength map object (wave, index, weights), already saved to
        ``<data dir>/../wavemaps/``; its path is in ``waveMap.filename``.
        The 1D and 2D calibration coefficients are stored in the FITS header
        (Q_WM1D*, Q_WM2D* keywords), and the diagnostic figures in a PDF next
        to it.
    datalist : list of DataCube
        The dark-subtracted (and flat-fielded, if a flat map was found) Neon
        data cubes used for the calibration.
    residual_rms_nm : float
        RMS of the residuals of the final pixel-to-wavelength polynomial fit,
        in nm, over the lines used in the fit.
    """

    # Set up file patterns
    if dark_patterns is None:
        dark_patterns = file_patterns
    if flat_patterns is None and file_patterns:
        folder = os.path.dirname(file_patterns[0])
        flat_patterns = file_patterns + [os.path.join(folder, "../flatmaps")]

    # The two Wollaston modes have different numbers of outputs: never mix them
    if wollaston is None:
        modes = find_wollaston_modes(file_patterns)
        if len(modes) > 1:
            raise ValueError(f"Neon files with several Wollaston modes {modes} were found: "
                             f"choose one with wollaston='IN' or 'OUT' (the command line "
                             f"processes each mode in turn when --wollaston is not given)")

    # Get file list and flat map
    fileList, flatMap = get_filelist_wave(file_patterns, dark_patterns, flat_patterns, wollaston)
    
    # Extract data
    datalist: List[DataCube] = fileList.extract_data_from_list(flatMap=flatMap)

    # Calculate optical aberration mapping
    neon = np.array([np.nanmean(d.data, axis=(0,1)) for d in datalist]).sum(axis=0)
    fiber_offsets = read_fiber_offsets(fiber_offsets_file) if fiber_offsets_file else None
    ref_pixels_lines, aberated_image, coef_2d, fiber_offsets, fig_aberations = \
        calculate_pixel_peaks_and_aberations(neon, fiber_offsets=fiber_offsets)

    # Calculate 1D wavelength mapping
    catalog = air_to_vacuum(neon_wavelengths_air) if vacuum else neon_wavelengths_air
    wave_1D_mapping, coef_1d, fig_1d_mapping, residual_rms_nm = calculate_the_pixel_to_wavelength_mapping(
        ref_pixels_lines, catalog, Nexclude, neon)

    # Compute final 2D wavelength map
    index_pixel_2d_float = np.arange(neon.shape[1]) + aberated_image

    index, weights = compute_interpolation_kernel(index_pixel_2d_float, interpolation)
    good_index = (index.min(axis=(0,1)) >= 0) & (index.max(axis=(0,1)) < neon.shape[1])
    index = index[:,:,good_index]
    weights = weights[:,:,good_index]
    wave = wave_1D_mapping[good_index]

    # Create WaveMap object
    waveMap = WaveMap()
    waveMap.create_from_data(wave, index, weights, medium='vacuum' if vacuum else 'air',
                             npixel=neon.shape[1], interpolation=interpolation)

    # Set up output directory
    header = datalist[-1].header
    folder = fileList.get_most_common_dir()
    output_dir = os.path.join(folder, "../wavemaps")

    # Save wavelength map
    output_filename = save_wavelength_map(waveMap, header, coef_1d, coef_2d, output_dir,
                                          fiber_offsets=fiber_offsets,
                                          fiber_offsets_source=fiber_offsets_file)

    # Create coefficient plot
    fig_coeffs = create_coefficients_plot(coef_1d, coef_2d)
    
    # Save all plots
    runlib_plots.save_pdf_in_file(output_filename)

    return waveMap, datalist, residual_rms_nm


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
        file_patterns = ["/Users/slacour/DATA/FIRST/20260609/preproc"]
        
        print(f"Development override: dark_patterns={dark_patterns}, flat_patterns={flat_patterns}, wollaston={wollaston}, Nexclude={Nexclude}")
        print(f"Development file patterns: {file_patterns}")

    # Process wavelength map data
    waveMap, datalist, residual_rms_nm = run_createWaveMap(
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
