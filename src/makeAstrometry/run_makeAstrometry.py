#%%
"""
FIRST Pipeline - Astrometric Analysis Core Algorithms

Core functions for performing precise astrometric measurements from preprocessed FIRST data.
Separated from CLI interface to enable interactive use in VS Code and notebooks.

Created on Wed May 21 22:56:25 2025
@author: slacour
"""

import sys
import os
# Add src directory to path for imports to work in both interactive and package contexts
if os.path.join(os.path.dirname(__file__), '..') not in sys.path:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from scipy.ndimage import convolve1d
from scipy.constants import speed_of_light
from typing import List, Tuple

import getpass
import warnings
import matplotlib
if "VSCODE_PID" in os.environ:
    matplotlib.use('macosx')
elif os.environ.get('SPYDER_DEBUG_FILE'):
    print("Running in Spyder")
else:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.pyplot import plot, hist, clf, figure, legend, imshow
plt.ion()

from tqdm import tqdm
from astroplan import Observer
from astropy.time import Time
from astropy.io import fits

from first_pipeline_shared.classes.runPL_class_flatMap import FlatMap  
from first_pipeline_shared.classes.runPL_class_waveMap import WaveMap
from first_pipeline_shared.classes.runPL_class_fileList import FileList
from first_pipeline_shared.classes.runPL_class_dataCube import DataCube
from first_pipeline_shared.classes.runPL_class_couplingMap import CouplingMap

from first_pipeline_shared.libraries import runPL_library_io as runlib_io
from first_pipeline_shared.libraries import runPL_library_plots as runlib_plots
from first_pipeline_shared.libraries import runPL_library_linalg as runlib_linalg
from makeAstrometry.astrometry_plots import plot_correlation_lag_histogram
from makeAstrometry.astrometry_plots import plot_jacobian_diagnostics


# Subaru Observatory instance for timing
subaru = Observer.at_site("Subaru")


def get_filelist_astrometry(file_patterns, dark_patterns=None, flat_patterns=None, 
                         wave_patterns=None, object_name=None, modID=None, 
                         modScale=None, wollaston=None):
    """
    Create file list for astrometry analysis with calibration associations.
    
    Parameters
    ----------
    file_patterns : list
        List of file patterns to search for OBJECT data
    dark_patterns : list, optional
        List of patterns for dark files
    flat_patterns : list, optional
        List of patterns for flat field files
    wave_patterns : list, optional
        List of patterns for wavelength map files
    object_name : str, optional
        Filter by object name
    modID : int or list, optional
        Modulation pattern ID(s)
    modScale : int, optional
        Modulation scale
    wollaston : str, optional
        Wollaston polarizer status
        
    Returns
    -------
    fileList : FileList
        Configured file list object
    flatMap : FlatMap or None
        Flat field map object
    waveMap : WaveMap or None
        Wavelength map object
    """
    # Set default modID if not provided
    if modID is None:
        modID = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

    # Create initial file list
    # fileList = FileList(file_patterns, data_type="OBJECT", first_type='PREPROC', 
    #                    wollaston=wollaston, object_name=object_name, 
    #                    modID=modID, modScale=modScale)
    fileList = FileList(file_patterns, data_type="OBJECT", first_type='PREPROC', 
                       wollaston=wollaston, object_name=object_name, 
                       modID=modID, modScale=modScale)

    # Get constraints from the dataset
    object_name = fileList.header.get('OBJECT', None)
    wollaston = fileList.header.get('X_FIRWOL', None)
    modID = fileList.header.get('X_FIRMID', None)
    modScale = fileList.header.get('X_FIRMSC', None)

    # Recreate with constraints
    fileList = FileList(file_patterns, data_type="OBJECT", first_type='PREPROC',
                       wollaston=wollaston, object_name=object_name, 
                       modID=modID, modScale=modScale)

    # Set up associations and maps
    fileList.make_association(dark_patterns=dark_patterns)
    file_flat = fileList.get_flatmap_file(flat_patterns)
    file_wave = fileList.get_wavemap_file(wave_patterns)

    flatMap = FlatMap(file_flat) if file_flat is not None else None
    waveMap = WaveMap(file_wave) if file_wave is not None else None

    return fileList, flatMap, waveMap, object_name

def check_observatory_status():
    """
    Check if it's currently night at Subaru Observatory.
    
    Returns
    -------
    str
        Status message about observatory conditions
    """
    now_time = Time.now()
    if subaru.is_night(now_time):
        return "It's night at Subaru Observatory."
    else:
        return "It's day at Subaru Observatory."


def plot_correlation_lag_histogram(data_corr_lag, good_data_flux,
                                   threshold_corr=0.5):
    """Plot lag-1 correlations between adjacent modulation steps.

    Parameters
    ----------
    data_corr_lag : ndarray, shape (Ncube, Nmod - 1)
        Correlations between adjacent modulation steps.
    good_data_flux : ndarray, shape (Ncube, Nmod)
        Flux-quality mask used to identify rejected adjacent pairs.
    threshold_corr : float, optional
        Minimum accepted adjacent-step correlation.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Created histogram figure.
    ax : matplotlib.axes.Axes
        Histogram axes.
    """
    correlation_values = data_corr_lag[np.isfinite(data_corr_lag)]
    rejected_flux_mask = ~(good_data_flux[:, 1:] & good_data_flux[:, :-1])
    rejected_correlation_values = data_corr_lag[
        rejected_flux_mask & np.isfinite(data_corr_lag)]

    below_threshold_percent = 100 * np.mean(correlation_values < threshold_corr)
    percentile_levels = np.array([5, 16, 50, 84, 95])
    correlation_percentiles = np.percentile(correlation_values, percentile_levels)
    fig, ax = plt.subplots(1, 1, figsize=(8, 6),
                           num="correlation_lag_histogram", clear=True)
    bin_edges = np.linspace(0, 1, 21)
    ax.hist(correlation_values, bins=bin_edges, color="steelblue", edgecolor="white",
            alpha=0.7, label="All data")
    if rejected_correlation_values.size:
        ax.hist(rejected_correlation_values, bins=bin_edges, color="tomato",
                edgecolor="white", alpha=0.7, label="Rejected by flux filter")
    ax.axvline(threshold_corr, color="goldenrod", linestyle="-", linewidth=2,
               label=f"Threshold: {threshold_corr:.2f} ({below_threshold_percent:.1f}% below)")
    for percentile, value in zip(percentile_levels, correlation_percentiles):
        ax.axvline(value, color="black", linestyle="--", linewidth=1,
                   label=f"P{percentile:g}: {value:.3f}")
    ax.set_xlabel("Correlation between adjacent modulation steps")
    ax.set_ylabel("Count")
    ax.set_title("Adjacent-step correlation across all cubes")
    ax.set_xlim(0, 1)
    ax.legend()
    return fig, ax
    

# def compute_smoothed_data(data_normalized, x_hanning):
#     """Compute the Hanning-smoothed (continuum) data cube for a given window size."""
#     Nwave = data_normalized.shape[-1]
#     x_zeros = x_hanning // 2
#     Nhanning = x_hanning * 2 + 1
#     hanning_window = np.hanning(Nhanning)
#     hanning_window[x_hanning - x_zeros:x_hanning + 1 + x_zeros] = 0
#     hanning_window /= hanning_window.sum()  # Normalize the window
#     # Edge normalization: weights overlapping the valid (zero-padded) region
#     edge_norm = np.convolve(np.ones(Nwave), hanning_window, mode='same')
#     # Vectorized convolution along the wavelength axis (zero-padded boundary,
#     # equivalent to np.convolve mode='same'); the symmetric odd window keeps
#     # the kernel centered.
#     data_smoothed = convolve1d(data_normalized, hanning_window, axis=-1,
#                                 mode='constant', cval=0.0)
#     data_smoothed = data_smoothed / edge_norm
#     return data_smoothed, hanning_window


def line_fit_region(line_aera):
    """Indices used to fit and evaluate the polynomial continuum under a line.

    The continuum is fitted on two side windows (`cont_idx`, each as wide as the
    line) and evaluated over the full boolean mask `fit_aera`, spanning from
    the left window start to the right window stop (i.e. the line plus both
    side windows).
    """
    line_idx = np.where(line_aera)[0]
    i0, i1 = line_idx[0], line_idx[-1]
    n_line = i1 - i0 + 1
    left = slice(max(i0 - n_line, 0), i0)
    right = slice(i1 + 1, i1 + 1 + n_line)
    cont_idx = np.r_[np.arange(left.start, left.stop),
                     np.arange(right.start, right.stop)]
    fit_aera = np.zeros_like(line_aera, dtype=bool)
    fit_aera[left.start:right.stop] = True
    return cont_idx, fit_aera


def compute_smoothed_line(data_b, wave_aera, fit_aera, poly_deg):
    """Estimate the continuum under a line with a low-order polynomial fit.

    The continuum is fitted on two side windows (each as wide as the line) on
    either side of the line and evaluated over the full span from the left to
    the right window (`fit_aera`). The wavelength axis may be last (data
    blocks) or penultimate (Jacobian blocks with a trailing RA/DEC axis).
    Inputs are already restricted to ``work_aera`` along the wavelength axis.
    """
    Nwork= wave_aera.size 
    wavelength_axis = -1 if data_b.shape[-1] == Nwork else -2
    if data_b.shape[wavelength_axis] != Nwork:
        raise ValueError("data_b has no axis matching the wavelength grid")

    data_by_wavelength = np.moveaxis(data_b, wavelength_axis, -1)
    y_cont = data_by_wavelength[..., fit_aera]
    x_cont = wave_aera[fit_aera]
    cont_shape = y_cont.shape[:-1]
    coeffs = np.polyfit(x_cont,
                        y_cont.reshape(-1, sum(fit_aera)).T, poly_deg)  # (poly_deg+1, Nseries)
    # Evaluate the polynomial continuum across the full left->right span
    V_line = np.vander(wave_aera, poly_deg + 1)                 # (n_fit, poly_deg+1)
    data_smoothed_line = (V_line @ coeffs).T.reshape(*cont_shape, -1)  # (..., n_fit)
    return np.moveaxis(data_smoothed_line, -1, wavelength_axis)


# def compute_smoothed_line_uncertainty(var_data, wave_aera, fit_aera, poly_deg):
#     """Propagate data variance through the unweighted polynomial continuum fit.

#     The fit is a linear operation with a polynomial smoothing matrix. This
#     returns its diagonal output variance and the same-wavelength covariance
#     with the input data. The input wavelength axis must be last and already
#     be restricted to ``work_aera``.
#     """
#     fit_in_work = fit_aera
#     x_cont = wave_aera[fit_aera]
#     x_work = wave_aera
#     V_cont = np.vander(x_cont, poly_deg + 1)
#     V_work = np.vander(x_work, poly_deg + 1)
#     smoothing_matrix = V_work @ np.linalg.pinv(V_cont)

#     var_sm = np.einsum('wc,...c->...w', smoothing_matrix ** 2,
#                        var_data[..., fit_in_work])
#     cov_ds = np.zeros_like(var_data)
#     cont_positions = np.flatnonzero(fit_in_work)
#     cov_ds[..., cont_positions] = (
#         var_data[..., cont_positions]
#         * smoothing_matrix[cont_positions, np.arange(cont_positions.size)])
#     return var_sm, cov_ds


def compute_smoothed_jacobian_uncertainty(C_J, wave_aera, fit_aera,
                                          poly_deg):
    """Propagate per-wavelength Jacobian covariance through continuum fitting.

    ``C_J`` contains the two-by-two RA/DEC covariance at each wavelength;
    inter-wavelength noise correlations are assumed negligible.
    """
    fit_in_work = fit_aera
    V_cont = np.vander(wave_aera[fit_aera], poly_deg + 1)
    V_work = np.vander(wave_aera, poly_deg + 1)
    smoothing_matrix = V_work @ np.linalg.pinv(V_cont)
    return np.einsum('wc,...cij->...wij', smoothing_matrix ** 2,
                     C_J[..., fit_in_work, :, :])


def compute_adjacent_jacobian_photon_covariance(
        var_center, var_forward, sky_step_basis_inv):
    """Compute photon cross-covariance between neighboring Jacobian blocks."""
    cross_diff_covariance = np.zeros(
        (*var_center[:, :-1].shape, 2, 2), dtype=float)
    cross_diff_covariance[..., 0, 0] = -var_forward[:, :-1]
    cross_diff_covariance[..., 0, 1] = -(
        var_center[:, :-1] + var_forward[:, :-1])
    cross_diff_covariance[..., 1, 1] = -var_center[:, :-1]
    return np.einsum(
        'cmji,cmowjk,cmkl->cmowil',
        sky_step_basis_inv[:, :-1], cross_diff_covariance,
        sky_step_basis_inv[:, 1:])


def compute_smoothed_jacobian_cross_covariance(
        cross_covariance, wave_aera, fit_aera, poly_deg):
    """Propagate neighboring-block Jacobian covariance through smoothing."""
    smoothing_matrix = np.vander(wave_aera, poly_deg + 1) @ np.linalg.pinv(
        np.vander(wave_aera[fit_aera], poly_deg + 1))
    return np.einsum('wc,...cij->...wij', smoothing_matrix ** 2,
                     cross_covariance[..., fit_aera, :, :])


def compute_smoothed_cross_covariances(cov_data_J, wave_aera, fit_aera,
                                       poly_deg_sm, poly_deg_J):
    """Return ``Cov(data, Jm)`` and ``Cov(sm, Jm)`` after continuum fits.

    The first term retains covariance only where the raw data contributes to
    the Jacobian fit. The second applies the data and Jacobian polynomial
    smoothing matrices to their shared measurement covariance.
    """
    fit_in_work = fit_aera
    x_cont = wave_aera[fit_aera]
    x_work = wave_aera
    smoothing_sm = np.vander(x_work, poly_deg_sm + 1) @ np.linalg.pinv(
        np.vander(x_cont, poly_deg_sm + 1))
    smoothing_J = np.vander(x_work, poly_deg_J + 1) @ np.linalg.pinv(
        np.vander(x_cont, poly_deg_J + 1))
    cov_data_Jm = np.zeros_like(cov_data_J)
    cont_positions = np.flatnonzero(fit_in_work)
    cov_data_Jm[..., cont_positions, :] = (
        cov_data_J[..., cont_positions, :]
        * smoothing_J[cont_positions, np.arange(cont_positions.size), None])
    cov_sm_Jm = np.einsum('wc,wc,...ci->...wi', smoothing_sm, smoothing_J,
                          cov_data_J[..., fit_in_work, :])
    return cov_data_Jm, cov_sm_Jm


def average_jacobian_over_nearest_cubes(jacobian_smoothed,
                                        jacobian_smoothed_covariance,
                                        n_cubes):
    """Average each Jacobian over its nearest odd number of cubes."""
    if n_cubes < 1 or n_cubes % 2 == 0:
        raise ValueError("n_cubes must be a positive odd integer")
    n_available = jacobian_smoothed.shape[0]
    if n_cubes > n_available:
        raise ValueError("n_cubes cannot exceed the number of cubes")

    half_window = n_cubes // 2
    cube_index = np.arange(n_available)
    window_start = np.maximum(cube_index - half_window, 0)
    window_end = np.minimum(cube_index + half_window + 1, n_available)

    jacobian_finite = np.isfinite(jacobian_smoothed)
    jacobian_values = np.where(jacobian_finite, jacobian_smoothed, 0.0)
    jacobian_count = jacobian_finite.astype(float)
    jacobian_sum = np.concatenate([
        np.zeros_like(jacobian_values[:1]),
        np.cumsum(jacobian_values, axis=0)])
    count_sum = np.concatenate([
        np.zeros_like(jacobian_count[:1]),
        np.cumsum(jacobian_count, axis=0)])
    jacobian_window_sum = jacobian_sum[window_end] - jacobian_sum[window_start]
    jacobian_window_count = count_sum[window_end] - count_sum[window_start]
    jacobian_average = jacobian_window_sum / np.maximum(jacobian_window_count, 1.0)

    covariance_finite = np.isfinite(jacobian_smoothed_covariance)
    covariance_values = np.where(
        covariance_finite, jacobian_smoothed_covariance, 0.0)
    covariance_count = covariance_finite.astype(float)
    covariance_sum = np.concatenate([
        np.zeros_like(covariance_values[:1]),
        np.cumsum(covariance_values, axis=0)])
    covariance_count_sum = np.concatenate([
        np.zeros_like(covariance_count[:1]),
        np.cumsum(covariance_count, axis=0)])
    covariance_window_sum = (
        covariance_sum[window_end] - covariance_sum[window_start])
    covariance_window_count = (
        covariance_count_sum[window_end] - covariance_count_sum[window_start])
    covariance_average = covariance_window_sum / np.maximum(
        covariance_window_count**2, 1.0)
    return jacobian_average, covariance_average


def validate_ncube_average(n_cubes, n_available=None):
    """Validate the cube-averaging window and fall back to one cube."""
    valid = (isinstance(n_cubes, (int, np.integer))
             and not isinstance(n_cubes, (bool, np.bool_))
             and n_cubes >= 1 and n_cubes % 2 == 1)
    if n_available is not None:
        valid = valid and n_cubes <= n_available
    if not valid:
        limit = ""
        if n_available is not None:
            limit = f" and no larger than the {n_available} available cubes"
        warnings.warn(
            f"Ncube_average={n_cubes!r} is invalid; it must be a positive "
            f"odd integer{limit}. Falling back to Ncube_average=1.",
            UserWarning,
            stacklevel=2)
        return 1
    return int(n_cubes)


def estimate_jacobian_systematic_variance(jacobian_smoothed,
                                          jacobian_smoothed_covariance,
                                          valid_basis=None):
    """Estimate systematic Jacobian variance after removing photon noise.

    The cube-to-cube scatter is measured after the wavelength smoothing. The
    expected photon variance of each cube difference is subtracted using the
    already propagated Jacobian covariance. For independent cubes with equal
    variance, ``Var(J[i+1] - J[i]) = 2 Var(J)``. The returned variance has
    shape ``(Ncube, Noutput)`` after averaging over blocks, wavelengths, and
    the two Jacobian components.
    """
    cube_difference = np.diff(jacobian_smoothed, axis=0)
    if valid_basis is not None:
        valid_cube_pair = valid_basis[:-1] & valid_basis[1:]
        cube_difference = np.where(
            valid_cube_pair[:, :, None, None, None], cube_difference, np.nan)
    difference_variance = np.nanvar(cube_difference, axis=(1,-1))
    measured_difference_variance = np.nanmean(difference_variance, axis=(-1))

    photon_difference_covariance = (
        jacobian_smoothed_covariance[:-1]
        + jacobian_smoothed_covariance[1:])
    valid_cube_pair = None
    if valid_basis is not None:
        valid_cube_pair = valid_basis[:-1] & valid_basis[1:]
        photon_difference_covariance = np.where(
            valid_cube_pair[:, :, None, None, None, None],
            photon_difference_covariance, np.nan)
    photon_difference_diagonal = np.diagonal(
        photon_difference_covariance, axis1=-2, axis2=-1)
    invalid_photon_values = ~np.isfinite(photon_difference_diagonal)
    if invalid_photon_values.any():
        print(f"* Warning: ignoring {invalid_photon_values.sum()} non-finite "
              "Jacobian covariance values in photon-noise estimate")
    photon_difference_diagonal = np.where(
        invalid_photon_values, np.nan, photon_difference_diagonal)
    photon_difference_variance = np.nanmean(
        photon_difference_diagonal, axis=(1, 3, 4))
    invalid_photon_variance = ~np.isfinite(photon_difference_variance)
    if invalid_photon_variance.any():
        print(f"* Warning: no finite photon-noise estimate for "
              f"{invalid_photon_variance.sum()} cube/output entries; "
              "using zero subtraction")
        photon_difference_variance = np.nan_to_num(
            photon_difference_variance, nan=0.0, posinf=0.0, neginf=0.0)

    systematic_difference_variance = (
        measured_difference_variance - photon_difference_variance)
    systematic_difference_variance = np.maximum(
        systematic_difference_variance, 0.0) / 2.0

    systematic_variance = np.empty(
        (jacobian_smoothed.shape[0], jacobian_smoothed.shape[2]))
    systematic_variance[0] = systematic_difference_variance[0]
    systematic_variance[-1] = systematic_difference_variance[-1]
    systematic_variance[1:-1] = (
        systematic_difference_variance[:-1]
        + systematic_difference_variance[1:]) / 2.0
    return systematic_variance


def estimate_jacobian_systematic_variance_2(
    jacobian_smoothed, jacobian_smoothed_covariance,
    adjacent_photon_covariance=None, valid_basis=None):
    """Estimate systematic variance from neighboring modulation blocks.

    Differences are formed along axis 1. If supplied, ``adjacent_photon_covariance``
    has shape ``(Ncube, Nblock - 1, Noutput, Nwave, 2, 2)`` and contains
    ``Cov(J[:, b], J[:, b + 1])``. The photon covariance of a difference is
    then ``C_b + C_b1 - K_b - K_b.T``. Without it, neighboring photon errors
    are treated as independent.
    """
    block_difference = np.diff(jacobian_smoothed, axis=1)
    if valid_basis is not None:
        valid_block_pair = valid_basis[:, :-1] & valid_basis[:, 1:]
        block_difference = np.where(
            valid_block_pair[:, :, None, None, None], block_difference, np.nan)
    difference_variance = np.nanvar(block_difference, axis=(1,-1))
    measured_difference_variance = np.nanmean(difference_variance, axis=(-1))

    photon_difference_covariance = (
        jacobian_smoothed_covariance[:, :-1]
        + jacobian_smoothed_covariance[:, 1:])
    valid_block_pair = None
    if valid_basis is not None:
        valid_block_pair = valid_basis[:, :-1] & valid_basis[:, 1:]
        photon_difference_covariance = np.where(
            valid_block_pair[:, :, None, None, None, None],
            photon_difference_covariance, np.nan)
    if adjacent_photon_covariance is not None:
        if adjacent_photon_covariance.shape != photon_difference_covariance.shape:
            raise ValueError(
                "adjacent_photon_covariance must match the neighboring "
                "Jacobian covariance shape")
        photon_difference_covariance -= adjacent_photon_covariance
        photon_difference_covariance -= np.swapaxes(
            adjacent_photon_covariance, -1, -2)
    photon_difference_diagonal = np.diagonal(
        photon_difference_covariance, axis1=-2, axis2=-1)
    invalid_photon_values = ~np.isfinite(photon_difference_diagonal)
    if invalid_photon_values.any():
        print(f"* Warning: ignoring {invalid_photon_values.sum()} non-finite "
              "Jacobian covariance values in photon-noise estimate")
    photon_difference_diagonal = np.where(
        invalid_photon_values, np.nan, photon_difference_diagonal)
    photon_difference_variance = np.nanmean(
        photon_difference_diagonal, axis=(1, 3, 4))
    invalid_photon_variance = ~np.isfinite(photon_difference_variance)
    if invalid_photon_variance.any():
        print(f"* Warning: no finite photon-noise estimate for "
              f"{invalid_photon_variance.sum()} cube/output entries; "
              "using zero subtraction")
        photon_difference_variance = np.nan_to_num(
            photon_difference_variance, nan=0.0, posinf=0.0, neginf=0.0)

    systematic_difference_variance = (
        measured_difference_variance - photon_difference_variance)
    systematic_difference_variance = np.maximum(
        systematic_difference_variance, 0.0) 

    return systematic_difference_variance


def diagnose_J(J, data, C_J):
    """Per-wavelength diagnostics of Jacobian SNR, geometric degeneracy, and error inflation.

    Returns
    -------
    snr_J : (Nwave,) Jacobian magnitude relative to its own error.
    r2    : (Nwave,) fraction of J retained after projecting out `data` (low
            means `a` and `flat` are nearly indistinguishable at that wavelength).
    lam   : (Nwave,) largest eigenvalue of ``M^-1 @ A``, the attenuation of the
            astrometric fit caused by Jacobian measurement error.
    """
    D2 = np.sum(data**2, axis=0)
    Gd = np.sum(data[..., None] * J, axis=0)
    J_proj = J - data[..., None] * (Gd / D2[..., None])[None]
    Pd = 1.0 - data**2 / D2[None]

    M = np.einsum('bowi,bowj->wij', J_proj, J_proj)
    A = np.einsum('bowij,bow->wij', C_J, Pd)

    snr_J = np.einsum('bowi,bowi->w', J, J) / np.einsum('bowii->w', C_J)
    r2    = np.einsum('bowi,bowi->w', J_proj, J_proj) / np.einsum('bowi,bowi->w', J, J)
    lam   = np.linalg.eigvalsh(np.linalg.solve(M, A)).max(axis=1)
    return snr_J, r2, lam


def compare_jacobian_covariance(J, C_J):
    """Compare normalized inter-block Jacobian scatter with normalized ``C_J``."""
    J_norm = np.linalg.norm(J, axis=-1, keepdims=True)
    J_normalized = J / (J_norm + 1e-30)
    empirical_variance = np.nanvar(J_normalized, axis=0).sum(axis=-1)
    normalized_covariance = C_J / (J_norm[..., None]**2 + 1e-30)
    covariance_trace = np.nanmean(
        np.trace(normalized_covariance, axis1=-2, axis2=-1), axis=0)
    variance_ratio = empirical_variance / (covariance_trace + 1e-30)
    return empirical_variance, covariance_trace, variance_ratio


def plot_jacobian_diagnostics(wave_aera, J, data, C_J,
                              smoothing_length, line_center=None,
                              line_width=None):
    """Plot Jacobian diagnostics for the selected cube-smoothing length."""
    fig, axes = plt.subplots(4, 1, figsize=(10, 14),
                             num="jacobian_diagnostics", clear=True,
                             sharex=True)
    snr_J, r2, lam = diagnose_J(J, data, C_J)
    _, _, variance_ratio = compare_jacobian_covariance(J, C_J)
    ratio = np.nanmedian(variance_ratio, axis=0)
    label = f"Ncube={smoothing_length}"
    axes[0].plot(wave_aera, snr_J, color='steelblue', label=label)
    axes[1].plot(wave_aera, r2, color='steelblue', label=label)
    axes[2].plot(wave_aera, np.max(lam, axis=-1) if lam.ndim > 1 else lam,
                 color='steelblue', label=label)
    axes[3].plot(wave_aera, ratio, color='steelblue', label=label)

    axes[0].axhline(1, color='k', linestyle=':', alpha=0.7, label="SNR = 1")
    axes[0].set_ylabel("Jacobian SNR")
    axes[0].set_yscale('log')
    axes[1].set_ylabel("R\u00b2 (J retained after data projection)")
    axes[1].set_ylim(0, 1.05)
    axes[2].set_ylabel("Max attenuation eigenvalue")
    axes[3].axhline(1/3, color='k', linestyle=':', alpha=0.7)
    axes[3].axhline(3, color='k', linestyle=':', alpha=0.7)
    axes[3].set_ylabel("empirical / C_J variance")
    axes[3].set_xlabel("Wavelength")

    if line_center is not None and line_width is not None:
        for ax in axes:
            ax.axvspan(line_center - line_width/2, line_center + line_width/2,
                       color='gray', alpha=0.2)
            ax.axvline(line_center, color='black', linewidth=1)

        line_mask = np.abs(wave_aera - line_center) <= line_width/2
        if not line_mask.any():
            line_mask = slice(None)
    else:
        line_mask = slice(None)

    axes[0].legend(fontsize=8)
    axes[0].set_title("Jacobian diagnostics for cube smoothing lengths")
    return fig, axes


def solve_eiv_J(J, data, sm, C_J, cov_Jsm, var_data=None):
    """Correct the projected astrometry fit for Jacobian measurement error.

    The direct-flat model is ``J @ a = data * flat - sm``. This method keeps
    the data and continuum values fixed, and corrects only the uncertainty in
    ``J`` and its covariance with ``sm``. It uses the diagonal-in-block
    covariance approximation for the projection onto the complement of
    ``data``.

    Parameters
    ----------
    J : (Nblocks, Noutput, Nwave, 2)
        Measured response Jacobian.
    data, sm : (Nblocks, Noutput, Nwave)
        Measured data and continuum estimate.
    C_J : (Nblocks, Noutput, Nwave, 2, 2)
        Covariance of the Jacobian error.
    cov_Jsm : (Nblocks, Noutput, Nwave, 2)
        Covariance between Jacobian error and continuum error.
    var_data : (Nblocks, Noutput, Nwave), optional
        Variance of the measured data. When supplied, the returned covariance
        contains only the propagated contribution from data noise.

    Returns
    -------
    astrometry_shift : (Nwave, 2)
        Jacobian-error-corrected astrometric shift.
    flat : (Noutput, Nwave)
        Direct multiplicative gain.
    M_corrected : (Nwave, 2, 2)
        Normal matrix after subtracting the projected Jacobian covariance.
    attenuation : (Nwave, 2)
        Eigenvalues of ``M^-1 @ A``, a diagnostic for the size of the
        Jacobian-error correction.
    astrometry_covariance : (Nwave, 2, 2)
        With ``var_data`` supplied, this is the propagated data-noise
        covariance; otherwise it is ``inv(M_corrected)`` under unit residual
        variance.
    """
    D2 = np.sum(data ** 2, axis=0)
    Gd = np.sum(data[..., None] * J, axis=0)
    H = np.sum(data * sm, axis=0)

    J_proj = J - data[..., None] * (Gd / D2[..., None])[None]
    d_proj = data * (H / D2)[None] - sm
    M = np.einsum('bowi,bowj->wij', J_proj, J_proj)
    rhs = np.einsum('bowi,bow->wi', J_proj, d_proj)

    # For P = I - data data.T / D2, use diag(P) for block-diagonal
    # covariance. A is the projected Jacobian-error contribution and c is
    # the projected J-sm covariance contribution to the right-hand side.
    projected_data_diagonal = 1.0 - data ** 2 / D2[None]
    A = np.einsum('bowij,bow->wij', C_J, projected_data_diagonal)
    c = -np.einsum('bowi,bow->wi', cov_Jsm, projected_data_diagonal) 

    M_corrected = M - A
    astrometry_shift = np.linalg.solve(
        M_corrected, (rhs - c)[..., None])[..., 0]
    flat = (H + np.einsum('owi,wi->ow', Gd, astrometry_shift)) / D2

    #diagnostics:
    # r2 faible (≪ 1) et attenuation grand → dégénérescence géométrique. Votre J est bon, mais a et flat sont quasi indistinguables dans cette configuration de blocs. Aucun traitement statistique n'y remédiera ; il faut plus de diversité de blocs, ou contraindre flat par ailleurs.
    # r2 normal et attenuation grand → J réellement mal connu. Il faut améliorer la calibration.

    r2 = np.einsum('bowi,bowi->w', J_proj, J_proj) / np.einsum('bowi,bowi->w', J, J)
    attenuation = np.linalg.eigvals(np.linalg.solve(M, A)).real
    M_inverse = np.linalg.inv(M_corrected)
    if var_data is None:
        astrometry_covariance = M_inverse
    else:
        projected_data_variance = var_data * (
            1.0 - data ** 2 / D2[None])**2
        rhs_covariance = np.einsum(
            'bowi,bow,bowj->wij', J_proj,
            projected_data_variance, J_proj)
        astrometry_covariance = np.einsum(
            'wij,wjk,wlk->wil', M_inverse, rhs_covariance, M_inverse)
    return astrometry_shift, flat, M_corrected, attenuation, astrometry_covariance


def process_astrometric_data(
    file_patterns, object_name=None, dark_patterns=None, flat_patterns=None, wave_patterns=None, modID=None, modScale=None, wollaston=None,
    line_center=656.28, line_width= 3.0, PA=137.0,
    Ncube_average=1):
    """
    Measure the wavelength-dependent photocenter shift (spectro-astrometry).

    Equation being solved
    ----------------------
    For a source at sky position p = (alpha, delta), each lantern output flux
    in `data_normalized` is locally linear in p:

        data_normalized(p) ~= data_normalized(p_k) + jacobian (p - p_k)

    where `jacobian` = d(data_normalized)/dp in R^(Nout x 2).

    The known modulation dither provides, around each interior point k, two
    sky steps and the corresponding measured output differences:

        sky_step_basis = [ sky_step_fwd , sky_step_bwd ]       (2 x 2)
                        = [ p_{k+1}-p_k , p_{k-1}-p_k ]

        data_diff_basis = [ data_diff_fwd , data_diff_bwd ]    (Nout x 2)
                        = [ D_{k+1}-D_k , D_{k-1}-D_k ]

    Since data_diff_basis = jacobian @ sky_step_basis, the local response
    Jacobian is recovered by inverting the (known) dither geometry:

        jacobian = data_diff_basis @ sky_step_basis_inv

    (only well-conditioned, non-collinear bases are kept, via `valid_basis`).

    The signal of interest is a small, wavelength-dependent astrometric shift
    `astrometry_shift`(lambda) = (d_alpha(lambda), d_delta(lambda)) shared by
    all outputs, steps and exposures. In addition, each output o carries an
    unknown multiplicative gain `flat`[o] (shape Noutput x Nwave, close to 1)
    on the measured normalized data. The forward model relating both unknowns
    to the data is:

        jacobian @ astrometry_shift = data_normalized * flat - data_smoothed

    Here `astrometry_shift` (2 values per wavelength) is shared by every output
    and block, while `flat` (one value per output per wavelength) is shared by
    every block but free across outputs. The gain is already linear:

        data_normalized * flat = data_smoothed + jacobian @ astrometry_shift

    The system decouples per wavelength; at each wavelength the unknowns are
    `astrometry_shift` (2) plus `flat` (Noutput). Because `flat[o]` enters
    linearly and only in the rows of output o, it is eliminated analytically
    for any given `astrometry_shift` (separable / variable-projection least
    squares):
        flat[o] = sum_b data_b[o] (data_smoothed_b[o] + jacobian[o] @ astrometry_shift)
                  / sum_b data_b[o]^2

    Substituting the optimal `flat` back projects the per-block Jacobian and
    continuum onto the complement of the measured-data directions (`J_proj`,
    `d_proj`) and leaves a single 2x2 normal system per wavelength:

        M(lambda) @ astrometry_shift(lambda) = sum_{b,o} J_proj * d_proj
        M(lambda) = sum_{b,o} J_proj @ J_proj^T

    solved with `np.linalg.solve`. The two columns of `astrometry_shift` are
    the RA and DEC astrometric signals versus wavelength. The projection is
    determined by `data_normalized`, so the normal matrix `M` is independent
    of the continuum fit; the helper `solve_astrometry_gain` performs the
    per-continuum solve.

    Identifiability: a single output cannot separate astrometry from its own
    flat gain, but `flat`[o] is constant across the dither blocks whereas the
    astrometric response `jacobian` varies block to block, so several dither
    positions are required to break the degeneracy (and the extra Noutput free
    parameters per wavelength do inflate the astrometric noise).

    `PA` (degrees) is used for plotting only: it draws a reference position
    angle line on the astrometry_scatter figure and does not affect any of the
    computed results.

    """

    # Polynomial continuum-fit degrees tested under the line (hard-coded)
    poly_deg_values = (2, 3, 4, 5)
    Ncube_average = validate_ncube_average(Ncube_average)

    # Set up default patterns
    if dark_patterns is None:
        dark_patterns = file_patterns
    if flat_patterns is None and file_patterns:
        folder = os.path.dirname(file_patterns[0])
        flat_patterns = file_patterns + [os.path.join(folder, "../flatmaps")] + [os.path.join(folder, "flatmaps")]
    if wave_patterns is None and file_patterns:
        folder = os.path.dirname(file_patterns[0])
        wave_patterns = file_patterns + [os.path.join(folder, "../wavemaps")] + [os.path.join(folder, "wavemaps")]

    # Get file list and calibration maps
    fileList, flatMap, waveMap, object_name = get_filelist_astrometry(
        file_patterns, dark_patterns, flat_patterns, wave_patterns,
        object_name, modID, modScale, wollaston
    )

    # Extract data
    datalist: List[DataCube] = fileList.extract_data_from_list(
        flatMap=flatMap,
        waveMap=waveMap
    )

    # Concatenate data arrays
    flux = np.concatenate([d.flux for d in datalist])
    datacube = np.concatenate([d.data for d in datalist])
    datacube_var = np.concatenate([d.variance for d in datalist])
    Ncube_average = validate_ncube_average(Ncube_average, datacube.shape[0])
    wave = datalist[0].wave  # Assuming all have the same wavelength grid
    xmod = np.concatenate([d.xmod for d in datalist])
    ymod = np.concatenate([d.ymod for d in datalist])
    ra_dec = np.concatenate([d.compute_xy_sky() for d in datalist])

    ####################
    # to remove later.
    # flux = flux*0 + 1
    # datacube_var *= 0.000
    # datacube_var += 0.008*1e-6
    # datacube = np.random.normal(datacube*0 + 80, np.sqrt(datacube_var))
    # systematic_noise = np.random.normal(0, np.sqrt(0.4),size=(40,188,19,1))  # Add small noise to avoid singularities
    # datacube += systematic_noise

    # Create filename associations
    basenames = []
    for d in datalist:
        n = d.data.shape[0]
        basenames.extend([d.basename] * n)

    # Data quality filtering based on flux threshold 
    goodData_flux, _ = runlib_linalg.flux_filtering(flux)
    print(f"* Percentage of good data: {np.sum(goodData_flux)/len(goodData_flux.ravel())*100:.1f} % (flux threshold)")

    # Data quality filtering based on correlation between adjacent modulation steps
    threshold_corr = 0.5
    low_correlation_pair_mask, data_corr_lag = runlib_linalg.correlation_filtering(
        datacube, threshold_corr=threshold_corr)

    # Plot diagnostic plots for flux map and correlation lag histogram  
    fig = runlib_plots.plot_flux_map(flux.mean(axis=(2))[0], xmod[0], ymod[0])
    figures_to_save = [fig]
    fig, ax = plot_correlation_lag_histogram(
        data_corr_lag, goodData_flux, threshold_corr=threshold_corr)
    figures_to_save.append(fig)

    ##########################
    # Taking care if TT stepping function:
    ##########################

    # Known sky steps from each interior modulation point to its two neighbours
    sky_step_fwd = ra_dec[:,2:] - ra_dec[:,1:-1]    # p_{k+1} - p_k
    sky_step_bwd = ra_dec[:,:-2] - ra_dec[:,1:-1]   # p_{k-1} - p_k

    # 2x2 basis of known sky steps (columns are the two step vectors)
    sky_step_basis = np.stack([sky_step_fwd, sky_step_bwd], axis=-1)
    sky_step_basis_inv = np.linalg.pinv(sky_step_basis)

    # Keep only well-conditioned (non-collinear) bases and good-quality data
    sky_step_basis_det = np.linalg.det(sky_step_basis)
    valid_basis = np.abs(sky_step_basis_det) > np.max(np.abs(sky_step_basis_det)) * 1e-2
    valid_basis &= goodData_flux[:,2:] & goodData_flux[:,:-2] & goodData_flux[:,1:-1]
    # valid_basis &= ~low_correlation_pair_mask[:,1:] & ~low_correlation_pair_mask[:,:-1] 
    print(f"* Percentage of valid triangles: {np.sum(valid_basis)/len(valid_basis.ravel())*100:.1f} % (correlation + determinant threshold)")

    ##########################
    #starting calculations of the Jacobian around the line of interest
    ##########################

    # Speed of light in km/s (precise CODATA value)
    # Doppler velocity (km/s)
    c = speed_of_light / 1e3
    velocity = c * (wave - line_center) / line_center

    # Define the wavelength regions for the line, the working area, and the fitting area
    work_aera = (wave > line_center - line_width*1.5) & (wave < line_center + line_width*1.5)
    wave_aera = wave[work_aera]
    line_aera = (wave_aera > line_center - line_width/2) & (wave_aera < line_center + line_width/2)
    fit_aera = ~line_aera

    mean_flux = np.nanmean(flux, axis=(0,1))
    datacube_normalized = datacube[...,work_aera] / mean_flux[...,work_aera]
    with np.errstate(divide='ignore', invalid='ignore'):
        datacube_var_normalized = datacube_var[...,work_aera] / mean_flux[...,work_aera]**2
    invalid_variance = ~np.isfinite(datacube_var_normalized)
    if invalid_variance.any():
        invalid_flux = (~np.isfinite(mean_flux[...,work_aera]) |
                        (mean_flux[...,work_aera] == 0))
        invalid_raw_variance = ~np.isfinite(datacube_var[...,work_aera])
        print(f"* Warning: {invalid_variance.sum()} non-finite normalized variance "
              f"values ({invalid_flux.sum()} from invalid/zero mean flux, "
              f"{invalid_raw_variance.sum()} from raw variance)")
    flux_scaled = mean_flux[...,work_aera]/ np.nanmax(mean_flux[...,work_aera])

    # Measured output changes for the same forward/backward steps
    # data_diff_diff= np.diff(np.diff(datacube_normalized, axis=1), axis=0) 
    data_diff_fwd = datacube_normalized[:,2:] - datacube_normalized[:,1:-1]    # D_{k+1} - D_k
    data_diff_bwd = datacube_normalized[:,:-2] - datacube_normalized[:,1:-1]   # D_{k-1} - D_k
    data_diff_basis = np.stack([data_diff_fwd,data_diff_bwd], axis=-1)
    jacobian = np.einsum('cbowj,cbjk->cbowk', data_diff_basis, sky_step_basis_inv)
    jacobian_smoothed = compute_smoothed_line(jacobian, wave_aera, fit_aera, 1)


    ##########################
    #Computing the error on the Jacobian, using the measured variance of the data differences and the known sky step basis
    ##########################


    # Measured variance / covariance of the same forward/backward differences (shared D_k)
    var_backward = datacube_var_normalized[:,:-2]
    var_center = datacube_var_normalized[:,1:-1]
    var_forward = datacube_var_normalized[:,2:]
    diff_covariance = np.empty((*data_diff_basis.shape, 2))
    diff_covariance[..., 0, 0] = var_forward + var_center
    diff_covariance[..., 0, 1] = var_center
    diff_covariance[..., 1, 0] = var_center
    diff_covariance[..., 1, 1] = var_backward + var_center
    # Calculating the covariance matrix of the Jacobian using the inverse of the sky step basis and the covariance of the data differences
    jacobian_covariance = np.einsum(
                'cmji,cmowjk,cmkl->cmowil', sky_step_basis_inv,
                diff_covariance, sky_step_basis_inv)
    # Calculating the covariance of the Jacobian with respect to the data center step 
    data_jacobian_covariance = np.einsum(
                'cmowj,cmji->cmowi', np.stack([-var_center, -var_center], axis=-1),
                sky_step_basis_inv)
    adjacent_photon_covariance = compute_adjacent_jacobian_photon_covariance(
        var_center, var_forward, sky_step_basis_inv)

    # smoothing the Jacobian over the working region (but outside the line) to gain snr on it.
    # Using for the smoothing a first oder fit.
    jacobian_smoothed_covariance = compute_smoothed_jacobian_uncertainty(
        jacobian_covariance, wave_aera, fit_aera, 1)
    adjacent_photon_covariance = compute_smoothed_jacobian_cross_covariance(
        adjacent_photon_covariance, wave_aera, fit_aera, 1)

    #estimating tip/tilt jitter -- can be ignored, just for info
    # tt = estimate_position_variance(
    #     jacobian_smoothed, jacobian_smoothed_covariance, sky_step_basis, valid_basis,
    #     ra_dec_center=ra_dec[:, 1:-1])
    # print(f"* Tip-tilt jitter: {tt['sigma']:.3f} mas "
    #     f"(pas de dither {tt['step_scale']:.3f} mas, "
    #     f"modele explique {tt['explained']*100:.0f}% du scatter)")
    
    # estimating jacobiasy      n systematic variance
    jacobian_systematic_variance = estimate_jacobian_systematic_variance(
        jacobian_smoothed, jacobian_smoothed_covariance, valid_basis)
    jacobian_systematic_variance_block = (
        estimate_jacobian_systematic_variance_2(
            jacobian_smoothed, jacobian_smoothed_covariance,
            adjacent_photon_covariance, valid_basis))
    comparison_mask = (
        np.isfinite(jacobian_systematic_variance)
        & np.isfinite(jacobian_systematic_variance_block)
        & (jacobian_systematic_variance > 0))
    if comparison_mask.any():
        cube_values = jacobian_systematic_variance[comparison_mask]
        block_values = jacobian_systematic_variance_block[comparison_mask]
        ratio_values = block_values / np.maximum(cube_values, 1e-30)
        print(
            "* Systematic variance comparison (cube differences / block differences): "
            f"median={np.median(cube_values):.3g} / {np.median(block_values):.3g}, "
            f"median ratio={np.median(ratio_values):.3g}, "
            f"P16-P84 ratio={np.percentile(ratio_values, 16):.3g}-"
            f"{np.percentile(ratio_values, 84):.3g}")

    # adding the systematic variance on the diagonal of the covariance matrix
    jacobian_smoothed_covariance[..., 0, 0] += (
        jacobian_systematic_variance[:, None, :, None])
    jacobian_smoothed_covariance[..., 1, 1] += (
        jacobian_systematic_variance[:, None, :, None])


    jacobian_smoothed, jacobian_smoothed_covariance = (
        average_jacobian_over_nearest_cubes(
            jacobian_smoothed, jacobian_smoothed_covariance, Ncube_average))

    # print(f"* Estimated Jacobian systematic variance: {jacobian_systematic_variance:.3g}")    
    # computed key data that will be used in the astrometry fit, and filtered to keep only the valid basis blocks (non-collinear triangles and good quality data)
    data_b = datacube_normalized[:, 1:-1][valid_basis]
    var_data_b = datacube_var_normalized[:, 1:-1][valid_basis]
    J_blocks = jacobian_smoothed[valid_basis]
    C_J_blocks = jacobian_smoothed_covariance[valid_basis]
    cov_data_J_blocks = data_jacobian_covariance[valid_basis]

    # removing outlier blocks based on the median and robust standard deviation of the Jacobian magnitude
    outlier_nsigma = 5.0
    J_magnitude = np.linalg.norm(J_blocks, axis=-1)
    block_score = np.median(J_magnitude, axis=(1, 2))
    median_score = np.median(block_score)
    robust_std = 1.4826 * np.median(np.abs(block_score - median_score))
    good_block = (block_score - median_score) <= outlier_nsigma * robust_std
    if not good_block.all():
        print(f"* Rejecting {np.sum(~good_block)} noisy-Jacobian block(s) out of {len(good_block)}")
        data_b = data_b[good_block]
        var_data_b = var_data_b[good_block]
        J_blocks = J_blocks[good_block]
        C_J_blocks = C_J_blocks[good_block]
        cov_data_J_blocks = cov_data_J_blocks[good_block]

    fig, axes = plot_jacobian_diagnostics(
        wave_aera, J_blocks, data_b, C_J_blocks, Ncube_average,
        line_center=line_center, line_width=line_width)
    figures_to_save.append(fig)  # page: Jacobian diagnostics by smoothing



    # Solve the variable-projection 2x2 system over the line for a list of
    # polynomial continuum degrees; each degree yields one astrometry_xy track.
    astrometry_xy_list = []
    astrometry_covariance_list = []
    attenuation_list = []
    # The Jacobian fit remains linear; repeat only the continuum fit and its
    # covariance propagation for every tested polynomial degree.
    for poly_deg in poly_deg_values:
        # Estimate the continuum under the line (polynomial fit on the side
        # windows) instead of the notch-Hanning smoothing.
        sm_b = compute_smoothed_line(data_b, wave_aera, fit_aera, poly_deg)
        _, cov_sm_Jm = compute_smoothed_cross_covariances(
            cov_data_J_blocks, wave_aera, fit_aera, poly_deg, 1)

        J, data, sm, C_J, cov_Jsm = J_blocks, data_b, sm_b, C_J_blocks, cov_sm_Jm
        astrometry_xy, flat_eiv, M_eiv, attenuation, astrometry_covariance = solve_eiv_J(
            J, data, sm, C_J, cov_Jsm, var_data=var_data_b,
        )

        astrometry_xy_list.append(astrometry_xy)
        astrometry_covariance_list.append(astrometry_covariance)
        attenuation_list.append(attenuation)



    # Compare RA and DEC astrometry over the line for the different poly_deg
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), num="astromet_comparison_poly",
                                clear=True, sharex=True)
    axes[1].sharey(axes[0])
    for poly_deg, astrometry_xy in zip(poly_deg_values, astrometry_xy_list):
        axes[0].plot(wave_aera, astrometry_xy[:, 0], alpha=0.8, label=f"{poly_deg}")
        axes[1].plot(wave_aera, astrometry_xy[:, 1], alpha=0.8, label=f"{poly_deg}")
    # Flux over the same wavelength span (fit_aera), shaded down to the
    # continuum trend interpolated from the fit_aera (line-excluded) points
    cont_order = np.argsort(wave_aera[fit_aera])
    continuum_flux = np.interp(
        wave_aera,
        wave_aera[fit_aera][cont_order],
        mean_flux[work_aera][fit_aera][cont_order])
    axes[2].fill_between(wave_aera, mean_flux[work_aera], continuum_flux,
                         color='r', alpha=0.3)
    axes[2].plot(wave_aera, mean_flux[work_aera].T, 'r', alpha=0.5)
    # Shade the line area
    for ax in axes:
        ax.axvspan(line_center - line_width/2, line_center + line_width/2,
                    color='gray', alpha=0.2)
        ax.axvline(line_center, color='black', linewidth=1)
    axes[0].set_ylabel("RA astrometric signal (mas)")
    axes[1].set_ylabel("DEC astrometric signal (mas)")
    axes[2].set_ylabel("Flux (scaled)")
    axes[2].set_xlabel("Wavelength")
    axes[0].set_title(f"{object_name} - RA astrometry (over the line)")
    axes[1].set_title(f"{object_name} - DEC astrometry (over the line)")
    axes[2].set_title(f"{object_name} - Flux (over the line)")
    axes[0].legend(title="polynomial degree of the continuum fit")
    # fig.savefig("astrometry_3.pdf")
    figures_to_save.append(fig)  # page 3: astrometry_3 (poly_deg comparison)


    # Compare separation and PA astrometry over the line for the different poly_deg
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), num="astromet_comparison_poly_sepPA",
                                clear=True, sharex=True)
    for poly_deg, astrometry_xy in zip(poly_deg_values, astrometry_xy_list):
        separation = np.hypot(astrometry_xy[:, 0], astrometry_xy[:, 1])
        PA_deg = np.degrees(np.arctan2(astrometry_xy[:, 0], astrometry_xy[:, 1]))
        axes[0].plot(wave_aera, separation, alpha=0.8, label=f"{poly_deg}")
        axes[1].plot(wave_aera, PA_deg, alpha=0.8, label=f"{poly_deg}")
    # Flux over the same wavelength span (fit_aera)
    axes[2].plot(wave_aera, mean_flux[work_aera].T, 'r', alpha=0.5)
    # Shade the line area
    for ax in axes:
        ax.axvspan(line_center - line_width/2, line_center + line_width/2,
                    color='gray', alpha=0.2)
    # Reference PA and -PA given to the function
    axes[1].axhline(PA, color='k', linestyle=':', alpha=0.7, label=f"PA={PA:.2f}°")
    axes[1].axhline(-PA, color='k', linestyle=':', alpha=0.7, label=f"-PA={-PA:.2f}°")
    axes[0].set_ylabel("Separation (mas)")
    axes[1].set_ylabel("PA (deg)")
    axes[2].set_ylabel("Flux (scaled)")
    axes[2].set_xlabel("Wavelength")
    axes[0].set_title(f"{object_name} - Separation (over the line)")
    axes[1].set_title(f"{object_name} - PA (over the line)")
    axes[2].set_title(f"{object_name} - Flux (over the line)")
    axes[0].legend(title="polynomial degree of the continuum fit")
    axes[1].legend(fontsize=8)
    figures_to_save.append(fig)  # page 4: astrometry_3 separation/PA (poly_deg comparison)



    fig, ax = plt.subplots(1, 1, figsize=(8, 6), num="astrometry_scatter", clear=True)

    flux_scaled_filtered = flux_scaled[line_aera]  - np.min(flux_scaled[line_aera])
    velocity_line = velocity[work_aera][line_aera]
    for astrometry_xy, covariance in zip(
            astrometry_xy_list[-2:-1], astrometry_covariance_list[-2:-1]):
        scatter = ax.scatter(astrometry_xy[line_aera, 0], astrometry_xy[line_aera, 1], c=velocity_line, s=flux_scaled_filtered*1000, cmap='RdBu_r', alpha=0.6)
        ax.plot(astrometry_xy[line_aera, 0], astrometry_xy[line_aera, 1], 'k-', alpha=0.3, linewidth=1)
        for point, point_covariance in zip(
                astrometry_xy[line_aera], covariance[line_aera]):
            eigenvalues, eigenvectors = np.linalg.eigh(point_covariance)
            eigenvalues = np.maximum(eigenvalues, 0.0)
            major_axis = np.argmax(eigenvalues)
            angle = np.degrees(np.arctan2(
                eigenvectors[1, major_axis], eigenvectors[0, major_axis]))
            ellipse = Ellipse(
                point, 2 * np.sqrt(eigenvalues[major_axis]),
                2 * np.sqrt(eigenvalues[1 - major_axis]), angle=angle,
                edgecolor='black', facecolor='none', linewidth=0.6, alpha=0.45)
            ax.add_patch(ellipse)
    ax.set_xlabel("RA (mas)")
    ax.set_ylabel("DEC (mas)")
    ax.set_title(f"{object_name} - Astrometry")
    ax.plot([], [], ' ', label=f"line center = {line_center:.6g}")
    ax.plot([], [], ' ', label=f"line width = {line_width:.6g}")
    ax.legend()
    ax.set_aspect('equal')
    lim = np.max(np.abs(ax.get_xlim() + ax.get_ylim()))
    ax.set_xlim(lim, -lim)
    ax.set_ylim(-lim, lim)
    fig.colorbar(scatter, ax=ax, label="Velocity (km/s)")
    mod_ids = sorted({d.modID for d in datalist})
    mod_scales = sorted({d.modScale for d in datalist})
    observation_dates = sorted({str(d.date) for d in datalist})
    ax.set_title(
        f"{object_name} - Astrometry vs Velocity, "
        f"poly deg={list(poly_deg_values)[-2]}\n"
        f"date={observation_dates}, modID={mod_ids}, "
        f"modScale={mod_scales}, files={len(datalist)}")
    # fig.savefig("astrometry_scatter.png", dpi=300)
    ax.grid(True, alpha=0.3)
    # ax.xaxis.set_major_locator(plt.MultipleLocator(0.05))
    # ax.yaxis.set_major_locator(plt.MultipleLocator(0.05))

    PA_rad = PA*np.pi/180
    y = np.linspace(-lim,lim,100)
    x = np.tan(PA_rad)*y
    ax.plot(x,y,'k--',label=f"PA={PA:.2f}°") 
    ax.legend() 

    figures_to_save.append(fig)  # page: astrometry_scatter_PA



    ##################################################
    # Save the astrometric results to a FITS file (mirrors run_createCouplingMap)
    ##################################################
    new_header = datalist[-1].header.copy()
    new_header['X_FIRTYP'] = 'ASTROMETRY'
    new_header['Q_ASLINE'] = (line_center, 'line center wavelength (nm)')
    new_header['Q_ASLWID'] = (line_width, 'line width (nm)')
    new_header['Q_ASPDEG'] = (str(list(poly_deg_values)), 'polynomial degrees of the continuum fit')
    new_header['Q_ASNAME'] = (runlib_io.create_basename(new_header), 'name of the astrometry file')

    output_dir = os.path.join(datalist[-1].dirname, "../astrometry")
    os.makedirs(output_dir, exist_ok=True)
    output_filename = os.path.join(output_dir, new_header['Q_ASNAME'])
    astrometry_xy_all = np.stack(astrometry_xy_list, axis=0)  # (n_poly, n_line, 2)
    hdul = fits.HDUList([
        fits.PrimaryHDU(header=new_header),
        fits.ImageHDU(data=np.asarray(wave, dtype=float), name='WAVE'),
        fits.ImageHDU(data=np.asarray(flux_scaled, dtype=float), name='FLUX_SCALED'),
        fits.ImageHDU(data=np.asarray(astrometry_xy_all, dtype=float), name='ASTROMETRY_XY'),
        fits.ImageHDU(data=np.asarray(poly_deg_values, dtype=float), name='POLY_DEG'),
    ])
    hdul.writeto(output_filename, overwrite=True)
    print(f"Astrometry results saved to {output_filename}")

    ##################################################
    # Save all collected figures to a single multi-page PDF next to the FITS file
    ##################################################
    pdf_filename = os.path.splitext(output_filename)[0] + ".pdf"
    from matplotlib.backends.backend_pdf import PdfPages
    pdf = PdfPages(pdf_filename)
    for fig in figures_to_save:
        pdf.savefig(fig)
    pdf.close()
    print(f"All figures saved to {pdf_filename}")



if __name__ == "__main__":
    """
    Run astrometric analysis with development defaults.
    Perfect for testing and direct execution of core functionality.
    """
    print("Running createCouplingMap core with development defaults...")
    

    # Development/interactive mode handling
    print("Running in compiler")
    if getpass.getuser() == "slacour":
        object_name = None
        dark_patterns = None
        flat_patterns = None
        wave_patterns = None
        modID = None
        modScale = None
        wollaston = None
        line_center=656.5
        line_width= 2
        PA=137  # for plotting only
        Ncube_average=1

        file_patterns = ["/Users/slacour/DATA/LANTERNE/tmp/firstpl_13:0*.fits"]
        file_patterns = ["/Users/slacour/DATA/LANTERNE/20251230/preproc/*T12?2*.fits"]
        file_patterns = ["/Users/slacour/DATA/FIRST/20260608/preproc/firstpl_2026-06-08T10h[1-2]*_RASALHAGUE_P.fits"]
        # file_patterns = ["/Users/slacour/DATA/FIRST/20260608/preproc/firstpl_2026-06-08T10h18*_RASALHAGUE_P.fits"]
        wave_patterns = ["/Users/slacour/DATA/FIRST/20260608/wavemaps/"]
        # flat_patterns = wave_patterns

        PA=137  # for plotting only
        modID = 9
        line_center=656.5
        line_width= 1.8
        file_patterns = ["/Users/slacour/DATA/FIRST/20260625/preproc/firstpl_2026-06-25T09h3[2-9]*_HD163296_P.fits"]
        wave_patterns = ["/Users/slacour/DATA/FIRST/20260625/wavemaps/"]
        file_patterns = ["/Users/slacour/DATA/FIRST/20260827/preproc/firstpl_2026-*_HD163296_P.fits"]
        wave_patterns = ["/Users/slacour/DATA/FIRST/20260827/wavemaps/"]

        # PA= 162
        # line_width= 1.3
        # line_center = 656.4
        # file_patterns = ["/Users/slacour/DATA/FIRST/20260625/preproc/firstpl_2026-06-25T08h59m59s_HD142527_P.fits",
        #                  "/Users/slacour/DATA/FIRST/20260625/preproc/firstpl_2026-06-25T09h01m49s_HD142527_P.fits",
        #                     "/Users/slacour/DATA/FIRST/20260625/preproc/firstpl_2026-06-25T09h19m55s_HD142527_P.fits",
        #                  ]

        #ALTAIR
        # PA= 25 
        # line_width= 1.7
        # line_center = 656.2
        # modID = 9
        # modScale = 25
        # file_patterns = ["/Users/slacour/DATA/FIRST/20260827/preproc/firstpl_2026-08-27T08h*fits",
        #                  ]
        
        
    print(f"Development file patterns: {file_patterns}")


    process_astrometric_data(
        file_patterns=file_patterns,
        object_name=object_name,
        dark_patterns=dark_patterns,
        flat_patterns=flat_patterns,
        wave_patterns=wave_patterns,
        modID=modID,
        modScale=modScale,
        wollaston=wollaston,
        line_center=line_center,
        line_width=line_width,
        PA=PA,
        Ncube_average=Ncube_average)
        # save_individual_frames=save_individual_frames,)
# %%

from scipy import odr

