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
import matplotlib
if "VSCODE_PID" in os.environ:
    matplotlib.use('macosx')
elif os.environ.get('SPYDER_DEBUG_FILE'):
    print("Running in Spyder")
else:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
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


def compute_smoothed_line(data_b, wave, fit_aera, work_aera, poly_deg):
    """Estimate the continuum under a line with a low-order polynomial fit.

    The continuum is fitted on two side windows (each as wide as the line) on
    either side of the line and evaluated over the full span from the left to
    the right window (`fit_aera`).
    """
    # Fit all (cube, step, output) continua at once on the side windows
    y_cont = data_b[..., fit_aera]     
    x_cont = wave[fit_aera]                     # (Ncube, Nstep, Noutput, Ncont)
    cont_shape = y_cont.shape[:-1]
    coeffs = np.polyfit(x_cont,
                        y_cont.reshape(-1, sum(fit_aera)).T, poly_deg)  # (poly_deg+1, Nseries)
    # Evaluate the polynomial continuum across the full left->right span
    V_line = np.vander(wave[work_aera], poly_deg + 1)                 # (n_fit, poly_deg+1)
    data_smoothed_line = (V_line @ coeffs).T.reshape(*cont_shape, -1)  # (..., n_fit)
    return data_smoothed_line


def solve_astrometry_gain(J_blocks_aera, data_b_aera, sm_b, outlier_nsigma=5.0):
    """Variable-projection solve of  J @ a = data - smoothed / flat.

    The per-output gain `flat` (close to 1) enters the forward model as
    `data_smoothed / flat`. Substituting g = 1/flat makes the model linear:

        J_b @ a = data_b - sm_b * g

    For a fixed astrometric shift `a`, the optimal per-output g is eliminated
    analytically by projecting onto the smoothed (`sm_b`) directions:

        g[o] = sum_b sm_b[o] (data_b[o] - J_b[o] @ a) / sum_b sm_b[o]^2

    leaving a single 2x2 normal system per wavelength. Because the projection
    is onto `sm_b`, the normal matrix `M` depends on the smoothing/continuum
    and must be rebuilt for every window (unlike the old data-projected form).

    Parameters
    ----------
    J_blocks : (Nblocks, Noutput, Nwave, 2) response Jacobian per block.
    data_b   : (Nblocks, Noutput, Nwave) measured (interior) data per block.
    sm_b     : (Nblocks, Noutput, Nwave) smoothed continuum per block.
    outlier_nsigma : float, optional
        Whole blocks (i_cube, j_step) are rejected if their response Jacobian
        `J_blocks` is unusually large there. `J_blocks` is built from the
        difference of just two exposures (`data_diff_basis @ sky_step_basis_inv`),
        so a noisy or ill-conditioned block can blow up its magnitude and bias
        the least-squares fit; since this affects every wavelength at once, the
        Jacobian magnitude is aggregated over (output, wavelength) into one
        robust score per block before thresholding (one-sided: too-large only).

    Returns
    -------
    astrometry_shift : (Nwave, 2) RA/DEC photocenter shift.
    flat             : (Noutput, Nwave) recovered per-output gain (= 1/g).
    d_proj, J_proj   : projected data/Jacobian, see `compute_astrometry_significance`.
    M                : (Nwave, 2, 2) normal matrix of the 2x2 system.
    """

    J_blocks  = J_blocks_aera
    data_b = data_b_aera

    # A noisy/ill-conditioned block blows up the Jacobian at every wavelength
    # at once, so aggregate its magnitude over (output, wavelength) into one
    # robust score per block before thresholding (one-sided: too-large only).
    J_magnitude = np.linalg.norm(J_blocks, axis=-1)
    block_score = np.median(J_magnitude, axis=(1, 2))
    median_score = np.median(block_score)
    robust_std = 1.4826 * np.median(np.abs(block_score - median_score))
    good_block = (block_score - median_score) <= outlier_nsigma * robust_std
    if not good_block.all():
        print(f"* Rejecting {np.sum(~good_block)} noisy-Jacobian block(s) out of {len(good_block)}")
        J_blocks = J_blocks[good_block]
        data_b = data_b[good_block]
        sm_b = sm_b[good_block]

    S2 = np.sum(sm_b ** 2, axis=0)                       # (Noutput, Nwave)
    Gs = np.sum(sm_b[..., None] * J_blocks, axis=0)      # (Noutput, Nwave, 2)
    J_proj = J_blocks - sm_b[..., None] * (Gs / S2[..., None])[None]
    M = np.einsum('bowi,bowj->wij', J_proj, J_proj)      # (Nwave, 2, 2)
    H = np.sum(sm_b * data_b, axis=0)                    # (Noutput, Nwave)
    d_proj = data_b - sm_b * (H / S2)[None]
    rhs = np.einsum('bowi,bow->wi', J_proj, d_proj)      # (Nwave, 2)
    astrometry_shift = np.linalg.solve(M, rhs[..., None])[..., 0]
    # Recover the eliminated gains: g = (H - Gs . astrometry) / S2, flat = 1/g
    g = (H - np.einsum('owi,wi->ow', Gs, astrometry_shift)) / S2
    flat = 1.0 / g
    return astrometry_shift, flat, d_proj, J_proj, M


def compute_astrometry_significance(J_proj, d_proj, astrometry_xy, M):
    """Per-wavelength goodness-of-fit and detection significance of the RA/DEC fit.

    `d_proj.std(axis=(0,1))` alone only shows that some scatter is present; it does
    not say whether that scatter lines up with the geometric RA/DEC response
    directions in `J_proj` rather than being unrelated noise. This computes, for
    every wavelength:

    - `correlation` (Nwave, 2): Pearson correlation of `d_proj` with each response
      direction (RA, DEC) of `J_proj`, across (block, output).
    - `r_squared` (Nwave,): fraction of the `d_proj` variance jointly explained by
      the 2D fit (`J_proj @ astrometry_shift`).
    - `significance` (Nwave,): amplitude of `astrometry_shift` in units of its
      formal 1-sigma uncertainty, i.e. sqrt(shift^T @ M @ shift / noise_variance) -
      a chi2-with-2-dof statistic under the null hypothesis of no offset.
    """
    predicted = np.einsum('bowi,wi->bow', J_proj, astrometry_xy)
    residual = d_proj - predicted

    d_mean = d_proj.mean(axis=(0, 1))
    J_mean = J_proj.mean(axis=(0, 1))
    covariance = np.mean((d_proj - d_mean)[..., None] * (J_proj - J_mean), axis=(0, 1))
    correlation = covariance / (d_proj.std(axis=(0, 1))[:, None] * J_proj.std(axis=(0, 1)))

    ss_res = np.sum(residual ** 2, axis=(0, 1))
    ss_tot = np.sum((d_proj - d_mean) ** 2, axis=(0, 1))
    r_squared = 1 - ss_res / ss_tot

    # Projecting out sm_b removes 1 dof per output from the Nblocks measurements.
    Nblocks, Noutput = d_proj.shape[0], d_proj.shape[1]
    dof = Noutput * (Nblocks - 1) - 2
    noise_variance = ss_res / dof
    chi2 = np.einsum('wi,wij,wj->w', astrometry_xy, M, astrometry_xy) / noise_variance
    significance = np.sqrt(chi2)

    return correlation, r_squared, significance


def process_astrometric_data(
    file_patterns, object_name=None, dark_patterns=None, flat_patterns=None, wave_patterns=None, modID=None, modScale=None, wollaston=None,
    Nsingular=19*6, line_center=656.28, line_width= 3.0, PA=137.0, fast=False):
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
    on the smooth spectral continuum (`data_smoothed`, Hanning smoothing). The
    forward model relating both unknowns to the data is:

        jacobian @ astrometry_shift = data_normalized - data_smoothed / flat

    Here `astrometry_shift` (2 values per wavelength) is shared by every output
    and block, while `flat` (one value per output per wavelength) is shared by
    every block but free across outputs. The gain enters non-linearly through
    1/`flat`, so we substitute g = 1/`flat` (also close to 1), which makes the
    model linear in the unknowns:

        jacobian @ astrometry_shift = data_normalized - data_smoothed * g

    The system decouples per wavelength; at each wavelength the unknowns are
    `astrometry_shift` (2) plus g (Noutput). Because g[o] enters linearly and
    only in the rows of output o, it is eliminated analytically for any given
    `astrometry_shift` (separable / variable-projection least squares):

        g[o] = sum_b data_smoothed_b[o] (data_b[o] - jacobian[o] @ astrometry_shift)
                    / sum_b data_smoothed_b[o]^2

    Substituting the optimal g back projects the per-block Jacobian and the
    measured data onto the complement of the smoothed directions (`J_proj`,
    `d_proj`) and leaves a single 2x2 normal system per wavelength:

        M(lambda) @ astrometry_shift(lambda) = sum_{b,o} J_proj * d_proj
        M(lambda) = sum_{b,o} J_proj @ J_proj^T

    solved with `np.linalg.solve`. The two columns of `astrometry_shift` are
    the RA and DEC astrometric signals versus wavelength. Because the gain now
    multiplies `data_smoothed`, the projection (and hence `M`/`J_proj`) depends
    on the smoothing window, so the 2x2 system is rebuilt for each Hanning
    window size (`x_hanning`); the helper `solve_astrometry_gain` performs the
    full per-window solve.

    Identifiability: a single output cannot separate astrometry from its own
    flat gain, but `flat`[o] is constant across the dither blocks whereas the
    astrometric response `jacobian` varies block to block, so several dither
    positions are required to break the degeneracy (and the extra Noutput free
    parameters per wavelength do inflate the astrometric noise).

    `PA` (degrees) is used for plotting only: it draws a reference position
    angle line on the astrometry_scatter figure and does not affect any of the
    computed results.

    `fast` (bool): when True, skip the (expensive) SVD filtering step and only
    use 3 Hanning window sizes (instead of 10) to speed up the computation at
    the cost of accuracy.
    """

    # Polynomial continuum-fit degrees tested under the line (hard-coded)
    poly_deg_values = (2, 3, 4, 5)

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
    wave = datalist[0].wave  # Assuming all have the same wavelength grid
    xmod = np.concatenate([d.xmod for d in datalist])
    ymod = np.concatenate([d.ymod for d in datalist])
    ra_dec = np.concatenate([d.compute_xy_sky() for d in datalist])
    Ncube = datacube.shape[0]
    Nmod = datacube.shape[1]
    Nwave = datacube.shape[3]


    # Create filename associations
    basenames = []
    for d in datalist:
        n = d.data.shape[0]
        basenames.extend([d.basename] * n)
    filenames = [d.filename for d in datalist]

    # Data quality filtering
    goodData_flux, _ = runlib_linalg.flux_filtering(flux)
    print(f"* Percentage of good data: {np.sum(goodData_flux)/len(goodData_flux.ravel())*100:.1f} % (flux threshold)")

    # Plot flux map
    fig = runlib_plots.plot_flux_map(flux.mean(axis=(2))[0], xmod[0], ymod[0])
    figures_to_save = [fig]

    mean_flux = np.nanmean(flux, axis=(0,1))
    datacube_normalized = datacube / mean_flux
    flux_scaled = mean_flux/ np.nanmax(mean_flux)

    # ra_dec = np.stack([xmod,ymod],axis=-1)
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

    # Measured output changes for the same forward/backward steps
    data_diff_fwd = datacube_normalized[:,2:] - datacube_normalized[:,1:-1]    # D_{k+1} - D_k
    data_diff_bwd = datacube_normalized[:,:-2] - datacube_normalized[:,1:-1]   # D_{k-1} - D_k

    # Measure lag-1 correlation between adjacent modulation steps (across outputs/wavelength)
    data_centered = datacube - datacube.mean(axis=(2,3), keepdims=True)
    data_std = np.nanstd(datacube, axis=(2,3))
    data_covariance_lag = np.nanmean(data_centered[:,1:] * data_centered[:,:-1],axis=(2,3))
    # mesure correlation
    data_corr_lag = data_covariance_lag / (data_std[:,1:] * data_std[:,:-1])

    # flag the correlation values that are too low (i.e. the corresponding data pairs are not correlated)
    threshold_corr = 0.5
    low_correlation_pair_mask = data_corr_lag < threshold_corr
    valid_basis &= ~low_correlation_pair_mask[:,1:] & ~low_correlation_pair_mask[:,:-1] 
    print(f"* Percentage of valid triangles: {np.sum(valid_basis)/len(valid_basis.ravel())*100:.1f} % (correlation + determinant threshold)")

    correlation_values = data_corr_lag[np.isfinite(data_corr_lag)]
    rejected_flux_mask = ~(goodData_flux[:, 1:] & goodData_flux[:, :-1])
    rejected_correlation_values = data_corr_lag[
        rejected_flux_mask & np.isfinite(data_corr_lag)]
    
    below_threshold_percent = 100 * np.mean(correlation_values < threshold_corr)
    percentile_levels = np.array([5, 16, 50, 84, 95])
    correlation_percentiles = np.percentile(correlation_values, percentile_levels)
    fig_2, ax = plt.subplots(1, 1, figsize=(8, 6), num="correlation_lag_histogram", clear=True)
    bin_edges = np.linspace(0, 1, 21)
    ax.hist(correlation_values, bins=bin_edges, color="steelblue", edgecolor="white",
        alpha=0.7, label="All data")
    if rejected_correlation_values.size:
        ax.hist(rejected_correlation_values, bins=bin_edges, color="tomato", edgecolor="white",
                alpha=0.7, label="Rejected by flux filter")
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

    # Build the per-block response Jacobian once: it does not depend on x_hanning.
    # Each block is a valid interior modulation point (i_cube, j_step); for that
    # block we also keep the measured (interior) data that multiplies the flat.
    jacobian_blocks = []
    data_blocks = []
    valid_indices = []
    data_interior = datacube_normalized[:, 1:-1]
    for i_cube in range(Ncube):
        for j_step in range(0, Nmod-2):

            if valid_basis[i_cube, j_step] == False:
                continue

            # Stack the two measured output differences: [D_{k+1}-D_k , D_{k-1}-D_k]
            data_diff_basis = np.stack(
                [data_diff_fwd[i_cube, j_step], data_diff_bwd[i_cube, j_step]], axis=-1)

            # Local response Jacobian J = (data differences) @ (sky-step basis)^-1
            jacobian = data_diff_basis @ sky_step_basis_inv[i_cube, j_step]

            jacobian_blocks.append(jacobian)
            data_blocks.append(data_interior[i_cube, j_step])
            valid_indices.append((i_cube, j_step))

    # (Nblocks, Noutput, Nwave, 2) and (Nblocks, Noutput, Nwave)
    J_blocks = np.stack(jacobian_blocks, axis=0)
    data_b = np.stack(data_blocks, axis=0)

    # Speed of light in km/s (precise CODATA value)
    # Doppler velocity (km/s)
    c = speed_of_light / 1e3
    velocity = c * (wave - line_center) / line_center

    line_aera = (wave > line_center - line_width/2) & (wave < line_center + line_width/2)
    work_aera = (wave > line_center - line_width*1.5) & (wave < line_center + line_width*1.5)
    fit_aera = work_aera & ~line_aera

    data_b_aera = data_b[..., work_aera]
    J_blocks_aera = J_blocks[..., work_aera, :]

    # Solve the variable-projection 2x2 system over the line for a list of
    # polynomial continuum degrees; each degree yields one astrometry_xy track.
    astrometry_xy_list = []
    correlation_list = []
    r_squared_list = []
    significance_list = []
    for poly_deg in poly_deg_values:
        # Estimate the continuum under the line (polynomial fit on the side
        # windows) instead of the notch-Hanning smoothing.
        sm_b = compute_smoothed_line(data_b, wave, fit_aera, work_aera, poly_deg)
        astrometry_xy, flat, d_proj, J_proj, M = solve_astrometry_gain(J_blocks_aera, data_b_aera, sm_b)  # (n_fit, 2)
        J_norm = np.linalg.norm(J_proj, axis=-1)
        correlation, r_squared, significance = compute_astrometry_significance(J_proj, d_proj, astrometry_xy, M)

        astrometry_xy_list.append(astrometry_xy)
        correlation_list.append(correlation)
        r_squared_list.append(r_squared)
        significance_list.append(significance)

    peak_idx = np.nanargmax(significance_list[-2])
    print(f"* Peak astrometric detection significance: {significance_list[-2][peak_idx]:.1f} sigma "
          f"at {wave[work_aera][peak_idx]:.3f} nm (R\u00b2={r_squared_list[-2][peak_idx]:.2f}, "
          f"poly_deg={poly_deg_values[-2]})")

    figures_to_save.append(fig_2) 

    # Compare RA and DEC astrometry over the line for the different poly_deg
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), num="astromet_comparison_poly",
                                clear=True, sharex=True)
    axes[1].sharey(axes[0])
    for poly_deg, astrometry_xy in zip(poly_deg_values, astrometry_xy_list):
        axes[0].plot(wave[work_aera], astrometry_xy[:, 0], alpha=0.8, label=f"{poly_deg}")
        axes[1].plot(wave[work_aera], astrometry_xy[:, 1], alpha=0.8, label=f"{poly_deg}")
    # Flux over the same wavelength span (fit_aera), shaded down to the
    # continuum trend interpolated from the fit_aera (line-excluded) points
    cont_order = np.argsort(wave[fit_aera])
    continuum_flux = np.interp(wave[work_aera], wave[fit_aera][cont_order], mean_flux[fit_aera][cont_order])
    axes[2].fill_between(wave[work_aera], mean_flux[work_aera], continuum_flux, color='r', alpha=0.3)
    axes[2].plot(wave[work_aera], mean_flux[work_aera].T, 'r', alpha=0.5)
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
        axes[0].plot(wave[work_aera], separation, alpha=0.8, label=f"{poly_deg}")
        axes[1].plot(wave[work_aera], PA_deg, alpha=0.8, label=f"{poly_deg}")
    # Flux over the same wavelength span (fit_aera)
    axes[2].plot(wave[work_aera], mean_flux[work_aera].T, 'r', alpha=0.5)
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


    # Compare detection significance, goodness of fit, and per-axis correlation
    # between J_proj (RA/DEC response) and d_proj (residual data) over the line
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), num="astromet_comparison_poly_significance",
                                clear=True, sharex=True)
    for poly_deg, significance, r_squared in zip(poly_deg_values, significance_list, r_squared_list):
        axes[0].plot(wave[work_aera], significance, alpha=0.8, label=f"{poly_deg}")
        axes[1].plot(wave[work_aera], r_squared, alpha=0.8, label=f"{poly_deg}")
    axes[0].axhline(3, color='k', linestyle=':', alpha=0.7, label="3 sigma")
    reference_correlation = correlation_list[-2]
    axes[2].plot(wave[work_aera], reference_correlation[:, 0], label="RA")
    axes[2].plot(wave[work_aera], reference_correlation[:, 1], label="DEC")
    axes[2].axhline(0, color='k', linewidth=1, alpha=0.5)
    # Shade the line area
    for ax in axes:
        ax.axvspan(line_center - line_width/2, line_center + line_width/2,
                    color='gray', alpha=0.2)
        ax.axvline(line_center, color='black', linewidth=1)
    axes[0].set_ylabel("Detection significance (sigma)")
    axes[1].set_ylabel("R\u00b2 (variance explained)")
    axes[2].set_ylabel(f"Correlation (poly_deg={poly_deg_values[-2]})")
    axes[2].set_xlabel("Wavelength")
    axes[0].set_title(f"{object_name} - Astrometric detection significance (over the line)")
    axes[1].set_title(f"{object_name} - Goodness of fit R\u00b2 (over the line)")
    axes[2].set_title(f"{object_name} - Correlation between J_proj and d_proj (over the line)")
    axes[0].legend(title="polynomial degree of the continuum fit")
    axes[2].legend()
    figures_to_save.append(fig)  # page: astromet_comparison_poly_significance


    fig, ax = plt.subplots(1, 1, figsize=(8, 6), num="astrometry_scatter", clear=True)

    flux_scaled_filtered = flux_scaled[line_aera]  - np.min(flux_scaled[line_aera])
    velocity_line = velocity[line_aera]
    for astrometry_xy in astrometry_xy_list[-2:-1]:
        scatter = ax.scatter(astrometry_xy[line_aera[work_aera], 0], astrometry_xy[line_aera[work_aera], 1], c=velocity_line, s=flux_scaled_filtered*1000, cmap='RdBu_r', alpha=0.6)
        ax.plot(astrometry_xy[:, 0], astrometry_xy[:, 1], 'k-', alpha=0.3, linewidth=1)
    ax.set_xlabel("RA (mas)")
    ax.set_ylabel("DEC (mas)")
    ax.plot([], [], ' ', label=f"line center = {line_center:.6g}")
    ax.plot([], [], ' ', label=f"line width = {line_width:.6g}")
    ax.legend()
    ax.set_aspect('equal')
    lim = np.max(np.abs(ax.get_xlim() + ax.get_ylim()))
    ax.set_xlim(lim, -lim)
    ax.set_ylim(-lim, lim)
    fig.colorbar(scatter, ax=ax, label="Velocity (km/s)")
    ax.set_title(f"{object_name} - Astrometry vs Velocity, poly deg={list(poly_deg_values)[-2]}")
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
    new_header['Q_ASSING'] = (Nsingular, 'number of singular values')
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
        Nsingular = 19*6
        modID = None
        modScale = None
        wollaston = None
        line_center=656.28
        line_width= 2
        PA=137  # for plotting only
        fast=True

        file_patterns = ["/Users/slacour/DATA/LANTERNE/tmp/firstpl_13:0*.fits"]
        file_patterns = ["/Users/slacour/DATA/LANTERNE/20251230/preproc/*T12?2*.fits"]
        file_patterns = ["/Users/slacour/DATA/FIRST/20260608/preproc/firstpl_2026-06-08T10h[1-2]*_RASALHAGUE_P.fits"]
        # file_patterns = ["/Users/slacour/DATA/FIRST/20260608/preproc/firstpl_2026-06-08T10h18*_RASALHAGUE_P.fits"]
        wave_patterns = ["/Users/slacour/DATA/FIRST/20260608/wavemaps/"]
        # flat_patterns = wave_patterns

        PA=137  # for plotting only
        modID = 9
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

        # #ALTAIR
        # PA= 162 
        # line_width= 1.7
        # line_center = 656.3
        # modID = 9
        # file_patterns = ["/Users/slacour/DATA/FIRST/20260827/preproc/firstpl_2026-08-27T09h58*fits",
        #                  "/Users/slacour/DATA/FIRST/20260827/preproc/firstpl_2026-08-27T09h58*fits",
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
        Nsingular=Nsingular,
        line_center=line_center,
        line_width=line_width,
        PA=PA,
        fast=fast)
        # save_individual_frames=save_individual_frames,)
# %%
