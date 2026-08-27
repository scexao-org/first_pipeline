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
    

def compute_smoothed_data(data_normalized, x_hanning):
    """Compute the Hanning-smoothed (continuum) data cube for a given window size."""
    Nwave = data_normalized.shape[-1]
    x_zeros = x_hanning // 2
    Nhanning = x_hanning * 2 + 1
    hanning_window = np.hanning(Nhanning)
    hanning_window[x_hanning - x_zeros:x_hanning + 1 + x_zeros] = 0
    hanning_window /= hanning_window.sum()  # Normalize the window
    # Edge normalization: weights overlapping the valid (zero-padded) region
    edge_norm = np.convolve(np.ones(Nwave), hanning_window, mode='same')
    # Vectorized convolution along the wavelength axis (zero-padded boundary,
    # equivalent to np.convolve mode='same'); the symmetric odd window keeps
    # the kernel centered.
    data_smoothed = convolve1d(data_normalized, hanning_window, axis=-1,
                                mode='constant', cval=0.0)
    data_smoothed = data_smoothed / edge_norm
    return data_smoothed, hanning_window


def line_fit_region(line_aera):
    """Indices used to fit and evaluate the polynomial continuum under a line.

    The continuum is fitted on two side windows (`cont_idx`, each as wide as the
    line) and evaluated over the full span `fit_aera` going from the left window
    start to the right window stop (i.e. the line plus both side windows).
    """
    line_idx = np.where(line_aera)[0]
    i0, i1 = line_idx[0], line_idx[-1]
    n_line = i1 - i0 + 1
    left = slice(max(i0 - n_line, 0), i0)
    right = slice(i1 + 1, i1 + 1 + n_line)
    cont_idx = np.r_[np.arange(left.start, left.stop),
                     np.arange(right.start, right.stop)]
    fit_aera = slice(left.start, right.stop)
    return cont_idx, fit_aera


def compute_smoothed_line(data_normalized, wave, line_aera, poly_deg):
    """Estimate the continuum under a line with a low-order polynomial fit.

    The continuum is fitted on two side windows (each as wide as the line) on
    either side of the line and evaluated over the full span from the left to
    the right window (`fit_aera`).
    """
    cont_idx, fit_aera = line_fit_region(line_aera)
    # Fit all (cube, step, output) continua at once on the side windows
    y_cont = data_normalized[..., cont_idx]                          # (Ncube, Nstep, Noutput, Ncont)
    cont_shape = y_cont.shape[:-1]
    coeffs = np.polyfit(wave[cont_idx],
                        y_cont.reshape(-1, len(cont_idx)).T, poly_deg)  # (poly_deg+1, Nseries)
    # Evaluate the polynomial continuum across the full left->right span
    V_line = np.vander(wave[fit_aera], poly_deg + 1)                 # (n_fit, poly_deg+1)
    data_smoothed_line = (V_line @ coeffs).T.reshape(*cont_shape, -1)  # (..., n_fit)
    return data_smoothed_line


def solve_astrometry_gain(J_blocks, data_b, sm_b):
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

    Returns
    -------
    astrometry_shift : (Nwave, 2) RA/DEC photocenter shift.
    flat             : (Noutput, Nwave) recovered per-output gain (= 1/g).
    """
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
    return astrometry_shift, flat


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

    # Create filename associations
    basenames = []
    for d in datalist:
        n = d.data.shape[0]
        basenames.extend([d.basename] * n)
    filenames = [d.filename for d in datalist]

    # Data quality filtering
    flux_goodData, flux_threshold = runlib_linalg.flux_filtering(flux)
    print(f"* Percentage of good data: {np.sum(flux_goodData)/len(flux_goodData.ravel())*100:.1f} % (flux threshold)")

    # SVD filtering (skipped in fast mode)
    if fast:
        print("* Fast mode: skipping SVD filtering")
        data_svdfiltered = datacube
        goodData = flux_goodData
    else:
        data_svdfiltered, fit_goodData, errors = runlib_linalg.svd_filtering(datacube, flux_goodData, Nsingular)
        goodData = flux_goodData & fit_goodData
        print(f"* Percentage of good data: {np.sum(goodData)/len(goodData.ravel())*100:.1f} % (flux and svd threshold)")

    # Plot flux map
    runlib_plots.plot_flux_map(flux.mean(axis=(2))[0], xmod[0], ymod[0])

    mean_flux = np.nanmean(flux, axis=(0,1))
    data_normalized = data_svdfiltered / mean_flux

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
    valid_basis &= goodData[:,2:] & goodData[:,:-2] & goodData[:,1:-1]

    # Measured output changes for the same forward/backward steps
    data_diff_fwd = data_normalized[:,2:] - data_normalized[:,1:-1]    # D_{k+1} - D_k
    data_diff_bwd = data_normalized[:,:-2] - data_normalized[:,1:-1]   # D_{k-1} - D_k


    # now do the calculations for a list of x_hanning values

    Ncube = data_normalized.shape[0]
    Nwave = data_normalized.shape[3]

    # Pixel-to-wavelength scale and central-notch half-width (in pixels) so that
    # the zeroed middle of the Hanning window spans the spectral line width.
    # Use abs() since `wave` may be stored in decreasing order.
    dwave = np.abs(np.median(np.diff(wave)))
    line_width_pix = line_width / dwave

    # Build the per-block response Jacobian once: it does not depend on x_hanning.
    # Each block is a valid interior modulation point (i_cube, j_step); for that
    # block we also keep the measured (interior) data that multiplies the flat.
    jacobian_blocks = []
    data_blocks = []
    valid_indices = []
    data_interior = data_normalized[:, 1:-1]
    for i_cube in range(Ncube):
        for j_step in range(data_diff_fwd.shape[1]):

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

    # Joint forward model, per block b, output o and wavelength (decouples per wave):
    #     jacobian @ astrometry = data - data_smoothed / flat
    # astrometry = (d_alpha, d_delta) is shared by all (b, o); flat[o] is an
    # unknown gain (close to 1) shared by all blocks b but free across outputs.
    # Substituting g = 1/flat linearises the model; for fixed astrometry the
    # optimal g is eliminated analytically (variable projection onto the
    # smoothed directions), leaving a 2x2 system per wavelength. Because the
    # projection is onto data_smoothed, the normal matrix now depends on the
    # smoothing window and is rebuilt for each one (see solve_astrometry_gain).

    # Solve for astrometry (and recover flat) for a list of x_hanning windows
    # Largest window is twice the line width (in pixels); 10 values over the
    # range (only 3 in fast mode)
    x_hanning_max = int(round(line_width_pix * 2))
    n_hanning = 3 if fast else 10
    x_hanning_values = [int(round(x)) for x in np.linspace(3, x_hanning_max, n_hanning)]
    astrometry_shift_list = []
    flat_list = []
    hanning_window_list = []
    for x_hanning in tqdm(x_hanning_values, desc="x_hanning"):
        data_smoothed, hanning_window = compute_smoothed_data(data_normalized, x_hanning)
        sm_b = np.stack([data_smoothed[:, 1:-1][i_cube, j_step]
                            for (i_cube, j_step) in valid_indices], axis=0)

        # Variable-projection solve (the 2x2 system depends on sm_b -> rebuilt here)
        astrometry_shift, flat = solve_astrometry_gain(J_blocks, data_b, sm_b)

        astrometry_shift_list.append(astrometry_shift)
        flat_list.append(flat)
        hanning_window_list.append(hanning_window)

    flux_scaled = np.nanmean(flux, axis=(0,1)) / np.nanmax(np.nanmean(flux, axis=(0,1)))

    # Speed of light in km/s (precise CODATA value)
    c = speed_of_light / 1e3
    line_aera = (wave > line_center - line_width/2) & (wave < line_center + line_width/2)

    # Prepare output paths (mirrors run_createCouplingMap save logic)
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
    pdf_filename = os.path.splitext(output_filename)[0] + ".pdf"

    # Collect the requested figures (astrometry_1, astrometry_2,
    # astrometry_scatter_PA) into a single multi-page PDF next to the FITS file.
    from matplotlib.backends.backend_pdf import PdfPages
    pdf = PdfPages(pdf_filename)

    # Compare RA and DEC astrometry for the different x_hanning values
    fig, axes = plt.subplots(2, 1, figsize=(10, 9), num="astromet_comparison_2",
                                clear=True, sharex=True)
    axes[1].sharey(axes[0])
    for x_hanning, astrometry_shift in zip(x_hanning_values, astrometry_shift_list):
        axes[0].plot(wave, astrometry_shift[:, 0], alpha=0.7, label=f"{x_hanning}")
        axes[1].plot(wave, astrometry_shift[:, 1], alpha=0.7, label=f"{x_hanning}")
    # Shade the line area
    for ax in axes:
        ax.axvspan(line_center - line_width/2, line_center + line_width/2,
                    color='gray', alpha=0.2)
        ax.axvline(line_center, color='black', linewidth=1)
    # Flux plotted on its own (right) y-axis in red
    ax0_flux = axes[0].twinx()
    ax1_flux = axes[1].twinx()
    for ax_flux in (ax0_flux, ax1_flux):
        ax_flux.plot(wave, flux_scaled.T, 'r', alpha=0.5)
        ax_flux.set_ylabel("Flux (scaled)", color='r')
        ax_flux.tick_params(axis='y', colors='r')
    axes[0].set_ylabel("RA astrometric signal (mas)")
    axes[1].set_ylabel("DEC astrometric signal (mas)")
    axes[1].set_xlabel("Wavelength")
    axes[0].set_title(f"{object_name} - RA astrometry")
    axes[1].set_title(f"{object_name} - DEC astrometry")
    axes[0].legend(title="size of Hanning smoothing window (in pixels)", fontsize=8, ncol=2)

    # fig.savefig("astrometry_1.pdf")
    pdf.savefig(fig)  # page 1: astrometry_1 (full-band view)


    axes[0].set_xlim(line_center - line_width*2, line_center + line_width*2)
    # Scale the y-axis to the peak |astrometry_shift| inside the plotted x-range
    wave_mask = (wave > line_center - line_width*2) & (wave < line_center + line_width*2)
    y_max = np.nanmax([np.nanmax(np.abs(a[wave_mask])) for a in astrometry_shift_list])
    axes[0].set_ylim(-y_max, y_max)
    # fig.savefig("astrometry_2.pdf")
    pdf.savefig(fig)  # page 2: astrometry_2 (zoom on the line)


    # here, do the work to plot the astrometry vs velocity, using the line_center and line_width to select the relevant wavelengths

    # Astrometry over the line, using a polynomial continuum
    # ------------------------------------------------------
    # The astrometry is estimated and plotted over the full left->right span
    # (`fit_aera`, the line plus its two side windows), matching the region
    # evaluated by compute_smoothed_line. Since the variable-projection now
    # projects onto the smoothed continuum, the 2x2 system depends on the
    # polynomial degree and is rebuilt for each one (solve_astrometry_gain).
    cont_idx, fit_aera = line_fit_region(line_aera)
    data_b_line = data_b[..., fit_aera]
    J_blocks_line = J_blocks[..., fit_aera, :]

    astrometry_wave = wave[fit_aera]
    # Doppler velocity (km/s)
    velocity = c * (astrometry_wave - line_center) / line_center

    # Solve the variable-projection 2x2 system over the line for a list of
    # polynomial continuum degrees; each degree yields one astrometry_xy track.
    astrometry_xy_list = []
    for poly_deg in poly_deg_values:
        # Estimate the continuum under the line (polynomial fit on the side
        # windows) instead of the notch-Hanning smoothing.
        data_smoothed_line = compute_smoothed_line(data_normalized, wave, line_aera, poly_deg)
        sm_b_line = np.stack([data_smoothed_line[:, 1:-1][i_cube, j_step]
                                for (i_cube, j_step) in valid_indices], axis=0)
        astrometry_xy, _ = solve_astrometry_gain(J_blocks_line, data_b_line, sm_b_line)  # (n_fit, 2)
        astrometry_xy_list.append(astrometry_xy)

    # Compare RA and DEC astrometry over the line for the different poly_deg
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), num="astromet_comparison_poly",
                                clear=True, sharex=True)
    axes[1].sharey(axes[0])
    for poly_deg, astrometry_xy in zip(poly_deg_values, astrometry_xy_list):
        axes[0].plot(astrometry_wave, astrometry_xy[:, 0], alpha=0.8, label=f"{poly_deg}")
        axes[1].plot(astrometry_wave, astrometry_xy[:, 1], alpha=0.8, label=f"{poly_deg}")
    # Flux over the same wavelength span (fit_aera)
    axes[2].plot(astrometry_wave, flux_scaled[fit_aera].T, 'r', alpha=0.5)
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
    pdf.savefig(fig)  # page 3: astrometry_3 (poly_deg comparison)


    # Compare separation and PA astrometry over the line for the different poly_deg
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), num="astromet_comparison_poly_sepPA",
                                clear=True, sharex=True)
    for poly_deg, astrometry_xy in zip(poly_deg_values, astrometry_xy_list):
        separation = np.hypot(astrometry_xy[:, 0], astrometry_xy[:, 1])
        PA_deg = np.degrees(np.arctan2(astrometry_xy[:, 0], astrometry_xy[:, 1]))
        axes[0].plot(astrometry_wave, separation, alpha=0.8, label=f"{poly_deg}")
        axes[1].plot(astrometry_wave, PA_deg, alpha=0.8, label=f"{poly_deg}")
    # Flux over the same wavelength span (fit_aera)
    axes[2].plot(astrometry_wave, flux_scaled[fit_aera].T, 'r', alpha=0.5)
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
    pdf.savefig(fig)  # page 4: astrometry_3 separation/PA (poly_deg comparison)



    fig, ax = plt.subplots(1, 1, figsize=(8, 6), num="astrometry_scatter", clear=True)
    # Widen the plotted region beyond the line by this factor (adjustable), so a
    # few extra points slightly outside the line area are shown too.
    mask_width_scale = 1.
    line_aera_wide = ((wave > line_center - line_width/2*mask_width_scale) &
                      (wave < line_center + line_width/2*mask_width_scale))
    # Only show the points falling inside the (widened) line region
    line_mask = line_aera_wide[fit_aera]
    flux_scaled_filtered = flux_scaled[fit_aera][line_mask]  - np.min(flux_scaled[fit_aera])
    velocity_line = velocity[line_mask]
    for astrometry_xy in astrometry_xy_list[-2:-1]:
        astrometry_xy = astrometry_xy[line_mask]
        scatter = ax.scatter(astrometry_xy[:, 0], astrometry_xy[:, 1], c=velocity_line, s=flux_scaled_filtered*1000, cmap='RdBu_r', alpha=0.6)
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

    # fig.savefig("astrometry_scatter_PA.png", dpi=300)
    pdf.savefig(fig)  # page 4: astrometry_scatter_PA
    pdf.close()
    print(f"All figures saved to {pdf_filename}")

    # Save the astrometric results to a FITS file (mirrors run_createCouplingMap)
    astrometry_shift_all = np.stack(astrometry_shift_list, axis=0)  # (n_hanning, Nwave, 2)
    astrometry_xy_all = np.stack(astrometry_xy_list, axis=0)        # (n_poly, n_line, 2)
    hdul = fits.HDUList([
        fits.PrimaryHDU(header=new_header),
        fits.ImageHDU(data=np.asarray(wave, dtype=float), name='WAVE'),
        fits.ImageHDU(data=np.asarray(flux_scaled, dtype=float), name='FLUX_SCALED'),
        fits.ImageHDU(data=np.asarray(astrometry_shift_all, dtype=float), name='ASTROMETRY_SHIFT'),
        fits.ImageHDU(data=np.asarray(astrometry_xy_all, dtype=float), name='ASTROMETRY_XY'),
        fits.ImageHDU(data=np.asarray(x_hanning_values, dtype=float), name='X_HANNING'),
        fits.ImageHDU(data=np.asarray(poly_deg_values, dtype=float), name='POLY_DEG'),
    ])
    hdul.writeto(output_filename, overwrite=True)
    print(f"Astrometry results saved to {output_filename}")



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
        file_patterns = ["/Users/slacour/DATA/FIRST/20260625/preproc/firstpl_2026-06-25T09h3[2-9]*_HD163296_P.fits"]
        wave_patterns = ["/Users/slacour/DATA/FIRST/20260625/wavemaps/"]

        PA= 162
        line_width= 1.3
        line_center = 656.4
        file_patterns = ["/Users/slacour/DATA/FIRST/20260625/preproc/firstpl_2026-06-25T08h59m59s_HD142527_P.fits",
                         "/Users/slacour/DATA/FIRST/20260625/preproc/firstpl_2026-06-25T09h01m49s_HD142527_P.fits",
                            "/Users/slacour/DATA/FIRST/20260625/preproc/firstpl_2026-06-25T09h19m55s_HD142527_P.fits",
                         ]
        # line_center=656.17
        # PA=28
        # file_patterns = ["/Users/slacour/DATA/FIRST/20260625/preproc/firstpl_2026-06-25T12h18*_ALTAIR_P.fits"]
        # file_patterns = ["/Users/slacour/DATA/FIRST/20260625/preproc/firstpl_2026-06-25T12h2*_ALTAIR_P.fits"]
        # file_patterns = ["/Users/slacour/DATA/FIRST/20260625/preproc/firstpl_2026-06-25T15h2[7-9]*_ALTAIR_P.fits"]

        # file_patterns = ["/Users/slacour/DATA/LANTERNE/20260114/preproc/*14T20h56*.fits"]
        # file_patterns = ["/Users/slacour/DATA/LANTERNE/20260114/preproc/*14T21h10*.fits"]
        # wave_patterns = ["/Users/slacour/DATA/LANTERNE/20251231/wavemaps/"]
        
        # #PDS70
        # file_patterns = ["/Users/slacour/DATA/LANTERNE/20260306/preproc/firstpl*_PDS70_P.fits"]
        # wave_patterns = ["/Users/slacour/DATA/LANTERNE/20260307/wavemaps/"]
        # flat_patterns = ["/Users/slacour/DATA/LANTERNE/20260114/flatmaps/"]

        # object_name = "HD163296"
        # modID = 2
        # file_patterns = ["/Users/slacour/DATA/LANTERNE/20260306/preproc/firstpl**.fits"]
        # wave_patterns = ["/Users/slacour/DATA/LANTERNE/20260307/wavemaps/"]

        
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
