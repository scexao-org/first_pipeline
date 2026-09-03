"""H-alpha continuum subtraction and spatial-correlation imaging."""
#%%

import getpass
import os
import sys

src_dir = os.path.join(os.path.dirname(__file__), '..')
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)


import getpass
import matplotlib
# if "VSCODE_PID" in os.environ:
#     matplotlib.use('macosx')
# elif os.environ.get('SPYDER_DEBUG_FILE'):
#     print("Running in Spyder")
# else:
#     matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import TwoSlopeNorm
from scipy.spatial import cKDTree

from first_pipeline_shared.classes.runPL_class_fileList import FileList
from first_pipeline_shared.classes.runPL_class_flatMap import FlatMap
from first_pipeline_shared.classes.runPL_class_waveMap import WaveMap


def line_fit_region(line_mask):
    """Return continuum side-band indices for a spectral-line mask."""
    line_indices = np.flatnonzero(line_mask)
    if len(line_indices) < 2:
        raise ValueError("The selected line region must contain at least two wavelength samples.")
    line_start, line_stop = line_indices[0], line_indices[-1] + 1
    line_width = line_stop - line_start
    continuum_width = 2 * line_width
    left = np.arange(max(0, line_start - continuum_width), line_start)
    right = np.arange(line_stop, min(len(line_mask), line_stop + continuum_width))
    continuum_indices = np.concatenate((left, right))
    if len(continuum_indices) < 3:
        raise ValueError("Not enough wavelength samples around the line to fit a continuum.")
    return continuum_indices


def fit_continuum(data, wave, line_mask, polynomial_degree):
    """Fit each position/output spectrum from side bands around the line."""
    continuum_indices = line_fit_region(line_mask)
    if len(continuum_indices) <= polynomial_degree:
        raise ValueError("Continuum fit requires more samples than the polynomial degree.")

    data_shape = data.shape
    coefficients = np.polyfit(
        wave[continuum_indices],
        data[..., continuum_indices].reshape(-1, len(continuum_indices)).T,
        polynomial_degree,
    )
    continuum = np.polynomial.polynomial.polyval(
        wave, coefficients[::-1], tensor=True).reshape(data_shape)
    return continuum


def decompose_data_svd(data):
    """Return a compact SVD of position-dependent spectra.

    ``data`` has shape ``(position, output, wavelength)``. Each position is
    represented by one row, with output and wavelength concatenated into the
    feature axis. Columns are mean-centered before decomposition.
    """
    if data.ndim != 3:
        raise ValueError("data must have shape (position, output, wavelength).")

    matrix = data.reshape(data.shape[0], -1)
    column_mean = np.nanmean(matrix, axis=0)
    matrix = np.where(np.isfinite(matrix), matrix, column_mean)
    matrix_centered = matrix - column_mean
    left_vectors, singular_values, right_vectors = np.linalg.svd(
        matrix_centered, full_matrices=False)
    return left_vectors, singular_values, right_vectors, column_mean


def plot_object_spectrum(data, continuum, wave, line_center, line_width):
    """Plot the mean object spectrum and fitted continuum."""
    object_spectrum = np.nanmean(data, axis=(0, 1))
    object_continuum = np.nanmean(continuum, axis=(0, 1))

    fig, spectrum_axis = plt.subplots(
        num="Object spectrum", clear=True, figsize=(11, 5))
    line_start = line_center - line_width / 2
    line_stop = line_center + line_width / 2
    line_mask = (wave >= line_start) & (wave <= line_stop)
    continuum_indices = line_fit_region(line_mask)
    fit_indices = np.union1d(np.flatnonzero(line_mask), continuum_indices)
    fit_half_width = max(np.abs(wave[fit_indices] - line_center))
    display_indices = np.flatnonzero(
        np.abs(wave - line_center) <= 1.22 * fit_half_width)
    spectrum_axis.plot(wave, object_spectrum, color='black', label='Object spectrum')
    spectrum_axis.plot(wave[display_indices], object_continuum[display_indices],
                       color='tab:orange', label='Continuum fit')
    spectrum_axis.axvspan(line_start, line_stop, color='tab:red', alpha=0.15)
    spectrum_axis.axvline(line_center, color='tab:red', linewidth=0.8)
    spectrum_axis.set(xlabel='Wavelength (nm)', ylabel='Mean flux')
    spectrum_axis.grid(True, alpha=0.3)
    spectrum_axis.legend()
    fig.tight_layout()
    return fig


def local_spatial_correlation(line_flux, continuum_flux, positions, neighbours):
    """Correlate line residual and continuum flux over nearby sampled positions."""
    if len(positions) < 3:
        raise ValueError("At least three observed positions are required for correlation.")
    neighbour_count = min(neighbours, len(positions))
    _, indices = cKDTree(positions).query(positions, k=neighbour_count)
    indices = np.atleast_2d(indices)
    correlation = np.full(len(positions), np.nan)
    for position_index, nearby_indices in enumerate(indices):
        x_values = line_flux[nearby_indices]
        y_values = continuum_flux[nearby_indices]
        valid = np.isfinite(x_values) & np.isfinite(y_values)
        if np.count_nonzero(valid) >= 3 and np.ptp(x_values[valid]) > 0 and np.ptp(y_values[valid]) > 0:
            correlation[position_index] = np.corrcoef(x_values[valid], y_values[valid])[0, 1]
    return correlation


def plot_offset_covariance(variance_by_offset, unique_offsets, sample_counts):
    """Plot the wavelength-averaged, count-normalized lagged covariance."""
    covariance = variance_by_offset / sample_counts
    nonzero_offsets = np.any(unique_offsets != 0, axis=1)
    background = covariance[nonzero_offsets]
    background_median = np.nanmedian(background)
    robust_scale = 1.4826 * np.nanmedian(np.abs(background - background_median))
    covariance_score = (covariance - background_median) / robust_scale
    candidate_indices = np.flatnonzero(nonzero_offsets & np.isfinite(covariance_score))
    candidate_index = candidate_indices[np.argmax(np.abs(covariance_score[candidate_indices]))]

    color_limit = np.nanmax(np.abs(covariance)+1)
    color_norm = TwoSlopeNorm(vcenter=0, vmin=-color_limit, vmax=color_limit)
    fig, covariance_axis = plt.subplots(
        num="Lagged covariance", clear=True, figsize=(7, 6))
    points = covariance_axis.scatter(
        unique_offsets[:, 0], unique_offsets[:, 1], c=covariance, s=70,
        cmap="RdBu_r", norm=color_norm)
    covariance_axis.scatter(
        unique_offsets[candidate_index, 0], unique_offsets[candidate_index, 1],
        facecolors="none", edgecolors="black", linewidths=1.5, s=150)
    fig.colorbar(points, ax=covariance_axis, label="Mean lagged covariance")
    covariance_axis.set(xlabel="x offset", ylabel="y offset", aspect="equal")
    covariance_axis.grid(True, alpha=0.3)
    fig.tight_layout()
    return (fig, covariance, candidate_index, unique_offsets[candidate_index],
            covariance_score[candidate_index])


def process_halpha_imaging(file_patterns, object_name=None, dark_patterns=None,
                           flat_patterns=None, wave_patterns=None, modID=None,
                           modScale=None, wollaston=None, line_center=656.28,
                           line_width=2.0, polynomial_degree=2, neighbours=12):
    """Create H-alpha residual and local spatial-correlation products.

    Each observed modulation position is continuum-subtracted output by output.
    The line residual is integrated across ``line_width``; its local Pearson
    correlation with the continuum flux is then evaluated over neighbouring
    observed sky positions.
    """
    if dark_patterns is None:
        dark_patterns = file_patterns
    if flat_patterns is None:
        folder = os.path.dirname(file_patterns[0])
        flat_patterns = file_patterns + [os.path.join(folder, "../flatmaps")]
    if wave_patterns is None:
        folder = os.path.dirname(file_patterns[0])
        wave_patterns = file_patterns + [os.path.join(folder, "../wavemaps")]

    file_list = FileList(file_patterns, data_type="OBJECT", first_type="PREPROC",
                         wollaston=wollaston, object_name=object_name, modID=modID,
                         modScale=modScale)
    file_list.make_association(dark_patterns=dark_patterns)
    flat_file = file_list.get_flatmap_file(flat_patterns)
    wave_file = file_list.get_wavemap_file(wave_patterns)
    if wave_file is None:
        raise FileNotFoundError("No wavelength map found; H-alpha imaging requires calibrated wavelengths.")

    flat_map = FlatMap(flat_file) if flat_file is not None else None
    wave_map = WaveMap(wave_file)
    datalist = file_list.extract_data_from_list(flatMap=flat_map, waveMap=wave_map)

    data = np.concatenate([cube.data.reshape(-1, cube.Noutput, cube.Nwave) for cube in datalist])
    positions = np.concatenate([cube.compute_xy_sky().reshape(-1, 2) for cube in datalist])
    wave = datalist[0].wave
    line_mask = np.abs(wave - line_center) <= line_width / 2
    continuum = fit_continuum(data, wave, line_mask, polynomial_degree)
    residual = data - continuum

    return datalist


if __name__ == '__main__':
    from matplotlib.pyplot import *
    ion()
    """Run H-alpha imaging with development defaults for interactive debugging."""
    if getpass.getuser() == 'slacour':
        file_patterns = [
            '/Users/slacour/DATA/FIRST/20260828/preproc/firstpl_2026-08-28T*'
        ]
        wave_patterns = ['/Users/slacour/DATA/FIRST/20260828/wavemaps/']
        flat_patterns = None
        dark_patterns = None
        object_name = "2MASSJ193918721455542"
        modID = None
        modScale = 30
        wollaston = None
        line_center = 655.58 
        line_width = 0.7
        polynomial_degree = 2
        neighbours = 12
    else:
        raise RuntimeError(
            'Set file_patterns and calibration paths in the __main__ block '
            'before running this module interactively.')

    print(f'Development file patterns: {file_patterns}')
    datalist = process_halpha_imaging(
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
        polynomial_degree=polynomial_degree,
        neighbours=neighbours,
    )

    data = np.concatenate([cube.data.reshape(-1, cube.Noutput, cube.Nwave) for cube in datalist])
    xmod = np.concatenate([cube.xmod.reshape(-1) for cube in datalist])
    ymod = np.concatenate([cube.ymod.reshape(-1) for cube in datalist])

    positions = np.concatenate([cube.compute_xy_sky().reshape(-1, 2) for cube in datalist])
    wave = datalist[0].wave
    line_mask = np.abs(wave - line_center) <= line_width / 2
    continuum = fit_continuum(data, wave, line_mask, polynomial_degree)
    plot_object_spectrum(data, continuum, wave, line_center, line_width)

    line_mask = (wave >= line_center - line_width) & (wave <= line_center + line_width)
    data = data[:,:,line_mask]
    continuum = continuum[:,:,line_mask]
    flat = (data*continuum).sum(axis=0)/(continuum**2).sum(axis=0)
    flat_mean = flat.mean(axis=0)
    residual = data - continuum*flat

    variance=[]
    xdiff=[]
    ydiff=[]
    for i in range(-3,3):
        v = (residual[i+18:i+-19]*residual[18:-19]).sum(axis=(1))
        x = xmod[i+18:i+-19] - xmod[18:-19]
        y = ymod[i+18:i+-19] - ymod[18:-19]
        if i !=0:
            variance.append(v)
            xdiff.append(x)
            ydiff.append(y)

    variance = np.array(variance)
    xdiff = np.array(xdiff)
    ydiff = np.array(ydiff)
    offsets = np.column_stack((xdiff.ravel(), ydiff.ravel()))
    offset_tolerance = 0.01
    offset_bins = np.rint(offsets / offset_tolerance).astype(np.int64)
    unique_bins = np.unique(offset_bins, axis=0)
    unique_offsets = unique_bins * offset_tolerance
    xdiff_bins = offset_bins[:, 0].reshape(xdiff.shape)
    ydiff_bins = offset_bins[:, 1].reshape(ydiff.shape)
    offset_masks = np.array([
        (xdiff_bins == x_bin) & (ydiff_bins == y_bin)
        for x_bin, y_bin in unique_bins
    ])
    sample_counts = offset_masks.sum(axis=(1, 2))
    variance_by_offset = np.array([
        variance[offset_mask].sum(axis=0)
        for offset_mask in offset_masks
    ])
    variance_by_offset = variance_by_offset[:, 5]
    _, covariance, candidate_index, candidate_offset, candidate_score = plot_offset_covariance(
        variance_by_offset, unique_offsets, sample_counts)
    print(
        "Strongest non-zero covariance candidate: "
        f"{candidate_offset} (robust score {candidate_score:.2f})"
    )

    

    



#%%
    # left_vectors, singular_values, right_vectors, column_mean = decompose_data_svd(data)

    # figure("Singular values", clear=True)
    # semilogy(singular_values, 'o-')
    # xlabel('SVD component')
    # ylabel('Singular value')

    # figure("First SVD spatial component", clear=True)
    # scatter(positions[:, 0], positions[:, 1], c=left_vectors[:, 0], cmap='RdBu_r')
    # colorbar(label='First left singular vector')
    # xlabel('RA offset (mas)')
    # ylabel('DEC offset (mas)')
    # axis('equal')




# %%
