"""H-alpha continuum subtraction and spatial-correlation imaging."""
#%%

import getpass
import os
import sys

src_dir = os.path.join(os.path.dirname(__file__), '..')
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from matplotlib.backends.backend_pdf import PdfPages
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
    width = line_stop - line_start
    left = np.arange(max(0, line_start - width), line_start)
    right = np.arange(line_stop, min(len(line_mask), line_stop + width))
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

    wavelength_step = np.abs(np.median(np.diff(wave)))
    line_flux = np.nansum(residual[..., line_mask], axis=(1, 2)) * wavelength_step
    continuum_flux = np.nanmean(continuum[..., line_mask], axis=(1, 2))
    correlation = local_spatial_correlation(line_flux, continuum_flux, positions, neighbours)

    header = datalist[-1].header.copy()
    header['X_FIRTYP'] = 'HALPHAIM'
    header['Q_HALINE'] = (line_center, 'H-alpha line center (nm)')
    header['Q_HALWID'] = (line_width, 'H-alpha integration width (nm)')
    header['Q_HAPDEG'] = (polynomial_degree, 'continuum polynomial degree')
    header['Q_HANEIG'] = (neighbours, 'local correlation neighbour count')
    header['Q_HAMETH'] = ('LOCALPEAR', 'local Pearson line/continuum correlation')

    output_dir = os.path.join(datalist[-1].dirname, '../halpha_imaging')
    os.makedirs(output_dir, exist_ok=True)
    filename_root = os.path.splitext(os.path.basename(datalist[-1].filename))[0]
    output_filename = os.path.join(output_dir, filename_root + '_HALPHA.fits')
    pdf_filename = os.path.splitext(output_filename)[0] + '.pdf'

    fits.HDUList([
        fits.PrimaryHDU(header=header),
        fits.ImageHDU(data=wave, name='WAVE'),
        fits.ImageHDU(data=positions, name='XY'),
        fits.ImageHDU(data=continuum, name='CONTINUUM'),
        fits.ImageHDU(data=residual, name='RESIDUAL'),
        fits.ImageHDU(data=line_flux, name='HALPHA_FLUX'),
        fits.ImageHDU(data=continuum_flux, name='CONTINUUM_FLUX'),
        fits.ImageHDU(data=correlation, name='CORRELATION'),
    ]).writeto(output_filename, overwrite=True)

    with PdfPages(pdf_filename) as pdf:
        mean_spectrum = np.nanmean(data, axis=(0, 1))
        mean_continuum = np.nanmean(continuum, axis=(0, 1))
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(wave, mean_spectrum, label='Observed spectrum')
        ax.plot(wave, mean_continuum, label='Polynomial continuum')
        ax.axvspan(line_center - line_width / 2, line_center + line_width / 2,
                   color='tab:red', alpha=0.15)
        ax.set(xlabel='Wavelength (nm)', ylabel='Flux', title='H-alpha continuum fit')
        ax.legend()
        pdf.savefig(fig)
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
        for ax, values, title, cmap, limits in (
            (axes[0], line_flux, 'H-alpha residual flux', 'RdBu_r', None),
            (axes[1], correlation, 'Local line-continuum correlation', 'coolwarm', (-1, 1)),
        ):
            scatter = ax.scatter(positions[:, 0], positions[:, 1], c=values, cmap=cmap,
                                 vmin=None if limits is None else limits[0],
                                 vmax=None if limits is None else limits[1])
            fig.colorbar(scatter, ax=ax)
            ax.set(xlabel='RA offset (mas)', ylabel='DEC offset (mas)', title=title, aspect='equal')
        pdf.savefig(fig)
        plt.close(fig)

    print(f'H-alpha imaging results saved to {output_filename}')
    print(f'H-alpha diagnostic figures saved to {pdf_filename}')
    return datalist


if __name__ == '__main__':
    """Run H-alpha imaging with development defaults for interactive debugging."""
    if getpass.getuser() == 'slacour':
        file_patterns = [
            '/Users/slacour/DATA/FIRST/20260625/preproc/'
            'firstpl_2026-06-25T09h3[2-9]*_HD163296_P.fits',
        ]
        wave_patterns = ['/Users/slacour/DATA/FIRST/20260625/wavemaps/']
        flat_patterns = None
        dark_patterns = None
        object_name = None
        modID = None
        modScale = None
        wollaston = None
        line_center = 656.28
        line_width = 2.0
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
    positions = np.concatenate([cube.compute_xy_sky().reshape(-1, 2) for cube in datalist])
    wave = datalist[0].wave
    line_mask = np.abs(wave - line_center) <= line_width / 2
    continuum = fit_continuum(data, wave, line_mask, polynomial_degree)
    residual = data - continuum

# %%
