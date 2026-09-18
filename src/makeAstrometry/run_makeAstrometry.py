#%%
"""
FIRST Pipeline - Astrometric Analysis (orchestration)

Measures the wavelength-dependent photocentre shift (spectro-astrometry) of a
source across a spectral line from preprocessed FIRST photonic-lantern data.

This module only does the I/O and the bookkeeping:

    load_astrometry_data   -> FITS files to arrays (data, variance, dither)
    select_good_data       -> flux / correlation quality masks (+ diagnostics)
    analyse_astrometry     -> step 1: normalisation and astrometry fit
                              (makeAstrometry.astrometry_core)
    astrometry_scale.calibrate_attenuation
                           -> step 2 (optional): PSF variability and the
                              attenuation factor kappa of the amplitude
                              (makeAstrometry.astrometry_scale)
    make_astrometry_figures, save_astrometry_results
    process_astrometric_data -> runs the steps above (CLI entry point)

All the numerics live in ``astrometry_core`` (estimation) and
``astrometry_scale`` (scale), pure numpy modules that can also be run on a
saved ``.npz`` data set or on ``simulate_lantern`` simulations.

Method (see astrometry_core for the equations)
-----------------------------------------------
Around each dither pose a *local* response Jacobian J (flux change per mas of
sky motion, per output and wavelength) is measured from the neighbouring
poses.  A separable least-squares fit then solves, per wavelength,

    gain * data = continuum + J . a(lambda)

for the RA/DEC photocentre shift ``a`` shared by all outputs and poses, the
per-(output, wavelength) gain being eliminated analytically.  Photon noise on
J is corrected (errors-in-variables).  PSF jitter and deformation between the
poses of a block are NOT photon noise: they attenuate ``a`` by an achromatic,
isotropic factor kappa that must be calibrated by simulation.  The wavelength
structure of ``a`` (line profile, position angle) is unbiased.

Created on Wed May 21 22:56:25 2025
@author: slacour
"""

import sys
import os
# Add src directory to path for imports to work in both interactive and package contexts
if os.path.join(os.path.dirname(__file__), '..') not in sys.path:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import getpass
import numpy as np
from scipy.constants import speed_of_light
from typing import List

import matplotlib
if "VSCODE_PID" in os.environ:
    matplotlib.use('macosx')
elif os.environ.get('SPYDER_DEBUG_FILE'):
    print("Running in Spyder")
else:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ion()

from astroplan import Observer
from astropy.time import Time
from astropy.io import fits

from first_pipeline_shared.classes.runPL_class_flatMap import FlatMap
from first_pipeline_shared.classes.runPL_class_waveMap import WaveMap
from first_pipeline_shared.classes.runPL_class_fileList import FileList
from first_pipeline_shared.classes.runPL_class_dataCube import DataCube
from first_pipeline_shared.libraries import runPL_library_io as runlib_io
from first_pipeline_shared.libraries import runPL_library_plots as runlib_plots
from first_pipeline_shared.libraries import runPL_library_linalg as runlib_linalg

from makeAstrometry import astrometry_core as core
from makeAstrometry import astrometry_scale as scale
from makeAstrometry.astrometry_plots import (
    plot_correlation_lag_histogram,
    plot_astrometry_comparison, plot_separation_pa, plot_astrometry_scatter,
    plot_astrometry_with_errors, plot_kappa_diagnostics)


# Subaru Observatory instance for timing
subaru = Observer.at_site("Subaru")

# Polynomial degrees tested for the continuum under the line
POLY_DEG_VALUES = (2, 3, 4, 5)
# Degree used for the reference track (figures with error bars, scatter plot)
POLY_DEG_REFERENCE = 3


def get_filelist_astrometry(file_patterns, dark_patterns=None, flat_patterns=None,
                            wave_patterns=None, object_name=None, modID=None,
                            modScale=None, wollaston=None):
    """
    Create file list for astrometry analysis with calibration associations.

    Parameters
    ----------
    file_patterns : list
        List of file patterns to search for OBJECT data
    dark_patterns, flat_patterns, wave_patterns : list, optional
        Patterns for dark files, flat field maps and wavelength maps
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
    flatMap : FlatMap or None
    waveMap : WaveMap or None
    object_name : str
    """
    if modID is None:
        modID = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

    fileList = FileList(file_patterns, data_type="OBJECT", first_type='PREPROC',
                        wollaston=wollaston, object_name=object_name,
                        modID=modID, modScale=modScale)

    # Constrain the selection to the first data set found
    object_name = fileList.header.get('OBJECT', None)
    wollaston = fileList.header.get('X_FIRWOL', None)
    modID = fileList.header.get('X_FIRMID', None)
    modScale = fileList.header.get('X_FIRMSC', None)
    fileList = FileList(file_patterns, data_type="OBJECT", first_type='PREPROC',
                        wollaston=wollaston, object_name=object_name,
                        modID=modID, modScale=modScale)

    fileList.make_association(dark_patterns=dark_patterns)
    file_flat = fileList.get_flatmap_file(flat_patterns)
    file_wave = fileList.get_wavemap_file(wave_patterns)
    flatMap = FlatMap(file_flat) if file_flat is not None else None
    waveMap = WaveMap(file_wave) if file_wave is not None else None
    return fileList, flatMap, waveMap, object_name


def check_observatory_status():
    """Return a message saying whether it is night at Subaru Observatory."""
    if subaru.is_night(Time.now()):
        return "It's night at Subaru Observatory."
    return "It's day at Subaru Observatory."


# ---------------------------------------------------------------------------
# Step 1: load
# ---------------------------------------------------------------------------

def load_astrometry_data(file_patterns, object_name=None, dark_patterns=None,
                         flat_patterns=None, wave_patterns=None, modID=None,
                         modScale=None, wollaston=None):
    """Read the preprocessed cubes and return the arrays needed by the fit.

    Returns a dict with ``datalist`` (list of DataCube), ``datacube`` and
    ``datacube_var`` (Ncube, Npose, Noutput, Nwave), ``flux`` (same shape,
    used for the quality masks and the spectrum normalisation), ``wave``
    (Nwave), ``xmod``/``ymod`` (modulation commands), ``ra_dec`` (Ncube,
    Npose, 2) sky positions in mas, and ``object_name``.
    """
    if dark_patterns is None:
        dark_patterns = file_patterns
    if flat_patterns is None and file_patterns:
        folder = os.path.dirname(file_patterns[0])
        flat_patterns = file_patterns + [os.path.join(folder, "../flatmaps"),
                                         os.path.join(folder, "flatmaps")]
    if wave_patterns is None and file_patterns:
        folder = os.path.dirname(file_patterns[0])
        wave_patterns = file_patterns + [os.path.join(folder, "../wavemaps"),
                                         os.path.join(folder, "wavemaps")]

    fileList, flatMap, waveMap, object_name = get_filelist_astrometry(
        file_patterns, dark_patterns, flat_patterns, wave_patterns,
        object_name, modID, modScale, wollaston)
    datalist: List[DataCube] = fileList.extract_data_from_list(
        flatMap=flatMap, waveMap=waveMap)

    return dict(
        datalist=datalist,
        object_name=object_name,
        flux=np.concatenate([d.flux for d in datalist]),
        datacube=np.concatenate([d.data for d in datalist]),
        datacube_var=np.concatenate([d.variance for d in datalist]),
        wave=datalist[0].wave,          # all cubes share the wavelength grid
        xmod=np.concatenate([d.xmod for d in datalist]),
        ymod=np.concatenate([d.ymod for d in datalist]),
        ra_dec=np.concatenate([d.compute_xy_sky() for d in datalist]),
    )


# ---------------------------------------------------------------------------
# Step 2: data quality
# ---------------------------------------------------------------------------

def select_good_data(flux, datacube, xmod, ymod, threshold_corr=0.5):
    """Flux-threshold and adjacent-step-correlation quality masks.

    Returns ``good_pose`` (Ncube, Npose) and the two diagnostic figures
    (flux map, correlation histogram).  The correlation mask is only shown,
    not applied (as in the previous version of the pipeline).
    """
    good_pose, _ = runlib_linalg.flux_filtering(flux)
    print(f"* Percentage of good data: {100 * np.mean(good_pose):.1f} % (flux threshold)")
    _, data_corr_lag = runlib_linalg.correlation_filtering(
        datacube, threshold_corr=threshold_corr)

    figures = [runlib_plots.plot_flux_map(flux.mean(axis=2)[0], xmod[0], ymod[0])]
    fig, _ = plot_correlation_lag_histogram(data_corr_lag, good_pose,
                                            threshold_corr=threshold_corr)
    figures.append(fig)
    return good_pose, figures


# ---------------------------------------------------------------------------
# Step 3: analysis
# ---------------------------------------------------------------------------

def analyse_astrometry(datacube, datacube_var, flux, ra_dec, wave, good_pose,
                       line_center, line_width, jacobian_method='local',
                       jac_half_window=1, jac_fit_order=1, n_cubes_average=1,
                       poly_deg_values=POLY_DEG_VALUES, verbose=True):
    """Normalise the data around the line and fit the astrometry.

    Parameters
    ----------
    datacube, datacube_var : (Ncube, Npose, Noutput, Nwave)
    flux : (Ncube, Npose, Nwave) total flux (figures only)
    ra_dec : (Ncube, Npose, 2) dither positions in mas
    wave : (Nwave,) wavelength grid in nm
    good_pose : (Ncube, Npose) boolean quality mask
    line_center, line_width : nm; the working window is +-1.5 line_width,
        the line (excluded from the continuum fits) is +-line_width/2
    jacobian_method : 'local' (finite differences on 2*jac_half_window+1
        poses, optionally averaged over n_cubes_average cubes) or 'spatial'
        (gradient of a polynomial model of the flux versus dither position)
    poly_deg_values : degrees of the continuum polynomial to try

    Returns
    -------
    dict : the output of ``astrometry_core.fit_astrometry`` plus the
        wavelength bookkeeping (``work_aera``, ``wave_aera``, ``velocity``,
        ``mean_flux``, ``flux_scaled``) and the parameters used.
    """
    # Wavelength window: continuum side windows + line
    work_aera = np.abs(wave - line_center) < 1.5 * line_width
    wave_aera = wave[work_aera]
    velocity = speed_of_light / 1e3 * (wave - line_center) / line_center

    # Normalise every output by its own mean spectrum (over cubes and poses):
    # the Jacobian at a given wavelength scales with the flux at that
    # wavelength (emission line!).  ``mean_flux`` (total flux spectrum) is
    # only used for the figures.
    mean_flux = np.nanmean(flux, axis=(0, 1))                  # (Nwave,)
    data_n, var_n, spectrum, good = core.normalize_by_spectrum(
        datacube[..., work_aera], datacube_var[..., work_aera])
    good &= good_pose[:, :, None, None]
    n_bad = np.sum(~good)
    if n_bad and verbose:
        print(f"* {n_bad} samples ignored (non-finite or flagged by the flux filter)")

    result = core.fit_astrometry(
        data_n, var_n, ra_dec, wave_aera, line_center, line_width,
        half_window=jac_half_window, fit_order=jac_fit_order,
        poly_deg_values=poly_deg_values, good=good,
        jacobian_method=jacobian_method, n_cubes_average=n_cubes_average,
        verbose=verbose)

    result.update(
        work_aera=work_aera, wave_aera=wave_aera, velocity=velocity,
        mean_flux=mean_flux, spectrum=spectrum,
        flux_scaled=mean_flux[..., work_aera] / np.nanmax(mean_flux[..., work_aera]),
        line_center=line_center, line_width=line_width,
        jac_half_window=jac_half_window, jac_fit_order=jac_fit_order)
    return result


# ---------------------------------------------------------------------------
# Step 4: figures and outputs
# ---------------------------------------------------------------------------

def make_astrometry_figures(result, datalist, object_name, PA):
    """Figures of the fitted astrometry (list of matplotlib figures)."""
    poly_deg_values = result['poly_deg_values']
    ref = POLY_DEG_REFERENCE if POLY_DEG_REFERENCE in poly_deg_values else poly_deg_values[0]
    astrometry_xy_list = [result[p]['astrometry_xy'] for p in poly_deg_values]
    wave_aera, line_aera, fit_aera = result['wave_aera'], result['line_aera'], result['fit_aera']
    lc, lw = result['line_center'], result['line_width']

    figures = []
    fig, _ = plot_astrometry_comparison(
        wave_aera, astrometry_xy_list, poly_deg_values, result['mean_flux'],
        result['work_aera'], fit_aera, object_name, lc, lw)
    figures.append(fig)

    fig, _ = plot_astrometry_with_errors(
        wave_aera, result[ref]['astrometry_xy'], result[ref]['covariance'],
        result['flux_scaled'], fit_aera, object_name, lc, lw, ref)
    figures.append(fig)

    fig, _ = plot_separation_pa(
        wave_aera, astrometry_xy_list, poly_deg_values, result['mean_flux'],
        result['work_aera'], lc, lw, PA)
    figures.append(fig)

    flux_line = result['flux_scaled'][line_aera]
    mod_ids = sorted({d.modID for d in datalist})
    mod_scales = sorted({d.modScale for d in datalist})
    dates = sorted({str(d.date) for d in datalist})
    subtitle = (f"date={dates}, modID={mod_ids}, modScale={mod_scales}, "
                f"files={len(datalist)}, Jacobian={result['jacobian_method']} "
                f"(window {2 * result['jac_half_window'] + 1} poses, "
                f"{result['n_cubes_average']} cube(s))")
    fig, _ = plot_astrometry_scatter(
        result[ref]['astrometry_xy'], result[ref]['covariance'], line_aera,
        result['velocity'][result['work_aera']][line_aera],
        flux_line - flux_line.min(), object_name, lc, lw, ref, PA, subtitle,
        kappa=result.get('kappa'), kappa_err=result.get('kappa_err'))
    figures.append(fig)
    # amplitude attenuation kappa (only when step 2 ran with calibrate_scale=True)
    if 'kappa_table' in result:
        fig, _ = plot_kappa_diagnostics(
            result, title=f"{object_name}: amplitude attenuation kappa of the "
                          f"local-Jacobian astrometry\n{subtitle}")
        figures.append(fig)
    return figures


def save_astrometry_results(result, datalist, figures):
    """Write the ASTROMETRY FITS file and the multi-page PDF next to the data."""
    poly_deg_values = result['poly_deg_values']
    header = datalist[-1].header.copy()
    header['X_FIRTYP'] = 'ASTROMETRY'
    header['Q_ASLINE'] = (result['line_center'], 'line center wavelength (nm)')
    header['Q_ASLWID'] = (result['line_width'], 'line width (nm)')
    header['Q_ASPDEG'] = (str(list(poly_deg_values)), 'polynomial degrees of the continuum fit')
    header['Q_ASJMET'] = (result['jacobian_method'], 'Jacobian estimator (local/spatial)')
    header['Q_ASJWIN'] = (2 * result['jac_half_window'] + 1, 'poses per local Jacobian block')
    header['Q_ASJORD'] = (result['jac_fit_order'], 'order of the local Jacobian fit')
    header['Q_ASNCUB'] = (result['n_cubes_average'], 'cubes averaged for the Jacobian')
    if result.get('kappa'):
        header['Q_ASKAPP'] = (result['kappa'], 'attenuation factor kappa (a_meas = kappa a_true)')
        header['Q_ASKERR'] = (result['kappa_err'], 'uncertainty on kappa')
        header['Q_ASJITT'] = (result['jitter'], 'measured pointing jitter (mas rms)')
        header['Q_ASDEFO'] = (result['deformation'], 'measured PSF flux deformation (rms fraction)')
    header['Q_ASNAME'] = (runlib_io.create_basename(header), 'name of the astrometry file')

    output_dir = os.path.join(datalist[-1].dirname, "../astrometry")
    os.makedirs(output_dir, exist_ok=True)
    output_filename = os.path.join(output_dir, header['Q_ASNAME'])
    astrometry_xy_all = np.stack([result[p]['astrometry_xy'] for p in poly_deg_values])
    covariance_all = np.stack([result[p]['covariance'] for p in poly_deg_values])
    hdul = fits.HDUList([
        fits.PrimaryHDU(header=header),
        fits.ImageHDU(data=np.asarray(result['wave_aera'], dtype=float), name='WAVE'),
        fits.ImageHDU(data=np.asarray(result['flux_scaled'], dtype=float), name='FLUX_SCALED'),
        fits.ImageHDU(data=np.asarray(astrometry_xy_all, dtype=float), name='ASTROMETRY_XY'),
        fits.ImageHDU(data=np.asarray(covariance_all, dtype=float), name='ASTROMETRY_COV'),
        fits.ImageHDU(data=np.asarray(poly_deg_values, dtype=float), name='POLY_DEG'),
        fits.ImageHDU(data=result['line_aera'].astype(np.uint8), name='LINE_MASK'),
    ])
    hdul.writeto(output_filename, overwrite=True)
    print(f"Astrometry results saved to {output_filename}")

    pdf_filename = os.path.splitext(output_filename)[0] + ".pdf"
    from matplotlib.backends.backend_pdf import PdfPages
    with PdfPages(pdf_filename) as pdf:
        for fig in figures:
            pdf.savefig(fig)
    print(f"All figures saved to {pdf_filename}")
    return output_filename, pdf_filename


def print_summary(result):
    """One line per continuum degree: mean shift on the line and noise."""
    line_aera, fit_aera = result['line_aera'], result['fit_aera']
    for poly_deg in result['poly_deg_values']:
        a = result[poly_deg]['astrometry_xy']
        sigma = np.sqrt(np.diagonal(result[poly_deg]['covariance'], axis1=-2, axis2=-1))
        w = 1.0 / sigma[line_aera] ** 2
        mean = (a[line_aera] * w).sum(0) / w.sum(0)
        print(f"* poly {poly_deg}: line mean (RA, DEC) = ({mean[0]:+.4f}, {mean[1]:+.4f}) mas, "
              f"PA = {np.degrees(np.arctan2(mean[0], mean[1])):+.0f} deg, "
              f"sigma/channel on line = {sigma[line_aera].mean():.4f}, "
              f"continuum rms = {a[fit_aera].std():.4f} mas")
        amplitude = np.hypot(*mean)
        if result.get('kappa'):
            k, dk = result['kappa'], result['kappa_err']
            print(f"          amplitude {amplitude:.4f} mas / kappa {k:.3f} = "
                  f"{amplitude / k:.3f} +- {amplitude * dk / k ** 2:.3f} mas (scale error only)")
        else:
            print(f"          amplitude {amplitude:.4f} mas, NOT corrected for the Jacobian "
                  f"attenuation kappa (use calibrate_scale=True)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def process_astrometric_data(
        file_patterns, object_name=None, dark_patterns=None, flat_patterns=None,
        wave_patterns=None, modID=None, modScale=None, wollaston=None,
        line_center=656.28, line_width=3.0, PA=137.0, Ncube_average=1,
        jacobian_method='local', jac_half_window=1, jac_fit_order=1,
        save_npz=None, calibrate_scale=False):
    """
    Measure the wavelength-dependent photocentre shift (spectro-astrometry).

    See the module docstring for the method.  ``PA`` (degrees) is only drawn
    on the figures.  ``Ncube_average`` (odd) averages the local Jacobian over
    that many neighbouring cubes at the same dither position.  ``save_npz``
    optionally saves the normalised working arrays for offline tests with
    ``astrometry_core`` (``datacube``, ``datacube_var``, ``ra_dec``, ``wave``).
    """
    Ncube_average = validate_ncube_average(Ncube_average)

    data = load_astrometry_data(file_patterns, object_name, dark_patterns,
                                flat_patterns, wave_patterns, modID, modScale,
                                wollaston)
    Ncube_average = validate_ncube_average(Ncube_average, data['datacube'].shape[0])

    good_pose, figures = select_good_data(data['flux'], data['datacube'],
                                          data['xmod'], data['ymod'])

    result = analyse_astrometry(
        data['datacube'], data['datacube_var'], data['flux'], data['ra_dec'],
        data['wave'], good_pose, line_center, line_width,
        jacobian_method=jacobian_method, jac_half_window=jac_half_window,
        jac_fit_order=jac_fit_order, n_cubes_average=Ncube_average)
    # ---- step 1 done: a(lambda), PA and statistical errors are final.
    # ---- step 2 (optional): amplitude scale from the PSF variability
    scale.report_jacobian_variability(result)
    if calibrate_scale:
        scale.calibrate_attenuation(result, line_center, line_width)
    print_summary(result)

    if save_npz:
        work = result['work_aera']
        np.savez(save_npz, datacube=data['datacube'][..., work],
                 datacube_var=data['datacube_var'][..., work],
                 ra_dec=data['ra_dec'], wave=data['wave'][work])
        print(f"Working arrays saved to {save_npz}")

    figures += make_astrometry_figures(result, data['datalist'],
                                       data['object_name'], PA)
    save_astrometry_results(result, data['datalist'], figures)
    return result


def validate_ncube_average(n_cubes, n_available=None):
    """Validate the cube-averaging window and fall back to one cube."""
    import warnings
    valid = (isinstance(n_cubes, (int, np.integer))
             and not isinstance(n_cubes, (bool, np.bool_))
             and n_cubes >= 1 and n_cubes % 2 == 1)
    if n_available is not None:
        valid = valid and n_cubes <= n_available
    if not valid:
        limit = f" and no larger than the {n_available} available cubes" if n_available else ""
        warnings.warn(f"Ncube_average={n_cubes!r} is invalid; it must be a positive "
                      f"odd integer{limit}. Falling back to Ncube_average=1.",
                      UserWarning, stacklevel=2)
        return 1
    return int(n_cubes)


if __name__ == "__main__":
    """Run the astrometric analysis with development defaults."""
    print("Running makeAstrometry with development defaults...")
    if getpass.getuser() == "slacour":
        object_name = None
        dark_patterns = None
        flat_patterns = None
        wave_patterns = None
        modID = 9
        modScale = None
        wollaston = None
        PA = 137            # for plotting only
        Ncube_average = 1
        line_center = 656.5
        line_width = 1.8
        calibrate_scale = True
        file_patterns = ["/Users/slacour/DATA/FIRST/20260827/preproc/firstpl_2026-*_HD163296_P.fits"]
        file_patterns = ["/Users/slacour/DATA/FIRST/20260827/preproc/firstpl_2026-08-27T08h[3-4]*P.fits"]
        wave_patterns = ["/Users/slacour/DATA/FIRST/20260827/wavemaps/"]

        # HD142527 (20260625): PA=162, line_width=1.3, line_center=656.4
        # ALTAIR (20260827):   
        object_name = "ALTAIR"
        PA=25
        line_width=1.7
        line_center=656.15
        modID=7 
        modScale=40


    print(f"Development file patterns: {file_patterns}")
    process_astrometric_data(
        file_patterns=file_patterns, object_name=object_name,
        dark_patterns=dark_patterns, flat_patterns=flat_patterns,
        wave_patterns=wave_patterns, modID=modID, modScale=modScale,
        wollaston=wollaston, line_center=line_center, line_width=line_width,
        calibrate_scale=calibrate_scale, PA=PA, Ncube_average=Ncube_average, save_npz="astrometry_working_arrays.npz")
# %%
