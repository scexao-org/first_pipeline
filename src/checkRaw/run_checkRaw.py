#! /usr/bin/env python3
# -*- coding: iso-8859-15 -*-
#%%
"""
FIRST Pipeline - Raw Data Coherence Check Core Algorithms

Looks at raw FIRST Visible Photonic Lantern detector cubes and estimates the
frame-to-frame coherence of the brightest pixels. For each raw file, the 5%
brightest pixels (by time-averaged flux) are selected, their flux time series
across frames is normalized, and the temporal autocorrelation function (ACF)
is computed and averaged over those pixels. A "coherence length" is then
estimated as the lag (in frame number and in seconds) at which the ACF first
drops to 1/e of its zero-lag value.

The same analysis is repeated on the 5% faintest pixels, which carry
essentially no signal, to provide an empirical detector-noise reference
(read noise / dark current). Comparing the two coherence lengths tells
whether the brightest pixels vary frame-to-frame beyond what pure detector
noise would produce (i.e. a real upstream instability) or are consistent
with noise alone.

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

from astropy.io import fits
from astropy.time import Time, TimeDelta
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from first_pipeline_shared.classes.runPL_class_fileList import FileList
plt.ion()


def load_raw_cube(filename):
    """
    Load a raw FITS cube and estimate the time sampling between frames.

    Args:
        filename (str): Path to the raw FITS file.

    Returns:
        tuple: (cube, header, frame_dt) where cube is a (Nframes, ny, nx)
               float32 array, header is the FITS header, and frame_dt is the
               estimated time (in seconds) between two consecutive frames.
    """
    with fits.open(filename) as hdul:
        cube = np.asarray(hdul[0].data, dtype=np.float32)
        header = hdul[0].header.copy()

    Nframes = cube.shape[0]
    frame_dt = header.get('EXPTIME', 1.0)

    # Prefer the real elapsed time between UT-STR and UT-END if available,
    # since readout overhead can make it larger than EXPTIME alone.
    try:
        date = header['DATE-OBS']
        t_start = Time(f"{date} {header['UT-STR']}")
        t_end = Time(f"{date} {header['UT-END']}")
        if t_end < t_start:
            t_end += TimeDelta(1, format='jd')
        frame_dt = (t_end - t_start).sec / Nframes
    except (KeyError, ValueError):
        pass

    return cube, header, frame_dt


def select_brightest_pixels(cube, fraction=0.05):
    """
    Select the brightest pixels of a cube based on their time-averaged flux.

    Args:
        cube (numpy.ndarray): Data cube of shape (Nframes, ny, nx).
        fraction (float): Fraction (0-1) of brightest pixels to keep.

    Returns:
        tuple: (pixel_timeseries, mask, mean_image) where pixel_timeseries has
               shape (Nframes, Npix), mask is a boolean (ny, nx) array and
               mean_image is the time-averaged (ny, nx) image.
    """
    mean_image = cube.mean(axis=0)
    threshold = np.percentile(mean_image, 100 * (1 - fraction))
    mask = mean_image >= threshold
    pixel_timeseries = cube[:, mask]
    return pixel_timeseries, mask, mean_image


def select_faintest_pixels(cube, fraction=0.05):
    """
    Select the faintest pixels of a cube based on their time-averaged flux.

    These pixels carry essentially no signal, so their frame-to-frame
    variations are an empirical measurement of the detector noise (read
    noise / dark current), used as a reference to detect real upstream
    instabilities affecting the brightest pixels.

    Args:
        cube (numpy.ndarray): Data cube of shape (Nframes, ny, nx).
        fraction (float): Fraction (0-1) of faintest pixels to keep.

    Returns:
        tuple: (pixel_timeseries, mask, mean_image), same layout as
               `select_brightest_pixels`.
    """
    mean_image = cube.mean(axis=0)
    threshold = np.percentile(mean_image, 100 * fraction)
    mask = mean_image <= threshold
    pixel_timeseries = cube[:, mask]
    return pixel_timeseries, mask, mean_image


def temporal_autocorrelation(pixel_timeseries, max_lag=None):
    """
    Compute the normalized temporal autocorrelation function (ACF), averaged
    over all pixel time series, using an FFT-based (unbiased) estimator.

    Args:
        pixel_timeseries (numpy.ndarray): Array of shape (Nframes, Npix).
        max_lag (int, optional): Maximum lag (in frames) to compute. Defaults
                                  to Nframes - 1.

    Returns:
        numpy.ndarray: ACF values for lag = 0..max_lag (length max_lag+1).
    """
    Nframes, Npix = pixel_timeseries.shape
    if max_lag is None:
        max_lag = Nframes - 1
    max_lag = min(max_lag, Nframes - 1)

    x = pixel_timeseries - pixel_timeseries.mean(axis=0, keepdims=True)
    variance = x.var(axis=0, keepdims=True)
    variance[variance == 0] = 1.0
    x = x / np.sqrt(variance)

    n = 2 * Nframes  # zero-pad to avoid circular wrap-around
    fx = np.fft.rfft(x, n=n, axis=0)
    acf_full = np.fft.irfft(fx * np.conj(fx), n=n, axis=0)[:max_lag + 1]
    counts_per_lag = (Nframes - np.arange(max_lag + 1))[:, None]
    acf = (acf_full / counts_per_lag).mean(axis=1)
    return acf


def estimate_coherence_length(acf, frame_dt=1.0, threshold=1 / np.e):
    """
    Estimate the coherence length as the first lag where the ACF drops to
    `threshold` times its zero-lag value, using linear interpolation.

    Args:
        acf (numpy.ndarray): Autocorrelation function values (lag=0, 1, 2, ...).
        frame_dt (float): Time between frames, in seconds.
        threshold (float): Fraction of the zero-lag value defining coherence.

    Returns:
        tuple: (coherence_frames, coherence_seconds). NaN if the ACF never
               reaches the threshold within the computed lags.
    """
    acf0 = acf[0]
    if acf0 <= 0:
        return np.nan, np.nan

    target = threshold * acf0
    below = np.where(acf <= target)[0]
    if len(below) == 0:
        return np.nan, np.nan

    idx = below[0]
    if idx == 0:
        return 0.0, 0.0

    y0, y1 = acf[idx - 1], acf[idx]
    frac = (y0 - target) / (y0 - y1) if y0 != y1 else 0.0
    coherence_frames = (idx - 1) + frac
    return coherence_frames, coherence_frames * frame_dt


def relative_frame_to_frame_variation(pixel_timeseries, lag=1):
    """
    Estimate how much the flux varies, on average, between two frames
    separated by `lag` frames, as a percentage of the mean flux.

    Computed per pixel as mean(abs(x[t+lag]-x[t])) / mean(abs(x)), then
    averaged over all pixels.

    Args:
        pixel_timeseries (numpy.ndarray): Array of shape (Nframes, Npix).
        lag (int): Frame separation used for the difference (1 = consecutive
                   frames, 2 = one frame apart, etc.).

    Returns:
        float: Mean relative flux variation at this lag, in percent.
    """
    mean_diff = np.abs(pixel_timeseries[lag:] - pixel_timeseries[:-lag]).mean(axis=0)
    mean_flux = np.abs(pixel_timeseries).mean(axis=0)
    mean_flux[mean_flux == 0] = np.nan
    return float(np.nanmean(mean_diff / mean_flux) * 100)


def analytical_noise_variation(pixel_timeseries, noise_timeseries, gain=1.0):
    """
    Predict the frame-to-frame relative flux variation expected from shot
    noise and detector noise alone (i.e. no real signal instability).

    The detector (read/dark) noise variance is measured empirically from the
    faintest pixels (`noise_timeseries`); the shot noise variance for each
    bright pixel is estimated as `gain * mean_flux` (ADU^2), following the
    same convention as `DataCube._subtract_dark_and_variance`. Since two
    frames separated by any lag are independent draws of this noise, the
    difference has twice that variance, and for a roughly Gaussian
    distribution E[|difference|] = sqrt(2*variance) * sqrt(2/pi).

    Args:
        pixel_timeseries (numpy.ndarray): Bright-pixel array (Nframes, Npix).
        noise_timeseries (numpy.ndarray): Faint/background-pixel array used
                                           as the detector noise reference.
        gain (float): Detector gain (ADU per electron), from the 'GAIN' header.

    Returns:
        float: Expected relative flux variation from noise alone, in percent.
    """
    read_noise_variance = noise_timeseries.var(axis=0).mean()
    mean_flux = np.abs(pixel_timeseries).mean(axis=0)
    mean_flux_safe = np.where(mean_flux == 0, np.nan, mean_flux)
    expected_variance = read_noise_variance + gain * mean_flux_safe
    expected_diff = np.sqrt(2 * expected_variance) * np.sqrt(2 / np.pi)
    return float(np.nanmean(expected_diff / mean_flux_safe) * 100)


def save_png(filename_out, mean_image, mask, pixel_timeseries, acf, frame_dt,
             coherence_frames, coherence_seconds, noise_mask, noise_acf,
             noise_coherence_frames, noise_coherence_seconds,
             variation_percent_lag1, variation_percent_lag2, variation_percent_expected, header):
    """Save a diagnostic figure: mean image with the bright/faint pixels, the
    average brightest-pixel flux vs time, and the two autocorrelation
    functions (brightest pixels vs faintest/noise reference)."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), clear=True, num="Check raw data")

    v1, v2 = np.percentile(mean_image, [1, 99])
    axes[0].imshow(mean_image, aspect='auto', origin='lower', cmap='viridis', vmin=v1, vmax=v2)
    ys, xs = np.where(mask)
    axes[0].plot(xs, ys, '.', color='red', markersize=1, alpha=0.5, label=f"{mask.sum()} brightest")
    ys_n, xs_n = np.where(noise_mask)
    axes[0].plot(xs_n, ys_n, '.', color='cyan', markersize=1, alpha=0.5, label=f"{noise_mask.sum()} faintest")
    axes[0].set_title("Mean image + selected pixels")
    axes[0].set_xlabel("Wavelength (pixel)")
    axes[0].set_ylabel("Spatial (pixel)")
    axes[0].legend(fontsize=7, markerscale=5)

    Nframes = pixel_timeseries.shape[0]
    time_axis = frame_dt * np.arange(Nframes)
    mean_flux = pixel_timeseries.mean(axis=1)
    axes[1].plot(time_axis, mean_flux, '-')
    axes[1].set_title(f"Mean flux (\u03941={variation_percent_lag1:.2f}%, \u03942={variation_percent_lag2:.2f}%, noise-only\u2248{variation_percent_expected:.2f}%)")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Flux (ADU)")

    lag_frames = np.arange(len(acf))
    lag_seconds = lag_frames * frame_dt
    axes[2].plot(lag_seconds, acf, '-', color='red', label="brightest pixels")
    noise_lag_seconds = np.arange(len(noise_acf)) * frame_dt
    axes[2].plot(noise_lag_seconds, noise_acf, '-', color='cyan', label="faintest pixels (noise)")
    if np.isfinite(coherence_seconds):
        axes[2].axvline(coherence_seconds, color='red', linestyle='--', linewidth=0.8,
                         label=f"coherence = {coherence_frames:.1f} frames ({coherence_seconds:.4f} s)")
    if np.isfinite(noise_coherence_seconds):
        axes[2].axvline(noise_coherence_seconds, color='cyan', linestyle='--', linewidth=0.8,
                         label=f"noise floor = {noise_coherence_frames:.1f} frames ({noise_coherence_seconds:.4f} s)")
    axes[2].legend(fontsize=7)
    axes[2].set_title("Temporal autocorrelation")
    axes[2].set_xlabel("Lag (s)")
    axes[2].set_ylabel("ACF")

    fig.suptitle(os.path.basename(filename_out).replace('_checkraw.png', ''))
    fig.tight_layout()
    fig.savefig(filename_out, dpi=150, bbox_inches='tight')
    plt.close(fig)


def process_one_file(filename, fraction=0.05, max_lag=None, output_dir=None):
    """
    Analyze the frame-to-frame coherence of the brightest pixels in a single
    raw FITS cube, and compare it to the coherence of the faintest pixels
    (an empirical detector-noise reference).

    Returns:
        dict: Text-friendly result summary for this file.
    """
    cube, header, frame_dt = load_raw_cube(filename)
    Nframes = cube.shape[0]

    pixel_timeseries, mask, mean_image = select_brightest_pixels(cube, fraction=fraction)
    acf = temporal_autocorrelation(pixel_timeseries, max_lag=max_lag)
    coherence_frames, coherence_seconds = estimate_coherence_length(acf, frame_dt=frame_dt)
    variation_percent_lag1 = relative_frame_to_frame_variation(pixel_timeseries, lag=1)
    variation_percent_lag2 = relative_frame_to_frame_variation(pixel_timeseries, lag=2)

    noise_timeseries, noise_mask, _ = select_faintest_pixels(cube, fraction=fraction)
    noise_acf = temporal_autocorrelation(noise_timeseries, max_lag=max_lag)
    noise_coherence_frames, noise_coherence_seconds = estimate_coherence_length(noise_acf, frame_dt=frame_dt)
    variation_percent_expected = analytical_noise_variation(pixel_timeseries, noise_timeseries,
                                                             gain=header.get('GAIN', 1.0))

    if noise_coherence_frames and np.isfinite(noise_coherence_frames) and noise_coherence_frames > 0:
        excess_ratio = coherence_frames / noise_coherence_frames
    else:
        excess_ratio = np.nan

    result = {
        'file': filename,
        'object': header.get('OBJECT', 'UNKNOWN'),
        'date_obs': header.get('DATE-OBS', 'UNKNOWN'),
        'ut_str': header.get('UT-STR', 'UNKNOWN'),
        'n_frames': Nframes,
        'frame_dt_s': frame_dt,
        'coherence_frames': coherence_frames,
        'coherence_seconds': coherence_seconds,
        'noise_coherence_frames': noise_coherence_frames,
        'noise_coherence_seconds': noise_coherence_seconds,
        'excess_ratio': excess_ratio,
        'variation_percent_lag1': variation_percent_lag1,
        'variation_percent_lag2': variation_percent_lag2,
        'variation_percent_expected': variation_percent_expected,
    }

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        png_name = os.path.splitext(os.path.basename(filename))[0] + "_checkraw.png"
        save_png(os.path.join(output_dir, png_name), mean_image, mask, pixel_timeseries,
                  acf, frame_dt, coherence_frames, coherence_seconds,
                  noise_mask, noise_acf, noise_coherence_frames, noise_coherence_seconds,
                  variation_percent_lag1, variation_percent_lag2, variation_percent_expected, header)

    return result


def format_result(result):
    """Format a single file's result as one text line."""
    return (f"{os.path.basename(result['file'])}\t"
            f" -> {result['object']:<12}\t"
            # f"date={result['date_obs']}T{result['ut_str']}\t"
            # f"n_frames={result['n_frames']}\t"
            f"frame_dt={result['frame_dt_s']:.6f}s\t"
            f"coherence={result['coherence_frames']:.2f}frames\t"
            f"coherence={result['coherence_seconds']:.4f}s\t"
            # f"noise_floor={result['noise_coherence_frames']:.2f}frames\t"
            # f"noise_floor={result['noise_coherence_seconds']:.4f}s\t"
            f"excess_ratio={result['excess_ratio']:.2f}\t"
            f"variation_lag1={result['variation_percent_lag1']:.2f}%\t"
            f"variation_lag2={result['variation_percent_lag2']:.2f}%\t"
            f"variation_noise_only={result['variation_percent_expected']:.2f}%")


def run_checkRaw(fraction=0.05, max_lag=None, file_patterns=None, object_name=None, modID=None):
    """
    High-level function to check the frame-to-frame coherence of the
    brightest pixels for each raw FITS file in the given file patterns.

    Args:
        fraction (float): Fraction (0-1) of brightest pixels to analyze.
        max_lag (int, optional): Maximum lag (in frames) for the ACF.
        file_patterns (list): File patterns to match raw FITS files.
        object_name (str, optional): Restrict to files with this OBJECT.
        modID (int, optional): Restrict to files with this modulation ID.

    Returns:
        list: One result dict per processed file.
    """
    fileList = FileList(file_patterns, first_type='RAW', object_name=object_name, modID=modID)
    filelist = fileList.filelist
    folder = fileList.get_most_common_dir()
    output_dir = os.path.join(folder, "../checkraw")

    results = []
    for filename in filelist:
        try:
            result = process_one_file(filename, fraction=fraction, max_lag=max_lag, output_dir=output_dir)
        except Exception as e:
            print(f"Error occurred while processing {filename}: {e}")
            continue
        results.append(result)
        print(format_result(result))

    os.makedirs(output_dir, exist_ok=True)
    txt_filename = os.path.join(output_dir, "checkraw_results.txt")
    with open(txt_filename, "w") as f:
        for result in results:
            f.write(format_result(result) + "\n")
    print(f"Text results saved: {txt_filename}")

    return results


if __name__ == "__main__":

    # Development/interactive mode handling
    print("Running in compiler")
    if getpass.getuser() == "slacour":
        fraction = 0.02
        max_lag = None
        object_name=None
        modID=None
        file_patterns = ["/Users/slacour/DATA/FIRST/*.fits"]

    print(f"Development file patterns: {file_patterns}")

    results = run_checkRaw(fraction=fraction, max_lag=max_lag, file_patterns=file_patterns)

# %%
