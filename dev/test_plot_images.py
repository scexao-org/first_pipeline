#!/usr/bin/env python3
#%%
"""
Test script to plot the images stored in the FIRST photonic-lantern FITS files
of a given directory.

Each FITS file contains a 3D data cube (frames, ny, nx). For every file the
script displays the mean image over a few frames. Files that are still being
recorded (and therefore truncated on disk) are handled gracefully by reading
only the frames that are actually available.

Usage
-----
    python test_plot_images.py [directory]

If no directory is given, it defaults to the 20260625 dataset.

@author: slacour
"""

import os
import sys
import glob
import warnings

import numpy as np
import matplotlib
if "VSCODE_PID" in os.environ:
    matplotlib.use("macosx")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from astropy.io import fits
from scipy.ndimage import median_filter
plt.ion()

def load_mean_image(filename, n_frames=10):
    """
    Return the mean image over up to ``n_frames`` frames of a FITS cube.

    The frames are read one at a time through ``hdu.section`` so that only the
    requested slices are loaded into memory. This also allows reading files
    that are still being written to (truncated cubes) without failing.

    Parameters
    ----------
    filename : str
        Path to the FITS file.
    n_frames : int, optional
        Maximum number of frames to average. Default is 10.

    Returns
    -------
    numpy.ndarray or None
        The 2D mean image, or None if no frame could be read.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # ignore truncation warnings
        with fits.open(filename) as hdul:
            hdu = hdul[0]
            shape = hdu.shape

            if hdu.data is None or len(shape) < 2:
                print(f"  Skipping {os.path.basename(filename)}: not an image cube")
                return None

            # Number of frames actually available on disk
            if len(shape) == 3:
                n_available = shape[0]
                # How many full frames the file can really provide
                bytes_per_frame = abs(hdu.header["BITPIX"]) // 8 * shape[1] * shape[2]
                data_bytes = os.path.getsize(filename) - hdu._data_offset
                n_on_disk = max(1, int(data_bytes // bytes_per_frame))
                n_use = min(n_frames, n_available, n_on_disk)

                frames = []
                for i in range(n_use):
                    try:
                        frames.append(np.asarray(hdu.section[i], dtype=float))
                    except Exception as exc:
                        print(f"  Stopped at frame {i}: {exc}")
                        break
                if not frames:
                    return None
                return np.mean(frames, axis=0)

            # 2D image
            return np.asarray(hdu.data, dtype=float)

def detect_bad_pixels(image, size=5, threshold=5.0):
    """
    Detect bad (hot/dead) pixels as strong outliers from a local median.

    The image is compared to a median-filtered version of itself. Pixels whose
    residual exceeds ``threshold`` times the robust noise estimate (MAD) are
    flagged.

    Parameters
    ----------
    image : numpy.ndarray
        2D image.
    size : int, optional
        Size of the median-filter window. Default is 5.
    threshold : float, optional
        Number of robust standard deviations above which a pixel is flagged.
        Default is 5.0.

    Returns
    -------
    numpy.ndarray
        Boolean mask, True where a pixel is flagged as bad.
    """
    smoothed = median_filter(image, size=size)
    residual = image - smoothed
    # Robust standard deviation from the median absolute deviation
    mad = np.median(np.abs(residual - np.median(residual)))
    robust_std = 1.4826 * mad if mad > 0 else residual.std()
    if robust_std == 0:
        return np.zeros_like(image, dtype=bool)
    return np.abs(residual) > threshold * robust_std

def plot_directory(directory, n_frames=1000):
    """
    Plot the mean image of every FITS file in ``directory``.

    Parameters
    ----------
    directory : str
        Directory containing the FITS files.
    n_frames : int, optional
        Number of frames to average per file. Default is 10.
    """
    files = sorted(glob.glob(os.path.join(directory, "*.fits")))
    if not files:
        print(f"No FITS files found in {directory}")
        return

    print(f"Found {len(files)} FITS file(s) in {directory}")

    n = len(files)
    ncols = min(2, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, clear=True, num="FIRST images",
        figsize=(8 * ncols, 5 * nrows), squeeze=False,
        sharex=True, sharey=True,
    )
    axes = axes.ravel()

    for ax, filename in zip(axes, files):
        basename = os.path.basename(filename)
        print(f"Processing {basename} ...")
        image = load_mean_image(filename, n_frames=n_frames) -150
        if image is None:
            ax.set_title(f"{basename}\n(no data)")
            ax.axis("off")
            continue

        # Logarithmic color scale (only positive values are meaningful in log)
        positive = image[image > 0]
        if positive.size == 0:
            ax.set_title(f"{basename}\n(no positive data)")
            ax.axis("off")
            continue
        vmin, vmax = np.percentile(positive, [1, 99])
        vmin = max(vmin, 1e-3)
        im = ax.imshow(
            image, aspect="auto", interpolation="none",
            norm=LogNorm(vmin=vmin, vmax=vmax), cmap="viridis",
        )
        fig.colorbar(im, ax=ax)

        # Detect and overlay bad pixels
        bad_mask = detect_bad_pixels(image)
        bad_y, bad_x = np.where(bad_mask)
        n_bad = bad_y.size
        if n_bad:
            ax.plot(bad_x, bad_y, "r.", markersize=2, alpha=0.7,
                    label=f"{n_bad} bad pixels")
            ax.legend(loc="upper right", fontsize=8, markerscale=3)
        print(f"  Detected {n_bad} bad pixels")

        ax.set_title(basename, fontsize=9)
        ax.set_xlabel("x [pixel]")
        ax.set_ylabel("y [pixel]")

    # Hide any unused axes
    for ax in axes[len(files):]:
        ax.axis("off")

    fig.tight_layout()

    out_path = os.path.join(directory, "test_plot_images.png")
    fig.savefig(out_path, dpi=150)
    print(f"Figure saved to {out_path}")

    if matplotlib.get_backend().lower() != "agg":
        plt.show()


if __name__ == "__main__":
    default_dir = "/Users/slacour/DATA/FIRST/20260625/firstpl"
    # Use the first CLI argument only if it is a real directory. This avoids
    # picking up the kernel connection file (e.g. --f=.../kernel-xxxx.json)
    # when the script is run inside a Jupyter/interactive session.
    directory = default_dir
    if len(sys.argv) > 1 and os.path.isdir(sys.argv[1]):
        directory = sys.argv[1]
    plot_directory(directory)

# %%
