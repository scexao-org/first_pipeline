#! /usr/bin/env python3
# -*- coding: iso-8859-15 -*-
"""
FIRST Pipeline - Raw Data Coherence Check CLI Interface

Command-line interface for checking the frame-to-frame coherence of the
brightest pixels in raw FIRST Visible Photonic Lantern data. This script
provides the CLI wrapper for the core coherence-check algorithms.

Created on Wed May 21 22:56:25 2025
@author: slacour
"""
#%%
import os
import sys
# Add src directory to path for imports to work in both interactive and package contexts
if os.path.join(os.path.dirname(__file__), '..') not in sys.path:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import matplotlib

if "VSCODE_PID" in os.environ:
    matplotlib.use('Qt5Agg')
else:
    matplotlib.use('Agg')


def main():
    """
    Main entry point for the raw data coherence check script.
    """
    parser = argparse.ArgumentParser(
        description="Check the frame-to-frame coherence of the brightest pixels in raw FIRST Pipeline data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
FIRST Pipeline Raw Data Coherence Check Tool

This script inspects raw FIRST Visible Photonic Lantern detector cubes and
estimates how many frames the brightest pixels stay correlated with
themselves (a proxy for the atmospheric/injection "coherence length").

For each raw file, the 5%% brightest pixels (by time-averaged flux) are
selected, their flux time series across frames is normalized, and the
temporal autocorrelation function (ACF) is computed and averaged over these
pixels. The coherence length is the lag (in frame number and in seconds) at
which the ACF first drops to 1/e of its zero-lag value.

The same analysis is repeated on the 5%% faintest pixels, which carry
essentially no signal, to provide an empirical detector-noise reference. The
reported excess_ratio (bright coherence / noise-floor coherence) close to 1
means the brightest pixels vary frame-to-frame within the detector noise
floor (stable upstream optics); a much larger value means the brightest
pixels show real correlated variability beyond detector noise (unstable
upstream optics).

The relative flux variation between one frame and the next (variation_lag1)
and between one frame and the one two frames later (variation_lag2) are
also reported, along with an analytical prediction of the variation
expected from shot noise and detector noise alone (variation_noise_only).
Measured values close to variation_noise_only, with lag1 ~ lag2, indicate a
stable upstream optical system; measured values well above it, or rising
with lag, indicate real instability.

Examples:
    %(prog)s *.fits
    %(prog)s --fraction=0.02 --max_lag=200 /data/raw/*.fits

Pipeline Workflow Integration:
    - This script processes RAW files (X_FIRTYP=RAW) and does not require any
      pixel map, wavelength map, or coupling map.
    - It is meant as a quick diagnostic check, run directly on raw data.

Input Files:
    - Raw FITS files with X_FIRTYP=RAW

Output:
    - One text result line per file printed to the console
    - A summary text file (checkraw_results.txt) in the ../checkraw directory
    - A diagnostic PNG per file (mean image, flux time series, ACF)

Output Fields (printed per file and written to checkraw_results.txt):
    - coherence: lag (in frames, and in seconds) at which the brightest
      pixels' ACF first drops to 1/e of its zero-lag value.
    - excess_ratio: coherence(brightest pixels) / coherence(faintest pixels).
      ~1 means the brightest pixels decorrelate frame-to-frame just as fast
      as pure detector noise (stable upstream optics). >> 1 means the
      brightest pixels stay correlated much longer than noise alone would
      allow (real frame-to-frame instability upstream, e.g. unstable
      injection/vibration/seeing).
    - variation_lag1 / variation_lag2: mean relative flux change (%%) between
      one frame and the next / the one two frames later.
    - variation_noise_only: analytical prediction (%%) of that same relative
      flux change from shot noise + detector noise alone, for comparison.
      Measured values close to variation_noise_only, with lag1 ~ lag2,
      indicate a stable system; values well above it, or rising with lag,
      indicate real instability.
        """,
        allow_abbrev=False
    )

    # needed to work in VSC:
    parser.add_argument("--f", help=argparse.SUPPRESS)

    # Add positional argument for file patterns
    parser.add_argument('file_patterns', nargs='*', default=['*.fits'],
                       help='One or more glob patterns for FITS files (default: *.fits)')

    # Add optional arguments
    parser.add_argument("--fraction", type=float, default=0.02,
                       help="Fraction of brightest pixels to analyze (default: %(default)s)")
    parser.add_argument("--max_lag", type=int, default=None,
                       help="Maximum lag in frames for the autocorrelation function (default: all frames)")
    parser.add_argument("--object_name",
                       help="Selection of the data by the Object name (default: all objects)")
    parser.add_argument("--modID", type=int,
                       help="Selection of the modulation pattern by user (default: all modIDs)")

    # Parse arguments
    args = parser.parse_args()

    # Import core functions
    from .run_checkRaw import run_checkRaw

    run_checkRaw(fraction=args.fraction, max_lag=args.max_lag, file_patterns=args.file_patterns,
                 object_name=args.object_name, modID=args.modID)


if __name__ == "__main__":
    main()
# %%
