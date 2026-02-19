#! /usr/bin/env python3
# -*- coding: iso-8859-15 -*-
#%%
"""
FIRST Pipeline - Coupling Map Generation

This script creates coupling maps from preprocessed FIRST Visible Photonic Lantern
data at SUBARU/SCEXAO. Coupling maps analyze the coupling efficiency between the 
telescope focal plane and individual photonic lantern channels using SVD-based 
decomposition techniques.

Coupling maps are essential for image reconstruction and astrometric analysis,
providing the relationship between sky position and fiber channel response.

Created on Wed May 21 22:56:25 2025
@author: slacour
"""

import os
import sys
from astropy.io import fits
from glob import glob
import argparse
import numpy as np
from scipy.signal import correlate
from scipy import linalg
from scipy.linalg import solve_triangular
from typing import List

import getpass
import matplotlib
if ("VSCODE_PID" in os.environ or os.environ.get('TERM_PROGRAM') == 'vscode'):
    matplotlib.use('Qt5Agg')
elif os.environ.get('SPYDER_DEBUG_FILE'):
    print("Running in Spyder")
else:
    matplotlib.use('Agg')
     
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot,hist,clf,figure,legend,imshow
from datetime import datetime
from tqdm import tqdm
from collections import defaultdict
from scipy import linalg
from matplotlib import animation
from itertools import product
from scipy.linalg import pinv

# Import FIRST pipeline classes
from .classes.runPL_class_flatMap import FlatMap
from .classes.runPL_class_waveMap import WaveMap
from .classes.runPL_class_fileList import FileList
from .classes.runPL_class_dataCube import DataCube
from .classes.runPL_class_couplingMap import CouplingMap

# Import FIRST pipeline libraries
from .libraries import runPL_library_io as runlib_io
from .libraries import runPL_library_plots as runlib_plots
from .libraries import runPL_library_linalg as runlib_linalg

from scipy.ndimage import zoom
from astropy.io import fits
import shutil
from scipy.interpolate import interpn
from astropy.table import Table
from scipy.interpolate import griddata
from scipy import odr
from scipy.optimize import least_squares
from scipy.ndimage import uniform_filter1d

from scipy.spatial.distance import pdist, squareform
from scipy.optimize import curve_fit
import signal

plt.ion()

DEBUG = True
# Add options
usage = """
    usage:  %prog [options] files.fits

    Goal: Create coupling maps from preprocessed photonic lantern data.

    Summary:
    This script processes preprocessed photonic lantern data (X_FIRTYP=PREPROC) to generate
    coupling maps. It performs SVD-based analysis, computes triangular and pyramidal coupling
    coefficients, and produces diagnostic plots for data quality assessment.

    Input:
    - Files with X_FIRTYP=PREPROC in the specified directory or pattern
    - Optional dark and flat calibration files
    - Wavelength maps for spectral calibration

    Output:
    - Coupling map files (X_FIRTYP=COUPLINGMAP) in "../couplingmaps" directory
    - PDF report with diagnostic plots and SVD analysis results

    Options:
    --object_name: Filter by object name (default: auto-detect from first file)
    --modID: Modulation pattern ID selection (default: auto-detect)
    --modScale: Modulation scale selection (default: auto-detect) 
    --wollaston: Wollaston prism setting (IN/OUT, default: auto-detect)
    --dark_files: Specific dark files to use (default: auto-detect)
    --flatMap: Specific flat map file (default: most recent in flatmaps/)
    --waveMap: Specific wavelength map file (default: most recent in wavemaps/)
    --wavelength_smooth: Spectral smoothing factor (default: 20)
    --wavelength_bin: Spectral binning factor (default: 10)
    --Nsingular: Number of SVD singular values (default: 114)

    Example:
    python runPL_createCouplingMaps.py preproc/*.fits --object_name HIP84212 --modScale 50
"""


def singular_vector_basis(data_svdfiltered,goodData,indexes, xmod, ymod):

    vectors_all_triangles = []
    center_all_triangles = []
    _,Nqr = indexes.shape
    if Nqr == 3:
        description = "Computing triangles singular vectors"
    else:
        description = "Computing pyramids singular vectors"

    Ntriangles = 0
    indexes_new = []
    for t in tqdm(np.arange(len(indexes)), desc=description):

        # as a first step 
        # extract the singular vectors for each triangle or pyramid
        i = indexes[t]

        data_triangle_svdfiltered, good_data_triangle ,_ = runlib_linalg.svd_filtering(data_svdfiltered[:,i],goodData[:,i],Nqr,verbose = False)

        data_triangle_svdfiltered = data_svdfiltered[:,i] # TO REMOVE when the svd filtering will be good enough to be used for the coupling map analysis. For now, we use it only for the selection of good triangles, but not for the actual computation of the singular vector basis, because it is not good enough yet and remove too much data.
        good_data_triangle = goodData[:,i] # TO REMOVE

        data_triangle = data_triangle_svdfiltered[good_data_triangle].reshape((data_triangle_svdfiltered[good_data_triangle].shape[0], -1))
        xmod_triangle = xmod[:,i][good_data_triangle]
        ymod_triangle = ymod[:,i][good_data_triangle]

        x = xmod_triangle
        y = ymod_triangle
        if Nqr == 3:
            X = np.array([x, y]).T
        else:
            X = np.array([x, y, x*y, x*x, y*y]).T

        Y = data_triangle

        # Centered copies for stats     
        mu_x = X.mean(axis=0)
        mu_y = Y.mean(axis=0)
        Xc = X - mu_x
        Yc = Y - mu_y

        # Covariance 
        Sxx =  Xc.T @ Xc
        Sxy =  Xc.T @ Yc

        # Check rank of Sxx
        rank_sxx = np.linalg.matrix_rank(Sxx)
        if rank_sxx < 2:
            continue

        B = pinv(Sxx) @ Sxy

        Vectors_triangle = np.hstack([mu_y[:,None], B.T])
        vectors_all_triangles.append(Vectors_triangle)
        center_all_triangles.append(mu_x)
        Ntriangles+=1
        indexes_new+=[i]

    Noutput, Nwave = data_svdfiltered.shape[2:]

    center_all_triangles = np.array(center_all_triangles)
    vectors_all_triangles = np.array(vectors_all_triangles).reshape((Ntriangles, Noutput, Nwave, Nqr))

    return vectors_all_triangles, center_all_triangles, indexes_new


def flux_matrices(singular_vectors):

    Ntriangles = singular_vectors.shape[0]
    Noutput = singular_vectors.shape[1]
    Nwave = singular_vectors.shape[2]

    flux_2_data = singular_vectors[:,:,:,0]
    flux_2_data = flux_2_data.transpose((2,1,0))
    data_2_flux = np.zeros((Nwave, Ntriangles, Noutput))
    data_2_flux = np.linalg.pinv(flux_2_data)

    return flux_2_data,data_2_flux


def Q_and_R_matrices(singular_vectors):

    Ntriangles = singular_vectors.shape[0]
    Noutput = singular_vectors.shape[1]
    Nwave = singular_vectors.shape[2]
    Nqr = singular_vectors.shape[3]

    QT_singular_vectors = np.zeros((Ntriangles,Nwave,Nqr,Noutput))
    R_singular_vectors = np.zeros((Ntriangles,Nwave,Nqr,Nqr))

    if Nqr == 3:
        description = "Calculating QR matrices for triangles"
    else:
        description = "Calculating QR matrices for pyramids"

    for t in tqdm(range(Ntriangles), desc = description):
        for w in range(Nwave):
            Q, R = np.linalg.qr(singular_vectors[t,:,w], mode="reduced")
            QT_singular_vectors[t,w] = Q.T
            R_singular_vectors[t,w] = R

    return QT_singular_vectors,R_singular_vectors

def quick_fits(data, title=""):
    if DEBUG:
        #For debugging purpose
        now = datetime.now()
        date_time_str = now.strftime("%Y_%m_%d_%H_%M_%S")
        if getpass.getuser() == "jsarrazin":
            runlib_io.save_fits_file(data, "/home/jsarrazin/Bureau/test zone/coupling_maps/"+title+"_"+date_time_str+".fits")
        print("Done")   

def quick_imshow(data, title=""):
    #For debugging purpose
    now = datetime.now()
    plt.imshow(data, aspect='auto')
    plt.title(title)
    print("Done")

def quick_plot(data,title =""):
    #For debugging purpose
    now = datetime.now()
    date_time_str = now.strftime("%Y_%m_%d_%H_%M_%S")
    plt.plot(data)
    plt.title(title)
    print("Done")


def main():
    """
    Main entry point for the coupling map generation script.
    """
    parser = argparse.ArgumentParser(
        description="Generate coupling efficiency maps from preprocessed FIRST Photonic Lantern data using SVD analysis.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
FIRST Pipeline Coupling Map Generation Tool

This script analyzes the coupling efficiency between the telescope focal plane 
and photonic lantern channels using advanced SVD-based decomposition. Coupling 
maps are essential for accurate image reconstruction and astrometric measurements.

Examples:
    %(prog)s --object_name="HD 164461" --wavelength_smooth=7 *.fits
    %(prog)s --modID=1 --modScale=2 --wollaston=IN data/*.fits
    %(prog)s --flatMap=/path/to/flat.fits --waveMap=/path/to/wave.fits *.fits

Pipeline Workflow Integration:
    1. Processes preprocessed data files (X_FIRTYP=PREPROC)
    2. Uses flat field and wavelength calibration maps
    3. Output coupling maps enable image reconstruction and astrometry
    4. Critical step for converting fiber measurements to sky coordinates

Input Files:
    - Preprocessed FITS files: X_FIRTYP=PREPROC
    - Flat field maps (automatic detection or manual selection)
    - Wavelength calibration maps (automatic detection or manual selection)
    - Dark frames for background subtraction
    - Files grouped by object name, modulation pattern, and Wollaston status

Output Files:
    - Coupling map FITS files: X_FIRTYP=COUPLINGMAP (../couplingmaps/ directory)
    - PDF diagnostic report with SVD analysis and quality plots
    - Triangular and pyramidal coupling coefficient matrices
    - Quality assessment metrics and validation plots

Processing Details:
    - SVD-based decomposition to extract coupling patterns
    - Wavelength smoothing and binning for noise reduction
    - Modulation pattern analysis for enhanced sensitivity
    - Automatic selection of singular values (configurable with --Nsingular)
    - Support for both polarimetry (Wollaston IN) and photometry (OUT) modes

Advanced Options:
    - object_name: Select specific science target for processing
    - modID/modScale: Choose specific modulation patterns
    - wavelength_smooth/bin: Control spectral processing parameters
    - Nsingular: Number of SVD modes to retain (affects map quality vs noise)

Technical Notes:
    - SVD analysis identifies dominant coupling modes
    - Coupling maps quantify spatial response of each fiber channel
    - Quality metrics assess map reliability and completeness
    - Results enable precise astrometric and photometric measurements

Note: Quality coupling maps are critical for accurate image reconstruction.
Review PDF diagnostics to ensure proper SVD convergence and coupling patterns.
        """
    )

    # needed to work in VSC:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--f", help=argparse.SUPPRESS)

    # Add positional argument for files
    parser.add_argument('files', nargs='*', default=['*.fits'],
                       help='FITS files to process (supports wildcards)')

    # Add optional arguments
    parser.add_argument("--object_name", 
                       help="Selection of the data by the Object name (default: first target in the list)")
    parser.add_argument("--dark_files", 
                       help="Select one or more specific dark(s) files to use")
    parser.add_argument("--flatMap", 
                       help="Select a specific flat Map to use (default: most recent in the flatmaps folder)")
    parser.add_argument("--waveMap", 
                       help="Select a specific wave Map to use (default: most recent in the wavemaps folder)")
    parser.add_argument("--wavelength_smooth", type=int, default=1,
                       help="Smoothing factor for wavelength (default: %(default)s)")
    parser.add_argument("--wavelength_bin", type=int, default=1,
                       help="Binning factor for wavelength (default: %(default)s)")
    parser.add_argument("--Nsingular", type=int, default=19*6,
                       help="Number of singular values to use (default: %(default)s)")
    parser.add_argument("--modID", type=int, 
                       help="Selection of the modulation pattern by user (default: first in the list)")
    parser.add_argument("--modScale", type=int, 
                       help="Selection of the modulation scale by user (default: first in the list)")
    parser.add_argument("--wollaston", 
                       help="Wollaston status. Use IN for internal or OUT for no wollaston (default: first in the list)")
    parser.add_argument("--use_pyramids", action="store_true",
                       help="Use pyramids instead of triangles for coupling map analysis (default: use triangles)")
    parser.add_argument("--center_data", action="store_true", default=True,
                       help="Center the data -- remove mean flux over the output -- before analysis (default: center the data)")

    # Parse the arguments
    args = parser.parse_args()
    file_patterns = args.files if args.files else ['*.fits','./preproc/*.fits']

    # Extract the parsed arguments
    modID = args.modID
    modScale = args.modScale
    object_name = args.object_name
    wollaston = args.wollaston
    Nsingular = args.Nsingular
    wavelength_smooth = args.wavelength_smooth
    wavelength_bin = args.wavelength_bin
    dark_patterns = args.dark_files
    flat_patterns = args.flatMap
    wave_patterns = args.waveMap
    use_pyramids = args.use_pyramids
    center_data = args.center_data
    
    if ("VSCODE_PID" in os.environ or os.environ.get('TERM_PROGRAM') == 'vscode' or os.environ.get('SPYDER_DEBUG_FILE')):
        print("Running in compiler")
        if getpass.getuser() == "slacour":
            file_patterns = "/Users/slacour/DATA/LANTERNE/20250514/preproc/firstpl_2025-05-14T11?3*fits"
            file_patterns = "/Users/slacour/DATA/LANTERNE/20250808/preproc/firstpl_2025-08-08T06:4?:??_HIP84212_P.fits"
            # file_patterns = "/Users/slacour/DATA/LANTERNE/20250808/preproc/firstpl_2025-08-08T06:4[3-4]:??_HIP84212_P.fits"
            # file_patterns = "/Users/slacour/DATA/LANTERNE/20250510/preproc/*10T09?2[0-3]*TETCRB_P.fits"
            # file_patterns = "/Users/slacour/DATA/LANTERNE/20250510/preproc/*10T09?21*TETCRB_P.fits"
            file_patterns = "/Users/slacour/DATA/LANTERNE/20250514/preproc/firstpl_2025-05-14T11?3*s"
            file_patterns = "/Users/slacour/Downloads/2025-07-14/"
            # dark_patterns = "/Users/slacour/DATA/LANTERNE/20250514/preproc"
            file_patterns = "/Users/slacour/DATA/LANTERNE/raw/20260114/preproc/firstpl_2026-01-14T20h56m24s_DAY_P.fits"
            file_patterns = "/Users/slacour/DATA/LANTERNE/20260103/preproc/*ALFLEO*"
            wave_patterns = "/Users/slacour/DATA/LANTERNE/20251231/wavemaps"
        if getpass.getuser() == "jsarrazin":
            file_patterns = "/home/jsarrazin/Bureau/PLDATA/moreTest/2024-11-21_13-48-32_science_copie/preproc"
            file_patterns = "/home/jsarrazin/Bureau/PLDATA/novembre/les_preproc"
        if getpass.getuser() == "ehuby":
            file_patterns = "/home/ehuby/WORK/DATA/FIRST-PL/2025-05-10/preproc/"
            modID, modScale, object_name = 3, 50, 'TETCRB'
            
            file_patterns = "/home/ehuby/WORK/DATA/FIRST-PL/2025-07-14/preproc/VEGA_0.002sec/"
            modID, modScale, object_name = 3, 80, 'VEGA'
        file_patterns = [file_patterns] if isinstance(file_patterns, str) else file_patterns

    # If the user specify a dark, use it. Otherwise, use the science file pattern
    if dark_patterns is None:
        dark_patterns = file_patterns
    # If the user specifies a specific map, use it, otherwise look into the arguments + default directories
    if flat_patterns is None:
        folder = os.path.dirname(file_patterns[0])
        flat_patterns = file_patterns + [os.path.join(folder,"../flatmaps")] + [os.path.join(folder,"flatmaps")]
    if wave_patterns is None:
        folder = os.path.dirname(file_patterns[0])
        wave_patterns = file_patterns + [os.path.join(folder,"../wavemaps")] + [os.path.join(folder,"wavemaps")]
    if modID is None:
        modID = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

    fileList = FileList(file_patterns, data_type= "OBJECT", first_type='PREPROC', wollaston=wollaston, object_name=object_name, modID=modID, modScale=modScale)

    # Adding constraints to make sure the dataset is coherent:
    object_name = fileList.header.get('OBJECT', None)
    wollaston = fileList.header.get('X_FIRWOL', None)
    modID = fileList.header.get('X_FIRMID', None)
    modScale = fileList.header.get('X_FIRMSC', None)

    fileList = FileList(file_patterns, data_type= "OBJECT", first_type='PREPROC', wollaston=wollaston, object_name=object_name, modID=modID, modScale=modScale)

    fileList.make_association(darks_pattern=dark_patterns)
    file_flat = fileList.get_flatmap_file(flat_patterns)
    file_wave = fileList.get_wavemap_file(wave_patterns)

    flatMap =  FlatMap(file_flat) if file_flat is not None else None
    waveMap =  WaveMap(file_wave) if file_wave is not None else None

    datalist : List[DataCube] = fileList.extract_data_from_list(Nsmooth=wavelength_smooth, 
                                                                Nbin = wavelength_bin, flatMap = flatMap, 
                                                                waveMap = waveMap, center = center_data)

    flux = np.concatenate([d.flux for d in datalist])
    datacube=np.concatenate([d.data for d in datalist])
    datacube_var=np.concatenate([d.variance for d in datalist])
    xmod=np.concatenate([d.xmod for d in datalist])
    ymod=np.concatenate([d.ymod for d in datalist])

    basenames = []
    for d in datalist:
        n = d.data.shape[0]  # first dimension of d.data
        basenames.extend([d.basename] * n)

    filenames = [d.filename for d in datalist]

    flux_goodData,flux_threshold = runlib_linalg.flux_filtering(flux)
    print(f"* Percentage of good data: {np.sum(flux_goodData)/len(flux_goodData.ravel())*100:.1f} % (flux threshold)")

    data_svdfiltered,fit_goodData,errors = runlib_linalg.svd_filtering(datacube,flux_goodData,Nsingular)
    goodData = flux_goodData & fit_goodData
    print(f"* Percentage of good data: {np.sum(goodData)/len(goodData.ravel())*100:.1f} % (flux and svd threshold)")

    runlib_plots.plot_flux_map(flux.mean(axis=(2))[0], xmod[0], ymod[0])

    goodPositions = goodData.mean(axis=0) > 0.3

    if use_pyramids == False:
        indexes  = datalist[0].get_triangles()
    else:
        indexes = datalist[0].get_pyramids()

    # Select only triangles with good data
    goodTriangles = goodPositions[indexes].mean(axis=1)  == 1
    indexes_good=indexes[goodTriangles]

    vectors_all, center_all, indexes_new = singular_vector_basis(data_svdfiltered,goodData,indexes_good, xmod, ymod)



    # Ntriangles = vectors_all_triangles.shape[0]
    # vectors_all_triangles = vectors_all_triangles.reshape((Ntriangles, Noutput, Nwave,6))
    spectra = flux[goodData].mean(axis=0)
    vectors_all_triangles = vectors_all/spectra[:,None]

    #getting the flux 2 data matrices
    flux_2_data,data_2_flux = flux_matrices(vectors_all_triangles)

    #getting the Q and R matrices
    QT,R = Q_and_R_matrices(vectors_all_triangles)

    ############### Save results ####################
    # Create CouplingMap object and save using the new save method
    
    couplingMap = CouplingMap()
    couplingMap.create_from_data(
        flux_2_data, data_2_flux, QT, R, center_all,
        spectra, wavelength_bin
    )
    
    new_header = datalist[-1].header.copy()
    
    # Définir le chemin complet du sous-dossier "output/couplingmaps"
    folder = datalist[-1].dirname
    output_dir = os.path.join(folder,"../couplingmaps")

    new_header['X_FIRTYP'] = 'COUPLINGMAP'

    # Add input parameters to the header
    new_header['Q_CMWSMO'] = (wavelength_smooth,  'wavelength smoothing factor')
    new_header['Q_CMWBIN'] = (wavelength_bin, 'wavelength binning factor')
    new_header['Q_CMSING'] = (Nsingular, 'number of singular values')
    new_header['Q_CMCENT'] = (center_data, 'whether the data was centered before analysis')
    new_header['Q_CMPYR'] = (use_pyramids, 'whether pyramids (True) or triangles (False) were used')
    new_header['Q_CM_FT'] = (flux_threshold, 'flux threshold')
    # new_header['CHI2THR'] = chi2_threshold  # Add chi2 threshold
    new_header['Q_CM_CK'] = (np.random.randint(0, 2**32, dtype=np.uint32), 'checksum')
    for i, filename in enumerate(filenames):
        new_header['Q_CM_F%i' % i] = (filename, 'filename of the extracted flux')

    new_header['Q_CMNAME'] = (runlib_io.create_basename(new_header), 'name of the coupling map file')
    # Créer les dossiers "output" et "pixel" s'ils n'existent pas déjà
    os.makedirs(output_dir, exist_ok=True)

    output_filename = os.path.join(output_dir, new_header['Q_CMNAME'])

    # Get modulation HDU
    modulation_hdu = fits.open(datalist[-1].filename)['MODULATION']
    
    # Save using the CouplingMap save method
    couplingMap.save(output_filename, new_header, flat_map=flatMap, wave_map=waveMap, modulation_hdu=modulation_hdu)


    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    ###############################################
    # Diagnostic plots
    ###############################################

    if flatMap is not None:
        runlib_plots.plot_detector_field(flatMap.flat, title="Flat Map for file"+flatMap.basename)


    dark = np.array([d.dark for d in datalist]).mean(axis=0)
    if dark.ndim == 0 or dark.shape == ():
        dark = np.full_like(datalist[0].data[0,0], dark)
    elif dark.ndim == 1:
        dark = np.tile(dark, (datalist[0].data.shape[2], 1))

    runlib_plots.plot_detector_field(dark, title="Dark Field")


    fig, axs = plt.subplots(2, 2, figsize=(12, 10), num="Flux/GoodData Selection", clear=True)

    # The data used to make them: mean flux per (wavelength, output)
    mean_flux = flux.mean(axis=(2))
    axs[0, 0].imshow(mean_flux, aspect='auto', origin='lower', cmap='viridis', interpolation='none', rasterized=True)
    axs[0, 0].set_title("Mean Flux (per wavelength/output)")
    axs[0, 0].set_xlabel("Output")
    axs[0, 0].set_ylabel("files")
    # Show the threshold as a horizontal line (if 1D), else as a contour
    # axs[0, 0].contour(flux_goodData, levels=[0.5], colors='r', linewidths=1, linestyles='--')

    # flux_goodData mask
    axs[0, 1].imshow(flux_goodData, aspect='auto', origin='lower', cmap='Greens', interpolation='none', rasterized=True, vmin=0, vmax=1)
    axs[0, 1].set_title("From flux, good Dataset (mask)")
    axs[0, 1].set_xlabel("Output")
    axs[0, 1].set_ylabel("Wavelength")

    # The data used to make them: mean flux per (wavelength, output)
    error_norm = errors.reshape((datacube.shape[0], -1))
    axs[1, 0].imshow(error_norm, aspect='auto', origin='lower', cmap='viridis', interpolation='none', rasterized=True)
    axs[1, 0].set_title("Amplitude of residuals after SVD filtering")
    axs[1, 0].set_xlabel("Output")
    axs[1, 0].set_ylabel("files")
    # Show the threshold as a horizontal line (if 1D), else as a contour
    # axs[1, 0].contour(fit_goodData, levels=[0.5], colors='r', linewidths=1, linestyles='--')

    # flux_goodData mask
    axs[1, 1].imshow(fit_goodData, aspect='auto', origin='lower', cmap='Greens', interpolation='none', rasterized=True, vmin=0, vmax=1)
    axs[1, 1].set_title("From SVD fits, good Dataset (mask)")
    axs[1, 1].set_xlabel("Output")
    axs[1, 1].set_ylabel("Wavelength")

    fig.tight_layout()

    fig, axs = plt.subplots(1, 2, num=" Positions fiber and of triangles" , figsize=(16, 6), sharex=True, sharey=True, clear=True)

    # 1. Plot positions (xmod, ymod) for all triangles
    axs[0].set_title("Positions of Fiber")
    axs[0].scatter(xmod, ymod, c='r', marker='.')
    axs[0].scatter(xmod[0,goodPositions], ymod[0,goodPositions], facecolors='g', marker='o', edgecolor='k', label='Good Positions')
    axs[0].set_xlabel("x [mas]")
    axs[0].set_ylabel("y [mas]")
    axs[0].set_aspect('equal')
    axs[0].legend()

    center_triangles=np.array([xmod[0,indexes].mean(axis=1), ymod[0,indexes].mean(axis=1)]).T
    # 1. Plot positions (xmod, ymod) for all triangles
    axs[1].set_title("Positions of Triangles")
    axs[1].scatter(center_triangles[:, 0], center_triangles[:, 1], c='r', marker='.')
    axs[1].scatter(center_all[:, 0], center_all[:, 1], facecolors='g', marker='o', edgecolor='k', label='Good Triangles')
    axs[1].set_xlabel("x [mas]")
    axs[1].set_ylabel("y [mas]")
    axs[1].set_aspect('equal')
    axs[1].legend()

    fig.tight_layout()

    ###############################################
    # Covariance and correlation matrix plot
    ###############################################

    if len(center_all) > 0:
        runlib_plots.plot_covariance(flux_2_data,center_all,"Triangles")

    R_amplitude = np.linalg.norm(R,axis=2)

    label = ["1","x","y","xy","x2","y2"]

    if use_pyramids:
        runlib_plots.plot_R_amplitude(R,name="pyramids")
    else:
        runlib_plots.plot_R_amplitude(R,name="triangle")

    ###############################################

    print("----------------------------------------------------")
    print("Coupling Map stored. You can quit by ctrl+C")
    print("----------------------------------------------------")
    print("Computing now additional health check plots.")
    def handle_sigint(signum, frame):
        print("\nCtrl+C detected. Exiting gracefully... after saving plots.")
        runlib_plots.save_pdf_in_file(output_filename)
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_sigint)



    couplingMap = CouplingMap(output_filename)

    QT= couplingMap.QT
    R= couplingMap.R
    centers = couplingMap.position

    datacube=np.concatenate([d.data for d in datalist])
    # datacube[np.isnan(datacube)] = 0

    datacube_T=datacube.transpose((3,2,0,1))
    # datacube_T=data_svdfiltered.transpose((3,2,0,1)) # TO REMOVE
    Nwave, Noutput, Ncube, Nmod = datacube_T.shape
    Ntriangles = QT.shape[0]

    if Ntriangles == 0:
        print(f"Bad coupling map, cannot continue the health check plots.")

    datacube_T=datacube_T.reshape((datacube_T.shape[0], datacube_T.shape[1], -1))
    chi2_max = np.sum(datacube_T**2, axis=(0,1))

    chi2_map = np.zeros((Ntriangles,Ncube * Nmod))
    chi2_map = np.zeros((Ntriangles, Ncube * Nmod))
    chi2_map[:] =  chi2_max
    # Here, the computation of the chi2 is simplified by the fact that QT is orthonormal
    # chi2 = ||data - Q @ Q.T @ data||^2 = ||data||^2 - ||Q.T @ data||^2
    for t in tqdm(range(Ntriangles), desc="Computing chi2 map"):
        k= QT[t] @ datacube_T
        chi2_map[t,:] -= np.sum(k ** 2, axis=(0,1))

    Nt,No,Nw,Nthree = vectors_all.shape
    vectors_all_inv = np.zeros((Nt,Nw,Nthree,No))
    vectors_all_kernel = np.zeros((Nt,Nw,No,No))
    for t in range(Nt):
        for w in range(Nw):
            v = vectors_all[t, :, w, :]
            v_inv = np.linalg.pinv(v)
            vectors_all_inv[t,w] = v_inv
            vectors_all_kernel[t,w] = np.eye(No) - v @ v_inv

    # chi2_map_2=chi2_map.copy()*0

    # for t in tqdm(range(Ntriangles), desc="Computing chi2 map"):
    #     k = vectors_all_kernel[t] @ datacube_T
    #     chi2_map_2[t] = np.sum(k ** 2, axis=(0,1))
                                                        
        

    # Handle all-NaN slices by using a masked approach
    chi2_argmin = np.zeros(chi2_map.shape[1], dtype=int)
    for i in range(chi2_map.shape[1]):
        valid_indices = ~np.isnan(chi2_map[:, i])
        if np.any(valid_indices):
            chi2_argmin[i] = np.nanargmin(chi2_map[:, i])
        else:
            # If all values are NaN, use the first triangle as fallback
            chi2_argmin[i] = -1

    # chi2_argmin[300] = 395  # manual fix for a weird outlier
    # chi2_argmin[300] = 412  # manual fix for a weird outlier

    QTdata = np.zeros((QT.shape[1],QT.shape[2],datacube_T.shape[2]))
    for i in tqdm(range(Ncube * Nmod), desc="Projection onto QT space"):
        t = chi2_argmin[i]
        if t >= 0:
            data = datacube_T[:,:,i]
            QTdata[:,:,i] = (QT[t] @ data[:,:,None])[:,:,0]
        else:
            QTdata[:,:,i] = np.nan  # or some other placeholder for invalid data



    QTdata_dxy = np.zeros_like(QTdata)
    Nqr = R.shape[2]

    Xpos = np.zeros((Ncube , Nmod))
    Ypos = np.zeros((Ncube , Nmod))
    Xcen = np.zeros((Ncube , Nmod))
    Ycen = np.zeros((Ncube , Nmod))
    Xdiff = np.zeros((Ncube , Nmod))
    Ydiff = np.zeros((Ncube , Nmod))
    ZXY_wave = np.zeros((Ncube * Nmod , Nwave, Nqr))
        
    for i in tqdm(range(Ncube * Nmod), desc="Computing XY positions"):
        t = chi2_argmin[i]
        center = couplingMap.position[t]

        if Nqr == 6:
            # x_hat_broadband, y_hat_broadband, k_hat_broadband, chi2_broadband, _ = runlib_linalg.fit_QR_6(QTdata_broadband, R_broadband[t])
            pass
        else:
            zxy_bd = np.zeros((Nwave,Nqr))
            if np.any(np.isnan(QTdata[:, :, i])):
                zxy_bd[:] = np.nan
            else:
                for w in range(Nwave):
                    zxy_bd[w] = solve_triangular(R[t, w], QTdata[w, :, i].T, lower=False)

        ZXY_wave[i] = zxy_bd  # or some other appropriate assignment
        Xpos.ravel()[i] = zxy_bd[:,1].mean(axis=0)/zxy_bd[:,0].mean(axis=0)  # x_hat_broadband
        Ypos.ravel()[i] = zxy_bd[:,2].mean(axis=0)/zxy_bd[:,0].mean(axis=0)  # y_hat_broadband

        Xcen.ravel()[i] = center[0]
        Ycen.ravel()[i] = center[1]

        Xdiff.ravel()[i] = Xpos.ravel()[i] + Xcen.ravel()[i] - xmod.ravel()[i]
        Ydiff.ravel()[i] = Ypos.ravel()[i] + Ycen.ravel()[i] - ymod.ravel()[i]

    coupling = "pyramids" if use_pyramids else "triangles"

    fig, axs = plt.subplots(2, Ncube, num="XY position -- using "+coupling, clear=True, figsize=(7*Ncube,12), squeeze=False)
    for i in range(Ncube):
        axs[0,i].plot(Xcen[i],Ycen[i],'.',label='Center of '+coupling)
        axs[0,i].set_ylim(axs[0,i].get_ylim()[0], axs[0,i].get_ylim()[1])
        axs[0,i].set_xlim(axs[0,i].get_xlim()[0], axs[0,i].get_xlim()[1])
        axs[0,i].plot((Xcen+Xpos)[i],(Ycen+Ypos)[i],'.-',label='Detected position')
        axs[0,i].plot((Xcen[i],(Xcen+Xpos)[i]),(Ycen[i],(Ycen+Ypos)[i]),'-k',alpha=0.3,linewidth=0.5)
        axs[0,i].set_title(basenames[i][8:])
        axs[0,i].set_xlabel("X [mas]")
        axs[0,i].set_ylabel("Y [mas]")
        axs[0,i].legend()
    for ax in axs[0]:
        ax.set_aspect('equal')
    for i in range(Ncube):
        x_median = np.nanmedian(Xdiff[i])
        y_median = np.nanmedian(Ydiff[i])
        x_1sigma = np.nanpercentile(Xdiff[i], [16, 84])
        y_1sigma = np.nanpercentile(Ydiff[i], [16, 84])
        range_max = np.nanmax((np.abs(x_1sigma), np.abs(y_1sigma))) * 2 +10
        axs[1,i].hist(Xdiff[i][~np.isnan(Xdiff[i])], bins=51, alpha=0.5, color='b', label='Xdiff', range=(-range_max, range_max))
        axs[1,i].hist(Ydiff[i][~np.isnan(Ydiff[i])], bins=51, alpha=0.5, color='r', label='Ydiff', range=(-range_max, range_max))
        x_median = np.nanmedian(Xdiff[i])
        y_median = np.nanmedian(Ydiff[i])
        x_1sigma = np.nanpercentile(Xdiff[i], [16, 84])
        y_1sigma = np.nanpercentile(Ydiff[i], [16, 84])
        axs[1,i].axvline(x_median, color='b', linestyle='--', label=f'X median: {x_median:.2f}')
        axs[1,i].axvline(y_median, color='r', linestyle='--', label=f'Y median: {y_median:.2f}')
        # axs[1,i].axvspan(x_1sigma[0], x_1sigma[1], color='b', alpha=0.2, label=f'X 1σ: [{x_1sigma[0]:.2f}, {x_1sigma[1]:.2f}]')
        # axs[1,i].axvspan(y_1sigma[0], y_1sigma[1], color='r', alpha=0.2, label=f'Y 1σ: [{y_1sigma[0]:.2f}, {y_1sigma[1]:.2f}]')
        axs[1,i].set_xlabel('Difference [mas]')
        axs[1,i].set_ylabel('Count')
        axs[1,i].legend()
    
    plt.tight_layout()

    wave=datalist[0].wave
    wave_label=datalist[0].wave_label
    fig, axs = plt.subplots(3, num="Spectra and xy offsets 3", clear=True, figsize=(10,18),sharex=True)
    
    # Calculate means and standard errors
    flux_mean = np.nanmean(ZXY_wave,axis=0)[:,0]
    flux_stderr = np.nanstd(ZXY_wave,axis=0)[:,0] / np.sqrt(np.sum(~np.isnan(ZXY_wave[:,:,0]),axis=0))
    
    x_offset_mean = np.nanmean(ZXY_wave,axis=0)[:,1]/np.nanmean(ZXY_wave,axis=0)[:,0]
    x_offset_stderr = np.nanstd(ZXY_wave[:,:,1]/ZXY_wave[:,:,0],axis=0) / np.sqrt(np.sum(~np.isnan(ZXY_wave[:,:,1]/ZXY_wave[:,:,0]),axis=0))
    
    y_offset_mean = np.nanmean(ZXY_wave,axis=0)[:,2]/np.nanmean(ZXY_wave,axis=0)[:,0]
    y_offset_stderr = np.nanstd(ZXY_wave[:,:,2]/ZXY_wave[:,:,0],axis=0) / np.sqrt(np.sum(~np.isnan(ZXY_wave[:,:,2]/ZXY_wave[:,:,0]),axis=0))
    
    # Plot flux with error bars
    axs[0].plot(wave, flux_mean)
    axs[0].set_ylim(*axs[0].get_ylim())
    axs[0].fill_between(wave, flux_mean - flux_stderr, flux_mean + flux_stderr, 
                        alpha=0.3, color='gray', label='±1σ')
    axs[0].set_title("Flux")
    
    # Plot X offsets with error bars
    axs[1].plot(wave, x_offset_mean)
    axs[1].set_ylim(*axs[1].get_ylim())
    axs[1].fill_between(wave, x_offset_mean - x_offset_stderr, x_offset_mean + x_offset_stderr, 
                        alpha=0.3, color='gray', label='±1σ')
    axs[1].set_title("X Offsets")
    # axs[1].legend()
    
    # Plot Y offsets with error bars
    axs[2].plot(wave, y_offset_mean)
    axs[2].set_ylim(*axs[2].get_ylim())
    axs[2].fill_between(wave, y_offset_mean - y_offset_stderr, y_offset_mean + y_offset_stderr, 
                        alpha=0.3, color='gray', label='±1σ')
    axs[2].set_title("Y Offsets")
    axs[2].set_xlabel(wave_label)
    # axs[2].legend()

    axs[0].set_xlim(wave.min(), wave.max())
    plt.tight_layout()

    runlib_plots.save_pdf_in_file(output_filename)


# %%

#ideal : 0.01 mas

if __name__ == "__main__":
    main()