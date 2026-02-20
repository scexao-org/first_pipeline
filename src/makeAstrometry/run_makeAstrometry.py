#! /usr/bin/env python3
# -*- coding: iso-8859-15 -*-
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
from typing import List, Tuple, Optional
from scipy.interpolate import griddata
from astropy.io import fits
from datetime import datetime
from scipy.optimize import least_squares

import getpass
import matplotlib
if ("VSCODE_PID" in os.environ or os.environ.get('TERM_PROGRAM') == 'vscode'):
    matplotlib.use('Qt5Agg')
elif os.environ.get('SPYDER_DEBUG_FILE'):
    print("Running in Spyder")
else:
    matplotlib.use('Agg')
     
import matplotlib.pyplot as plt
from tqdm import tqdm
from astroplan import Observer
from astropy.time import Time

from first_pipeline_shared.libraries import runPL_library_io as runlib_io
from first_pipeline_shared.libraries import runPL_library_plots as runlib_plots
from first_pipeline_shared.libraries import runPL_library_linalg as runlib_linalg
from first_pipeline_shared.classes.runPL_class_fileList import FileList
from first_pipeline_shared.classes.runPL_class_dataCube import DataCube
from first_pipeline_shared.classes.runPL_class_couplingMap import CouplingMap

# Subaru Observatory instance for timing
subaru = Observer.at_site("Subaru")


def get_development_defaults():
    """
    Get development environment default parameters.
    
    Returns
    -------
    dict
        Dictionary containing default parameters for development environment
    """
    defaults = {
        'file_patterns': ['*.fits'],
        'object_name': None,
        'dark_patterns': None,
        'coupling_map': None,
        'wavelength_smooth': 1,
        'wollaston': None,
        'save_individual_frames': True,
        'save_individual_wavelength': False,
        'pyramids': False
    }
    
    # Check if running in development environment
    if (("VSCODE_PID" in os.environ or os.environ.get('TERM_PROGRAM') == 'vscode') 
        or os.environ.get('SPYDER_DEBUG_FILE')):
        
        user = getpass.getuser()
        if user == "slacour":
            defaults['file_patterns'] = ["/Users/slacour/DATA/LANTERNE/20250514/preproc/firstpl_2025-05-14T10:10*.fits"]
            defaults['dark_patterns'] = ["/Users/slacour/DATA/LANTERNE/20250514/preproc"]
            defaults['coupling_map'] = "/Users/slacour/DATA/LANTERNE/20250514/couplingmaps/firstpl_2025-05-14T11:39:58_HIP85819_CM.fits"
        elif user == "ehuby":
            defaults['file_patterns'] = ["/home/ehuby/WORK/DATA/FIRST-PL/2025-05-10/preproc/firstpl_*.fits"]
    
    return defaults


def get_filelist_astrometry(file_patterns, dark_patterns=None, cmap_patterns=None,
                           object_name=None, wollaston=None):
    """
    Create file list for astrometric analysis with coupling map association.
    
    Parameters
    ----------
    file_patterns : list
        List of file patterns to search for OBJECT data
    dark_patterns : list, optional
        List of patterns for dark files
    cmap_patterns : list, optional
        List of patterns for coupling map files
    object_name : str, optional
        Filter by object name
    wollaston : str, optional
        Wollaston polarizer status
        
    Returns
    -------
    fileList : FileList
        Configured file list object
    couplingMap : CouplingMap
        Coupling map object for astrometric analysis
    """
    # Set default patterns
    if dark_patterns is None:
        dark_patterns = file_patterns
    if cmap_patterns is None and file_patterns:
        folder = os.path.dirname(file_patterns[0])
        cmap_patterns = file_patterns + [os.path.join(folder, "../couplingmaps")] + [os.path.join(folder, "couplingmaps")]

    # Create file list with OBJECT type data
    fileList = FileList(file_patterns, data_type="OBJECT", first_type='PREPROC',
                       wollaston=wollaston, object_name=object_name)

    fileList.make_association(dark_patterns=dark_patterns)

    # Load coupling map
    file_coup = fileList.get_couplingmap_file(cmap_patterns)
    couplingMap = CouplingMap(file_coup, pyramids=True)

    return fileList, couplingMap


def apply_data_filtering(flux, datacube):
    """
    Apply flux and SVD filtering to data for quality control.
    
    Parameters
    ----------
    flux : numpy.ndarray
        Flux measurements for filtering
    datacube : numpy.ndarray
        Data cube to be filtered
        
    Returns
    -------
    goodData : numpy.ndarray
        Boolean mask indicating good data points
    flux_threshold : float
        Flux filtering threshold used
    """
    # Flux-based filtering
    flux_goodData, flux_threshold = runlib_linalg.flux_filtering(flux)
    print(f"* Percentage of good data: {np.sum(flux_goodData)/len(flux_goodData.ravel())*100:.1f} % (flux threshold)")
    
    # SVD-based filtering
    data_svdfiltered, fit_goodData, errors = runlib_linalg.svd_filtering(datacube, flux_goodData)
    goodData = flux_goodData & fit_goodData
    print(f"* Percentage of good data: {np.sum(goodData)/len(goodData.ravel())*100:.1f} % (flux and svd threshold)")

    return goodData, flux_threshold


def compute_astrometric_positions(datacube_T, couplingMap, ra_dec, star_detected, star_index, Nqr=3):
    """
    Compute high-precision astrometric positions using coupling map analysis.
    
    Parameters
    ----------
    datacube_T : numpy.ndarray
        Transposed data cube with shape (Nwave, Noutput, Nimages)
    couplingMap : CouplingMap
        Coupling map object for astrometric analysis
    ra_dec : numpy.ndarray
        Sky coordinate positions
    star_detected : numpy.ndarray
        Boolean mask of frames with detected stars
    star_index : numpy.ndarray
        Triangle indices for each detection
    Nqr : int, optional
        Number of QR components to use (default: 3)
        
    Returns
    -------
    xy_dev : numpy.ndarray
        Astrometric position deviations
    QTdata_star_removed : numpy.ndarray  
        Data with stellar signal removed
    R_dxy : numpy.ndarray
        Position derivative matrices
    """
    Nwave, Noutput, Nimages = datacube_T.shape
    
    # Compute broadband QR matrices for enhanced precision
    spectra = np.mean(datacube_T, axis=2)  # Average spectrum
    wmin = len(spectra) // 4
    wmax = 3 * len(spectra) // 4
    QT_broadband, R_broadband = couplingMap.compute_broadband_QR(wmin, wmax, spectra)

    # Transform data to QT space
    QTdata = couplingMap.QT_dot_data(star_index, datacube_T)
    
    # Initialize arrays for star removal and position derivatives
    QTdata_star_removed = QTdata.copy()
    R_dxy = np.zeros((Nwave, Nqr, Nimages, 2))
    
    # Process each frame with detected star
    for i in range(Nimages):
        if star_detected[i]:
            t = star_index[i]
            
            # Get star position for this frame
            x_hat_broadband = ra_dec[i, 0]
            y_hat_broadband = ra_dec[i, 1]
            
            # Set up position model (polynomial basis)
            if Nqr == 6:  # Full quadratic model
                v = np.array([1.0, x_hat_broadband, y_hat_broadband, 
                             x_hat_broadband*y_hat_broadband, 
                             x_hat_broadband**2, y_hat_broadband**2])
                dv_dx = np.array([0.0, 1.0, 0.0, y_hat_broadband, 
                                 2.0*x_hat_broadband, 0.0])
                dv_dy = np.array([0.0, 0.0, 1.0, x_hat_broadband, 
                                 0.0, 2.0*y_hat_broadband])
            else:  # Linear model
                v = np.array([1.0, x_hat_broadband, y_hat_broadband])
                dv_dx = np.array([0.0, 1.0, 0.0])
                dv_dy = np.array([0.0, 0.0, 1.0])

            # Compute R matrix for this triangle
            R = couplingMap.R[t]
            r = R @ v
            
            # Remove stellar signal using kernel projection
            Kernel_v = np.identity(len(v)) - (r[:,:,None] @ r[:,None]) / (r[:,None] @ r[:,:,None])
            QTdata_star_removed[:,:,i] = (Kernel_v @ QTdata[:,:,i,None])[...,0]

            # Compute position derivative matrices
            dev_phi = np.array((dv_dx, dv_dy)).T
            R_dxy[:,:,i] = Kernel_v @ (R @ dev_phi)

    # Transform back to data space
    data_star_removed = couplingMap.Q_dot_QTdata(star_index, QTdata_star_removed)

    # Solve for position deviations
    xy_dev = np.linalg.pinv(R_dxy.reshape((Nwave, -1, 2))) @ QTdata_star_removed.reshape((Nwave, -1, 1))
    xy_dev = xy_dev[..., 0]

    return xy_dev, QTdata_star_removed, R_dxy


def save_astrometric_results(xy_dev, header, output_dir, save_individual_frames=True, 
                           save_individual_wavelength=False, wavelength_smooth=1):
    """
    Save astrometric measurements to FITS file.
    
    Parameters
    ----------
    xy_dev : numpy.ndarray
        Astrometric position deviations
    header : astropy.io.fits.Header
        FITS header to be updated
    output_dir : str
        Output directory path
    save_individual_frames : bool, optional
        Save individual frame measurements (default: True)
    save_individual_wavelength : bool, optional
        Save wavelength-resolved measurements (default: False)
    wavelength_smooth : int, optional
        Wavelength smoothing parameter
        
    Returns
    -------
    str
        Path to saved astrometry file
    """
    # Update header
    header = header.copy()
    header['X_FIRTYP'] = 'ASTROMETRY'
    current_time = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
    header['DATE-PRO'] = current_time
    header['WLSMOOTH'] = wavelength_smooth

    # Create HDU list
    list_of_hdus = []
    
    # Primary HDU with astrometric measurements
    hdu_primary = fits.PrimaryHDU(xy_dev)
    
    # Residual map (same as primary for now)
    hdu_residual = fits.ImageHDU(xy_dev, name="RESIDUAL")
    list_of_hdus += [hdu_primary, hdu_residual]

    # Additional extensions for individual frames/wavelengths if requested
    if save_individual_frames:
        # Add frame-by-frame analysis if needed
        pass
    
    if save_individual_wavelength:
        # Add wavelength-resolved analysis if needed
        pass

    # Create output directory and save
    os.makedirs(output_dir, exist_ok=True)
    
    # Add header to primary HDU
    hdu_primary.header.extend(header, strip=True)
    
    # Create and save FITS file
    hdul = fits.HDUList(list_of_hdus)
    output_filename = os.path.join(output_dir, runlib_io.create_basename(header))
    hdul.writeto(output_filename, overwrite=True)
    
    return output_filename


def create_astrometric_plots(xy_dev, ra_dec, star_detected):
    """
    Generate diagnostic plots for astrometric quality assessment.
    
    Parameters
    ----------
    xy_dev : numpy.ndarray
        Astrometric position deviations
    ra_dec : numpy.ndarray
        Original sky coordinates
    star_detected : numpy.ndarray
        Boolean mask of detected stars
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure showing astrometric results
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), num="Astrometric Results")
    
    # Plot X position deviations vs wavelength
    axes[0,0].plot(xy_dev[:, 0], '.-')
    axes[0,0].set_title('X Position Deviations')
    axes[0,0].set_xlabel('Wavelength channel')
    axes[0,0].set_ylabel('X deviation (mas)')
    axes[0,0].grid(True)
    
    # Plot Y position deviations vs wavelength
    axes[0,1].plot(xy_dev[:, 1], '.-')
    axes[0,1].set_title('Y Position Deviations')
    axes[0,1].set_xlabel('Wavelength channel')
    axes[0,1].set_ylabel('Y deviation (mas)')
    axes[0,1].grid(True)
    
    # Position scatter plot
    good_positions = ra_dec[star_detected] if len(ra_dec) == len(star_detected) else ra_dec
    if len(good_positions) > 0:
        axes[1,0].scatter(good_positions[:, 0], good_positions[:, 1], alpha=0.6)
        axes[1,0].set_title('Sky Positions')
        axes[1,0].set_xlabel('RA offset (mas)')
        axes[1,0].set_ylabel('Dec offset (mas)')
        axes[1,0].axis('equal')
        axes[1,0].grid(True)
    
    # Detection rate vs time
    if len(star_detected) > 0:
        detection_rate = np.convolve(star_detected.astype(float), 
                                   np.ones(min(50, len(star_detected))))/min(50, len(star_detected))
        axes[1,1].plot(detection_rate)
        axes[1,1].set_title('Star Detection Rate')
        axes[1,1].set_xlabel('Frame number')
        axes[1,1].set_ylabel('Detection rate')
        axes[1,1].grid(True)
    
    plt.tight_layout()
    return fig


def process_astrometric_data(file_patterns=None, object_name=None, dark_patterns=None,
                           coupling_map=None, wavelength_smooth=None, wollaston=None,
                           save_individual_frames=None, save_individual_wavelength=None,
                           pyramids=None):
    """
    Complete workflow for astrometric analysis from coupling maps and preprocessed data.
    
    This is the main processing function that orchestrates the entire astrometric
    analysis workflow from file loading through position measurement.
    
    Parameters
    ----------
    file_patterns : list, optional
        List of file patterns to search for OBJECT data files.
        If None, uses development defaults.
    object_name : str, optional
        Filter by object name
    dark_patterns : list, optional
        List of patterns for dark files
    coupling_map : str, optional
        Specific coupling map file to use
    wavelength_smooth : int, optional
        Wavelength smoothing factor. If None, uses development defaults.
    wollaston : str, optional
        Wollaston polarizer status
    save_individual_frames : bool, optional
        Save individual frame measurements. If None, uses development defaults.
    save_individual_wavelength : bool, optional
        Save wavelength-resolved measurements. If None, uses development defaults.
    pyramids : bool, optional
        Use pyramidal fitting. If None, uses development defaults.
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'output_filename': path to saved astrometry file
        - 'xy_dev': astrometric position measurements
        - 'star_detected': star detection results  
        - 'figures': list of diagnostic figures
    """
    # Use development defaults if parameters are None
    if any(param is None for param in [file_patterns, wavelength_smooth, save_individual_frames, save_individual_wavelength, pyramids]):
        defaults = get_development_defaults()
        if file_patterns is None:
            file_patterns = defaults['file_patterns']
        if wavelength_smooth is None:
            wavelength_smooth = defaults['wavelength_smooth']
        if save_individual_frames is None:
            save_individual_frames = defaults['save_individual_frames']
        if save_individual_wavelength is None:
            save_individual_wavelength = defaults['save_individual_wavelength']
        if pyramids is None:
            pyramids = defaults['pyramids']
        
        # Also use default coupling_map if not specified
        if coupling_map is None and defaults['coupling_map'] is not None:
            coupling_map = defaults['coupling_map']
        if dark_patterns is None and defaults['dark_patterns'] is not None:
            dark_patterns = defaults['dark_patterns']
    
    # Set up coupling map patterns
    cmap_patterns = [coupling_map] if coupling_map else None

    # Get file list and coupling map
    fileList, couplingMap = get_filelist_astrometry(
        file_patterns, dark_patterns, cmap_patterns, object_name, wollaston
    )

    # Extract data with coupling map wavelength binning
    datalist: List[DataCube] = fileList.extract_data_from_list(
        Nsmooth=wavelength_smooth, Nbin=couplingMap.wavelength_bin,
        flatMap=None, waveMap=None, center=False
    )

    results = []
    figures = []

    for i, d in enumerate(datalist[:]):
        print(f"---->  Processing file {i+1}/{len(datalist)}: {d.filename}")

        # Extract data arrays
        flux = d.flux
        datacube = d.data
        datacube_var = d.variance
        ra_dec = d.compute_xy_sky(couplingMap)

        # Get dimensions
        Ncube = ra_dec.shape[0]
        Nmod = ra_dec.shape[1]
        Npos = ra_dec.shape[2]
        Nwave = datacube.shape[3]
        Noutput = datacube.shape[2]
        Nimages = Ncube * Nmod

        # Apply data quality filtering
        goodData, flux_threshold = apply_data_filtering(flux, datacube)

        # Select only good data
        datacube_T = datacube[goodData].transpose((2, 1, 0))
        ra_dec = ra_dec.reshape((-1, *ra_dec.shape[2:]))[goodData]
        flux = flux.reshape((-1, *flux.shape[2:]))[goodData]

        # Detect stars using coupling map chi-squared filtering
        star_detected, star_index, star_radec, chi2 = couplingMap.chi2_filtering(datacube_T, ra_dec)
        print(f"* Percentage of data with star detected: {np.sum(star_detected)/len(star_detected)*100:.1f} % (flux, svd and chi2 threshold)")

        # Select only frames with detected stars
        datacube_T = datacube_T[:, :, star_detected]
        ra_dec = ra_dec[star_detected]
        flux = flux[star_detected]
        star_index = star_index[star_detected]

        # Determine QR size based on pyramid/triangle mode
        Nqr = 6 if pyramids else 3

        # Compute astrometric positions
        xy_dev, QTdata_star_removed, R_dxy = compute_astrometric_positions(
            datacube_T, couplingMap, ra_dec, star_detected, star_index, Nqr
        )

        # Set up output directory
        output_dir = os.path.join(d.dirname, "../astrometry")

        # Save astrometric results
        output_filename = save_astrometric_results(
            xy_dev, d.header, output_dir, save_individual_frames,
            save_individual_wavelength, wavelength_smooth
        )

        print(f"Astrometry saved to {output_filename}")

        # Create diagnostic plots
        fig = create_astrometric_plots(xy_dev, ra_dec, star_detected)
        figures.append(fig)

        # Save plots
        runlib_plots.save_pdf_in_file(output_filename)

        results.append({
            'output_filename': output_filename,
            'xy_dev': xy_dev,
            'star_detected': star_detected,
            'star_radec': star_radec,
            'chi2': chi2
        })

    return {
        'results': results,
        'figures': figures,
        'couplingMap': couplingMap
    }


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


if __name__ == "__main__":
    """
    Run astrometric analysis with development defaults.
    Perfect for testing and direct execution of core functionality.
    """
    print("Running makeAstrometry core with development defaults...")
    
    # Check observatory status
    status = check_observatory_status()
    print(status)
    
    # Get development defaults first
    defaults = get_development_defaults()
    
    # Run astrometric analysis with defaults
    try:
        result = process_astrometric_data()
        
        print(f"Astrometric analysis completed successfully!")
        print(f"Processed {len(result['results'])} file(s)")
        
        for i, file_result in enumerate(result['results']):
            print(f"  File {i+1}: {file_result['output_filename']}")
            print(f"    Stars detected: {file_result['star_detected'].sum() if hasattr(file_result['star_detected'], 'sum') else len(file_result['star_detected'])} frames")
            
            if 'xy_dev' in file_result:
                xy_dev = file_result['xy_dev']
                print(f"    Position measurements: {xy_dev.shape[0]} wavelength channels")
                print(f"    X deviation RMS: {xy_dev[:, 0].std():.3f} mas")
                print(f"    Y deviation RMS: {xy_dev[:, 1].std():.3f} mas")
        
    except Exception as e:
        print(f"Error running astrometric analysis: {e}")
        print("This may be due to missing preprocessed data files or coupling maps in default paths")
        
        # Show default paths being used
        print(f"Default file patterns: {defaults['file_patterns']}")
        if defaults['coupling_map']:
            print(f"Default coupling map: {defaults['coupling_map']}")
        print("Note: Requires preprocessed files and coupling maps")