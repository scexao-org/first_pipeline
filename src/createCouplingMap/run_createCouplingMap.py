#%%

"""
FIRST Pipeline - Coupling Map Generation Core Algorithms

Core functions for creating coupling maps from preprocessed FIRST Visible Photonic Lantern data.
Separated from CLI interface to enable interactive use in VS Code and notebooks.

Created on Wed May 21 22:56:25 2025
@author: slacour
"""

import os
import getpass
import matplotlib
if "VSCODE_PID" in os.environ:
    matplotlib.use('Qt5Agg')
elif os.environ.get('SPYDER_DEBUG_FILE'):
    print("Running in Spyder")
else:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, hist, clf, figure, legend, imshow
plt.ion()

import numpy as np
from typing import List, Tuple
from scipy.signal import correlate
from scipy import linalg
from scipy.linalg import solve_triangular, pinv
     
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict
from itertools import product
import signal

from first_pipeline_shared.classes.runPL_class_flatMap import FlatMap  
from first_pipeline_shared.classes.runPL_class_waveMap import WaveMap
from first_pipeline_shared.classes.runPL_class_fileList import FileList
from first_pipeline_shared.classes.runPL_class_dataCube import DataCube
from first_pipeline_shared.classes.runPL_class_couplingMap import CouplingMap

from first_pipeline_shared.libraries import runPL_library_io as runlib_io
from first_pipeline_shared.libraries import runPL_library_plots as runlib_plots
from first_pipeline_shared.libraries import runPL_library_linalg as runlib_linalg

from astropy.io import fits


def singular_vector_basis(data_svdfiltered, goodData, indexes, xmod, ymod):
    """
    Compute singular vector basis for triangles or pyramids using SVD filtering.
    
    This function processes SVD-filtered data to extract singular vectors for each
    triangle/pyramid using least squares fitting with spatial modulation positions.
    
    Parameters
    ----------
    data_svdfiltered : numpy.ndarray
        SVD-filtered data array with shape (Nframes, Noutput, Nwave)
    goodData : numpy.ndarray
        Boolean mask indicating good data points
    indexes : numpy.ndarray
        Triangle or pyramid indices for fiber combinations
    xmod : numpy.ndarray
        X-direction modulation positions
    ymod : numpy.ndarray
        Y-direction modulation positions
        
    Returns
    -------
    vectors_all_triangles : numpy.ndarray
        Singular vectors for all valid triangles/pyramids
    center_all_triangles : numpy.ndarray
        Center positions for all valid triangles/pyramids
    indexes_new : list
        Updated indexes for triangles/pyramids with valid data
    """
    vectors_all_triangles = []
    center_all_triangles = []
    _, Nqr = indexes.shape
    
    if Nqr == 3:
        description = "Computing triangles singular vectors"
    else:
        description = "Computing pyramids singular vectors"

    Ntriangles = 0
    indexes_new = []
    
    for t in tqdm(np.arange(len(indexes)), desc=description):
        # Extract singular vectors for each triangle or pyramid
        i = indexes[t]

        # Use SVD filtering to get good triangle data
        data_triangle_svdfiltered, good_data_triangle, _ = runlib_linalg.svd_filtering(
            data_svdfiltered[:, i], goodData[:, i], Nqr, verbose=False
        )
        
        # Use original data for now (SVD filtering needs improvement)
        data_triangle_svdfiltered = data_svdfiltered[:, i]
        good_data_triangle = goodData[:, i]

        data_triangle = data_triangle_svdfiltered[good_data_triangle].reshape(
            (data_triangle_svdfiltered[good_data_triangle].shape[0], -1)
        )
        xmod_triangle = xmod[:, i][good_data_triangle]
        ymod_triangle = ymod[:, i][good_data_triangle]

        x = xmod_triangle
        y = ymod_triangle
        
        # Set up design matrix based on triangle (3) or pyramid (5) configuration
        if Nqr == 3:
            X = np.array([x, y]).T
        else:
            X = np.array([x, y, x*y, x*x, y*y]).T

        Y = data_triangle

        # Centered statistics
        mu_x = X.mean(axis=0)
        mu_y = Y.mean(axis=0)
        Xc = X - mu_x
        Yc = Y - mu_y

        # Covariance matrices
        Sxx = Xc.T @ Xc
        Sxy = Xc.T @ Yc

        # Check rank of design matrix
        rank_sxx = np.linalg.matrix_rank(Sxx)
        if rank_sxx < 2:
            continue

        # Least squares solution
        B = pinv(Sxx) @ Sxy

        # Store vectors: [mean, coefficients]
        Vectors_triangle = np.hstack([mu_y[:, None], B.T])
        vectors_all_triangles.append(Vectors_triangle)
        center_all_triangles.append(mu_x)
        Ntriangles += 1
        indexes_new += [i]

    Noutput, Nwave = data_svdfiltered.shape[2:]

    center_all_triangles = np.array(center_all_triangles)
    vectors_all_triangles = np.array(vectors_all_triangles).reshape((Ntriangles, Noutput, Nwave, Nqr))

    return vectors_all_triangles, center_all_triangles, indexes_new


def flux_matrices(singular_vectors):
    """
    Compute flux-to-data and data-to-flux transformation matrices.
    
    Parameters
    ----------
    singular_vectors : numpy.ndarray
        Singular vectors array with shape (Ntriangles, Noutput, Nwave, Nqr)
        
    Returns
    -------
    flux_2_data : numpy.ndarray
        Transformation matrix from flux to data space
    data_2_flux : numpy.ndarray
        Pseudo-inverse transformation from data to flux space
    """
    Ntriangles = singular_vectors.shape[0]
    Noutput = singular_vectors.shape[1]
    Nwave = singular_vectors.shape[2]

    flux_2_data = singular_vectors[:, :, :, 0]
    flux_2_data = flux_2_data.transpose((2, 1, 0))
    data_2_flux = np.zeros((Nwave, Ntriangles, Noutput))
    data_2_flux = np.linalg.pinv(flux_2_data)

    return flux_2_data, data_2_flux


def Q_and_R_matrices(singular_vectors):
    """
    Compute QR decomposition matrices for singular vectors.
    
    Parameters
    ----------
    singular_vectors : numpy.ndarray
        Singular vectors array with shape (Ntriangles, Noutput, Nwave, Nqr)
        
    Returns
    -------
    QT_singular_vectors : numpy.ndarray
        Transpose of Q matrices from QR decomposition
    R_singular_vectors : numpy.ndarray
        R matrices from QR decomposition
    """
    Ntriangles = singular_vectors.shape[0]
    Noutput = singular_vectors.shape[1]
    Nwave = singular_vectors.shape[2]
    Nqr = singular_vectors.shape[3]

    QT_singular_vectors = np.zeros((Ntriangles, Nwave, Nqr, Noutput))
    R_singular_vectors = np.zeros((Ntriangles, Nwave, Nqr, Nqr))

    if Nqr == 3:
        description = "Calculating QR matrices for triangles"
    else:
        description = "Calculating QR matrices for pyramids"

    for t in tqdm(range(Ntriangles), desc=description):
        for w in range(Nwave):
            Q, R = np.linalg.qr(singular_vectors[t, :, w], mode="reduced")
            QT_singular_vectors[t, w] = Q.T
            R_singular_vectors[t, w] = R

    return QT_singular_vectors, R_singular_vectors


def get_filelist_coupling(file_patterns, dark_patterns=None, flat_patterns=None, 
                         wave_patterns=None, object_name=None, modID=None, 
                         modScale=None, wollaston=None):
    """
    Create file list for coupling map generation with calibration associations.
    
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

    return fileList, flatMap, waveMap


def create_diagnostic_plots(flux, data_svdfiltered, flux_2_data, flux_goodData,
                           fit_goodData, errors, center_all, QT, R,
                           flatMap=None):
    """
    Generate comprehensive diagnostic plots for coupling map analysis.
    
    Parameters
    ----------
    flux : numpy.ndarray
        Flux data array
    data_svdfiltered : numpy.ndarray
        SVD-filtered data
    goodData : numpy.ndarray
        Combined good data mask
    flux_goodData : numpy.ndarray
        Flux filtering mask
    fit_goodData : numpy.ndarray
        SVD fitting mask
    errors : numpy.ndarray
        SVD reconstruction errors
    center_all : numpy.ndarray
        Triangle/pyramid center positions
    QT : numpy.ndarray
        QT matrices
    R : numpy.ndarray
        R matrices
    flatMap : FlatMap, optional
        Flat field map for additional diagnostics
        
    Returns
    -------
    list
        List of created matplotlib figures
    """
    figures = []

    # Plot detector flat field if available
    if flatMap is not None:
        fig_flat = runlib_plots.plot_detector_field(flatMap.flat, 
                                                   title="Flat Map for file" + flatMap.basename)
        figures.append(fig_flat)

    # Data quality plots
    datacube = data_svdfiltered  # Alias for clarity
    fig, axs = plt.subplots(2, 2, num="Dataset Information", figsize=(16, 12), clear=True)

    # Mean flux per (wavelength, output)
    axs[0, 0].imshow(flux.mean(axis=(2)), aspect='auto', origin='lower', 
                     cmap='viridis', interpolation='none', rasterized=True)
    axs[0, 0].set_title("From flux, Dataset (flux)")
    axs[0, 0].set_xlabel("Output")
    axs[0, 0].set_ylabel("Wavelength")

    # Good data mask from flux
    axs[0, 1].imshow(flux_goodData, aspect='auto', origin='lower', 
                     cmap='Greens', interpolation='none', rasterized=True, vmin=0, vmax=1)
    axs[0, 1].set_title("From flux, good Dataset (mask)")
    axs[0, 1].set_xlabel("Output")
    axs[0, 1].set_ylabel("Wavelength")

    # SVD reconstruction errors
    error_norm = errors.reshape((datacube.shape[0], -1))
    axs[1, 0].imshow(error_norm, aspect='auto', origin='lower', 
                     cmap='viridis', interpolation='none', rasterized=True)
    axs[1, 0].set_title("Amplitude of residuals after SVD filtering")
    axs[1, 0].set_xlabel("Output")
    axs[1, 0].set_ylabel("files")

    # SVD fit quality mask
    axs[1, 1].imshow(fit_goodData, aspect='auto', origin='lower',
                     cmap='Greens', interpolation='none', rasterized=True, vmin=0, vmax=1)
    axs[1, 1].set_title("From SVD fits, good Dataset (mask)")
    axs[1, 1].set_xlabel("Output")
    axs[1, 1].set_ylabel("Wavelength")

    fig.tight_layout()
    figures.append(fig)

    # Position plots
    fig, axs = plt.subplots(1, 2, num="Positions fiber and of triangles", 
                           figsize=(16, 6), sharex=True, sharey=True, clear=True)

    # Plot triangle centers
    axs[1].scatter(center_all[:, 0], center_all[:, 1], facecolors='g', 
                   marker='o', edgecolor='k', label='Good Triangles')
    axs[1].set_xlabel("x [mas]")
    axs[1].set_ylabel("y [mas]")
    axs[1].set_aspect('equal')
    axs[1].legend()

    fig.tight_layout()
    figures.append(fig)

    # Covariance and correlation matrix plots
    if len(center_all) > 0:
        fig_cov = runlib_plots.plot_covariance(flux_2_data, center_all, "Triangles")
        figures.append(fig_cov)

    # R amplitude analysis
    R_amplitude = np.linalg.norm(R, axis=2)
    fig_r = runlib_plots.plot_R_amplitude(R, name="coupling_analysis")
    figures.append(fig_r)

    return figures


def save_coupling_map(couplingMap, header, output_dir, flatMap=None, waveMap=None, 
                     wavelength_smooth=1, wavelength_bin=1, Nsingular=114,
                     center_data=True, use_pyramids=False, flux_threshold=None,
                     filenames=None, modulation_hdu=None):
    """
    Save coupling map with metadata to FITS file.
    
    Parameters
    ----------
    couplingMap : CouplingMap
        Coupling map object containing matrices and data
    header : astropy.io.fits.Header
        FITS header to be updated
    output_dir : str
        Output directory path
    flatMap : FlatMap, optional
        Associated flat field map
    waveMap : WaveMap, optional
        Associated wavelength map
    wavelength_smooth : int, optional
        Wavelength smoothing factor
    wavelength_bin : int, optional
        Wavelength binning factor
    Nsingular : int, optional
        Number of singular values used
    center_data : bool, optional
        Whether data was centered
    use_pyramids : bool, optional
        Whether pyramids were used instead of triangles
    flux_threshold : float, optional
        Flux filtering threshold
    filenames : list, optional
        List of input filenames
    modulation_hdu : astropy.io.fits.HDU, optional
        Modulation HDU from input files
        
    Returns
    -------
    str
        Path to saved coupling map file
    """
    # Update header with processing parameters
    new_header = header.copy()
    new_header['X_FIRTYP'] = 'COUPLINGMAP'
    new_header['Q_CMWSMO'] = (wavelength_smooth, 'wavelength smoothing factor')
    new_header['Q_CMWBIN'] = (wavelength_bin, 'wavelength binning factor') 
    new_header['Q_CMSING'] = (Nsingular, 'number of singular values')
    new_header['Q_CMCENT'] = (center_data, 'whether the data was centered before analysis')
    new_header['Q_CMPYR'] = (use_pyramids, 'whether pyramids (True) or triangles (False) were used')
    
    if flux_threshold is not None:
        new_header['Q_CM_FT'] = (flux_threshold, 'flux threshold')
    new_header['Q_CM_CK'] = (np.random.randint(0, 2**32, dtype=np.uint32), 'checksum')
    
    # Add input filenames
    if filenames is not None:
        for i, filename in enumerate(filenames):
            new_header[f'Q_CM_F{i}'] = (filename, 'filename of the extracted flux')

    new_header['Q_CMNAME'] = (runlib_io.create_basename(new_header), 'name of the coupling map file')

    # Create output directory and save
    os.makedirs(output_dir, exist_ok=True)
    output_filename = os.path.join(output_dir, new_header['Q_CMNAME'])

    # Save using CouplingMap save method
    couplingMap.save(output_filename, new_header, flat_map=flatMap, 
                    wave_map=waveMap, modulation_hdu=modulation_hdu)

    return couplingMap


def run_createCouplingMap(file_patterns=None, object_name=None, dark_patterns=None,
                             flat_patterns=None, wave_patterns=None, 
                             wavelength_smooth=None, wavelength_bin=None, 
                             Nsingular=None, modID=None, modScale=None,
                             wollaston=None, use_pyramids=None, center_data=None):
    """
    Complete workflow for coupling map generation from preprocessed data.
    
    This is the main processing function that orchestrates the entire coupling
    map generation workflow from file loading through final map creation.
    
    Parameters
    ----------
    file_patterns : list, optional
        List of file patterns to search for OBJECT data files.
        If None, uses development defaults.
    object_name : str, optional
        Filter by object name
    dark_patterns : list, optional
        List of patterns for dark files
    flat_patterns : list, optional
        List of patterns for flat field files  
    wave_patterns : list, optional
        List of patterns for wavelength map files
    wavelength_smooth : int, optional
        Wavelength smoothing factor. If None, uses development defaults.
    wavelength_bin : int, optional
        Wavelength binning factor. If None, uses development defaults.
    Nsingular : int, optional
        Number of SVD singular values to retain. If None, uses development defaults.
    modID : int or list, optional
        Modulation pattern ID(s)
    modScale : int, optional
        Modulation scale
    wollaston : str, optional
        Wollaston polarizer status
    use_pyramids : bool, optional
        Use pyramids instead of triangles. If None, uses development defaults.
    center_data : bool, optional
        Center data before analysis. If None, uses development defaults.
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'output_filename': path to saved coupling map
        - 'couplingMap': CouplingMap object
        - 'QT': QT matrices
        - 'R': R matrices
        - 'center_all': triangle/pyramid centers
        - 'figures': list of diagnostic figures
    """

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
    fileList, flatMap, waveMap = get_filelist_coupling(
        file_patterns, dark_patterns, flat_patterns, wave_patterns,
        object_name, modID, modScale, wollaston
    )

    # Extract data
    datalist: List[DataCube] = fileList.extract_data_from_list(
        Nsmooth=wavelength_smooth, Nbin=wavelength_bin, flatMap=flatMap,
        waveMap=waveMap, center=center_data
    )

    # Concatenate data arrays
    flux = np.concatenate([d.flux for d in datalist])
    datacube = np.concatenate([d.data for d in datalist])
    datacube_var = np.concatenate([d.variance for d in datalist])
    xmod = np.concatenate([d.xmod for d in datalist])
    ymod = np.concatenate([d.ymod for d in datalist])

    # Create filename associations
    basenames = []
    for d in datalist:
        n = d.data.shape[0]
        basenames.extend([d.basename] * n)
    filenames = [d.filename for d in datalist]

    # Data quality filtering
    flux_goodData, flux_threshold = runlib_linalg.flux_filtering(flux)
    print(f"* Percentage of good data: {np.sum(flux_goodData)/len(flux_goodData.ravel())*100:.1f} % (flux threshold)")

    # SVD filtering
    data_svdfiltered, fit_goodData, errors = runlib_linalg.svd_filtering(datacube, flux_goodData, Nsingular)
    goodData = flux_goodData & fit_goodData
    print(f"* Percentage of good data: {np.sum(goodData)/len(goodData.ravel())*100:.1f} % (flux and svd threshold)")

    # Plot flux map
    runlib_plots.plot_flux_map(flux.mean(axis=(2))[0], xmod[0], ymod[0])

    # Select good positions and triangles/pyramids
    goodPositions = goodData.mean(axis=0) > 0.3

    if use_pyramids == False:
        indexes = datalist[0].get_triangles()
    else:
        indexes = datalist[0].get_pyramids()

    # Filter triangles with good data
    goodTriangles = goodPositions[indexes].mean(axis=1) == 1
    indexes_good = indexes[goodTriangles]

    # Compute singular vector basis
    vectors_all, center_all, indexes_new = singular_vector_basis(
        data_svdfiltered, goodData, indexes_good, xmod, ymod
    )

    # Normalize by spectra
    spectra = flux[goodData].mean(axis=0)
    vectors_all_triangles = vectors_all / spectra[:, None]

    # Compute transformation matrices
    flux_2_data, data_2_flux = flux_matrices(vectors_all_triangles)
    QT, R = Q_and_R_matrices(vectors_all_triangles)

    # Create CouplingMap object
    couplingMap = CouplingMap()
    couplingMap.create_from_data(
        flux_2_data, data_2_flux, QT, R, center_all,
        spectra, wavelength_bin
    )

    # Set up output
    header = datalist[-1].header.copy()
    folder = datalist[-1].dirname
    output_dir = os.path.join(folder, "../couplingmaps")

    # Get modulation HDU if available
    modulation_hdu = None
    try:
        modulation_hdu = fits.open(datalist[-1].filename)['MODULATION']
    except (KeyError, FileNotFoundError):
        pass

    # Save coupling map
    couplingMap = save_coupling_map(
        couplingMap, header, output_dir, flatMap, waveMap,
        wavelength_smooth, wavelength_bin, Nsingular,
        center_data, use_pyramids, flux_threshold,
        filenames, modulation_hdu
    )

    # Create diagnostic plots
    figures = create_diagnostic_plots(
        flux, data_svdfiltered, flux_2_data, flux_goodData,
        fit_goodData, errors, center_all, QT, R, flatMap
    )

    # Save plots
    runlib_plots.save_pdf_in_file(couplingMap.filename)

    print("----------------------------------------------------")
    print("Coupling Map stored. You can quit by ctrl+C")
    print("----------------------------------------------------")

    return couplingMap, datalist


if __name__ == "__main__":
    """
    Run coupling map creation with development defaults.
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
        wavelength_smooth = 1
        wavelength_bin = 1
        Nsingular = 19*6
        modID = None
        modScale = None
        wollaston = None
        use_pyramids = False
        center_data = False

        file_patterns = ["/Users/slacour/DATA/LANTERNE/tmp/firstpl_13:0*.fits"]
        file_patterns = ["/Users/slacour/DATA/LANTERNE/20251230/preproc/*T12?2*.fits"]
        wave_patterns = ["/Users/slacour/DATA/LANTERNE/20251231/wavemaps/"]
        # file_patterns += ["/Users/slacour/DATA/LANTERNE/20251230/preproc/*T12?1[5-9]*.fits"]
        
    print(f"Development override: wavelength_smooth={wavelength_smooth}, wavelength_bin={wavelength_bin}, Nsingular={Nsingular}")
    print(f"Development file patterns: {file_patterns}")


    # Process coupling map data
    couplingMap,datalist = run_createCouplingMap(
        file_patterns=file_patterns,
        object_name=object_name,
        dark_patterns=dark_patterns,
        flat_patterns=flat_patterns,
        wave_patterns=wave_patterns,
        wavelength_smooth=wavelength_smooth,
        wavelength_bin=wavelength_bin,
        Nsingular=Nsingular,
        modID=modID,
        modScale=modScale,
        wollaston=wollaston,
        use_pyramids=use_pyramids,
        center_data=center_data
    )

    print(f"Coupling map created successfully: {couplingMap.filename}")
    print(f"Number of triangles/pyramids: {couplingMap.QT.shape[0]}")
    print(f"QT shape: {couplingMap.QT.shape}")
    print(f"R shape: {couplingMap.R.shape}")

    couplingMap_2 = CouplingMap()
    couplingMap_2.load(couplingMap.filename)
    print(f"Coupling map loaded successfully: {couplingMap_2.filename}")

    QT = couplingMap.QT
    R = couplingMap.R
    position = couplingMap.position
    Ntriangles = QT.shape[0]
    Nqr = QT.shape[2]

    datacube=np.concatenate([d.data for d in datalist])
    # datacube[np.isnan(datacube)] = 0

    datacube_T=datacube.transpose((3,2,0,1))
    # datacube_T=data_svdfiltered.transpose((3,2,0,1))
    Nwave, Noutput, Ncube, Nmod = datacube_T.shape

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

    chi2_argmin = np.nanargmin(chi2_map,axis=0)
    plot(position[chi2_argmin])

    QTdata = np.zeros((Nwave,Nqr,Ncube * Nmod))
    ZXY = np.zeros((Nwave,Nqr,Ncube * Nmod))
    for i in tqdm(range(Ncube * Nmod), desc="Projection onto QT space"):
        t = chi2_argmin[i]
        data = datacube_T[:,:,i]
        QTdata[:,:,i] = (QT[t] @ data[:,:,None])[:,:,0]
        for w in range(Nwave):
            ZXY[w,:,i] = solve_triangular(R[t,w], QTdata[w,:,i], lower=False)


    # for i in tqdm(range(Ncube * Nmod), desc="Computing XY positions"):
    #     t = chi2_argmin[i]
    #     solve_triangular(R, b, lower=False)
    # XY = 


# %%
