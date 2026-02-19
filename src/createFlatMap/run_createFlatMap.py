#%%

"""
FIRST Pipeline - Flat Field Map Generation Core Algorithms

Core algorithms for creating flat field maps from SuperK calibration data. 
Contains the main processing functions separated from CLI interface for 
interactive use and modularity.

Created on Wed May 21 22:56:25 2025
@author: slacour
"""

import os
import getpass
import matplotlib
if "VSCODE_PID" in os.environ:
    matplotlib.use('Qt5Agg')
else:
    matplotlib.use('Agg')

import numpy as np
from typing import List
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, xlim  
from datetime import datetime
from astropy.io import fits

from first_pipeline_shared.classes.runPL_class_flatMap import FlatMap
from first_pipeline_shared.classes.runPL_class_fileList import FileList
from first_pipeline_shared.classes.runPL_class_dataCube import DataCube
from first_pipeline_shared.libraries import runPL_library_io as runlib_io
from first_pipeline_shared.libraries import runPL_library_plots as runlib_plots


def get_filelist_wave(flat_patterns, dark_patterns, wollaston):
    """
    Get file lists for flat field and dark files with appropriate filtering.
    
    Parameters
    ----------
    flat_patterns : str or list
        File patterns for flat field files
    dark_patterns : str or list
        File patterns for dark files  
    wollaston : str, optional
        Wollaston status filter ('IN' or 'OUT')
        
    Returns
    -------
    tuple
        (flat_files, dark_files) - lists of matching FITS files
    """
    fits_keywords = {
        'X_FIRTYP': ['PREPROC'],
        'DATA-TYP': ['FLAT'],
    }    
    
    # Adding other constraints if asked by user
    if wollaston is not None:
        fits_keywords['X_FIRWOL'] = [wollaston]
    
    print(flat_patterns)
    filelist = runlib_io.get_filelist(flat_patterns, fits_keywords)

    # Adding new constraints if not asked by user
    hd = fits.getheader(filelist[0])
    wollaston = hd.get('X_FIRWOL', None)
    if wollaston is not None:
        fits_keywords['X_FIRWOL'] = [wollaston]

    print("----------------")
    print(f"Selected wollaston={wollaston}")

    filelist = runlib_io.get_filelist(flat_patterns, fits_keywords)

    print(f"Found {len(filelist)} files matching criteria.")
    print("----------------")

    # Finding dark files
    darks_keywords = {
        'X_FIRTYP': ['PREPROC'],
        'DATA-TYP': ['DARK'],
    }
    
    if wollaston is not None:
        darks_keywords['X_FIRWOL'] = [wollaston]
    
    darks_with_flats = runlib_io.get_filelist(dark_patterns, darks_keywords)
    print(f"Found {len(darks_with_flats)} dark files.")

    return filelist, darks_with_flats


def compute_flat(datalist, Nflat_smooth=25):
    """
    Compute flat field from a list of DataCube objects.
    
    This function calculates the flat field by fitting gain coefficients across
    spectral channels, using smoothing to reduce noise and improve calibration accuracy.
    
    Parameters
    ----------
    datalist : List[DataCube]
        List of DataCube objects containing flat field data
    Nflat_smooth : int, optional
        Smoothing parameter for flat field computation (default: 25)
        
    Returns
    -------
    tuple
        (flat_full, flat_individual) where:
        - flat_full: Combined flat field array
        - flat_individual: Individual file flat fields
    """
    
    flats = np.array([np.nansum(d.data, axis=(0,1)) for d in datalist])    
    valid_mask = ~np.isnan(flats[:,0,0])
    flats = flats[valid_mask]

    # Create smoothing kernel
    flats_smooth = np.zeros_like(flats)
    window = np.hanning(Nflat_smooth)
    window[Nflat_smooth//2] = 0.0
    window /= window.sum()
    conv_ref = np.convolve(np.ones(len(flats[0,0])), window, mode='same')
    
    # Apply smoothing to each file and output
    for f in range(flats_smooth.shape[0]):
        for o in range(flats_smooth.shape[1]):
            flats_smooth[f, o, :] = np.convolve(flats[f, o, :], window, mode='same') / conv_ref

    # Calculate individual and combined flat fields
    flat_individual = flats / flats_smooth
    flat_full = flats.sum(axis=0) / flats_smooth.sum(axis=0)

    return flat_full, flat_individual


def create_diagnostic_plots(flat_full, flat_individual, datalist):
    """
    Create diagnostic plots for flat field quality assessment.
    
    Parameters
    ----------
    flat_full : numpy.ndarray
        Combined flat field array
    flat_individual : numpy.ndarray
        Individual file flat fields
    datalist : List[DataCube]
        List of DataCube objects used for flat field computation
    """
    
    # Plot combined flat field
    fig = figure("Flat Field Computation", clear=True, figsize=(18,6))
    plt.plot(flat_full.T + np.arange(flat_full.shape[0]) * 0.1)
    xlim((0, len(flat_full.T)))
    plt.ylim((0.85, 1.15 + len(flat_full) * 0.1))
    plt.title(fig.get_label())
    plt.xlabel("x pixel")
    plt.ylabel("gain _ output * 10%")
    plt.tight_layout()
    
    # Plot individual flat fields for each output
    Noutput = len(flat_full)
    for o in range(Noutput):
        fig = figure(f"Flat values for output {o}", clear=True, figsize=(18,6))
        flat_output = flat_individual[:,o] / flat_full[o]
        plt.plot(flat_output.T + np.arange(flat_output.shape[0]) * 0.05)
        xlim((0, len(flat_output.T)))
        plt.ylim((0.85, 1.15 + len(flat_output) * 0.05))
        plt.title(fig.get_label())
        plt.xlabel("x pixel")
        plt.ylabel("gain -<mean gain> _ file number * 5%")
        plt.tight_layout()


def create_flat_comparison_plots(datalist, flatMap_loaded, output_filename):
    """
    Create comparison plots showing data before and after flat field correction.
    
    Parameters
    ----------
    datalist : List[DataCube]
        Original data list  
    flatMap_loaded : FlatMap
        Loaded flat field map for correction
    output_filename : str
        Output filename for saving plots
    """
    
    # Create FileList for comparison
    fileList = FileList([d.filename for d in datalist[:3]])  # Limit to first 3 for clarity
    
    # Extract data without and with flat correction
    datalist_1 = fileList.extract_data_from_list(flatMap=None, center=False)
    datalist_2 = fileList.extract_data_from_list(flatMap=flatMap_loaded, center=False)

    data_noflat = np.array([np.nanmean(d.data, axis=(0,1)) for d in datalist_1])
    data_withflat = np.array([np.nanmean(d.data, axis=(0,1)) for d in datalist_2])

    for i in range(len(data_noflat)):
        d = datalist_1[0]
        fig = plt.figure(figsize=(18,10), clear=True)
        
        # Plot without flat correction
        plt.subplot(2, 2, 1)
        vmin, vmax = np.percentile(data_noflat[i], (5,95))
        im0 = plt.imshow(data_noflat[i], origin='lower', aspect='auto', vmin=vmin, vmax=vmax, 
                        interpolation='none', rasterized=True)
        plt.title(f'Without flat - File {i}')
        plt.colorbar(im0)
        
        # Plot with flat correction
        plt.subplot(2, 2, 2)
        im1 = plt.imshow(data_withflat[i], origin='lower', aspect='auto', vmin=vmin, vmax=vmax, 
                        interpolation='none', rasterized=True)
        plt.title(f'With flat - File {i}')
        plt.colorbar(im1)
        
        # Plot spectral comparison
        plt.subplot(2, 1, 2)
        plt.plot(data_withflat[i].T, 'k')
        plt.plot(data_noflat[i].T)
        plt.suptitle(f'Flat correction comparison for file {d.basename}')
        plt.tight_layout()

    # Save plots to PDF
    runlib_plots.save_pdf_in_file(output_filename)


def run_createFlatMap(file_patterns=None, dark_patterns=None, wollaston=None, 
                           Nflat_smooth=25, override_flat_keyword=False):
    """
    Complete processing pipeline for flat field map generation.
    
    Parameters
    ----------
    file_patterns : list, optional
        File patterns for input data. If None, uses development defaults.
    dark_patterns : list, optional
        Dark file patterns (default: same as file_patterns)
    wollaston : str, optional
        Wollaston status filter
    Nflat_smooth : int, optional
        Smoothing parameter for flat computation
    override_flat_keyword : bool, optional
        Override DATA-TYP=FLAT requirement
        
    Returns
    -------
    str
        Path to saved flat field map file
    """
    
    # Set default dark patterns
    if dark_patterns is None:
        dark_patterns = file_patterns

    # Set data_type based on override flag
    data_type = None if override_flat_keyword else 'FLAT'
    if override_flat_keyword:
        print("WARNING: Overriding FLAT keyword requirement. Processing files without DATA-TYP=FLAT constraint.")
    
    # Create file list and extract data
    fileList = FileList(file_patterns, data_type=data_type, first_type='PREPROC', wollaston=wollaston)
    fileList.make_association(darks_pattern=dark_patterns)

    datalist = fileList.extract_data_from_list(center=False)

    # Compute flat field
    flat_full, flat_individual = compute_flat(datalist, Nflat_smooth)

    # Create diagnostic plots
    create_diagnostic_plots(flat_full, flat_individual, datalist)

    # Save flat field map
    flatMap = FlatMap()
    flatMap.create_from_data(1/flat_full)  # Inverse for correction

    # Create header with metadata
    header = datalist[-1].header.copy()
    header['X_FIRTYP'] = 'FLATMAP'

    # Add processing parameters to header
    header['Q_FMNFSM'] = (Nflat_smooth, 'flat smoothing parameter')
    header['Q_FM_CK'] = (np.random.randint(0, 2**32, dtype=np.uint32), 'checksum')

    # Set output directory and filename
    folder = fileList.get_most_common_dir()
    output_dir = os.path.join(folder, "../flatmaps")

    # Add input filenames to header
    filenames = [d.filename for d in datalist]
    for i, filename in enumerate(filenames):
        header[f'Q_FM_F{i}'] = (filename, 'filename of the extracted flux')

    header['Q_FMNAME'] = (runlib_io.create_basename(header), 'name of the flatmap file')

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    output_filename = os.path.join(output_dir, header['Q_FMNAME'])

    # Save flat field map
    flatMap.save(output_filename, header)

    # Create flat field comparison plots
    flatMap_loaded = FlatMap(output_filename)
    # create_flat_comparison_plots(datalist, flatMap_loaded, output_filename)

    print(f"Flat field map saved to: {output_filename}")
    
    return output_filename


if __name__ == "__main__":
    """
    Run flat field map creation with development defaults.
    Perfect for testing and direct execution of core functionality.
    """


    if getpass.getuser() == "slacour":
        dark_patterns = None
        flat_patterns = None
        wollaston = None
        Nflat_smooth = 25
        override_flat_keyword = True
        file_patterns = ["/Users/slacour/DATA/LANTERNE/raw/20260114/preproc/"]
        
        print(f"Development override: dark_patterns={dark_patterns}, flat_patterns={flat_patterns}, wollaston={wollaston}, Nflat_smooth={Nflat_smooth}")
        print(f"Development file patterns: {file_patterns}")

    # Process flat field data
    output_filename = run_createFlatMap(
        file_patterns=file_patterns,
        dark_patterns=dark_patterns,
        wollaston=wollaston,
        Nflat_smooth=Nflat_smooth,
        override_flat_keyword=override_flat_keyword
    )
    
    print(f"Successfully created flat field map: {output_filename}")
    
# %%
