#! /usr/bin/env python3
# -*- coding: iso-8859-15 -*-
"""
FIRST Pipeline - FITS Header Keyword Management Core Functions

Core algorithms for modifying FITS header keywords for the FIRST Visible 
Photonic Lantern pipeline classification and processing control.

Created on Wed May 21 22:56:25 2025
@author: slacour
"""

import os
from astropy.io import fits
from glob import glob
import getpass
from first_pipeline_shared.libraries import runPL_library_io as runlib


def get_development_defaults():
    """
    Get development environment default parameters.
    
    Returns
    -------
    dict
        Dictionary containing default parameters for development environment
    """
    defaults = {
        'files': ['*.fits'],
        'header_updates': {},
        'extract_date_from_filename': False
    }
    
    # Check if running in development environment
    if ("VSCODE_PID" in os.environ or os.environ.get('TERM_PROGRAM') == 'vscode' or 
        os.environ.get('SPYDER_DEBUG_FILE')):
        
        user = getpass.getuser()
        if user == "slacour":
            defaults['files'] = ["/Users/slacour/DATA/LANTERNE/tmp/*.fits"]
        elif user == "jsarrazin":
            defaults['files'] = ["/home/jsarrazin/Bureau/PLDATA/novembre/*.fits"]
        elif user == "ehuby":
            defaults['files'] = ["/home/ehuby/WORK/DATA/FIRST-PL/2025-05-10/*.fits"]
    
    return defaults


def update_fits_headers(files=None, header_updates=None, extract_date_from_filename=None):
    """
    Update FITS headers based on provided options.
    
    Parameters:
    -----------
    files : list of str, optional
        List of FITS filenames to process. If None, uses development defaults.
    header_updates : dict, optional
        Dictionary of keyword-value pairs to update. If None, uses development defaults.
    extract_date_from_filename : bool, optional
        Whether to extract DATE from filename. If None, uses development defaults.
        
    Returns:
    --------
    list of str
        List of processing messages for each file
    """
    # Use development defaults if parameters are None
    if any(param is None for param in [files, header_updates, extract_date_from_filename]):
        defaults = get_development_defaults()
        if files is None:
            files = defaults['files']
        if header_updates is None:
            header_updates = defaults['header_updates']
        if extract_date_from_filename is None:
            extract_date_from_filename = defaults['extract_date_from_filename']
    
    messages = []
    
    for filename in files:
        updates = header_updates.copy()
        if extract_date_from_filename:
            updates['DATE'] = runlib.get_date_from_filename(filename)
            
        string_print = filename + "   ----->"
        with fits.open(filename, mode='update') as filehandle:
            for key, value in updates.items():
                if value is not None:
                    filehandle[0].header[key] = value
                    string_print += f'   {key}={value}'
        messages.append(string_print)
        
    return messages


def collect_files(file_patterns):
    """
    Collect FITS files based on file patterns or current directory.
    
    Parameters:
    -----------
    file_patterns : list of str
        List of file patterns/wildcards
        
    Returns:
    --------
    list of str
        Sorted list of FITS filenames
    """
    filelist = []
    
    # If the user specifies file names or wildcards
    if len(file_patterns) > 0:
        for f in file_patterns:
            filelist += [file for file in glob(f) if file.endswith(".fits")]
    # Processing of the full current directory
    else:
        for file in os.listdir("."):
            if file.endswith(".fits"):
                filelist.append(file)
                
    return sorted(filelist)


if __name__ == "__main__":
    """
    Run FITS header keyword updates with development defaults.
    Perfect for testing and direct execution of core functionality.
    """
    print("Running changeKeyword core with development defaults...")
    
    # Get development defaults
    defaults = get_development_defaults()
    
    # Collect files using the existing collect_files function
    files = collect_files(defaults['files'])
    print(f"Found {len(files)} FITS files")
    
    if files:
        # Example header updates
        header_updates = {'X_FIRTYP': 'RAW', 'DATA-TYP': 'OBJECT'}
        
        # Update headers
        messages = update_fits_headers(
            files=files[:5],  # Process first 5 files only for safety
            header_updates=header_updates,
            extract_date_from_filename=defaults['extract_date_from_filename']
        )
        
        print("Processing complete:")
        for msg in messages:
            print(f"  {msg}")
    else:
        print("No FITS files found with default patterns")
        print(f"Searched patterns: {defaults['files']}")
