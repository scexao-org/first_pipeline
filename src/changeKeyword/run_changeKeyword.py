#! /usr/bin/env python3
# -*- coding: iso-8859-15 -*-
"""
FIRST Pipeline - FITS Header Keyword Management Core Functions

Core algorithms for modifying FITS header keywords for the FIRST Visible 
Photonic Lantern pipeline classification and processing control.

Created on Wed May 21 22:56:25 2025
@author: slacour
"""

import sys
import os
# Add src directory to path for imports to work in both interactive and package contexts
if os.path.join(os.path.dirname(__file__), '..') not in sys.path:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from astropy.io import fits
from glob import glob
import getpass
from first_pipeline_shared.libraries import runPL_library_io as runlib


def run_changeKeyword(files=None, header_updates=None, extract_date_from_filename=None):
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
    
    # Get development default