#! /usr/bin/env python3
# -*- coding: iso-8859-15 -*-
#%%
"""
Created on Sun May 24 22:56:25 2015

@author: slacour
"""

import os
from astropy.io import fits
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import re


# def associate_dark(filelist, filelist_dark):
#     """
#     Filters the input file list to separate coupling map files and dark files based on FITS keywords.
#     Raises an error if no valid files are found.
#     Returns a dictionary mapping coupling map files to their closest dark files.
#     """

#     # raise an error if filelist is empty
#     if len(filelist) == 0:
#         raise FileNotFoundError("No good files to process. Please check the file patterns and keywords.")
#     # raise an error if filelist_dark is empty
#     if len(filelist_dark) == 0:
#         print("WARNING: No good dark to substract to data files")

#     # Check if all files have the same value for header['PM_CHECK']
#     pm_check_values = set()
#     combined_filelist = []
#     combined_filelist.extend(filelist_dark)
#     combined_filelist.extend(filelist)
#     for file in combined_filelist:
#         header = fits.getheader(file)
#         pm_check_values.add(header.get('PM_CHECK', 0))
        
#     if len(pm_check_values) > 1:
#         print("WARNING: The 'PM_CHECK' values (ie, the pixel map used to preprocess the files) \n are not consistent across all files!")
#         print(f"Found values: {pm_check_values}")

#     # for each file in filelist_cmap find the closest dark file in filelist_dark with, by priority, first the directory in which the file is, and then by the date in the "DATE" fits keyword, and second, the directory in which the file is

#     filelist = np.sort(np.unique(filelist))
#     files_with_dark = {file: find_closest_dark(file, filelist_dark) for file in filelist}

#     return files_with_dark


# def associate_pixelmap(filelist , filelist_pixelmap):

#     # raise an error if filelist is empty
#     if len(filelist) == 0:
#         raise FileNotFoundError("No good files to process. Please check the file patterns and keywords.")
#     # raise an error if filelist_dark is empty
#     if len(filelist_pixelmap) == 0:
#         raise FileNotFoundError("No good pixelmap to process the files. Please check the file patterns and keywords.")

#     filelist = np.sort(np.unique(filelist))
#     files_with_pixelmap = {file: find_closest_pixelmap(file, filelist_pixelmap) for file in filelist}
#     # Remove entries where no pixel map was found
#     files_with_pixelmap = {file: pixelmap for file, pixelmap in files_with_pixelmap.items() if pixelmap is not None}

#     if len(files_with_pixelmap) == 0:
#         raise FileNotFoundError("The pixel map does not have the good wollaston configuration.")
    
#     if len(files_with_pixelmap) != len(filelist):
#         print("WARNING: Some files do not have a corresponding pixel map.")
#         for file in filelist:
#             if file not in files_with_pixelmap:
#                 print(f"No pixel map found for {file}")

#     return files_with_pixelmap


def make_figure_of_trace(raw_image,traces_loc,pixel_wide,pixel_min,pixel_max):
    output_channels=traces_loc.shape[1]
    fig=plt.figure("Extract fitted traces",clear=True,figsize=(18,10))
    v1,v2=np.percentile(raw_image.ravel(),[1,99])
    plt.imshow(raw_image,aspect="auto",interpolation='none',vmin=v1,vmax=v2)
    plt.colorbar()
    for i in range(output_channels): 
            plt.plot(traces_loc[:,i],'r',linewidth=1)
            plt.plot(traces_loc[:,i]+pixel_wide,'g',linewidth=0.3)
            plt.plot(traces_loc[:,i]-pixel_wide,'g',linewidth=0.3)
    plt.plot(np.ones(2)*pixel_min,[0,raw_image.shape[0]],'w')
    plt.plot(np.ones(2)*pixel_max,[0,raw_image.shape[0]],'w')
    plt.xlim(0, raw_image.shape[1])
    plt.ylim(0, raw_image.shape[0])
    plt.tight_layout()
    plt.xlabel("Wavelength")
    ax = plt.gca()
    return fig, ax


def update_header_date(filelist):
    for file in filelist:
        date = get_date_from_filename(file)
        update_anything_in_fits(file, 'DATE', date)
    print("Date updated in all files")


def create_output_filename(header):
    date = header.get('DATE', 'NODATE')
    object = header.get('OBJECT', "NONAME")
    type = header.get('DATA-TYP',None)
    cat = header.get('X_FIRTYP',None)

    name_extension = object
    data_type_extension = ["DARK", "WAVE", "WAVELENGTHMAP", "SKY"]

    processing_extension = {"PIXELMAP": "PM", "COUPLINGMAP": "CM", "WAVELENGTHMAP": "WM", "PREPROC": "P", "SPECTRA": "S", "IMAGE": "I", "ASTROMETRY": "A"}

    if type in data_type_extension:
        name_extension = type
    if cat in processing_extension.keys():
        name_extension = name_extension + "_" + processing_extension[cat]

    output_filename = 'firstpl_' + date + '_' + name_extension + '.fits'
    return output_filename

def get_date_from_filename(filename):
    match = re.search(r"(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})", filename)
    match2 = re.search(r"(\d{4}-\d{2}-\d{2})T(\d{2}:\d{2}:\d{2})", filename)
    if match:
        # Extract date and time parts
        date_part = match.group(1)
        time_part = match.group(2).replace('-', ':')  # Replace '-' with ':' for time
        return f"{date_part}T{time_part}"
    elif match2:
        return f"{match.group(1)}T{match.group(2)}"
    else:
        return None  # Return None if no match is found
    
def latest_file(filelist):

    if filelist==[]:
        return None  # Return None if no valid files are found
    
    # Find the file with the most recent creation time
    last_created_file = max(filelist, key=os.path.getctime)
    
    return last_created_file

def get_fits_date(fits_file):
    try:
        with fits.open(fits_file) as hdul:
            header = hdul[0].header
            date_str = header.get('DATE', None)
            
            if date_str:
                # Parse the DATE value
                file_date = datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%S")
    except Exception as e:
        print(f"Error reading file {fits_file}: {e}")
    return file_date

def get_latest_date_fits(fits_files):
    """
    Finds the FITS file with the latest 'DATE' value in its header.

    Parameters:
        fits_files (list): A list of paths to FITS files.

    Returns:
        str: The path to the FITS file with the latest 'DATE' value.
        None: If no valid 'DATE' values are found in the files.
    """
    latest_file = None
    latest_date = None
    
    for file in fits_files:
        try:
            with fits.open(file) as hdul:
                header = hdul[0].header
                date_str = header.get('DATE', None)
                
                if date_str:
                    # Parse the DATE value
                    file_date = datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%S")
                    
                    # Update the latest file if this file has a newer date
                    if latest_date is None or file_date > latest_date:
                        latest_date = file_date
                        latest_file = file
        except Exception as e:
            print(f"Error reading file {file}: {e}")
    
    return latest_file

def update_anything_in_fits(file_path, header, header_value):
    # Update the chosen keyword 
    with fits.open(file_path, mode='update') as hdul:
        hdr = hdul[0].header
        hdr[header] = header_value
        hdul.flush()
        print(f"Updated {header} in {file_path} to {header_value}")

def update_anything_in_multiple_fits(folder_path, header, header_value):
    """
    Updates the chosen value in the header of all .fits files in a specified folder.

    Parameters:
        folder_path (str): Path to the folder containing .fits files.
    """
    # Iterate over all files in the folder
    for file_name in os.listdir(folder_path):
        # Check if the file has a .fits extension
        if file_name.lower().endswith(".fits"):
            file_path = os.path.join(folder_path, file_name)
            
            try:
                # Open the FITS file
                with fits.open(file_path, mode='update') as hdul:
                    update_anything_in_fits(file_path, header, header_value)
            except Exception as e:
                print(f"Failed to update {file_name}: {e}")


def save_fits_file(data, filepath, headerDict=None):
    fits.writeto(filepath, np.array(data), overwrite=True)

    hdu = fits.PrimaryHDU(data)
    if headerDict is not None:
        for val in headerDict:
            hdu.header[val]= headerDict[val]
    hdul = fits.HDUList([hdu])
    hdul.writeto(filepath, overwrite=True)
    print("Fit file saved in : ", filepath)



