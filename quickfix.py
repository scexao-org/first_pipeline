import os
import sys
from astropy.io import fits
from glob import glob

from runPL_createPixelMap import run_createPixelMap
from runPL_preprocess import preprocess
import runPL_library_io as runlib
import numpy as np
import matplotlib
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt

def update_date_in_fits(folder_path):
    """
    Updates the DATE value in the header of all .fits files based on the date in their file names.

    Parameters:
        folder_path (str): Path to the folder containing .fits files.
    """
    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith(".fits"):
            # Extract the date and time from the file name
            try:
                # File name format: im_cube_2024-08-18_11-01-23_altair.fits
                parts = file_name.split("_")
                date_part = parts[2]  # "2024-08-18"
                time_part = parts[3]  # "11-01-23"
                time_part = time_part.split(".")[0]  # Remove ".fits" or extra extensions
                
                # Format as '2024-08-18T11:01:23'
                formatted_date = f"{date_part}T{time_part.replace('-', ':')}"
                
                # Full file path
                file_path = os.path.join(folder_path, file_name)
                
                # Update the DATE keyword in the header
                with fits.open(file_path, mode='update') as hdul:
                    hdr = hdul[0].header
                    hdr['DATE'] = formatted_date
                    hdul.flush()
                    print(f"Updated DATE in {file_name} to {formatted_date}")
            
            except Exception as e:
                print(f"Failed to update DATE in {file_name}: {e}")



def update_header_value(file, which_header, what_value):
    """
    Updates the GAIN value in the header of all .fits files in a specified folder.

    Parameters:
        folder_path (str): Path to the folder containing .fits files.
    """
    # Check if the file has a .fits extensio
    try:
        # Open the FITS file
        with fits.open(file, mode='update') as hdul:
            # Access the primary header
            hdr = hdul[0].header
            
            # Update the GAIN value
            hdr[which_header] = what_value
            
            # Save the changes
            hdul.flush()
            print(f"Updated {which_header} to {what_value} in {file}")
    except Exception as e:
        print(f"Failed to update {file}: {e}")

update_header_value("/home/jsarrazin/Bureau/PLDATA/InitData/Neon4/firstpl_00:57:11.727002706.fits", "DATA-CAT", "RAW")
update_header_value("/home/jsarrazin/Bureau/PLDATA/InitData/Neon4/firstpl_00:59:38.528185978.fits", "DATA-CAT", "RAW")
update_header_value("/home/jsarrazin/Bureau/PLDATA/InitData/Neon4/firstpl_00:57:11.727002706.fits", "DATA-TYP", "WAVE")
update_header_value("/home/jsarrazin/Bureau/PLDATA/InitData/Neon4/firstpl_00:59:38.528185978.fits", "DATA-TYP", "DARK")


'''
    fits_keywords = {'DATA-CAT': ['PREPROC'], 
                    'DATA-TYP': ['DARK']}

    fits_keywords = {'DATA-CAT': ['PREPROC'], 
                    'DATA-TYP': ['WAVE']}

        fits_keywords = {'DATA-CAT': ['PREPROC'], 
                        'DATA-TYP': ['OBJECT']}


'''
def compare_neon_calibration():
    fits_keywords = {'DATA-CAT': ['RAW'], 
                        'DATA-TYP': ['WAVE']}

    all_fits = runlib.get_all_fits_files("/home/jsarrazin/Bureau/PLDATA/InitData")
    all_raw_neon = runlib.clean_filelist(fits_keywords, all_fits)
    latest_neon = runlib.get_latest_date_fits(all_raw_neon)
    print(latest_neon)
    all_neon_path = [os.path.dirname(neon) for neon in all_raw_neon]

    dict_date = {
    "Neon1": "07/2024",
    "Neon2": "09/2024",
    "Neon3": "07/2023",
    "Neon4": "12/2024"
    }

    plt.figure()
    mega_neon = np.zeros(1500)
    number_of_files = 0
    for neon in all_neon_path:
        if os.path.basename(neon)=="Neon5" :
            continue
        filename = os.path.join(neon, "calibration_result","pixels_to_wavelength.txt")
        data = np.loadtxt(filename, skiprows=1) 
        pixels = data[:, 0][:1500]       # First column: Pixels
        wavelengths = data[:, 1][:1500]  # Second column: Wavelengths
        mega_neon=np.add(mega_neon, wavelengths)
        number_of_files +=1
        plt.plot(pixels, wavelengths, label=dict_date[os.path.basename(neon)])
    plt.title("Pixels to wavelenght for all Neon")
    plt.xlabel("Pixels")
    plt.ylabel("Wavelength (nm)")
    plt.legend()
    plt.show()

    plt.figure()
    mega_neon=mega_neon/number_of_files
    for neon in all_neon_path:
        if os.path.basename(neon)=="Neon5" :
            continue
        filename = os.path.join(neon, "calibration_result","pixels_to_wavelength.txt")
        data = np.loadtxt(filename, skiprows=1) 
        pixels = data[:, 0][:1500]       # First column: Pixels
        wavelengths = (data[:, 1][:1500]-mega_neon)/mega_neon  # Second column: Wavelengths
        wavelengths = wavelengths*100
        plt.plot(pixels, wavelengths, label=dict_date[os.path.basename(neon)])
    plt.title("Relative deviation of pixels to wavelength for all Neon")
    plt.xlabel("Pixels")
    plt.ylabel("Relative Deviation (%)")
    plt.legend()
    plt.show()
    print("buffer")

def most_recent_stars():
    fits_keywords = {'DATA-CAT': ['RAW'], 
                        'DATA-TYP': ['OBJECT']}

    all_fits = runlib.get_all_fits_files("/home/jsarrazin/Bureau/PLDATA/InitData")
    all_raw_star = runlib.clean_filelist(fits_keywords, all_fits)
    latest_star = runlib.get_latest_date_fits(all_raw_star)
    print(latest_star)

if __name__ == "__main__":
    recent = "/home/jsarrazin/Bureau/PLDATA/2025_03_14"

    filelist = runlib.get_all_fits_files(recent)
    for file in filelist:
        if "optim" not in file:
            runlib.update_anything_in_fits(file, "DATA-CAT", "RAW")

            if "cube" in file:
                runlib.update_anything_in_fits(file, "DATA-TYP", "OBJECT")
            elif "dark" in file:
                runlib.update_anything_in_fits(file, "DATA-TYP", "DARK")