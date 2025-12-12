#! /usr/bin/env python3
# -*- coding: iso-8859-15 -*-
#%%
"""
Created on Sun May 24 22:56:25 2015

@author: slacour
"""

import os
import sys
from astropy.io import fits
from glob import glob
import argparse
import numpy as np
import peakutils

import getpass
import matplotlib
if "VSCODE_PID" in os.environ:
    matplotlib.use('Qt5Agg')
else:
    matplotlib.use('Agg')

from classes.runPL_class_fileList import FileList     
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot,hist,clf,figure,legend,imshow
from datetime import datetime
from tqdm import tqdm
import libraries.runPL_library_io as runlib_io
import shutil
from scipy.signal import find_peaks


# plt.ion()

# Add options
usage = """
Usage: %prog [options] [file_patterns]

Goal: Create the pixel map needed to preprocess the data.

Arguments:
    file_patterns       One or more glob patterns for FITS files (default: *.fits)

Input:
    - Files of type X_FIRTYP=RAW in the directory.

Output:
    - A FITS file with the pixel map.
    - A PNG file with the pixel map.
    
Options:
    --pixel_min         Minimum pixel value along wavelength axis (default: 100)
    --pixel_max         Maximum pixel value along wavelength axis (default: 2100)
    --pixel_wide        Window half width (default: 2) (full width = 2*pixel_wide+1)
    --filter_files      Flag to filter out files that don't have enough flux. Can be long, recommended only if previous run failed.

Note:
    - Output channels are automatically determined based on wollaston status (38 for 'IN', 19 for 'OUT')
    - Files are processed separately by wollaston status

Examples:
    runPL_createPixelMap.py --pixel_min=100 --pixel_max=2100 --pixel_wide=2 --filter_files *.fits
    runPL_createPixelMap.py --pixel_min=50 --pixel_max=1500 data/*.fits

"""


def process_files(folder=".", file_patterns=["**/*.fits"]):
    """
    Processes files based on the given parameters.

    Args:
        pixel_min (int): Minimum pixel value.
        pixel_max (int): Maximum pixel value.
        pixel_wide (int): Pixel width.
        output_channels (int): Number of output channels.
        file_patterns (list): List of file patterns to process (e.g., ["*.fits"]).
    
    Returns:
        list: A list of files to process.
    """
    filelist = []
    if folder.endswith("*fits"):
        folder = folder[:-5]

    # If file patterns are provided, use glob to find matching files
    for pattern in file_patterns:
        filelist += glob(os.path.join(folder, pattern))

    
    # Sort the file list for consistent processing order
    filelist.sort()
    return filelist


def raw_image_clean(filelist):
        '''
        Process all raw files and sum them into one image
        By summing all cubes into one picture 
        '''

        # raise an error if filelist_cleaned is empty
        if len(filelist) == 0:
            raise FileNotFoundError("No good file to process")

        header = fits.getheader(filelist[-1])

        raw_image = np.zeros((header['NAXIS2'], header['NAXIS1']), dtype=np.double)
        print("Processing files: ", filelist)
        for filename in tqdm(filelist, desc="Co-adding files"):
            if not "optim" in filename:
                try:
                    data_summed = fits.getdata(filename).sum(axis=0)
                    raw_image += data_summed
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
        
        return raw_image, header

def quick_fits(data, title=""):
    #For debugging purpose
    now = datetime.now()
    date_time_str = now.strftime("%Y_%m_%d_%H_%M_%S")
    runlib_io.save_fits_file(data, "/home/jsarrazin/Bureau/test zone/coupling_maps/"+title+"_"+date_time_str+".fits")
    print("check")

def loop_lowering_my_treshold( sampling, peaks_number, raw_image, peaks, output_channels, filelist, start = 0.01, stop = 0.1, num = 50):
    # if instance==1:
    #     print("Cant find flux at the moment.... Running additional tests on your files.")
    #     filter_only_good_files(filelist)
    threshold_array = np.linspace(start, stop, num)
    solution_found=[]
    for i in (range(sampling.shape[0])): #from 0 to the number of samples
        #Sum 10 values of x (wavelenght=columns) of the pic
        sum_image = raw_image[:,sampling[i]-25:sampling[i]+25].sum(axis=1)
        detectedWavePeaks=np.zeros(output_channels)
        found = False
        #Search for the 38 modes expected
        for t in threshold_array:
            detectedWavePeaks_tmp = peakutils.peak.indexes(sum_image,thres=t, min_dist=7)
            if len(detectedWavePeaks_tmp) == peaks_number:
                detectedWavePeaks = detectedWavePeaks_tmp
                found = True
                break
        solution_found+=[found]
        #The values will be saved at the index i of the sample
        peaks[:,i]=detectedWavePeaks

    return solution_found, peaks


def peaks_using_scipy(raw_image, sampling, Nsampling, output_channels):
    # if instance==1:
    #     print("Cant find flux at the moment.... Running additional tests on your files.")
    #     filter_only_good_files(filelist)
    solution_found=[]
    peaks_all_samples = np.zeros([output_channels, sampling.shape[0]])
    for i in (range(sampling.shape[0])): #from 0 to the number of samples
        #Sum 50 values of x (wavelenght=columns) of the pic
        sum_image = raw_image[:,sampling[i]-Nsampling:sampling[i]+Nsampling].sum(axis=1)
        #slightly smooth data along y axis
        sum_image = np.convolve(sum_image, [0.25,0.5,0.25], mode='same')
        #normalize for prominence calculation
        sum_image /= max(sum_image)

        prominence = 0.02
        peaks, props = find_peaks(sum_image/max(sum_image), prominence=prominence, distance=5)

        detectedWavePeaks=np.zeros(output_channels)
        found = False

        if len(peaks) >= output_channels:
            detectedWavePeaks = peaks[np.argsort(props['prominences'])[-output_channels:]]
            detectedWavePeaks.sort()
            found = True

        solution_found+=[found]
        #The values will be saved at the index i of the sample
        peaks_all_samples[:,i]=detectedWavePeaks

    return solution_found, peaks_all_samples


def generate_pixelmap(raw_image, pixel_min, pixel_max, output_channels, filelist):
    """
    Generate pixel map by detecting and tracing fiber peaks across wavelength range.
    
    This function analyzes a summed raw image to detect fiber peaks at different wavelength
    positions, fits polynomial traces to connect the peaks, and creates a pixel map
    for data preprocessing.
    
    :param raw_image: 2D numpy array of summed raw detector data (y-axis: spatial, x-axis: wavelength)
    :param pixel_min: Minimum pixel value along wavelength axis to start peak detection
    :param pixel_max: Maximum pixel value along wavelength axis to end peak detection  
    :param output_channels: Number of expected fiber channels/peaks to detect (38 for wollaston IN, 19 for OUT)
    :param filelist: List of input FITS files being processed (used for error reporting)
    :return: Tuple of (traces_loc, x_found, y_found, x_none, y_none) where:
             - traces_loc: 2D array mapping each wavelength pixel to spatial pixel for each channel
             - x_found: List of wavelength positions where peaks were successfully detected
             - y_found: List of spatial positions of detected peaks
             - x_none: List with None values for outlier/rejected peak positions
             - y_none: List with None values for outlier/rejected peak positions
    """
    

    pixel_length=raw_image.shape[1]

    #300 values of pixels between pixelmin and pixelmax
    Nsampling = 25
    sampling        = np.linspace(pixel_min+Nsampling,pixel_max-Nsampling,300,dtype=int)
    peaks           = np.zeros([output_channels, sampling.shape[0]])

    solution_found, peaks = peaks_using_scipy(raw_image, sampling, Nsampling, output_channels)
    
    true_count = sum(solution_found)  # because True == 1, False == 0
    percentage = true_count / len(solution_found)
    print(f"Percentage of successful detections: {percentage}")
    if percentage<0.1 : 
        raise ValueError("Too many runs, no solution found. Verify your pixelmap or run with --filter_files True.")

    traces_loc= np.ones([pixel_length,output_channels],dtype=int)

    x_found=[]
    y_found=[]
    x_none = []
    y_none = []

    #Once we've picked each detected peak, we need to verify that they all belong to the same mode,
    #and that there is no outlier
    diff_y_median = np.median(np.diff(peaks[:,solution_found],axis=0))

    for i in range(output_channels):
        # x is a list of all the pixels/wavelength at which 38 peaks were detected
        x = sampling[solution_found]
        # y the corresponding positions of each peak/mode
        y = peaks[i][solution_found]

        inliers = np.abs(y-np.median(y)) < diff_y_median * 2 / 3
        poly_coeffs = np.polyfit(x[inliers], y[inliers], 1)

        if i==11:
            print("check")

        # To check for outlier, we make a 1D polyfit between x and y
        for b in range(5): # The process is repeated 5 times to refine the polyfit each time

            # Calculate residuals of the function
            y_fit = np.polyval(poly_coeffs, x)
            residuals = y - y_fit

            # Calculate standard deviation of residuals
            std_residuals = np.std(residuals)
            # keep it above half a pixel
            if std_residuals < 1/2:
                std_residuals = 1/2

            # Identify inliers (points with residuals within the threshold)
            inliers = np.abs(residuals) < 3 * std_residuals
            inliers &= np.abs(residuals) < diff_y_median * 2 / 3
            

            # Remove outliers
            x = x[inliers]
            y = y[inliers]

            # Fit the polynomial to the cleaned data
            poly_coeffs = np.polyfit(x, y, 2)

            # Replace outliers with None
            x_with_none = [xi if inlier else None for xi, inlier in zip(x, inliers)]
            y_with_none = [yi if inlier else None for yi, inlier in zip(y, inliers)]

        # We stop considering solo pixels and consider the 1D polyfit to trace over all of them.
        traces_loc[:,i] = np.polyval(poly_coeffs, np.arange(pixel_length))+0.5
        # x is a list of all the pixels/wavelength at which 38 peaks were detected
        # y the corresponding positions of each peak/mode
        x_found += [x]
        y_found += [y]
        x_none +=[x_with_none]
        y_none +=[y_with_none]
        plot(x,y,'o')

    imshow(raw_image, aspect='auto', origin='lower', cmap='viridis',vmax = 1e6)
    plot(traces_loc, '-', linewidth=2.5)

    return traces_loc, x_found,y_found, x_none, y_none


def checking_wavelength_aligment_in_modes(x_none, y_none):
    matplotlib.use('TkAgg')
    fig, ax = plt.subplots()

    # Find the maximum number of columns
    max_columns = max(len(row) for row in x_none)

    # Iterate over each column index
    for j in range(max_columns):
        x_vals = []
        y_vals = []
        for i in range(len(x_none)):  # Loop through rows (modes)
            if j < len(x_none[i]) and y_none[i][j] is not None:  # Ensure valid x and y
                x_vals.append(x_none[i][j])
                y_vals.append(y_none[i][j])
        if len(x_vals) > 1:  # Plot only if there's at least two points to connect
            ax.plot(x_vals, y_vals, marker='o', label=f'Column {j+1}')

    # Add a legend and labels
    ax.legend()
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Plots Across Y Columns (Handling Missing Values)")
    plt.show()
    print("buffer")

def save_fits_and_png(raw_image,traces_loc, header, x_found,y_found, pixel_min, pixel_max,pixel_wide,output_channels, folder):

    # Handle case when traces_loc is None (failed pixelmap generation)
    if traces_loc is None:
        print("Warning: traces_loc is None, creating empty pixelmap data for output")
        # Create a dummy array with the same shape as the raw image
        traces_loc_data = np.zeros((raw_image.shape[1], output_channels), dtype=int)
    else:
        traces_loc_data = traces_loc.copy()

    # Save fits file with traces_loc inside
    hdu = fits.PrimaryHDU(traces_loc_data)
    header['X_FIRTYP'] = 'PIXELMAP'
    # Add date and time to the header
    current_time = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
    header['DATE-PRO'] = current_time
    if 'DATE' not in header:
        header['DATE'] = current_time

    # Add input parameters to the header
    header['Q_PMXMIN'] = pixel_min
    header['Q_PMXMAX'] = pixel_max
    header['Q_PMWIDE'] = pixel_wide
    header['Q_PMCHAN'] = output_channels
    header['Q_PM_CK'] = np.random.randint(0, 2**32, dtype=np.uint32)
    basename = runlib_io.create_output_filename(header)
    header['Q_PMNAME'] = basename

    # Définir le chemin complet du sous-dossier
    if folder.endswith("*fits"):
        folder = folder[:-5]
    output_dir = os.path.join(folder,"../pixelmaps")

    # Créer le dossier "pixelmaps" s'il n'existe pas déjà
    os.makedirs(output_dir, exist_ok=True)

    hdu.header.extend(header, strip=True)
    hdul = fits.HDUList([hdu])
    filename_out = os.path.join(output_dir, basename)

    if traces_loc is not None:
        hdul.writeto(filename_out, overwrite=True)
        hdul.close()
        print("File saved as: "+filename_out)

    traces_loc_for_plot = traces_loc_data
    fig,ax=runlib_io.make_figure_of_trace(raw_image,traces_loc_for_plot,pixel_wide,pixel_min,pixel_max)
    
    annotation = False
    y_trace = False
    if not y_trace and x_found is not None and y_found is not None:
        for i in range(output_channels):
            if i < len(x_found) and i < len(y_found):
                ax.plot(x_found[i],y_found[i],'w-',linewidth=0.5)
                if annotation :
                    # Annotate each point
                    for j, (x, y) in enumerate(zip(x_found[i], y_found[i])):
                        offset = (5, -5) if j % 2 == 0 else (-5, 5)  # Alternate offsets
                        ax.annotate(f'({x}, {y})', xy=(x, y), xytext=offset, textcoords='offset points', 
                                    fontsize=6, color='white')
    elif x_found is None or y_found is None:
        print("Warning: x_found or y_found is None, skipping trace plotting")
    
    
    if y_trace and x_found is not None and y_found is not None:
        max_columns = max(len(row) for row in x_found)

        # Iterate over each column index
        for j in range(max_columns):
            x_vals = []
            y_vals = []
            for i in range(len(x_found)):  # Loop through rows (modes)
                if j < len(x_found[i]):  # Ensure the column exists in the current row
                    x_vals.append(x_found[i][j])
                    y_vals.append(y_found[i][j])
            if x_vals and y_vals:  # Check if there is data to plot
                ax.plot(x_vals, y_vals, marker='o', label=f'Column {j+1}')


    fig.savefig(filename_out[:-4]+"png",dpi=300)
    print("PNG saved as: "+filename_out[:-4]+"png")

def quick_fits(data, title=""):
    #For debugging purpose
    now = datetime.now()
    date_time_str = now.strftime("%Y_%m_%d_%H_%M_%S")
    runlib_io.save_fits_file(data, "/home/jsarrazin/Bureau/test zone/coupling_maps/"+title+"_"+date_time_str+".fits")
    print("check")

def run_createPixelMap(folder, destination, pixel_min=20, pixel_max=1600, pixel_wide=3, output_channels=38, file_patterns=["**/*.fits"]):
    filelist = process_files(folder, file_patterns)
    raw_image, header = raw_image_clean(filelist)
    # quick_fits(raw_image)
    traces_loc, x_found,y_found, x_none, y_none = generate_pixelmap(raw_image, pixel_min, pixel_max, output_channels, filelist)
    #checking_wavelength_aligment_in_modes(x_none, y_none) # TESTING ONLY, TO REMOVE
    save_fits_and_png(raw_image, traces_loc, header, x_found,y_found, pixel_min, pixel_max,pixel_wide,output_channels, folder)
    save_fits_and_png(raw_image,traces_loc, header, x_found,y_found, pixel_min, pixel_max,pixel_wide,output_channels, destination)
    return raw_image, traces_loc, header,  x_found,y_found


def get_fits_statistics(filepath):
    """
    Opens a FITS file and returns general statistics about the data.
    
    Parameters:
        filepath (str): Path to the FITS file.
        
    Returns:
        dict: Dictionary containing statistical measures.
    """
    with fits.open(filepath) as hdul:
        data = hdul[0].data

    # Ensure it's a NumPy array and mask NaNs if present
    data = np.array(data)
    data = data[np.isfinite(data)]

    stats = {
        'std_dev': np.std(data),
        'variance': np.var(data),
        'coefficient_of_variation': np.std(data) / np.mean(data) if np.mean(data) != 0 else float('inf')
    }

    return stats


def filter_only_good_files(filelist, filter_files=False):
    good_files = []
    bad_files = []
    stats = {}
    #evaluate_shapes(filelist)
    for file in tqdm(filelist, desc="Filtering files"):
        with fits.open(file) as hdul:
            
            data = hdul[0].data
            #maybe measure the value of individiual scans then discriminate scans ? Or like take 100 at random and discriminate on those
            var_coef = np.std(data) / np.mean(data)
            stats = {file : var_coef}
    
            if var_coef >=0.031 : #approx value for which it works
                good_files.append(file)
            else : 
                bad_files.append(file)
    
    if len(bad_files)==0: 
        print("All files have the correct flux")
        return filelist
    elif filter_files==False:
        print(len(bad_files), " bad files detected out of ", len(filelist), ", you might want to rerun with filter_files=True. Now proceeding with all files.")
        return filelist
    elif filter_files==True and len(good_files)==0:
        print("No flux detected in any files ! Are you running on darks ?")
        return 0
    else :
        print(f"{len(good_files)} have enough flux out of the {len(filelist)}, now using them.")
        return good_files
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create the pixel map needed to preprocess the data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    %(prog)s --pixel_min=20 --pixel_max=1600 --pixel_wide=2 --filter_files *.fits
    %(prog)s --pixel_min=50 --pixel_max=1500  data/*.fits

Input:
    - Files of type X_FIRTYP=RAW in the directory.

Output:
    - A FITS file with the pixel map.
    - A PNG file with the pixel map.
        """
    )

    # needed to work in VSC:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--f", help=argparse.SUPPRESS)

    # Add positional argument for file patterns
    parser.add_argument('file_patterns', nargs='*', default=['*.fits'],
                       help='One or more glob patterns for FITS files (default: *.fits)')

    # Add optional arguments with better defaults and validation
    parser.add_argument("--pixel_min", type=int, default=100,
                       help="Minimum pixel value along wavelength axis (default: %(default)s)")
    parser.add_argument("--pixel_max", type=int, default=2100,
                       help="Maximum pixel value along wavelength axis (default: %(default)s)")
    parser.add_argument("--pixel_wide", type=int, default=2,
                       help="Window half width (default: %(default)s) (full width = 2*pixel_wide+1)")
    parser.add_argument("--filter_files", action='store_true',
                       help="Flag to filter out files that don't have enough flux. Can be long, recommended only if previous run failed.")
    
    # Extract the parsed arguments
    args = parser.parse_args()
    pixel_min = args.pixel_min
    pixel_max = args.pixel_max
    pixel_wide = args.pixel_wide
    file_patterns = args.file_patterns
    filter_files = args.filter_files

    if ("VSCODE_PID" in os.environ or os.environ.get('TERM_PROGRAM') == 'vscode' or 
        os.environ.get('SPYDER_DEBUG_FILE')):
        print("Running in compiler")
        if getpass.getuser() == "slacour":
            pixel_min=50
            pixel_max=1500
            pixel_wide=2
            filter_files=True
            file_patterns=["/Users/slacour/DATA/LANTERNE/tmp/firstpl_13:0*.fits"]
            file_patterns=["/Users/slacour/DATA/LANTERNE/raw/20251119/firstpl"]
            
            # Create clean argument list for development environment
            dev_args = [
                f"--pixel_min={pixel_min}", 
                f"--pixel_max={pixel_max}", 
                f"--pixel_wide={pixel_wide}"
            ] + file_patterns
            
            print(f"Development arguments: {dev_args}")
            # Parse with custom arguments to avoid Jupyter kernel conflicts
            args = parser.parse_args(dev_args)

    
    fileList = FileList(file_patterns, first_type='RAW')
    filelist = fileList.filelist

    wollastons = np.unique([fits.getheader(f).get('X_FIRWOL', 'UNKNOWN') for f in filelist])

    # Check for UNKNOWN wollaston status and raise error if found
    valid_wollastons = {'IN', 'OUT'}
    if not set(wollastons).issubset(valid_wollastons):
        invalid_wollastons = set(wollastons) - valid_wollastons
        unknown_files = [f for f in filelist if fits.getheader(f).get('X_FIRWOL', 'UNKNOWN') == 'UNKNOWN']
        raise ValueError(f"Found {len(unknown_files)} files with UNKNOWN wollaston status. Update manually wollaston status with runPL_changeKeyword.py")

    for wollaston in wollastons:
        print(f"Processing files with wollaston status: {wollaston}")
        # Filter files based on wollaston status

        fileList = FileList(file_patterns, first_type='RAW', wollaston=wollaston)

        if wollaston == 'IN':
            output_channels = 38
        else:
            output_channels = 19

        raw_image, header = raw_image_clean(fileList.filelist)
        try:
            traces_loc, x_found,y_found, x_none, y_none = generate_pixelmap(raw_image, pixel_min, pixel_max, output_channels, fileList.filelist)
        except Exception as e:
            print(f"Error occurred while generating pixelmap: {e}")
            traces_loc, x_found,y_found, x_none, y_none = None, None, None, None, None
        
        folder = fileList.get_most_common_dir()
        save_fits_and_png(raw_image, traces_loc, header, x_found,y_found, pixel_min, pixel_max,pixel_wide,output_channels, folder)

# %%
