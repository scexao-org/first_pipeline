#! /usr/bin/env python3
# -*- coding: iso-8859-15 -*-
#%%
"""
FIRST Pipeline - Pixel Map Generation Core Algorithms

Core algorithms for creating pixel maps essential for preprocessing raw FIRST 
Visible Photonic Lantern data. Contains the main processing functions separated 
from CLI interface for interactive use and modularity.

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

# import sys
# import os
# # Add src directory to path for imports to work in both interactive and package contexts
# if os.path.join(os.path.dirname(__file__), '..') not in sys.path:
#     sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from astropy.io import fits
from glob import glob
import numpy as np
from datetime import datetime
from tqdm import tqdm
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, hist, clf, figure, legend, imshow

from first_pipeline_shared.classes.runPL_class_fileList import FileList
from first_pipeline_shared.classes.runPL_class_pixelMap import PixelMap
from first_pipeline_shared.libraries import runPL_library_io as runlib_io
plt.ion()

def process_files(folder=".", file_patterns=["**/*.fits"]):
    """
    Processes files based on the given parameters.

    Args:
        folder (str): Directory to process files from.
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
    """
    Create a clean, bias-subtracted, and summed image from a list of FITS files.
    
    Args:
        filelist (list): List of FITS file paths to process
        
    Returns:
        tuple: (cleaned_image, header) where cleaned_image is 2D numpy array
               and header is the FITS header from the first file
    """
    header = None  # Will store header from the first file
    summed_image = None  # Will store the summed image
    collapsed_images = []
    
    for filename in tqdm(filelist,desc="Summing raw images", unit="file"):
        with fits.open(filename) as hdul:
            if header is None:
                header = hdul[0].header.copy()  # Copy header from first file
            
            data = hdul[0].data
            if data is not None and data.size > 0:
                collapsed = data.sum(axis=0, dtype=np.uint32)
                collapsed_images.append(collapsed)

    stack = np.stack(collapsed_images, axis=0).astype(np.float32)
    N = stack.shape[0]

    trim_fraction = 0.1  # 10% trim
    k = int(N * trim_fraction)

    # Sort along stack axis
    sorted_stack = np.sort(stack, axis=0)

    # Remove lowest and highest k samples
    trimmed = sorted_stack[k:N-k]

    trimmed_mean = trimmed.mean(axis=0)
    return trimmed_mean, header


def quick_fits(data, title=""):
    """For debugging purpose - saves FITS file for inspection."""
    now = datetime.now()
    date_time_str = now.strftime("%Y_%m_%d_%H_%M_%S")
    runlib_io.save_fits_file(data, "/home/jsarrazin/Bureau/test zone/coupling_maps/"+title+"_"+date_time_str+".fits")
    print("check")


def peaks_using_scipy(raw_image, sampling, Nsampling, output_channels):
    """
    Detect fiber peaks using scipy peak finding algorithm.
    
    Args:
        raw_image: 2D numpy array of detector data
        sampling: Array of wavelength pixel positions to sample
        Nsampling: Half-width of sampling window
        output_channels: Number of expected peaks/channels
        
    Returns:
        tuple: (solution_found, peaks_all_samples) where solution_found is boolean array
               indicating successful detection and peaks_all_samples contains peak positions
    """
    solution_found = []
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
        solution_found+=[len(peaks) == output_channels]
        
        if len(peaks) == output_channels:
            peaks_all_samples[:, i] = peaks
        else:
            # Better strategy when number of peaks doesn't match expected channels
            if len(peaks) > output_channels:
                # Too many peaks detected - select strongest ones
                prominence_values = props['prominences']
                strongest_indices = np.argsort(prominence_values)[-output_channels:]
                peaks_all_samples[:, i] = peaks[strongest_indices]
                print(f"Warning: {len(peaks)} peaks detected at position {sampling[i]}, expected {output_channels}. Using {output_channels} strongest peaks.")
            else:
                # Too few peaks detected 
                peaks_all_samples[:len(peaks), i] = peaks
                print(f"Warning: Only {len(peaks)} peaks detected at position {sampling[i]}, expected {output_channels}")

    return solution_found, peaks_all_samples


def generate_pixelmap(raw_image, pixel_min, pixel_max, output_channels, filelist):
    """
    Generate pixel map by detecting and tracing fiber peaks across wavelength range.
    
    This function analyzes a summed raw image to detect fiber peaks at different wavelength
    positions, fits polynomial traces to connect the peaks, and creates a pixel map
    for data preprocessing.
    
    Args:
        raw_image: 2D numpy array of summed raw detector data (y: spatial, x: wavelength)
        pixel_min: Minimum pixel value along wavelength axis to start peak detection
        pixel_max: Maximum pixel value along wavelength axis to end peak detection  
        output_channels: Number of expected fiber channels/peaks to detect
        filelist: List of input FITS files being processed (for error reporting)
        
    Returns:
        tuple: (traces_loc, traces_loc_double, x_found, y_found, x_none, y_none) where:
               - traces_loc: 2D array mapping each wavelength pixel to spatial pixel for each channel
               - traces_loc_double: Same as traces_loc but with sub-pixel precision
               - x_found: List of wavelength positions where peaks were successfully detected
               - y_found: List of spatial positions of detected peaks
               - x_none: List with None values for outlier/rejected peak positions
               - y_none: List with None values for outlier/rejected peak positions
    """
    
    pixel_length = raw_image.shape[1]

    # 300 values of pixels between pixelmin and pixelmax
    Nsampling = 25
    sampling = np.linspace(pixel_min+Nsampling, pixel_max-Nsampling, 300, dtype=int)
    peaks = np.zeros([output_channels, sampling.shape[0]])

    solution_found, peaks = peaks_using_scipy(raw_image, sampling, Nsampling, output_channels)
    
    true_count = sum(solution_found)  # because True == 1, False == 0
    percentage = true_count / len(solution_found)
    print(f"Percentage of successful detections: {percentage}")
    
    if percentage < 0.1:
        print("Very low detection rate. Consider adjusting parameters or checking data quality.")
        return None, None, None, None, None, None

    traces_loc = np.zeros((pixel_length, output_channels),dtype=int)
    traces_loc_double = np.zeros((pixel_length, output_channels))

    x_found = []
    y_found = []
    x_none = []
    y_none = []

    for i in range(output_channels):
        # Extract valid peak positions for this channel
        valid_detections = np.array(solution_found)
        x = sampling[valid_detections]
        y = peaks[i][valid_detections]
        
        if len(x) < 3:  # Need at least 3 points for polynomial fitting
            print(f"Warning: Channel {i} has insufficient valid detections ({len(x)} points)")
            continue
            
        # Initial polynomial fit (1st order)
        poly_coeffs = np.polyfit(x, y, 1)

        # Iterative outlier removal
        for iteration in range(5):
            # Calculate residuals of the function
            y_fit = np.polyval(poly_coeffs, x)
            residuals = y - y_fit

            # Calculate standard deviation of residuals
            std_residuals = np.std(residuals)
            # Keep it above half a pixel
            if std_residuals < 1/2:
                std_residuals = 1/2

            # Identify inliers (points with residuals within the threshold)
            diff_y_median = np.median(np.diff(y))
            inliers = np.abs(residuals) < 3 * std_residuals
            inliers &= np.abs(residuals) < diff_y_median * 2 / 3

            # Remove outliers and refit if we have enough points
            if np.sum(inliers) >= 3:
                x = x[inliers]
                y = y[inliers]
                poly_coeffs = np.polyfit(x, y, 1)
            else:
                break

        # Generate full wavelength trace using polynomial
        x_full = np.arange(pixel_length)
        y_trace = np.polyval(poly_coeffs, x_full)
        
        # Clip to valid pixel range
        y_trace = np.clip(y_trace, 0, raw_image.shape[0]-1)
        
        traces_loc[:, i] = y_trace
        traces_loc_double[:, i] = y_trace  # For now, same as integer version
        
        # Store found and rejected points for diagnostics
        x_found.append(x)
        y_found.append(y)
        
        # Create lists with None for rejected points (for diagnostics)
        x_with_none = sampling.copy().astype(float)
        y_with_none = peaks[i].copy()
        rejected_mask = ~np.isin(sampling, x)
        x_with_none[rejected_mask] = None
        y_with_none[rejected_mask] = None
        
        x_none.append(x_with_none)
        y_none.append(y_with_none)
        
        plot(x, y, 'o')

    imshow(raw_image, aspect='auto', origin='lower', cmap='viridis', vmax=1e6)
    plot(traces_loc, '-', linewidth=2.5)

    return traces_loc, traces_loc_double, x_found, y_found, x_none, y_none


def checking_wavelength_aligment_in_modes(x_none, y_none):
    """Check wavelength alignment diagnostic plots."""
    plt.figure()
    for i in range(len(x_none)):
        plt.plot(x_none[i], y_none[i], 'o-', alpha=0.7, label=f'Mode {i}')
    plt.xlabel('Wavelength Pixel')
    plt.ylabel('Spatial Pixel') 
    plt.title('Wavelength Alignment Check')
    plt.legend()
    plt.show()
    print("buffer")


def save_fits_and_png(raw_image, traces_loc, traces_loc_double, header, x_found, y_found, 
                     pixel_min, pixel_max, pixel_wide, output_channels, folder):
    """
    Save pixel map results as FITS file and PNG visualization.
    
    Args:
        raw_image: Original summed detector image
        traces_loc: Integer pixel map traces 
        traces_loc_double: Sub-pixel precision traces
        header: FITS header for output file
        x_found, y_found: Detected peak positions
        pixel_min, pixel_max, pixel_wide: Processing parameters
        output_channels: Number of channels processed
        folder: Output directory path
    """

    # Handle case when traces_loc is None (failed pixelmap generation)
    if traces_loc is None:
        print("Warning: traces_loc is None, creating empty pixelmap data for output")
        # Create a dummy array with the same shape as the raw image
        traces_loc_data = np.zeros((raw_image.shape[1], output_channels), dtype=int)
        traces_loc_double = np.zeros((raw_image.shape[1], output_channels), dtype=float)
    else:
        traces_loc_data = traces_loc.copy()
        traces_loc_double = traces_loc_double.copy()

    # Create PixelMap object and save using the new save method
    pixelMap = PixelMap()
    pixelMap.create_from_data(traces_loc_data, traces_loc_double, pixel_min, pixel_max, pixel_wide, output_channels)

    # Prepare header with additional information
    save_header = header.copy()
    save_header['X_FIRTYP'] = 'PIXELMAP'
    basename = runlib_io.create_basename(save_header)
    save_header['Q_PMNAME'] = basename
    save_header['X_FIRWOL'] = header.get('X_FIRWOL', 'UNKNOWN') 

    # Définir le chemin complet du sous-dossier
    if folder.endswith("*fits"):
        folder = folder[:-5]
    output_dir = os.path.join(folder, "../pixelmaps")

    # Créer le dossier "pixelmaps" s'il n'existe pas déjà
    os.makedirs(output_dir, exist_ok=True)

    filename_out = os.path.join(output_dir, basename)

    if traces_loc is not None:
        pixelMap.save(filename_out, save_header)

    traces_loc_for_plot = traces_loc_data
    fig, ax = runlib_io.make_figure_of_trace(raw_image, traces_loc_for_plot, pixel_wide, pixel_min, pixel_max)
    
    annotation = False
    y_trace = False

    # Save PNG visualization
    if annotation:
        for i in range(len(x_found)):
            ax.plot(x_found[i], y_found[i], 'r.', markersize=1)
            ax.annotate(str(i), (x_found[i][0], y_found[i][0]), textcoords="offset points", 
                       xytext=(0,10), ha='center', fontsize=8, color='white')
    if y_trace and traces_loc is not None:
        ax.plot(traces_loc, '-', color='red', linewidth=1, alpha=0.7)

    png_filename = filename_out.replace('.fits', '.png')
    fig.savefig(png_filename, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Pixel map saved: {filename_out}")
    print(f"Visualization saved: {png_filename}")

    return pixelMap


def get_fits_statistics(filepath):
    """Get basic statistics from a FITS file for quality assessment."""
    try:
        with fits.open(filepath) as hdul:
            data = hdul[0].data
            if data is not None:
                return {
                    'mean': np.mean(data),
                    'median': np.median(data),
                    'std': np.std(data),
                    'max': np.max(data),
                    'min': np.min(data)
                }
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None


def filter_only_good_files(filelist, filter_files=False):
    """
    Filter files based on flux quality if requested.
    
    Args:
        filelist: List of FITS file paths
        filter_files: Whether to apply flux filtering
        
    Returns:
        list: Filtered list of files with sufficient flux
    """
    if not filter_files:
        return filelist
        
    print("Filtering files based on flux quality...")
    good_files = []
    flux_threshold = 1000  # Minimum flux threshold
    
    for filepath in filelist:
        stats = get_fits_statistics(filepath)
        if stats and stats['median'] > flux_threshold:
            good_files.append(filepath)
        else:
            print(f"Excluding {filepath}: insufficient flux")
    
    if len(good_files) < len(filelist) * 0.5:
        print(f"Warning: Only {len(good_files)}/{len(filelist)} files passed flux filter")
    else:
        print(f"{len(good_files)} files have sufficient flux out of {len(filelist)}")
        
    return good_files


def run_createPixelMap(pixel_min=None, pixel_max=None, pixel_wide=None, file_patterns=None):
    """
    High-level function to run the complete pixel map creation process.
    
    Args:
        folder: Input directory with raw FITS files.
               If None, uses development defaults.
        destination: Output directory for pixel maps.
                   If None, uses development defaults.
        pixel_min, pixel_max: Wavelength axis range for processing.
                              If None, uses development defaults.
        pixel_wide: Detection window half-width.
                   If None, uses development defaults.
        output_channels: Number of expected fiber channels
        file_patterns: File patterns to match.
                      If None, uses development defaults.
        
    Returns:
        tuple: (raw_image, traces_loc, header, x_found, y_found)
    """

    # Process files
    fileList = FileList(file_patterns, first_type='RAW')
    filelist = fileList.filelist

    # Check Wollaston status
    wollastons = np.unique([fits.getheader(f).get('X_FIRWOL', 'UNKNOWN') for f in filelist])

    # Validate Wollaston status
    valid_wollastons = {'IN', 'OUT', 'UNKNOWN'}
    if not set(wollastons).issubset(valid_wollastons):
        invalid_wollastons = set(wollastons) - valid_wollastons
        unknown_files = [f for f in filelist if fits.getheader(f).get('X_FIRWOL', 'UNKNOWN') == 'UNKNOWN']
        raise ValueError(f"Found {len(unknown_files)} files with UNKNOWN wollaston status. Update manually wollaston status with runPL_changeKeyword.py")

    # Process each Wollaston configuration separately
    for wollaston in wollastons:
        print(f"Processing files with wollaston status: {wollaston}")
        
        # Filter files based on wollaston status
        fileList = FileList(file_patterns, first_type='RAW', wollaston=wollaston)

        # Determine output channels based on Wollaston status
        if wollaston == 'IN':
            output_channels = 38
        else:
            output_channels = 19

        # Validate and adjust pixel range
        raw_image, header = raw_image_clean(fileList.filelist)

        ny, nx = raw_image.shape
        if pixel_min < 0:
            print(f"Warning: pixel_min ({pixel_min}) is below 0, setting to 0")
            pixel_min = 0
        if pixel_max >= nx:
            print(f"Warning: pixel_max ({pixel_max}) is >= image width ({nx}), setting to {nx-1}")
            pixel_max = nx - 1
        if pixel_min >= pixel_max:
            print(f"Warning: pixel_min ({pixel_min}) >= pixel_max ({pixel_max}), adjusting to maintain valid range")
            pixel_min = max(0, pixel_max - 100)  # Ensure at least 100 pixel range

        # Generate pixel map
        try:
            traces_loc, traces_loc_double, x_found, y_found, x_none, y_none = generate_pixelmap(
                raw_image, pixel_min, pixel_max, output_channels, fileList.filelist)
        except Exception as e:
            print(f"Error occurred while generating pixelmap: {e}")
            traces_loc, traces_loc_double, x_found, y_found, x_none, y_none = None, None, None, None, None, None
        
        # Save results
        folder = fileList.get_most_common_dir()
        pixelMap = save_fits_and_png(raw_image, traces_loc, traces_loc_double, header, x_found, y_found, 
                            pixel_min, pixel_max, pixel_wide, output_channels, folder)

    return pixelMap, raw_image

if __name__ == "__main__":

    # Development/interactive mode handling
    print("Running in compiler")
    if getpass.getuser() == "slacour":
        pixel_min = 50
        pixel_max = 1500
        pixel_wide = 2
        filter_files = True
        file_patterns = ["/Users/slacour/DATA/LANTERNE/tmp/firstpl_13:0*.fits"]
        file_patterns = ["/Users/slacour/DATA/LANTERNE/raw/20260114/firstpl/*3.fits"]
        
        
    print(f"Development override: pixel_min={pixel_min}, pixel_max={pixel_max}, pixel_wide={pixel_wide}")
    print(f"Development file patterns: {file_patterns}")

    pixelMap, raw_image = run_createPixelMap(pixel_min=pixel_min, pixel_max=pixel_max, pixel_wide=pixel_wide, file_patterns=file_patterns)

    pixelMap2= PixelMap(pixelMap.filename)

    data_cut_pixels, data_dark_pixels, data_edge_pixels = pixelMap2.preprocess_cutData(raw_image/1000, True)



# %%
