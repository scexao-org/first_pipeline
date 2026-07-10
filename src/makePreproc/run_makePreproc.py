#%%

"""
FIRST Pipeline - Data Preprocessing Core Functions

Core algorithms for preprocessing raw FIRST Visible Photonic Lantern data using 
pixel maps. Contains the main processing functions separated from CLI interface 
for interactive use and modularity.

Created on Wed May 21 22:56:25 2025
@author: slacour
"""

import os
import getpass
import glob
import matplotlib
if "VSCODE_PID" in os.environ:
    matplotlib.use('macosx')
else:
    matplotlib.use('Agg')
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from astropy.io import fits
from tqdm import tqdm
from datetime import datetime

from first_pipeline_shared.classes.runPL_class_fileList import FileList
from first_pipeline_shared.classes.runPL_class_pixelMap import PixelMap
from first_pipeline_shared.classes.runPL_class_preproc import Preproc
from first_pipeline_shared.libraries import runPL_library_io as runlib_io


def preprocess_files(fileList, overwrite=False, plot_sum=False):
    """
    Preprocesses raw FITS files using provided pixel map(s), extracts and aggregates spectral
    traces, computes basic quality-control metrics, and writes preprocessed FITS files and
    diagnostic PNG figures into a per-directory "preproc" folder.
    
    This function uses the Preproc class to handle individual file processing.
    
    Parameters
    ----------
    fileList : FileList
        FileList object containing raw files and their associated pixel maps
    overwrite : bool, optional
        Whether to overwrite existing preprocessed files. Default is False.
    plot_sum : bool, optional
        If True, a summary PNG showing the vertical offset of extracted windows across all
        processed files will be produced and saved. Default is False.
        
    Returns
    -------
    list
        A list of output filenames (basename of created preprocessed FITS files) that were
        created during this call. If no files were processed, an empty list is returned.
    """
    
    dir_path_0 = fileList.get_most_common_dir()
    files_out = []  # Store preprocessed file paths for summary plot

    # Process each file using the Preproc class
    for file_withpixelmap in tqdm(fileList.files_with_associated_files, desc=f"Pre-processing of files in {dir_path_0}"):
        
        file = file_withpixelmap['file']
        pixelmap_file = file_withpixelmap['pixelMap']
        output_dir = os.path.join(os.path.dirname(file), "../preproc")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        if pixelmap_file is None:
            print(f"No pixel map associated with {file}, skipping.")
            continue

        # Check if a txt file with same name exists
        file_base = os.path.splitext(file)[0]
        txt_file = file_base + ".txt"
        if not os.path.exists(txt_file):
            txt_file = None

        

        pixelmap = PixelMap(pixelmap_file)
            
        try:
            # Create Preproc instance and process the file
            preproc = Preproc()
            
            preproc_created = preproc.create_from_raw(
                file,
                pixelmap,
                output_dir,
                check_if_exist=not overwrite,
                telemetry_txt_file=txt_file,
            )
            
            if preproc_created:
                # Generate diagnostic figures
                preproc.generate_diagnostic_figures(pixelmap)

                # Collect file path for summary plot
                if preproc.quality_metrics:
                    files_out.append(preproc.filename)
                    
                # Save the preprocessed file
                preproc.save()
                
        except Exception as e:
            print(f"Error processing {file}: {e}")
            continue
    
    if len(files_out) == 0:
        print(f"No files to process in {dir_path_0}.")
        return []

    # Create summary centroid shift plot if requested
    if plot_sum and len(files_out) > 0:
        create_centroid_summary_plot(files_out, dir_path_0)
            


    return files_out


def create_centroid_summary_plot(files_in, dir_path_0):
    """
    Create a summary plot for centroid shifts across all processed files.
    
    Parameters
    ----------
    files_in : list
        List of preprocessed FITS file paths. The centroid shift (``Q_P_CENT``)
        and modulation-to-frame shift (``X_FIRGSH``) values are read from each
        file's header.
    dir_path_0 : str
        Base directory path for saving the plot
    """
    # Read the values to plot from the FITS headers
    files_out = []
    centroid_data = []
    shift_data = []
    for f in files_in:
        try:
            header = fits.getheader(f)
        except Exception as e:
            print(f"Error reading header from {f}: {e}")
            continue
        files_out.append(os.path.basename(f))
        centroid_data.append(header.get('Q_P_CENT', 0))
        shift_data.append(header.get('X_FIRGSH', np.nan))

    if len(files_out) == 0:
        print("No headers could be read for centroid shift summary.")
        return

    preproc_dir_path = os.path.join(dir_path_0, "../preproc")
    filename_out = files_out[-1] if files_out else "summary"
    filename_out = "_".join(filename_out.split("_")[:-2]) if "_" in filename_out else filename_out
    filename_out_full = os.path.join(preproc_dir_path, filename_out)
    
    try:
        fig, (ax_centroid, ax_shift) = plt.subplots(
            2, 1, sharex=True, clear=True,
            num="Centroid shift summary",
            figsize=(max(8, len(files_out) * 0.25)+1, 9),
        )

        # Centroid shift (top axis)
        ax_centroid.plot(range(len(centroid_data)), centroid_data, 'o-', color='red', markersize=4)
        ax_centroid.axhline(y=0, color='black', linestyle=':', alpha=0.7)
        ax_centroid.set_title("Vertical offset of extracted windows (centroid shift)")
        ax_centroid.set_ylabel("Pixel shift")
        ax_centroid.grid(True, alpha=0.3)

        # Modulation-to-frame shift (bottom axis)
        ax_shift.plot(range(len(shift_data)), shift_data, 'o-', color='blue', markersize=4)
        ax_shift.axhline(y=0, color='black', linestyle=':', alpha=0.7)
        ax_shift.set_title("Modulation-to-frame shift (metrology glitch)")
        ax_shift.set_xlabel("File index")
        ax_shift.set_ylabel("Frame shift")
        ax_shift.grid(True, alpha=0.3)

        # Show all file names on the x-axis (shared x-axis -> set on bottom)
        ax_shift.set_xticks(range(len(files_out)))
        ax_shift.set_xticklabels(files_out, rotation=90)
        ax_shift.set_xlim(0, len(files_out) - 1)

        fig.tight_layout()
        fig.savefig(filename_out_full + "_PREPROCSHIFT.png", dpi=300)
        plt.close(fig)
        print("PNG saved as: " + filename_out_full + "_PREPROCSHIFT.png")
    except Exception as e:
        print(f"Error while plotting centroid shift summary: {e}")



def run_preprocess(file_patterns=None, pixel_map=None, object_name=None,
                           only_with_modulation=None, overwrite=None):
    """
    Process files with full parameter control (used by CLI interface).
    
    Parameters
    ----------
    file_patterns : list
        List of file patterns to process
    pixel_map : str
        Path to pixel map file or directory 
    object_name : str, optional
        Object name to filter files
    only_with_modulation : bool
        Only process files with modulation data
    overwrite : bool
        Whether to overwrite existing files
        
    Returns
    -------
    list
        List of processed filenames
    """

    # Auto-detect pixel map if not specified
    if pixel_map is None:
        folder = os.path.dirname(file_patterns[0]) if file_patterns else "."
        print(f"Using pixel map folder: {folder}")
        pixel_map = file_patterns + [os.path.join(folder, "../pixelmaps")]

    # Set modulation ID filter if requested
    modID = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20] if only_with_modulation else None
    
    # Create FileList object with filters
    fileList = FileList(file_patterns, first_type='RAW', object_name=object_name, modID=modID)
    fileList.make_association(pixelMap=pixel_map)

    print(f"Found {len(fileList.filelist)} files to process in {file_patterns}")
    print(f"Overwrite existing already preprocessed files: {overwrite}")
    
    processed_files = preprocess_files(fileList, overwrite=overwrite, plot_sum=True)

    print(f"Successfully processed {len(processed_files)} files")
    if processed_files:
        print("Processed files:", processed_files[:5])  # Show first 5 files
        if len(processed_files) > 5:
            print(f"... and {len(processed_files) - 5} more files")

    return fileList


if __name__ == "__main__":
    """
    Run data preprocessing with development defaults.
    Perfect for testing and direct execution of core functionality.
    """
    print("Running makePreproc core with development defaults...")

    if getpass.getuser() == "slacour":
        pixel_map = None
        object_name = None
        only_with_modulation = False
        overwrite = True
        file_patterns = ["/Users/slacour/DATA/LANTERNE/tmp/firstpl_13:0*.fits"]
        file_patterns = ["/Users/slacour/DATA/FIRST/20260608/firstpl/firstpl_10:10:12.570875507.fits"]
        file_patterns = ["/Users/slacour/DATA/FIRST/20260625/firstpl/firstpl_10:10:12.570875507.fits"]
        
        print(f"Development override: pixel_map={pixel_map}, object_name={object_name}, only_with_modulation={only_with_modulation}, overwrite={overwrite}")
        print(f"Development file patterns: {file_patterns}")

    # run_preprocess(file_patterns=file_patterns, pixel_map=pixel_map, object_name=object_name,
    #                        only_with_modulation=only_with_modulation, overwrite=overwrite)

    # Build the centroid shift summary from all preprocessed FITS files in the preproc directory
    raw_dir = os.path.dirname(file_patterns[0])
    preproc_dir = os.path.join(raw_dir, "../preproc")
    preproc_files = sorted(glob.glob(os.path.join(preproc_dir, "*.fits")))
    if preproc_files:
        create_centroid_summary_plot(preproc_files, raw_dir)
    else:
        print(f"No preprocessed FITS files found in {preproc_dir}")
    
# %%



