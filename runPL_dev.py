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
from scipy.signal import correlate
from scipy import linalg

from scipy.interpolate import griddata

import getpass
import matplotlib
if ("VSCODE_PID" in os.environ or os.environ.get('TERM_PROGRAM') == 'vscode'):
    matplotlib.use('Qt5Agg')
elif os.environ.get('SPYDER_DEBUG_FILE'):
    print("Running in Spyder")
else:
    matplotlib.use('Agg')
     
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot,hist,clf,figure,legend,imshow
from datetime import datetime
from tqdm import tqdm
from collections import defaultdict
from scipy import linalg
from matplotlib import animation
from itertools import product
from scipy.linalg import pinv
from scipy.optimize import curve_fit
import libraries.runPL_library_io as runlib
import libraries.runPL_library_plots as runlib_i
import libraries.runPL_library_basic as basic
from scipy.ndimage import zoom
from astropy.io import fits
import shutil
from scipy.interpolate import interpn
from scipy.spatial import cKDTree
from scipy.spatial import Delaunay
from itertools import combinations
from scipy.optimize import minimize

plt.ion()

DEBUG = False

# Add options
usage = """
Goal:
    Reconstruct images from FIRST Photonic Lantern data using specified coupling maps and options.

Summary:
    This script processes preprocessed FITS files, applies coupling maps, reconstructs images, and saves the results.
    It supports selection by object name, modulation pattern, and smoothing options. The script can save individual frames,
    wavelength slices, and residuals, and allows explicit selection of coupling map files.

Input files:
    - Preprocessed data FITS files (e.g., with DPR_CATG=OBJECT and DPR_TYPE=PREPROC)
    - Coupling map FITS files (e.g., with X_FIRTYP=COUPLINGMAP)

Output files:
    - Reconstructed image FITS files, including summed images, residuals, and optionally individual frames and wavelength slices.

Options:
    --object_name <str>           Selection of the data by the Object name (default: NONE)
    --modID <int>                 Selection of the modulation pattern by user [0 == first in the list] (default: 0)
    --modScale <int>              Selection of the modulation scale by user [0 == first in the list] (default: 0)
    --coupling_map <str>          Force to select which coupling map file to use (default: the one in the directory)
    --wavelength_smooth <int>     Smoothing factor for wavelength (default: 1)
    --save_individual_frames      Save individual frames (default: True)
    --save_individual_wavelength  Save individual wavelength slices (default: False)

Example:
    python runPL_imageReconstruction.py --object_name=HIP81126 --modID=1 --coupling_map=path/to/couplingmap.fits *.fits
"""


def get_filelist(file_patterns, cmap_patterns, modID, modScale, object_name, wollaston):

        fits_keywords = {'X_FIRTYP': ['PREPROC'],
                        'DATA-TYP': ['OBJECT','OJECT','TEST'],
                        }    
        
        # Adding other constraints if asked by user
        if modID is not None:
            fits_keywords['X_FIRMID'] = [modID]
        if modScale is not None:
            fits_keywords['X_FIRMSC'] = [modScale]
        if object_name is not None:
            fits_keywords['OBJECT'] = [object_name]
        if wollaston is not None:
            fits_keywords['X_FIRWOL'] = [wollaston]
        
        print(file_patterns)
        filelist = runlib.get_filelist(file_patterns, fits_keywords)

        if len(filelist) == 0:
            raise FileNotFoundError("No files found with the specified patterns and keywords.")

        # Adding new constraints if not asked by user
        hd=fits.getheader(filelist[0])
        modID = hd.get('X_FIRMID', 0)
        modScale = hd.get('X_FIRMSC', 0)
        object_name = hd.get('OBJECT', 'NONE')
        wollaston = hd.get('X_FIRWOL', 'IN')
        fits_keywords['OBJECT'] = [object_name]
        fits_keywords['X_FIRMID'] = [modID]
        fits_keywords['X_FIRMSC'] = [modScale]
        fits_keywords['X_FIRWOL'] = [wollaston]

        print("----------------")
        print(f"Selected object='{object_name}' with modScale={modScale} and modID={modID}")
        print("----------------")

        filelist = runlib.get_filelist(file_patterns, fits_keywords)

        if len(filelist) == 0:
            raise FileNotFoundError("No files found with the specified patterns and keywords.")

        fits_keywords = {'X_FIRTYP': ['PREPROC'],
                        'DATA-TYP': ['DARK'],
                        'X_FIRWOL': [wollaston],
                        }    
    
        try:
            filelist_dark = runlib.get_filelist(file_patterns, fits_keywords, name_search="dark")
        except FileNotFoundError as e:
            print(f"No darks: {e}")
            filelist_dark = []

        fits_keywords = {'X_FIRTYP': ['COUPLINGMAP'],
                        'X_FIRWOL': [wollaston],
                        }    

        filelist_cmap = runlib.get_filelist(cmap_patterns, fits_keywords, name_search="coupling map")

        files_with_dark = runlib.associate_dark(filelist, filelist_dark)

        return files_with_dark, filelist_cmap




def filter_couplingmapfile(filelist_cmap):
    """
    Filters the input file list to separate coupling map files and dark files based on FITS keywords.
    Raises an error if no valid files are found.
    Returns a dictionary mapping coupling map files to their closest dark files.
    """

    print("runPL object filelist : ", filelist_cmap)

    # raise an error if filelist_cleaned is empty
    if len(filelist_cmap) == 0:
        raise FileNotFoundError("No coupling map to use.\n Please specify which one to use with the option --coupling_map")

    # raise an error if filelist_cleaned is more than one
    if len(filelist_cmap) > 1:
        raise ValueError("Too many coupling maps to use! I can only use one.\n Please specify which one to use with the option --coupling_map")

    # Check if all files have the same value for header['PM_CHECK']
    pm_check_values = set()
    combined_filelist = []
    combined_filelist.extend(filelist_cmap)
    for file in combined_filelist:
        header = fits.getheader(file)
        pm_check_values.add(int(header.get('PM_CHECK', 0)))
        
    if len(pm_check_values) > 1:
        print("WARNING: The 'PM_CHECK' values (ie, the pixel map used to preprocess the files) \n are not consistent across all files!")
        print(f"Found values: {pm_check_values}")

    # for each file in filelist_cmap find the closest dark file in filelist_dark with, by priority, first the directory in which the file is, and then by the date in the "DATE" fits keyword, and second, the directory in which the file is
    # files_with_dark = {cmap: runlib.find_closest_dark(cmap, filelist_dark) for cmap in filelist_data}

    return filelist_cmap



def quick_fits(data, title=""):
    if DEBUG:
        #For debugging purpose
        now = datetime.now()
        date_time_str = now.strftime("%Y_%m_%d_%H_%M_%S")
        if getpass.getuser() == "jsarrazin":
            runlib.save_fits_file(data, "/home/jsarrazin/Bureau/test zone/coupling_maps/"+title+"_"+date_time_str+".fits")
        print("Done")

def quick_imshow(data, title=""):
    #For debugging purpose
    now = datetime.now()
    plt.imshow(data, aspect='auto')
    plt.title(title)
    print("Done")

def quick_plot(data,title =""):
    #For debugging purpose
    now = datetime.now()
    date_time_str = now.strftime("%Y_%m_%d_%H_%M_%S")
    plt.plot(data)
    plt.title(title)
    print("Done")


def getWaves_from_polytxt(pixels = np.arange(0, 5001) ):
    file_name="WavePolyBest.txt"
    if os.path.exists(file_name):
        # File exists, read it
        WavePolyBest_loaded = np.loadtxt(file_name)
        print("Loaded coefficients : ", WavePolyBest_loaded)
        
        bestFit = np.poly1d(WavePolyBest_loaded)
        wavelengths = bestFit(pixels)
        data = np.column_stack((pixels, wavelengths))
        return data
    else:
    # File does not exist
        print(f"Error: File '{file_name}' does not exist.")
    return 

def identify_halpha_ray(modScale, data_2_postiptilt, ray=656.3):

    waves_to_pixels = getWaves_from_polytxt(pixels=modScale) #if we start at 625
    wave_binning_to = data_2_postiptilt.shape[1] #then get to like 50
    
    #Retrieve Nbin
    for q in range(modScale// 1, 0, -1):
        if wave_binning_to % q == 0:
            Nbin = wave_binning_to // q
            if (modScale // Nbin) * Nbin == wave_binning_to:
                break
    print(waves_to_pixels)
    waves_to_pixels[:,:,:(Nwave//Nbin)*Nbin]
    print(waves_to_pixels)
    #For a binning Nbin we apply waveDim = (waveInit//Nbin)*Nbin
    

    return

def interpolate_halpha(data_2_postiptilt, postiptilt_2_data, pix_to_waves=""):
    """
    Takes the data_2_postiptilt and postiptilt_2_data model 
    
    """
    if pix_to_waves=="": #temporary, it needs to be changed to an accurate dictionnary (made with calibration and adapted to the bin)
        pix_to_waves={i : i for i in range(0, data_2_postiptilt.shape[1])}

    #data_2_postiptilt : mask, waves, 3, outputs
    #postiptilt_2_data :  mask, waves, outputs, 3
    quick_fits(data_2_postiptilt, "pre_inter_data")
    pre_data_2_postiptilt = data_2_postiptilt.copy()
    pre_postiptilt_2_data = postiptilt_2_data.copy()

    #Looking for the index corresponding to H_alpha
    H_alpha = 50.1#test value
    first_index = next((k for k, v in reversed(list(pix_to_waves.items())) if v < H_alpha-5), None)
    second_index = next((k for k, v in list(pix_to_waves.items()) if v > H_alpha+5), None)

    #We're naning these values to build our model
    data_2_postiptilt[:,first_index:second_index, 0 , :] = np.nan
    postiptilt_2_data[:,first_index:second_index, : , 0] = np.nan

    for i in range(data_2_postiptilt.shape[3]):
        

        x, y = np.indices((data_2_postiptilt.shape[0],data_2_postiptilt.shape[1]))

        # Known data (non-NaN)
        known_points = np.array([x[~np.isnan(data_2_postiptilt[:,:,0,i])], y[~np.isnan(data_2_postiptilt[:,:,0,i])]]).T
        known_values = data_2_postiptilt[~np.isnan(data_2_postiptilt[:,:,0,i])][:,0,i]
        # Points to interpolate (NaNs)
        missing_points = np.array([x[np.isnan(data_2_postiptilt[:,:,0,i])], y[np.isnan(data_2_postiptilt[:,:,0,i])]]).T
        # Interpolation
        interpolated_values = griddata(
            points=known_points,
            values=known_values,
            xi=missing_points,
            method='linear'  # or 'nearest', 'cubic'
        )
        # Fill in the interpolated values
        data_2_postiptilt[np.isnan(data_2_postiptilt[:,:,0,i]), 0,i] = interpolated_values


        # Known data (non-NaN)
        known_points = np.array([x[~np.isnan(postiptilt_2_data[:,:,i,0])], y[~np.isnan(postiptilt_2_data[:,:,i,0])]]).T
        known_values = postiptilt_2_data[~np.isnan(postiptilt_2_data[:,:,i,0])][:,i,0]
        # Points to interpolate (NaNs)
        missing_points = np.array([x[np.isnan(postiptilt_2_data[:,:,i,0])], y[np.isnan(postiptilt_2_data[:,:,i,0])]]).T
        # Interpolation
        interpolated_values = griddata(
            points=known_points,
            values=known_values,
            xi=missing_points,
            method='linear'  # or 'nearest', 'cubic'
        )
        # Fill in the interpolated values
        postiptilt_2_data[np.isnan(postiptilt_2_data[:,:,i,0]), i,0] = interpolated_values

    quick_fits(data_2_postiptilt, "post_inter_data")


    # Reading the error
    erreur = pre_data_2_postiptilt[:,:,0,:]-data_2_postiptilt[:,:,0,:]
    fig, axes = plt.subplots(6, 7, figsize=(18, 12))
    axes = axes.flatten()

    for i in range(38):
        ax = axes[i]
        l1, = ax.plot(erreur[:, first_index:second_index, i].sum(axis=1), label="erreur", color="r")
        l2, = ax.plot(data_2_postiptilt[:, first_index:second_index, 0, i].sum(axis=1), label="corrected", alpha=0.5, color="g", linestyle="dashed")
        l3, = ax.plot(pre_data_2_postiptilt[:, first_index:second_index, 0, i].sum(axis=1), label="original", alpha=0.5, color="grey", linestyle="dashed")
        ax.set_title(f'Output {i+1}', fontsize=8)
        ax.tick_params(labelsize=6)

    ax = axes[i+1]
    ax.plot(erreur[:, first_index:second_index,:].sum(axis=(1,2)), color="r")
    ax.plot(data_2_postiptilt[:, first_index:second_index, 0,:].sum(axis=(1,2)), alpha=0.5, color="g", linestyle="dashed")
    ax.plot(pre_data_2_postiptilt[:, first_index:second_index, 0,:].sum(axis=(1,2)), alpha=0.5, color="grey", linestyle="dashed")
    ax.set_title(f'All summed', fontsize=8)

    ax = axes[i+2]
    ax.plot(erreur[:, first_index:second_index].mean(axis=(1,2)), color="r")
    ax.plot(data_2_postiptilt[:, first_index:second_index, 0,:].mean(axis=(1,2)), alpha=0.5, color="g", linestyle="dashed")
    ax.plot(pre_data_2_postiptilt[:, first_index:second_index, 0,:].mean(axis=(1,2)), alpha=0.5, color="grey", linestyle="dashed")
    ax.set_title(f'All meaned', fontsize=8)

    fig.legend(handles=[l1, l2, l3], loc='upper left', ncol=3, fontsize=10)
    fig.suptitle("data_2_postiptilt : All corrected wavelenght, summmed", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  
    plt.show()


    # Reading the error
    erreur = pre_postiptilt_2_data[:,:,:,0]-postiptilt_2_data[:,:,:,0]
    fig, axes = plt.subplots(6, 7, figsize=(18, 12))
    axes = axes.flatten()

    for i in range(38):
        ax = axes[i]
        l1, = ax.plot(erreur[:, first_index:second_index, i].sum(axis=1), label="erreur", color="r")
        l2, = ax.plot(postiptilt_2_data[:, first_index:second_index, i,0].sum(axis=1), label="corrected", alpha=0.5, color="g", linestyle="dashed")
        l3, = ax.plot(pre_postiptilt_2_data[:, first_index:second_index, i,0].sum(axis=1), label="original", alpha=0.5, color="grey", linestyle="dashed")
        ax.set_title(f'Output {i+1}', fontsize=8)
        ax.tick_params(labelsize=6)

    ax = axes[i+1]
    ax.plot(erreur[:, first_index:second_index].sum(axis=(1,2)), color="r")
    ax.plot(postiptilt_2_data[:, first_index:second_index, :,0].sum(axis=(1,2)), alpha=0.5, color="g", linestyle="dashed")
    ax.plot(pre_postiptilt_2_data[:, first_index:second_index, :,0].sum(axis=(1,2)), alpha=0.5, color="grey", linestyle="dashed")
    ax.set_title(f'All summed', fontsize=8)

    ax = axes[i+2]
    ax.plot(erreur[:, first_index:second_index].mean(axis=(1,2)), color="r")
    ax.plot(postiptilt_2_data[:, first_index:second_index,:, 0].mean(axis=(1,2)), alpha=0.5, color="g", linestyle="dashed")
    ax.plot(pre_postiptilt_2_data[:, first_index:second_index,:, 0].mean(axis=(1,2)), alpha=0.5, color="grey", linestyle="dashed")
    ax.set_title(f'All meaned', fontsize=8)

    # Use legend handles from the first subplot only
    fig.legend(handles=[l1, l2, l3], loc='upper left', ncol=3, fontsize=10)
    fig.suptitle("postiptilt_2_data : All corrected wavelenght, summmed", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space at top for legend
    plt.show()


    return data_2_postiptilt, postiptilt_2_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Development script for FIRST Photonic Lantern data analysis.",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Add positional argument for files
    parser.add_argument('files', nargs='*', default=['*.fits'],
                       help='FITS files to process (supports wildcards)')

    # Add optional arguments
    parser.add_argument("--object_name", 
                       help="Selection of the data by the Object name (default: no selection)")
    parser.add_argument("--modID", type=int,
                       help="Selection of the modulation pattern by user [0 == first in the list]")
    parser.add_argument("--modScale", type=int,
                       help="Selection of the modulation scale by user [0 == first in the list]")
    parser.add_argument("--wollaston", 
                       help="Wollaston status. Use IN for internal or OUT for no wollaston (default: first in the list)")
    parser.add_argument("--coupling_map", 
                       help="Force to select which coupling map file to use (default: the one in the directory)")
    parser.add_argument("--wavelength_smooth", type=int, default=1,
                       help="Smoothing factor for wavelength (default: %(default)s)")
    parser.add_argument("--save_individual_frames", action="store_true", default=True,
                       help="Save individual frames (default: %(default)s)")
    parser.add_argument("--save_individual_wavelength", action="store_true", default=False,
                       help="Save individual wavelength (default: %(default)s)")
    

    if (("VSCODE_PID" in os.environ or os.environ.get('TERM_PROGRAM') == 'vscode') or os.environ.get('SPYDER_DEBUG_FILEfile =')):
        print("Running in compiler")
        modID = None
        modScale = None
        object_name = None
        wollaston = None

        if getpass.getuser() == "slacour":
            file_patterns = "/Users/slacour/DATA/LANTERNE/20250614/preproc/firstpl_2025-06-14T01:38*fits"
            coupling_map = "/Users/slacour/DATA/LANTERNE/20250614/preproc/../couplingmaps/firstpl_2025-06-14T01:42:19_COUPLINGMAP.fits"
        if getpass.getuser() == "ehuby" :
            file_patterns = "/home/ehuby/WORK/DATA/FIRST-PL/2025-05-10/preproc/firstpl_*.fits"
            coupling_map = "/home/ehuby/WORK/DATA/FIRST-PL/2025-05-10/preproc/couplingmaps_TETCRB/"
            object_name = 'HIP81126'
    else:

        args = parser.parse_args()
        file_patterns = args.files if args.files else ['*.fits']

        wavelength_smooth = args.wavelength_smooth
        modID = args.modID
        modScale = args.modScale
        object_name = args.object_name

        save_individual_frames = args.save_individual_frames
        save_individual_wavelength = args.save_individual_wavelength

        # If the user specifies a coupling map, use it, otherwise look into the arguments
        coupling_map = args.coupling_map
        if coupling_map is None:
            coupling_map = file_patterns +['../couplingmaps/*.fits']

    dir_files="/Users/slacour/DATA/LANTERNE/20250808/preproc/"
    coupling_map = dir_files+"../couplingmaps"
    # file_patterns = "/Users/slacour/DATA/LANTERNE/20250614/preproc/firstpl_2025-06-14T01:38*fits"
    # file_patterns = "/Users/slacour/DATA/LANTERNE/20250614/preproc/firstpl_2025-06-14T01:48*fits"
    file_patterns = dir_files+"firstpl_2025-08-08T07:17:??_HIP84212_P.fits"
    # file_patterns = dir_files+"firstpl_2025-08-08T07:07:??_HIP84212_P.fits"
    files_with_dark_1, filelist_cmap = get_filelist(file_patterns, coupling_map, modID, modScale, object_name, wollaston)
    # file_patterns = "/Users/slacour/DATA/LANTERNE/20250614/preproc/firstpl_2025-06-14T01:42*fits"
    # file_patterns = "/Users/slacour/DATA/LANTERNE/20250614/preproc/firstpl_2025-06-14T01:37*fits"
    file_patterns = dir_files+"firstpl_2025-08-08T07:16:??_HIP84212_P.fits"

    files_with_dark_2, filelist_cmap = get_filelist(file_patterns, coupling_map, modID, modScale, object_name, wollaston)

    filelist_cmap=filter_couplingmapfile(filelist_cmap)
    couplingMap = basic.CouplingMap(filelist_cmap[0])
    flat = couplingMap.flat  

    datalist_10=runlib_i.extract_datacube(files_with_dark_1, Nsmooth=wavelength_smooth, Nbin=couplingMap.wavelength_bin, flat = couplingMap.flat)
    datalist_20=runlib_i.extract_datacube(files_with_dark_2, Nsmooth=wavelength_smooth, Nbin=couplingMap.wavelength_bin, flat = couplingMap.flat)

    #################################
    #%%
    datalist_1 = datalist_10
    datalist_2 = datalist_10
    datacube_1=np.concatenate([d.data for d in datalist_1])
    datacube_1=datacube_1.transpose((3,2,0,1))[:,:,0]
    datacube_2=np.concatenate([d.data for d in datalist_2])
    datacube_2=datacube_2.transpose((3,2,0,1))[:,:,0]

    xmod_1=datalist_1[0].xmod
    ymod_1=datalist_1[0].ymod
    xmod_2=datalist_2[0].xmod
    ymod_2=datalist_2[0].ymod

    xy_1=(xmod_1, ymod_1*1e3)
    xy_2=(xmod_2, ymod_2*1e3)
    xy_1=(xmod_1*1.02, ymod_1*1e3)
    xy_2=(xmod_2*1.02, ymod_2*1e3)

    # Stack coordinates for efficient search
    xy_1_points = np.column_stack(xy_1)
    xy_2_points = np.column_stack(xy_2)

    # Build KDTree for xy_2
    tree = cKDTree(xy_2_points)
    singular_total = []

    ik_range= range(2, 15)  # Range of k for nearest neighbors
    for ik in ik_range:
        # Query for the 2 nearest neighbors for each point in xy_1
        distances, indices = tree.query(xy_1_points, k=ik)

        # indices: shape (len(xy_1[0]), 2), each row contains indices of the 2 closest points in xy_2
        # distances: corresponding distances
        singular = []
        # Example: print the closest pairs
        for i, (idxs, dists) in enumerate(zip(indices[:,1:], distances)):
            # Ensure idxs is always an array for consistent indexing
            if np.isscalar(idxs):
                idxs = np.array([idxs])
                dists = np.array([dists])
            # print(f"xy_1[{i}] closest xy_2 indices: {idxs}, distances: {dists}")
            # a=datacube_2[:,:,idxs[0]].T.ravel()
            # b=datacube_2[:,:,idxs[1]].T.ravel()
            c=datacube_1[:,:,i].ravel()
            # Fit a and b together on c: c = p0 * a + p1 * b
            A = datacube_2[:,:,idxs].reshape((-1,len(idxs)))
            # A[:,1:2]=0
            # A[:,1:2]=0
            # Orthogonalize all columns of A to column 0
            A_ortho = A.copy()
            col0 = A_ortho[:, 0]
            for j in range(1, A_ortho.shape[1]):
                proj = np.dot(A_ortho[:, j], col0) / np.dot(col0, col0)
                A_ortho[:, j] = A_ortho[:, j] - proj * col0
            A = A_ortho
            params = np.dot(np.linalg.pinv(A),c)
            fit_values = params  # [p0, p1]
            fit_residuals = c - (np.dot(A,params))
            if dists[0] > 3:
                fit_residuals*=0
            res_norm=np.linalg.norm(fit_residuals)
            c_norm = np.linalg.norm(c)
            # Store params, res_norm, and c_norm into an array
            if i == 0:
                fit_results = np.zeros((len(xy_1_points), 2+len(idxs)))
                fit_residuals_arr = np.zeros((len(xy_1_points), len(c)))  # Store all fit_residuals
            fit_results[i, :len(idxs)] = fit_values
            fit_results[i, -2] = res_norm
            fit_results[i, -1] = c_norm
            fit_residuals_arr[i] = fit_residuals
            # U,s,Vh=np.linalg.svd(A,full_matrices=False)
            singular+=[params]
        singular_total+=[np.array(singular)]

        res_max=fit_results[:, -2].max()
        # Plot xy_1_points with c_norm as color
        plt.figure(num=len(idxs),figsize=(8, 6), clear=True)
        scatter = plt.scatter(xy_1_points[:, 0], xy_1_points[:, 1], c=fit_results[:, -2], cmap='viridis', s=40)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(str(res_max))
        cbar = plt.colorbar(scatter)
        cbar.set_label('c_norm')
        plt.tight_layout()
        plt.show()
        # Store res_max for each ik
        if ik == ik_range[0]:
            res_max_list = [fit_results[:, -1]]
        res_max_list.append(fit_results[:, -2])

    res_max_list=np.array(res_max_list)
    plt.figure(num=70,figsize=(6, 4))
    plt.plot(np.append(0,ik_range), res_max_list.mean(axis=1), marker='o')
    plt.plot(np.append(0,ik_range), res_max_list.max(axis=1), marker='.')
    plt.xlabel('ik')
    plt.ylabel('res_max')
    plt.yscale('log')
    plt.title('res_max vs ik')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# %%

res_max_arg=res_max_list[3].argmax()-3
# res_max_arg=524
# res_max_arg=525

poly_order=[1,2,3]
indices = [res_max_arg+2, res_max_arg-3, res_max_arg-2]
for ik in poly_order:
    # Query for the 2 nearest neighbors for each point in xy_1
    # distances, indices = tree.query(xy_1_points, k=ik+1)
    i = res_max_arg
    idxs = indices[:ik]
    if np.isscalar(idxs):
        idxs = np.array([idxs])
    c=datacube_1[:,:,i].transpose((1,0)).ravel()
    # Fit a and b together on c: c = p0 * a + p1 * b
    A = datacube_2[:,:,idxs].transpose((1,0,2)).reshape((-1,len(idxs)))
    params = np.dot(np.linalg.pinv(A),c)
    print("params", params)
    fit_values = np.dot(A,params)  # [p0, p1]}
    fit_residuals = c - fit_values
    
    # Store fit_values for each ik
    if ik == poly_order[0]:
        fit_values_arr = np.zeros((len(poly_order), len(c)))
        fit_residuals_arr = np.zeros((len(poly_order), len(c)))
    fit_values_arr[poly_order.index(ik)] = fit_values
    fit_residuals_arr[poly_order.index(ik)] = fit_residuals

# Plot the fit_values for each ik and residuals below
fig, (ax1, ax2) = plt.subplots(2, 1, num = 75, figsize=(10, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})

# Top plot: fit values
ax1.plot(c, label='data', color='k')
for idx, ik in enumerate(poly_order):
    ax1.plot(fit_values_arr[idx], "--", label=f'N neighbours={ik}')
ax1.set_ylabel('Fit Value')
ax1.set_title('Fit Values with different number of neighbors')
ax1.legend()
ax1.grid(True)

# Bottom plot: residuals
for idx, ik in enumerate(poly_order):
    ax2.plot(fit_residuals_arr[idx], label=f'N neighbours={ik}')
ax2.set_xlabel('Pixel Index')
ax2.set_ylabel('Residual')
ax2.set_title('Residuals')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()

# fig.savefig("fit_values_residuals_zoom.png")

# %%

y_target = c 
Y = A.T

# Matrice de Gram et vecteur k, et scalaire s:
G = Y @ Y.T     
k = Y @ y_target               
s = float(y_target @ y_target)

x_nodes = xy_1_points[idxs, 0]

Phi = np.column_stack([
    x_nodes**2, x_nodes,  np.ones_like(x_nodes)
])  # shape (6,6)

M = np.linalg.inv(Phi.T)

xmin, xmax = x_nodes.min(), x_nodes.max()
dx = xmax - xmin

box_pad = 0.5
starts_grid = 10

xmin -= box_pad * dx; xmax += box_pad * dx
bounds = (xmin, xmax)
xs = np.linspace(xmin, xmax, starts_grid)
starts = xs


def _phi_x(x):
    return np.array([x*x, x, 1.0], dtype=float)

def _dphi_x(x):
    return np.array([2*x, 1.0, 0.0], dtype=float)

def fun(z):
    x = float(z)
    phi = _phi_x(x)
    ell = M @ phi
    return s - 2.0*(ell @ k) + ell @ (G @ ell)


def jac(z):
    x = z
    dphix = _dphi_x(x)
    ex = M @ dphix
    # On évite de recalculer ell via fun pour rester auto-contenu
    phi = _phi_x(x)
    ell = M @ phi
    dJx = -2.0*(ex @ k) + 2.0*(ex @ (G @ ell))
    return np.array(dJx)

    # multi-start sur une grille
max_iter=200
tol=1e-10
verbose = True

for x0 in starts:

    res = minimize(fun, x0=x0, method="L-BFGS-B",
                    options={"maxiter": max_iter, "ftol": tol, "gtol": tol})

    print("Result for start ", x0, " : ", res.x)

# phi   = np.array([x*x, x*y, y*y, x, y, 1.0], dtype=float)
# dphix = np.array([2*x,  y , 0.0, 1.0, 0.0, 0.0], dtype=float)
# dphiy = np.array([0.0,  x , 2*y, 0.0, 1.0, 0.0], dtype=float)


# dphi_x = np.column_stack([
#     2*x_nodes, np.ones_like(x_nodes), np.zeros_like(x_nodes)
# ])  # shape (6,2)


# ell   = M @ phi
# ex    = M @ dphix
# ey    = M @ dphiy

# J = s - 

# %%
