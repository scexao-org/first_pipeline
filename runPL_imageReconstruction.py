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
from optparse import OptionParser
import numpy as np
from scipy.signal import correlate
from scipy import linalg

from scipy.interpolate import griddata

import getpass
import matplotlib
if "VSCODE_PID" in os.environ or os.environ.get('TERM_PROGRAM') == 'vscode':
    matplotlib.use('Qt5Agg')
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
import runPL_library_io as runlib
import runPL_library_imaging as runlib_i
import runPL_library_basic as basic
from scipy.ndimage import zoom
from astropy.io import fits
import shutil
from scipy.interpolate import interpn

plt.ion()

DEBUG = False

# Add options
usage = """
    usage:  %prog [options] files.fits

    Goal: Compare different coupling maps and make a movie of the correlation between them. Also, plot the deconvolved images.

    It will get as input a list of files with DPR_CATG=CMAP and DPR_TYPE=PREPROC keywords. 
    On those, it will find which ones have the keyword DPR_OPT=DARK and which ones have nothing for DPR_OPT.
    It will read the files which have nothing in the DPR_OPT keyword, and it will subtract from them the files which have the DARK keyword.
    
    Example:
    runPL_compareCouplingMaps.py --cmap_size=25 *.fits

    Options:
    --cmap_size: Width of cmap size, in pixels (default: 25)
"""

def filter_filelist(filelist,coupling_map,modID):
    """
    Filters the input file list to separate coupling map files and dark files based on FITS keywords.
    Raises an error if no valid files are found.
    Returns a dictionary mapping coupling map files to their closest dark files.
    """

    # Use the function to clean the filelist
    if modID == 0:
        fits_keywords = {'X_FIRTYP': ['PREPROC'],
                        'DATA-TYP': ['OBJECT','TEST']}
    else:
        fits_keywords = {'X_FIRTYP': ['PREPROC'],
                        'DATA-TYP': ['OBJECT','TEST'],
                        'MOD_ID': [modID]}
    filelist_data = runlib.clean_filelist(fits_keywords, filelist)
    print("runPL object filelist : ", filelist_data)

    fits_keywords = {'X_FIRTYP': ['PREPROC'],
                    'DATA-TYP': ['DARK']}
    filelist_dark = runlib.clean_filelist(fits_keywords, filelist)
    print("runPL dark filelist : ", filelist_dark)

    fits_keywords = {'X_FIRTYP': ['COUPLINGMAP']}

    filelist_cmap = runlib.clean_filelist(fits_keywords, coupling_map)
    print("runPL object filelist : ", filelist_cmap)

    # raise an error if filelist_cleaned is empty
    if len(filelist_data) == 0:
        raise FileNotFoundError("No good file to process")

    # raise an error if filelist_cleaned is empty
    if len(filelist_cmap) == 0:
        raise FileNotFoundError("No coupling map to use.\n Please specify which one to use with the option --coupling_map")

    # raise an error if filelist_cleaned is more than one
    if len(filelist_cmap) > 1:
        raise ValueError("Two coupling maps to use! I can only use one.\n Please specify which one to use with the option --coupling_map")

    # Check if all files have the same value for header['PM_CHECK']
    pm_check_values = set()
    combined_filelist = []
    combined_filelist.extend(filelist_data)
    combined_filelist.extend(filelist_dark)
    combined_filelist.extend(filelist_cmap)
    for file in combined_filelist:
        header = fits.getheader(file)
        pm_check_values.add(int(header.get('PM_CHECK', 0)))
        
    if len(pm_check_values) > 1:
        print("WARNING: The 'PM_CHECK' values (ie, the pixel map used to preprocess the files) \n are not consistent across all files!")
        print(f"Found values: {pm_check_values}")

    # for each file in filelist_cmap find the closest dark file in filelist_dark with, by priority, first the directory in which the file is, and then by the date in the "DATE" fits keyword, and second, the directory in which the file is

    files_with_dark = {cmap: runlib.find_closest_dark(cmap, filelist_dark) for cmap in filelist_data}

    return files_with_dark,filelist_cmap



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
    parser = OptionParser(usage)

    # default values
    wavelength_smooth = 1
    modID = 0
    save_individual_frames = True
    save_individual_wavelength = True


    # Add options for these values
    parser.add_option("--modID", type="int", default=modID,
                      help="Selection of the modulation pattern by user [0 == first in the list of file] (default: %default)")
    parser.add_option("--coupling_map", type="string", default=None,
                    help="Force to select which coupling map file to use (default: the one in the directory)")
    parser.add_option("--wavelength_smooth", type="int", default=wavelength_smooth,
                    help="smoothing factor for wavelength (default: %default)")
    parser.add_option("--save_individual_frames", action="store_true", default=save_individual_frames,
                    help="Save individual frames (default: %default)")
    parser.add_option("--save_individual_wavelength", action="store_true", default=save_individual_wavelength,
                    help="Save individual wavelength (default: %default)")
    

    if "VSCODE_PID" in os.environ or os.environ.get('TERM_PROGRAM') == 'vscode':
        if getpass.getuser() == "slacour":
            file_patterns = "/Users/slacour/DATA/LANTERNE/Mai3/preproc/firstpl_2025-05-09T03:26:36_BETUMA.fits"
            coupling_map = "/Users/slacour/DATA/LANTERNE/Mai/preproc2/couplingmaps"
    else:

        (options, args) = parser.parse_args()
        file_patterns=args if args else ['*.fits','preproc/*.fits','preproc/couplingmaps/*.fits']

        wavelength_smooth=options.wavelength_smooth
        modID=options.modID
        save_individual_frames=options.save_individual_frames
        save_individual_wavelength=options.save_individual_wavelength

        # If the user specifies a coupling map, use it, otherwise look into the arguments
        coupling_map = options.coupling_map
        if coupling_map is None:
            coupling_map = ['./preproc/couplingmaps/*.fits']#file_patterns


    filelist=runlib.get_filelist( file_patterns )
    filelist_pixelmap=runlib.get_filelist(coupling_map)

    files_with_dark,filelist_cmap = filter_filelist(filelist,filelist_pixelmap,modID)


    couplingMap = basic.CouplingMap(filelist_cmap[0])

    #Input preproc
    #clean and sum all data

    datalist=runlib_i.extract_datacube(files_with_dark,wavelength_smooth,Nbin=couplingMap.wavelength_bin)

    #datacube (625, 38, 100)
    #select only the data in datalist which has the same modulation pattern
    if modID == 0:
        modID = datalist[0].modID
        datalist = [d for d in datalist if d.modID == modID]
        if len(datalist) == 0:
            print("No data with the selected modulation pattern")
            exit()

    datacube=np.concatenate([d.data for d in datalist])
    datacube=datacube.transpose((3,2,0,1))

    xmod=datalist[0].xmod
    ymod=datalist[0].ymod

    quick_fits(datacube, 'datacube')

    Nwave=datalist[0].Nwave
    Noutput=datalist[0].Noutput
    Ncube=len(datalist)

    # Convert arg_model values into 2D indices of size cmap_size
    # Utilise pour selectionner les bonnes images 
    # ===> Pas utile pour le quick look

    datacube_cleaned,arg_triangle = runlib_i.chi2_cleaning(datacube,couplingMap)

    # getting the residual after substracting a point source
    # using the flux tip tilt matrices
    # Utilise pour soustraire une source ponctuelle 
    # ===> Pas utile pour le quick look

    residual, fft_fit = basic.make_image_source_removal(datacube,arg_triangle,couplingMap)

    # Define the grid for interpolation
    # calcul de la grille de l'image que l'on souhaite reconstruire
    # if it is for a quick look of the real time display, use xmod=ymod=0
    Npixel = 100
    grid_x, grid_y = basic.make_image_grid(couplingMap, Npixel, xmod, ymod)

    # create the image maps
    flux_maps, fluxes = basic.make_image_maps(datacube_cleaned, couplingMap, grid_x, grid_y, xmod, ymod, wavelength=False)
    residuals_maps, fluxes_residuals = basic.make_image_maps(residual, couplingMap, grid_x, grid_y, xmod, ymod, wavelength=False)

    flux_maps_sum = np.nansum(flux_maps,axis=1)
    residuals_maps_sum = np.nansum(residuals_maps,axis=1)
    # Save image and residual maps to FITS files :


    # Coupling maps for inspection
    couplingmaps = np.mean(datacube, axis=(0,1))
    # Define the grid for interpolation
    grid_x, grid_y = np.mgrid[np.min(xmod):np.max(xmod):Npixel*1j, np.min(ymod):np.max(ymod):Npixel*1j]  # 500x500 grid
    # Interpolate the fluxes onto the grid
    couplingmaps_interp= np.zeros((len(couplingmaps), Npixel, Npixel))
    for i,fm in enumerate(couplingmaps):
        couplingmaps_interp[i] = griddata((xmod, ymod), fm, (grid_x, grid_y), method='cubic').T
    

    for i,d in enumerate(datalist):
        header = d.header
        header['X_FIRTYP'] = 'IMAGE'

        list_of_hdus = []
        # Create a primary HDU with the data
        hdu_primary = fits.PrimaryHDU(flux_maps_sum[i,0])
        hdu_residual = fits.ImageHDU(residuals_maps_sum[i,0], name="RESIDUAL")
        list_of_hdus += [hdu_primary, hdu_residual]

        # Create a primary HDU with no data, just the header
        if save_individual_frames:
            hdu_frame = fits.ImageHDU(flux_maps[i,:,0], name="FRAMES")
            hdu_frame_residual = fits.ImageHDU(residuals_maps[i,:,0], name="FRAMES_RESIDUAL")
            list_of_hdus += [hdu_frame, hdu_frame_residual]

        if save_individual_wavelength:
            flux_maps_wave, fluxes = basic.make_image_maps(datacube_cleaned[:,:,i,None], couplingMap, grid_x, grid_y, xmod, ymod, wavelength=True)
            flux_maps_wave = np.nansum(flux_maps_wave[0],axis=0)
            residuals_maps_wave, fluxes_residuals = basic.make_image_maps(residual[:,:,i,None], couplingMap, grid_x, grid_y, xmod, ymod, wavelength=True)
            residuals_maps_wave = np.nansum(residuals_maps_wave[0],axis=0)

            hdu_wave = fits.ImageHDU(flux_maps_wave, name="3D_IMAGE")
            hdu_wave_residual = fits.ImageHDU(residuals_maps_wave, name="3D_IMAGE_RESIDUAL")
            list_of_hdus += [hdu_wave, hdu_wave_residual]
            header['X_FIRTYP'] = 'WDIMAGE'

        hdu_coupling = fits.ImageHDU(couplingmaps_interp[i], name="COUPLING")
        list_of_hdus += [hdu_coupling]

        # Add date and time to the header
        current_time = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
        header['DATE-PRO'] = current_time

        # Add input parameters to the header
        header['WLSMOOTH'] = wavelength_smooth  # Add wavelength smoothing factor

        # Définir le chemin complet du sous-dossier "images"
        output_dir = os.path.join(d.dirname,"images")

        #if os.path.exists(output_dir) and os.path.isdir(output_dir):
        #    shutil.rmtree(output_dir)

        # Créer les dossiers "output" et "pixel" s'ils n'existent pas déjà
        os.makedirs(output_dir, exist_ok=True)

        hdu_primary.header.extend(header, strip=True)

        # Combine all HDUs into an HDUList
        hdul = fits.HDUList(list_of_hdus)

        output_filename = os.path.join(output_dir, runlib.create_output_filename(header))

        # Write to a FITS file
        hdul.writeto(output_filename, overwrite=True)
        print(f"Image saved to {output_filename}")









# %%
