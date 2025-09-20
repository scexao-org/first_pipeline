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
import runPL_library_io as runlib
import runPL_library_imaging as runlib_i
import runPL_library_basic as basic
from scipy.ndimage import zoom
from astropy.io import fits
import shutil
from scipy.interpolate import interpn
from astroplan import Observer
from astropy.time import Time
subaru = Observer.at_site("Subaru")
now_time = Time.now()
if subaru.is_night(now_time):
    print("It's night at Subaru Observatory.")
else:
    print("It's day at Subaru Observatory.")

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


def get_filelist(file_patterns, dark_patterns, cmap_patterns, modID, modScale, object_name, wollaston):

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

        # Adding new constraints if not asked by user
        hd=fits.getheader(filelist[0])
        modID = hd.get('X_FIRMID', 0)
        modScale = hd.get('X_FIRMSC', 0)
        object_name = hd.get('OBJECT', 'NONE')
        wollaston = hd.get('X_FIRWOL', 'IN')
        fits_keywords['OBJECT'] = [object_name]
        fits_keywords['X_FIRMID'] = [modID]
        fits_keywords['X_FIRMSC'] = [modScale]
        # fits_keywords['X_FIRWOL'] = [wollaston] # to be added later

        print("----------------")
        print(f"Selected object='{object_name}' with modScale={modScale} and modID={modID}")
        print("----------------")

        filelist = runlib.get_filelist(file_patterns, fits_keywords)

        fits_keywords = {'X_FIRTYP': ['PREPROC'],
                        'DATA-TYP': ['DARK'],
                        'X_FIRWOL': [wollaston],
                        }    
        
        try:
            filelist_dark = runlib.get_filelist(dark_patterns, fits_keywords, name_search="dark")
        except FileNotFoundError as e:
            print(f"NO DARKS: {e}")
            filelist_dark = []

        if len(filelist_dark) == 0:
            print("WARNING: No dark files found with the specified patterns and keywords.")

        fits_keywords = {'X_FIRTYP': ['COUPLINGMAP'],
                         'X_FIRWOL': [wollaston],
                        }    

        filelist_cmap = runlib.get_filelist(cmap_patterns, fits_keywords, name_search="coupling map")
        filelist_cmap  = filter_couplingmapfile(filelist_cmap)

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
    parser = OptionParser(usage)

    # default values
    wavelength_smooth = 1
    save_individual_frames = True
    save_individual_wavelength = False

    # Add options for these values
    parser.add_option("--object_name", type="string", 
                    help="Selection of the data by the Object name (default: %default -- no selection)")
    parser.add_option("--modID", type="int", 
                      help="Selection of the modulation pattern by user [0 == first in the list] (default: %default)")
    parser.add_option("--modScale", type="int", 
                      help="Selection of the modulation pattern by user [0 == first in the list] (default: %default)")
    parser.add_option("--wollaston", type="string", 
                      help="Wollaston status. Use IN for internal or OUT for no wollaston (default: first in the list)")
    parser.add_option("--coupling_map", type="string", 
                    help="Force to select which coupling map file to use (default: the one in the directory)")
    parser.add_option("--dark_files", type="string", 
                help="Select one or more specific dark(s) files to use")
    parser.add_option("--wavelength_smooth", type="int", default=wavelength_smooth,
                    help="smoothing factor for wavelength (default: %default)")
    parser.add_option("--save_individual_frames", action="store_true", default=save_individual_frames,
                    help="Save individual frames (default: %default)")
    parser.add_option("--save_individual_wavelength", action="store_true", default=save_individual_wavelength,
                    help="Save individual wavelength (default: %default)")
    

    if (("VSCODE_PID" in os.environ or os.environ.get('TERM_PROGRAM') == 'vscode') or os.environ.get('SPYDER_DEBUG_FILEfile =')):
        print("Running in compiler")
        modID = None
        modScale = None
        object_name = None
        wollaston = None
        dark_patterns = None

        if getpass.getuser() == "slacour":
            # file_patterns = "/Users/slacour/DATA/LANTERNE/20250614/preproc/firstpl_2025-06-14T01:38*fits"
            file_patterns = "/Users/slacour/DATA/LANTERNE/20250510/preproc/*10T05?5*BETACMI_P.fits"
            file_patterns = "/Users/slacour/DATA/LANTERNE/20250510/preproc/*10T05?53*BETACMI_P.fits"
            file_patterns = "/Users/slacour/DATA/LANTERNE/20250510/preproc/*10T09?2[0-2]*TETCRB_P.fits"
            cmap_patterns = "/Users/slacour/DATA/LANTERNE/20250510/preproc/../couplingmaps/firstpl_2025-05-10T09:23:36_COUPLINGMAP.fits"
            dark_patterns = "/Users/slacour/DATA/LANTERNE/20250808/preproc"
        if getpass.getuser() == "ehuby" :
            file_patterns = "/home/ehuby/WORK/DATA/FIRST-PL/2025-05-10/preproc/firstpl_*.fits"
            cmap_patterns = "/home/ehuby/WORK/DATA/FIRST-PL/2025-05-10/preproc/couplingmaps_TETCRB/"
            object_name = 'HIP81126'
    else:

        (options, args) = parser.parse_args()
        file_patterns=args if args else ['*.fits']

        wavelength_smooth=options.wavelength_smooth
        modID=options.modID
        modScale=options.modScale
        object_name=options.object_name
        dark_patterns = options.dark_files
        wollaston = options.wollaston

        save_individual_frames=options.save_individual_frames
        save_individual_wavelength=options.save_individual_wavelength

        cmap_patterns = options.coupling_map

    # If the user specifies a coupling map, use it, otherwise use the science file pattern
    if cmap_patterns is None:
        cmap_patterns = file_patterns + ['../couplingmaps/*.fits']
    # If the user specify a dark, use it. Otherwise, use the science file pattern
    if dark_patterns is None:
        dark_patterns = file_patterns


    files_with_dark, filelist_cmap = get_filelist(file_patterns, dark_patterns, cmap_patterns, modID, modScale, object_name, wollaston)

    couplingMap = basic.CouplingMap(filelist_cmap[0])

    #%%

    #Input preproc
    #clean and sum all data

    datalist=runlib_i.extract_datacube(files_with_dark,Nsmooth=wavelength_smooth,Nbin=couplingMap.wavelength_bin,flat = couplingMap.flat)

    datacube=np.concatenate([d.data for d in datalist])
    flux = np.concatenate([np.repeat(np.expand_dims(d.get_spectra(), axis=0), len(d.data), axis=0) for d in datalist])
    # datacube=datacube.transpose((3,2,0,1))
    xmod=datalist[0].xmod
    ymod=datalist[0].ymod

    flux_2_data = couplingMap.flux_2_data * spectra.mean(axis=(0))
    Pos = couplingMap.Pos
    Npos = len(Pos)


    Ncube=datacube.shape[0]
    Nmod=datacube.shape[1]
    Noutput=datacube.shape[2]
    Nwave=datacube.shape[3]

    data_2_flux = linalg.pinv(flux_2_data.reshape(Npos, -1))
    flux = np.dot(datacube.reshape(Ncube*Nmod, -1), data_2_flux)
    residual = datacube - (flux @ flux_2_data.reshape((-1,  Noutput * Nwave))).reshape(datacube.shape)

    # Define the grid for interpolation
    # calcul de la grille de l'image que l'on souhaite reconstruire
    # if it is for a quick look of the real time display, use xmod=ymod=0
    Npixel = 150
    grid_x, grid_y = basic.make_image_grid(couplingMap, Npixel, xmod, ymod)


    xpos = couplingMap.Pos[:,0]
    ypos = couplingMap.Pos[:,1]
    flux = flux.reshape((Ncube,Nmod,Npos))

    flux_maps = []
    for c in range(Ncube):
        for m in tqdm(range(Nmod), desc="Reconstruction of the 3D image"):
            # Interpolate the fluxes onto the grid
            flux_map = griddata((xpos-xmod[m], ypos-ymod[m]), flux[c,m,:], (grid_x, grid_y), method='cubic')
            flux_maps += [flux_map]
    flux_maps = np.array(flux_maps).reshape((Ncube,Nmod,Nwave,len(grid_x),len(grid_y)))
    

    # create the image maps


    Ncube=datacube.shape[0]
    Nmod=datacube.shape[1]
    Noutput=datacube.shape[2]
    Nwave=datacube.shape[3]

    xpos = couplingMap.Pos[:,0]
    ypos = couplingMap.Pos[:,1]

    Npositions = couplingMap.Npositions

    data_2_flux = couplingMap.data_2_flux

    fluxes = np.matmul(data_2_flux, datacube.reshape((Nwave,Noutput,Ncube*Nmod)))
    fluxes = fluxes.reshape((Nwave,Npositions,Ncube,Nmod))

    # developement mode, or quick look mode :
    if wavelength == False:
        fluxes = fluxes.mean(axis=0, keepdims=True)
        Nwave = 1

    flux_maps = []
    if wavelength == False:
        for c in range(Ncube):
            for m in range(Nmod):
                for w in range(Nwave):
                    # Interpolate the fluxes onto the grid
                    flux_map = griddata((xpos-xmod[m], ypos-ymod[m]), fluxes[w,:,c,m], (grid_x, grid_y), method='cubic')
                    flux_maps += [flux_map]
    else:
        for c in range(Ncube):
            for m in tqdm(range(Nmod), desc="Reconstruction of the 3D image"):
                for w in range(Nwave):
                    # Interpolate the fluxes onto the grid
                    flux_map = griddata((xpos-xmod[m], ypos-ymod[m]), fluxes[w,:,c,m], (grid_x, grid_y), method='cubic')
                    flux_maps += [flux_map]
    flux_maps = np.array(flux_maps).reshape((Ncube,Nmod,Nwave,len(grid_x),len(grid_y)))
    

    # create the image maps
    flux_maps, fluxes = basic.make_image_maps(datacube_cleaned, couplingMap, grid_x, grid_y, xmod, ymod, wavelength=False)

    #%%



    xmod=datalist[0].xmod
    ymod=datalist[0].ymod

    # quick_fits(datacube, 'datacube')

    Nwave=datalist[0].Nwave
    Noutput=datalist[0].Noutput
    Ncube=len(datalist)



    # Coupling maps for inspection
    Npixel = 100
    couplingmaps = np.mean(datacube, axis=(0,1))
    # Define the grid for interpolation
    grid_x, grid_y = np.mgrid[np.min(xmod):np.max(xmod):Npixel*1j, np.min(ymod):np.max(ymod):Npixel*1j]  # 500x500 grid
    # Interpolate the fluxes onto the grid
    couplingmaps_interp= np.zeros((len(couplingmaps), Npixel, Npixel))
    for i,fm in enumerate(couplingmaps):
        couplingmaps_interp[i] = griddata((xmod, ymod), fm, (grid_x, grid_y), method='cubic').T
    
    # Convert arg_model values into 2D indices of size cmap_size
    # Utilise pour selectionner les bonnes images 
    # ===> Pas utile pour le quick look

    datacube_cleaned,arg_triangle = runlib_i.chi2_cleaning(datacube,couplingMap)

    # getting the residual after substracting a point source
    # using the flux tip tilt matrices
    # Utilise pour soustraire une source ponctuelle 
    # ===> Pas utile pour le quick look

    residual, fft_fit = basic.make_image_source_removal(datacube,arg_triangle,couplingMap, mode="crosses")

    # Define the grid for interpolation
    # calcul de la grille de l'image que l'on souhaite reconstruire
    # if it is for a quick look of the real time display, use xmod=ymod=0
    Npixel = 150
    grid_x, grid_y = basic.make_image_grid(couplingMap, Npixel, xmod, ymod)

    # create the image maps
    flux_maps, fluxes = basic.make_image_maps(datacube_cleaned, couplingMap, grid_x, grid_y, xmod, ymod, wavelength=False)
    residuals_maps, fluxes_residuals = basic.make_image_maps(residual, couplingMap, grid_x, grid_y, xmod, ymod, wavelength=False)

    flux_maps_sum = np.nansum(flux_maps,axis=1)
    residuals_maps_sum = np.nansum(residuals_maps,axis=1)
    # Save image and residual maps to FITS files :


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
        output_dir = os.path.join(d.dirname,"../images")

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
