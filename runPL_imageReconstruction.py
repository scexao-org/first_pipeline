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

def filter_filelist(filelist,coupling_map):
    """
    Filters the input file list to separate coupling map files and dark files based on FITS keywords.
    Raises an error if no valid files are found.
    Returns a dictionary mapping coupling map files to their closest dark files.
    """

    # Use the function to clean the filelist
    fits_keywords = {'DATA-CAT': ['PREPROC'],
                    'DATA-TYP': ['OBJECT','TEST']}
    filelist_data = runlib.clean_filelist(fits_keywords, filelist)
    print("runPL object filelist : ", filelist_data)

    fits_keywords = {'DATA-CAT': ['PREPROC'],
                    'DATA-TYP': ['DARK']}
    filelist_dark = runlib.clean_filelist(fits_keywords, filelist)
    print("runPL dark filelist : ", filelist_dark)

    fits_keywords = {'DATA-CAT': ['COUPLINGMAP']}

    filelist_cmap = runlib.clean_filelist(fits_keywords, coupling_map)
    print("runPL object filelist : ", filelist_cmap)

    # raise an error if filelist_cleaned is empty
    if len(filelist_data) == 0:
        raise ValueError("No good file to process")

    # raise an error if filelist_cleaned is empty
    if len(filelist_cmap) == 0:
        raise ValueError("No coupling map to use.\n Please specify which one to use with the option --coupling_map")

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

    wavelength_smooth = 1

    # Add options for these values
    parser.add_option("--coupling_map", type="string", default=None,
                    help="Force to select which coupling map file to use (default: the one in the directory)")
    parser.add_option("--wavelength_smooth", type="int", default=wavelength_smooth,
                    help="smoothing factor for wavelength (default: %default)")

    if "VSCODE_PID" in os.environ or os.environ.get('TERM_PROGRAM') == 'vscode':
        if getpass.getuser() == "slacour":
            file_patterns = "/Users/slacour/DATA/LANTERNE/Mai/preproc2/firstpl_2025-05-06T09:52:?1_BETUMA.fits"
            coupling_map = "/Users/slacour/DATA/LANTERNE/Mai/preproc2/couplingmaps"
    else:

        (options, args) = parser.parse_args()
        file_patterns=args if args else ['*.fits']

        wavelength_smooth=options.wavelength_smooth
        # If the user specifies a coupling map, use it, otherwise look into the arguments
        coupling_map = options.coupling_map
        if coupling_map is None:
            coupling_map = file_patterns


    filelist=runlib.get_filelist( file_patterns )
    filelist_pixelmap=runlib.get_filelist(coupling_map)

    files_with_dark,filelist_cmap = filter_filelist(filelist,filelist_pixelmap)

    cmap_file=fits.open(filelist_cmap[0])
    header = cmap_file[0].header
    flux_2_data=cmap_file['F2DATA'].data
    data_2_flux=cmap_file['DATA2F'].data
    fluxtiptilt_2_data=cmap_file['FTT2DATA'].data
    data_2_fluxtiptilt=cmap_file['DATA2FTT'].data
    xpos=cmap_file['POSITIONS'].data.field('X_POS')
    ypos=cmap_file['POSITIONS'].data.field('Y_POS')
    Npositions=cmap_file['POSITIONS'].header['NAXIS2']
    Ntriangles=cmap_file['TRIANGLES'].header['NAXIS2']
    cmap_file.close()

    wavelength_bin = header['WL_BIN']

    try:
        files_with_dark.pop('/Users/slacour/DATA/LANTERNE/Optim_maps/November2024/preproc/firstpl_2025-01-14T15:34:08_NONAME.fits')

        for _ in range(7):
            files_with_dark.pop(next(iter(files_with_dark)))

        # closest_dark_files.pop(next(reversed(closest_dark_files)))
        # closest_dark_files.pop(next(reversed(closest_dark_files)))
        # closest_dark_files.pop(next(reversed(closest_dark_files)))
    except:
        pass

    #Input preproc
    #clean and sum all data

    datalist=runlib_i.extract_datacube(files_with_dark,wavelength_smooth,Nbin=wavelength_bin)
    #datacube (625, 38, 100)
    #select only the data in datalist which has the same modulation pattern
    Nmod = datalist[0].Nmod
    datalist = [d for d in datalist if d.Nmod == Nmod]

    datacube=np.concatenate([d.data for d in datalist])
    datacube=datacube.transpose((3,2,0,1))

    xmod=datalist[0].xmod
    ymod=datalist[0].ymod

    quick_fits(datacube, 'datacube')

    Nwave=datalist[0].Nwave
    Noutput=datalist[0].Noutput
    Ncube=len(datalist)

    # Convert arg_model values into 2D indices of size cmap_size
    chi2_min,chi2_max,arg_triangle=runlib_i.get_chi2_maps(datacube,fluxtiptilt_2_data,data_2_fluxtiptilt)

    flux_thresold=np.percentile(datacube.mean(axis=(0,1)),80)/5
    flux_goodData=datacube.mean(axis=(0,1)) > flux_thresold
    chi2_delta=chi2_min/chi2_max
    percents=np.nanpercentile(chi2_delta[flux_goodData],[16,50,84])
    chi2_threshold=percents[1]+(percents[2]-percents[0])*3/2

    chi2_goodData = (chi2_delta < chi2_threshold)&flux_goodData

    # getting the residual after substracting a point source
    # using the flux tip tilt matrices
    residual = datacube.copy()
    fft_fit = np.zeros((Nwave,3,Ncube,Nmod))
    for c in range(Ncube):
        for m in range(Nmod):
            i = arg_triangle[c,m]
            fft = np.matmul(data_2_fluxtiptilt[i],datacube[:,:,c,m,None]) #flux tip tilt
            fft_fit[:,:,c,m] = fft[:,:,0]
            residual[:,:,c,m] -= np.matmul(fluxtiptilt_2_data[i],fft)[:,:,0]

    datacube_cleaned = datacube.copy()
    datacube_cleaned[:,:,~chi2_goodData]=0
    residual[:,:,~chi2_goodData]=0

    
    xmin, xmax   = np.min(xpos)-np.max(xmod), np.max(xpos)-np.min(xmod)
    ymin, ymax   = np.min(ypos)-np.max(ymod), np.max(ypos)-np.min(ymod)
    # Define the grid for interpolation
    grid_x, grid_y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]  # 500x500 grid

    # Interpolate the fluxes onto the grid
    fluxes = np.matmul(data_2_flux, datacube_cleaned.reshape((Nwave,Noutput,Ncube*Nmod)))
    fluxes = fluxes.reshape((Nwave,Npositions,Ncube,Nmod))
    flux_maps = []
    for c in tqdm(range(Ncube)):
        for m in range(Nmod):
            for w in range(Nwave):
                # Interpolate the fluxes onto the grid
                flux_map = griddata((xpos-xmod[m], ypos-ymod[m]), fluxes[w,:,c,m], (grid_x, grid_y), method='cubic')
                flux_maps += [flux_map]
    flux_maps = np.array(flux_maps).reshape((Ncube,Nmod,Nwave,100,100))
    flux_maps_sum = np.nansum(flux_maps,axis=1)

    # Interpolate residuals onto the grid
    fluxes = np.matmul(data_2_flux, residual.reshape((Nwave,Noutput,Ncube*Nmod)))
    fluxes = fluxes.reshape((Nwave,Npositions,Ncube,Nmod))
    residuals_maps = []
    for c in tqdm(range(Ncube)):
        for m in range(Nmod):
            for w in range(Nwave):
                # Interpolate the fluxes onto the grid
                flux_map = griddata((xpos-xmod[m], ypos-ymod[m]), fluxes[w,:,c,m], (grid_x, grid_y), method='cubic')
                residuals_maps += [flux_map]
    residuals_maps = np.array(residuals_maps).reshape((Ncube,Nmod,Nwave,100,100))
    residuals_maps_sum = np.nansum(residuals_maps,axis=1)

    # Save image_2d and residual_2d to a FITS file

    for i,d in enumerate(datalist):
        header = d.header


        # Create a primary HDU with no data, just the header
        hdu_primary = fits.PrimaryHDU(flux_maps_sum[i])
        hdu_residual = fits.ImageHDU(residuals_maps_sum[i], name="RESIDUAL")

        header['DATA-CAT'] = 'IMAGE'
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
        hdul = fits.HDUList([hdu_primary, hdu_residual])

        output_filename = os.path.join(output_dir, runlib.create_output_filename(header))

        # Write to a FITS file
        hdul.writeto(output_filename, overwrite=True)
        print(f"Image saved to {output_filename}")

















    #%% now just images and plots to be saved for information

    image_2d_T = image_2d.transpose(3, 2, 0,1)
    quick_fits(image_2d_T, "transposed")
    residual_2d_T = residual_2d.transpose(3, 2, 0,1)
    quick_fits(residual_2d_T, "transposed residual")


    # Plot all the images in a single figure

    fig, axes = plt.subplots(2, len(images_broad), figsize=(15, 6), squeeze=False)

    # Normalize color scale across all images
    vmin = 0
    vmax = max(images_broad.max(), residual_broad.max())/10

    # Plot images_broad in the first row
    for i, img in enumerate(images_broad):
        #i is the image number, img is the image 49x49
        ax = axes[0, i]
        im = ax.imshow(img, vmin=vmin, vmax=vmax, cmap='viridis')
        ax.set_title(f"Image {i+1}")
        ax.axis('off')

    # Plot residual_broad in the second row
    for i, res in enumerate(residual_broad):
        ax = axes[1, i]
        im = ax.imshow(res, vmin=vmin, vmax=vmax, cmap='viridis')
        ax.set_title(f"Residual {i+1}")
        ax.axis('off')

    # Add a colorbar
    # fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.02, pad=0.04)


    #with interpolation 
    data_2_postiptilt, postiptilt_2_data = interpolate_halpha(data_2_postiptilt, postiptilt_2_data)


    residual = datacube.copy()
    fft_fit = np.zeros((Nwave,3,Ncube,Npos))
    for c in range(Ncube):
        for p in range(Npos):
            i = arg_model[c,p]
            fft = np.matmul(data_2_postiptilt[i],datacube[:,:,c,p,None]) #flux tip tilt
            fft_fit[:,:,c,p] = fft[:,:,0]
            residual[:,:,c,p] -= np.matmul(postiptilt_2_data[i],fft)[:,:,0]

    datacube_cleaned = datacube.copy()
    datacube_cleaned[:,:,~chi2_goodData]=0
    residual[:,:,~chi2_goodData]=0

    image = np.matmul(data_2_flux, datacube_cleaned.reshape((Nwave,Noutput,Ncube*Npos)))
    image = image.reshape((Nwave,Nmodel,Ncube,Npos)).transpose((3,1,2,0))
    image_2d= runlib_i.resize_and_shift(image,masque, dither_x, dither_y).sum(axis=0)
    images_broad=image_2d.sum(axis=3).transpose((2,0,1))

    image_residual = np.matmul(data_2_flux, residual.reshape((Nwave,Noutput,Ncube*Npos)))
    image_residual = image_residual.reshape((Nwave,Nmodel,Ncube,Npos)).transpose((3,1,2,0))
    residual_2d= runlib_i.resize_and_shift(image_residual,masque, dither_x, dither_y).sum(axis=0)
    residual_broad=residual_2d.sum(axis=3).transpose((2,0,1))



    image_2d_T = image_2d.transpose(3, 2, 0,1)
    quick_fits(image_2d_T, "transposed")
    residual_2d_T = residual_2d.transpose(3, 2, 0,1)
    quick_fits(residual_2d_T, "transposed residual")


    # Plot all the images in a single figure

    fig, axes = plt.subplots(2, len(images_broad), figsize=(15, 6), squeeze=False)

    # Normalize color scale across all images
    vmin = 0
    vmax = max(images_broad.max(), residual_broad.max())/10

    # Plot images_broad in the first row
    for i, img in enumerate(images_broad):
        #i is the image number, img is the image 49x49
        ax = axes[0, i]
        im = ax.imshow(img, vmin=vmin, vmax=vmax, cmap='viridis')
        ax.set_title(f"Image {i+1} with interpolated data")
        ax.axis('off')

    # Plot residual_broad in the second row
    for i, res in enumerate(residual_broad):
        ax = axes[1, i]
        im = ax.imshow(res, vmin=vmin, vmax=vmax, cmap='viridis')
        ax.set_title(f"Residual {i+1} with interpolated data")
        ax.axis('off')


    plt.tight_layout()
    plt.show()

    runlib_i.save_all_as_PDF(output_dir = output_dir)

