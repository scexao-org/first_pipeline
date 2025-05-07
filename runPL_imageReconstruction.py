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
                    'DATA-TYP': ['OBJECT']}
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
    if len(filelist_dark) == 0:
        raise ValueError("No good dark to substract to cmap files")

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

    def find_closest_in_time_dark(cmap_file, dark_files):
        """
        Finds the closest dark file to a given coupling map file based on the 'DATE' FITS keyword.
        """

        cmap_date = fits.getheader(cmap_file)['DATE']
        
        # find the closest by date
        dark_dates = [(dark, fits.getheader(dark)['DATE']) for dark in dark_files]
        dark_dates.sort(key=lambda x: abs(datetime.strptime(x[1], '%Y-%m-%dT%H:%M:%S') - datetime.strptime(cmap_date, '%Y-%m-%dT%H:%M:%S')))
        
        return dark_dates[0][0]  # Return the closest dark file by date

    def find_closest_dark(cmap_file, dark_files):
        """
        Finds the closest dark file to a given coupling map file, prioritizing files in the same directory.
        """

        cmap_dir = os.path.dirname(cmap_file)
        
        # Filter dark files by the same directory
        same_dir_darks = [dark for dark in dark_files if os.path.dirname(dark) == cmap_dir]
        
        if same_dir_darks:
            return find_closest_in_time_dark(cmap_file, same_dir_darks)  # Return the first match in the same directory    
        else:
            return find_closest_in_time_dark(cmap_file, dark_files) 

    files_with_dark = {cmap: find_closest_dark(cmap, filelist_dark) for cmap in filelist_data}

    return files_with_dark,filelist_cmap


def dithering_of_image(cmap_size, step_size=1):
    """
    Generates dithering offsets for an image based on the cmap size and step size.
    Returns arrays of x and y offsets.
    """

    dither_x=[]
    dither_y=[]
    for i in range(cmap_size):
        for j in range(cmap_size):
            dither_x+=[int((i-cmap_size//2)/step_size)]
            dither_y+=[int((j-cmap_size//2)/step_size)]

    dither_x=np.array(dither_x)
    dither_y=np.array(dither_y)
    return (dither_x,dither_y)


def get_shift_between_image(projdata):
    """
    Calculates the shift between images in a dataset using 2D cross-correlation.
    Returns the x and y shifts, and the cross-correlated data.
    """

    def distance_median(dist):
        dist_mean = dist.mean(axis=1)
        dist -= np.round(dist_mean+0.001).astype(int)[:,None]
        return np.round(np.median(dist,axis=0)).astype(int)

    Nsingular=projdata.shape[0]
    Ncube=projdata.shape[1]
    Npos=projdata.shape[2]  
    cmap_size=int((Npos)**.5)

    projdata = projdata.reshape((Nsingular, Ncube, cmap_size, cmap_size))
    # Perform 2D cross-correlation along the last two dimensions
    cross_correlated_projected_data = np.zeros((Nsingular, Ncube, Ncube, cmap_size, cmap_size))

    for i in tqdm(range(Nsingular)):
        for j in range(Ncube):
            for k in range(Ncube):
                cross_correlated_projected_data[i, j, k] = correlate(projdata[i, j], projdata[i, k], mode='same')


    c=cross_correlated_projected_data
    dist=c.sum(axis=0).reshape((Ncube,Ncube,-1)).argmax(axis=2)
    dist_2d_x,dist_2d_y=np.array(np.unravel_index(dist,(cmap_size,cmap_size)))-cmap_size//2
    dist_2d_x=distance_median(dist_2d_x)
    dist_2d_y=distance_median(dist_2d_y)

    print("shift in x --> ",dist_2d_x)
    print("shift in y --> ",dist_2d_y)

    return dist_2d_x,dist_2d_y,c


def get_projection_matrice(datacube,flux_goodData,Nsingular):
    """
    Computes the projection matrix and singular values using Singular Value Decomposition (SVD).
    datacube is a flux_2_data matrix
    
        flux_2_data == projdata_2_data @ s @ flux_2_data
        data_2_projdata is the transpose of projdata_2_data

    Returns the projection matrix data_2_projdata and singular values.
    """

    Nwave=datacube.shape[0] #100
    Noutput=datacube.shape[1] #38
    Ncube=datacube.shape[2] #10
    Npos=datacube.shape[3] #625
    datacube=datacube.reshape((Nwave*Noutput,Ncube,Npos)) #reshape to (3800, 10, 625)

    pos_2_data = datacube[:,flux_goodData] #(3800, 3017) datacube is (3800, 10, 625), flux_good is (10, 625)

    U,s,Vh=linalg.svd(pos_2_data,full_matrices=False)

    #pos_2_singular = Vh[:Nsingular]*s[:Nsingular,None]
    singular_2_data = U[:,:Nsingular] #(3800, 57)
    pos_2_singular = singular_2_data.T @ datacube.reshape((Nwave*Noutput,Ncube*Npos)) #(57, 6250)

    singular_values = s #(3017,)
    pos_2_singular = pos_2_singular.reshape((Nsingular,Ncube,Npos)) #reshape to (57, 10, 625)
    singular_2_data = singular_2_data.reshape((Nwave,Noutput,Nsingular))

    return pos_2_singular,singular_values,singular_2_data

def shift_and_add(data, dist_2d_x, dist_2d_y):
    """
    Shifts and averages data cubes based on calculated offsets.
    Applies zero padding for areas with no information.
    Returns the averaged model and shifted data.
    """
    
    Nsingular=data.shape[0]
    Ncube=data.shape[1]
    Npos=data.shape[2]
    cmap_size = int(np.sqrt(Npos))
    data=data.reshape((Nsingular,Ncube,cmap_size,cmap_size))

    shifted_data = np.zeros_like(data)
    for i in range(Nsingular):
        for j in range(Ncube):
            x_offset = dist_2d_x[j]
            y_offset = dist_2d_y[j]
            shifted_data[i, j] = np.roll(data[i, j], shift=(x_offset, y_offset), axis=(0, 1))

            # Zero padding where we have no information
            if x_offset > 0:
                shifted_data[i, j, :x_offset, :] = np.nan
            elif x_offset < 0:
                shifted_data[i, j, x_offset:, :] = np.nan
            if y_offset > 0:
                shifted_data[i, j, :, :y_offset] = np.nan
            elif y_offset < 0:
                shifted_data[i, j, :, y_offset:] = np.nan

    added_data=np.nanmean(shifted_data,axis=1)

    return added_data,shifted_data

def get_postiptilt_model(projected_model,projdata_2_data):
    """
    Computes the flux and tip-tilt model from the projected data.
    Returns matrices for converting between projected data and flux/tip-tilt, and a mask.
    """

    Nsingular=projected_model.shape[0]
    cmap_size=projected_model.shape[1]
    Nwave=projdata_2_data.shape[0]
    Noutput=projdata_2_data.shape[1]

    postiptilt_model_vectors=np.zeros((cmap_size,cmap_size,Nsingular,3))
    postiptilt_masque=np.zeros((cmap_size,cmap_size),dtype=bool)
    for i in range(cmap_size-1):
        for j in range(cmap_size-1):
            postiptilt_model_vectors[i,j,:,0]=projected_model[:,i,j]
            postiptilt_model_vectors[i,j,:,1]=projected_model[:,i,j]-projected_model[:,i+1,j]
            postiptilt_model_vectors[i,j,:,2]=projected_model[:,i,j]-projected_model[:,i,j+1]
            if np.isnan(postiptilt_model_vectors[i,j]).sum() != 0:
                postiptilt_model_vectors[i,j]=0.0
            else:
                postiptilt_masque[i,j]=True

    postiptilt_model_vectors=postiptilt_model_vectors[postiptilt_masque]
    postiptilt_2_projdata= postiptilt_model_vectors

    Nmodel=len(postiptilt_model_vectors)


    a=projdata_2_data.reshape((Nwave*Noutput,Nsingular))
    b=postiptilt_2_projdata.reshape((Nmodel,Nsingular,3))
    postiptilt_2_data = np.matmul(a,b).reshape((Nmodel,Nwave,Noutput,3))
    flux_norm_wave = postiptilt_2_data[:,:,:,0].sum(axis=(0,2), keepdims=True)[:,:,:,None]
    postiptilt_2_data /= flux_norm_wave

    data_2_postiptilt=np.zeros((Nmodel,Nwave,3,Noutput))
    for w in tqdm(range(Nwave)):
        for i in range(Nmodel):
            data_2_postiptilt[i,w]=pinv(postiptilt_2_data[i,w])


    return postiptilt_2_data,data_2_postiptilt,postiptilt_masque

def get_chi2_maps(datacube,postiptilt_2_data,data_2_postiptilt):
    """
    Calculates chi-squared maps to evaluate the fit of the data to the model.
    Returns the minimum chi-squared, maximum chi-squared, and the chi-squared map.
    """

    print("Computing chi2 of individual observations")
    Nwave=datacube.shape[0]
    Noutput=datacube.shape[1]
    Ncube=datacube.shape[2]
    Npos=datacube.shape[3]
    Nmodel = postiptilt_2_data.shape[0]

    b=datacube.reshape(Nwave,Noutput,Ncube*Npos)
    ftt = np.matmul(data_2_postiptilt,b) #long

    chi2=np.zeros((Nmodel,Ncube*Npos))
    for i in tqdm(range(Nmodel)):
        residual = (b-np.matmul(postiptilt_2_data[i],ftt[i]))**2
        chi2[i]= residual.sum(axis=(0,1))

    arg_model=chi2.argmin(axis=0)
    # best_ftt = np.array([ftt[best_model[n],:,:,n] for n in range(Ncube*Npos)])

    chi2_min=chi2.min(axis=0).reshape((Ncube,Npos))
    chi2_max=chi2.max(axis=0).reshape((Ncube,Npos))
    # best_ftt=best_ftt.reshape((Ncube,Npos,Nwave,3))
    arg_model=arg_model.reshape((Ncube,Npos))

    return chi2_min,chi2_max,arg_model


def get_flux_model(postiptilt_2_data):
    Nmodel=postiptilt_2_data.shape[0]
    Nwave=postiptilt_2_data.shape[1]
    Noutput=postiptilt_2_data.shape[2]

    flux_2_data=postiptilt_2_data[:,:,:,0].transpose((1,2,0))
    data_2_flux=np.zeros((Nwave,Nmodel,Noutput))
    for w in tqdm(range(Nwave)):
        data_2_flux[w]=pinv(flux_2_data[w])

    return flux_2_data,data_2_flux

def get_flux_tip_tilt_model(postiptilt_2_data, dim=0):
    Nmodel=postiptilt_2_data.shape[0]
    Nwave=postiptilt_2_data.shape[1]
    Noutput=postiptilt_2_data.shape[2]

    flux_2_data=postiptilt_2_data[:,:,:,dim].transpose((1,2,0))
    data_2_flux=np.zeros((Nwave,Nmodel,Noutput))
    for w in tqdm(range(Nwave)):
        data_2_flux[w]=pinv(flux_2_data[w])

    return flux_2_data,data_2_flux


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
            file_patterns = "/Users/slacour/DATA/LANTERNE/Optim_maps/November2024/preproc"
            coupling_map = file_patterns+"/couplingmaps/"
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
    masque=(cmap_file['MASQUE'].data) ==1
    flux_2_data=cmap_file['F2DATA'].data
    data_2_flux=cmap_file['DATA2F'].data
    postiptilt_2_data=cmap_file['FTT2DATA'].data
    data_2_postiptilt=cmap_file['DATA2FTT'].data
    cmap_file.close()

    wavelength_bin = header['WL_BIN']
    Nmodel = postiptilt_2_data.shape[0]



    try:
        files_with_dark.pop('/Users/slacour/DATA/LANTERNE/Optim_maps/November2024/preproc/firstpl_2025-01-14T15:34:08_NONAME.fits')

        for _ in range(7):
            files_with_dark.pop(next(iter(files_with_dark)))

        # closest_dark_files.pop(next(reversed(closest_dark_files)))
        # closest_dark_files.pop(next(reversed(closest_dark_files)))
        # closest_dark_files.pop(next(reversed(closest_dark_files)))
    except:
        pass


    # to be reaplaced by the real values
    dither_x, dither_y = dithering_of_image(25)
    Npos = 625 # number of positions in the dithering patterm


    #Input preproc
    #clean and sum all data
    datalist=runlib_i.extract_datacube(files_with_dark,wavelength_smooth,Nbin=wavelength_bin)
    datalist = [d for d in datalist if d.Npos == Npos]

    datacube=[d.data for d in datalist]
    #datacube (625, 38, 100)
    quick_fits(datacube, 'datacube')

    # output_filename is coupling map file
    datacube=np.array(datacube).transpose((3,2,0,1))

    Nwave=datalist[0].Nwave
    Noutput=datalist[0].Noutput
    Ncube=len(datalist)
    Npos=datalist[0].Npos

    # Convert arg_model values into 2D indices of size cmap_size
    chi2_min,chi2_max,arg_model = get_chi2_maps(datacube,postiptilt_2_data,data_2_postiptilt)

    flux_thresold=np.percentile(datacube.mean(axis=(0,1)),80)/5/10000
    flux_goodData=datacube.mean(axis=(0,1)) > flux_thresold
    chi2_delta=chi2_min/chi2_max
    percents=np.nanpercentile(chi2_delta[flux_goodData],[16,50,84])
    chi2_threshold=percents[1]+(percents[2]-percents[0])*3/2


    chi2_goodData = (chi2_delta < chi2_threshold)&flux_goodData

    arg_model_masques = np.where(masque.ravel())[0][arg_model]

    arg_model_indices = np.unravel_index(arg_model_masques, (len(masque), len(masque[0])))
    arg_model_indices = np.array(arg_model_indices)

    fig,ax=plt.subplots(3,num="Position4",clear=True,sharex=True)
    x=np.arange(Npos)
    for c in range(Ncube):
        ax[0].plot(x[chi2_goodData[c]],arg_model_indices[0][c,chi2_goodData[c]],'.', label="Cube "+str(c))
        ax[1].plot(x[chi2_goodData[c]],arg_model_indices[1][c,chi2_goodData[c]],'.', label="Cube "+str(c))
        ax[2].plot(x[chi2_goodData[c]],chi2_delta[c,chi2_goodData[c]],'.-', label="Cube "+str(c))

    ax[0].plot(dither_x, label="dither_x")
    ax[1].plot(dither_y, label="dither_y")

    ax[2].set_yscale('log')
    ax[0].legend()
    ax[0].set_title("Position on x")
    ax[1].legend()
    ax[1].set_title("Position on y")
    ax[2].legend()
    ax[2].set_title("Chi2_delta good data")

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

    # Save image_2d and residual_2d to a FITS file

    for i,d in enumerate(datalist):
        header = d.header


        # Create a primary HDU with no data, just the header
        hdu_primary = fits.PrimaryHDU(image_2d[:,:,i].transpose((2,0,1)))
        hdu_residual = fits.ImageHDU(residual_2d[:,:,i].transpose((2,0,1)), name="RESIDUAL")

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

