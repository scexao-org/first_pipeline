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

DEBUG = True

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

def filter_filelist(filelist,cmap_size=25):
    """
    Filters the input file list to separate coupling map files and dark files based on FITS keywords.
    Raises an error if no valid files are found.
    Returns a dictionary mapping coupling map files to their closest dark files.
    """

    # Use the function to clean the filelist
    fits_keywords = {'DATA-CAT': ['PREPROC'],
                    'DATA-TYP': ['OBJECT'],
                    'NAXIS3': [cmap_size*cmap_size]}
    filelist_cmap = runlib.clean_filelist(fits_keywords, filelist)
    print("runPL cmap filelist : ", filelist_cmap)

    fits_keywords = {'DATA-CAT': ['PREPROC'],
                    'DATA-TYP': ['DARK']}
    filelist_dark = runlib.clean_filelist(fits_keywords, filelist)
    print("runPL dark filelist : ", filelist_dark)


    # raise an error if filelist_cleaned is empty
    if len(filelist_cmap) == 0:
        raise ValueError("No good file to run cmap")
    # raise an error if filelist_cleaned is empty
    if len(filelist_dark) == 0:
        raise ValueError("No good dark to substract to cmap files")

    # Check if all files have the same value for header['PM_CHECK']
    pm_check_values = set()
    combined_filelist = []
    combined_filelist.extend(filelist_dark)
    combined_filelist.extend(filelist_cmap)
    for file in combined_filelist:
        header = fits.getheader(file)
        pm_check_values.add(header.get('PM_CHECK', 0))
        
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

    files_with_dark = {cmap: find_closest_dark(cmap, filelist_dark) for cmap in filelist_cmap}

    return files_with_dark


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

def run_create_coupling_maps(files_with_dark, 
                                cmap_size = 25,
                                wavelength_smooth = 20,
                                wavelength_bin = 15,
                                make_movie = False,
                                Nsingular=19*3):
    """
    Used in lancementserie.py for global generation
    
    """
    
    plt.close("all")

    #Input preproc
    #clean and sum all data
    datalist=runlib_i.extract_datacube(files_with_dark,wavelength_smooth,Nbin=wavelength_bin)
    #datacube (625, 38, 100)
    datacube=[d.data for d in datalist]
    quick_fits(datacube, 'datacube')

    datacube=np.array(datacube).transpose((3,2,0,1))

    if make_movie:
        runlib_i.create_movie_cross(datacube)

        plt.close('all')

    # select data only above a threshold based on flux
    flux_thresold=np.percentile(datacube.mean(axis=(0,1)),80)/5
    flux_goodData=datacube.mean(axis=(0,1)) > flux_thresold
    plt.imshow(flux_goodData)
    if np.sum(flux_goodData)<57:
        #too little good data, we need to lower the bar
        flux_goodData=datacube.mean(axis=(0,1)) > flux_thresold/2

    # get the Nsingulat highest singular values and the projection vectors into that space 
    #VSD
    #datacube : (100, 38, 10, 625)
    #flux_gooddata : (10, 625)
    #Nsingular : 57
    pos_2_singular,singular_values,singular_2_data=get_projection_matrice(datacube,flux_goodData,Nsingular)


    # cross correlate the dataset to see if there is a significant offset between the different datasets
    dist_2d_x,dist_2d_y,cross_correlated_projected_data = get_shift_between_image(pos_2_singular)

    # shift and average all the datacubes, do not includes the bad frames
    pos_2_singular[:,~flux_goodData]=np.nan
    pos_2_singular_mean,shifted_pos_2_singular = shift_and_add(pos_2_singular, dist_2d_x, dist_2d_y)

    # compute the matrices to go from the projected data to the flux and tip tilt (and inverse)
    postiptilt_2_data,data_2_postiptilt,postiptilt_masque = get_postiptilt_model(pos_2_singular_mean,singular_2_data)

    #use datamodel to check if the observations are point like
    # To do so, fits the vector model and check if the chi2 decrease resonably
    chi2_min,chi2_max,arg_model=get_chi2_maps(datacube,postiptilt_2_data,data_2_postiptilt)
    chi2_delta=chi2_min/chi2_max
    percents=np.nanpercentile(chi2_delta[flux_goodData],[16,50,84])
    chi2_threshold=percents[1]+(percents[2]-percents[0])*3/2
    chi2_goodData = (chi2_delta < chi2_threshold)&flux_goodData

    #redo most of the work above but with flagged datasets
    pos_2_singular,singular_values,singular_2_data=get_projection_matrice(datacube,chi2_goodData,Nsingular)
    dist_2d_x,dist_2d_y,cross_correlated_projected_data = get_shift_between_image(pos_2_singular)
    pos_2_singular[:,~chi2_goodData]=np.nan
    pos_2_singular_mean,shifted_pos_2_singular = shift_and_add(pos_2_singular, dist_2d_x, dist_2d_y)
    postiptilt_2_data,data_2_postiptilt,postiptilt_masque = get_postiptilt_model(pos_2_singular_mean,singular_2_data)

    flux_2_data,data_2_flux = get_flux_model(postiptilt_2_data)
    # Save arrays into a FITS file

    # Create a primary HDU with no data, just the header
    hdu_primary = fits.PrimaryHDU()

    # Create HDUs for each array
    hdu_0 = fits.ImageHDU(data=postiptilt_masque.astype(np.uint8), name='MASQUE')  # Save masque as uint8 to save space
    hdu_1 = fits.ImageHDU(data=flux_2_data, name='F2DATA')
    hdu_2 = fits.ImageHDU(data=data_2_flux, name='DATA2F')
    hdu_3 = fits.ImageHDU(data=postiptilt_2_data, name='FTT2DATA')
    hdu_4 = fits.ImageHDU(data=data_2_postiptilt, name='DATA2FTT')

    header = datalist[-1].header
    # Définir le chemin complet du sous-dossier "output/couplingmaps"
    folder = datalist[-1].dirname
    output_dir = os.path.join(folder,"couplingmaps")

    header['DATA-CAT'] = 'COUPLINGMAP'
    # Add date and time to the header
    current_time = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
    header['DATE-PRO'] = current_time
    if 'DATE' not in header:
        header['DATE'] = current_time

    # Add input parameters to the header
    header['CMAPSIZE'] = cmap_size  # Add cmap size
    header['WLSMOOTH'] = wavelength_smooth  # Add wavelength smoothing factor
    header['WL_BIN'] = wavelength_bin
    header['NSINGUL'] = Nsingular  # Add number of singular values
    header['FLUXTHR'] = flux_thresold  # Add flux threshold
    header['CHI2THR'] = chi2_threshold  # Add chi2 threshold

    # Créer les dossiers "output" et "pixel" s'ils n'existent pas déjà
    os.makedirs(output_dir, exist_ok=True)

    hdu_primary.header.extend(header, strip=True)

    # Combine all HDUs into an HDUList
    hdul = fits.HDUList([hdu_primary, hdu_0, hdu_1, hdu_2, hdu_3, hdu_4])

    output_filename = os.path.join(output_dir, runlib.create_output_filename(header))

    # Write to a FITS file
    hdul.writeto(output_filename, overwrite=True)
    print(f"Data saved to {output_filename}")

    runlib_i.generate_plots(singular_values, chi2_delta, flux_goodData, chi2_goodData, chi2_threshold, cross_correlated_projected_data, shifted_pos_2_singular, postiptilt_2_data, output_dir)


if __name__ == "__main__":
    parser = OptionParser(usage)


    # Default values
    cmap_size = 25
    wavelength_smooth = 20
    wavelength_bin = 15
    make_movie = False
    Nsingular=19*3 #for cmap=7, 57 is too high (34, 19 for plots is max for novemeber data in cmap=7)

    # Add options for these values
    parser.add_option("--cmap_size", type="int", default=cmap_size,
                    help="step numbers of modulation (default: %default)")
    parser.add_option("--Nsingular", type="int", default=Nsingular,
                      help="Number of singular values to use (default: %default)")
    parser.add_option("--wavelength_smooth", type="int", default=wavelength_smooth,
                    help="smoothing factor for wavelength (default: %default)")
    parser.add_option("--wavelength_bin", type="int", default=wavelength_bin,
                    help="binning factor for wavelength (default: %default)")
    parser.add_option("--make_movie", action="store_true", default=make_movie,
                    help="Create a nice mp4 with all datacubes -- can be long (default: %default)")
    
    if "VSCODE_PID" in os.environ or os.environ.get('TERM_PROGRAM') == 'vscode':
        if getpass.getuser() == "slacour":
            file_patterns = "/Users/slacour/DATA/LANTERNE/Optim_maps/November2024/preproc"
        if getpass.getuser() == "jsarrazin":
            file_patterns = "/home/jsarrazin/Bureau/PLDATA/moreTest/2024-11-21_13-48-32_science_copie/preproc"
            file_patterns = "/home/jsarrazin/Bureau/PLDATA/novembre/les_preproc"
        #file_patterns = "/home/jsarrazin/Bureau/PLDATA/2025_03_14"
        #file_patterns = "/home/jsarrazin/Bureau/PLDATA/selection_prises_15_mars"
    else:
        # Parse the options
        (options, args) = parser.parse_args()

        # Pass the parsed options to the function
        cmap_size=options.cmap_size
        Nsingular=options.Nsingular
        wavelength_smooth=options.wavelength_smooth
        make_movie=options.make_movie
        wavelength_bin=options.wavelength_bin
        file_patterns=args if args else ['*.fits']

    print(file_patterns)
    filelist = runlib.get_filelist(file_patterns)
    print(filelist)
    files_with_dark = filter_filelist(filelist,cmap_size)

    try:
        files_with_dark.pop('/Users/slacour/DATA/LANTERNE/Optim_maps/November2024/preproc/firstpl_2025-01-14T15:34:08_NONAME.fits')

        for _ in range(7):
            files_with_dark.pop(next(iter(files_with_dark)))

        # files_with_dark.pop(next(reversed(files_with_dark)))
        # files_with_dark.pop(next(reversed(files_with_dark)))
        # files_with_dark.pop(next(reversed(files_with_dark)))
    except:
        pass

    try:
        files_with_dark.pop('/Users/slacour/DATA/LANTERNE/Optim_maps/May2024/preproc/firstpl_2025-02-19T11:25:12_NONAME.fits')
        files_with_dark.pop('/Users/slacour/DATA/LANTERNE/Optim_maps/May2024/preproc/firstpl_2025-02-19T11:25:13_NONAME.fits')
        files_with_dark.pop('/Users/slacour/DATA/LANTERNE/Optim_maps/May2024/preproc/firstpl_2025-02-19T11:25:14_NONAME.fits')
        files_with_dark.pop('/Users/slacour/DATA/LANTERNE/Optim_maps/May2024/preproc/firstpl_2025-02-19T11:25:15_NONAME.fits')
    except:
        pass
    

    run_create_coupling_maps(files_with_dark, 
                                cmap_size = cmap_size,
                                wavelength_smooth = wavelength_smooth,
                                wavelength_bin = wavelength_bin,
                                make_movie = make_movie,
                                Nsingular=19*3)


# %%
