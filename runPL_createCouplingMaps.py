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
import runPL_library_io as runlib
import runPL_library_imaging as runlib_i
from scipy.ndimage import zoom
from astropy.io import fits
import shutil
from scipy.interpolate import interpn
from astropy.table import Table

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
    runPL_compareCouplingMaps.py  *.fits

    Options:
    --cmap_size: Width of cmap size, in pixels (default: 25) (removed)
"""

def filter_filelist(filelist):
    """
    Filters the input file list to separate coupling map files and dark files based on FITS keywords.
    Raises an error if no valid files are found.
    Returns a dictionary mapping coupling map files to their closest dark files.
    """

    # Use the function to clean the filelist
    fits_keywords = {'DATA-CAT': ['PREPROC'],
                    'DATA-TYP': ['OBJECT','TEST']}
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
        print("WARNING: No good dark to substract to cmap files")

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

    files_with_dark = {cmap: runlib.find_closest_dark(cmap, filelist_dark) for cmap in filelist_cmap}

    return files_with_dark

def filter_data(datacube,flux_goodData,Nsingular):
    """
    Filters the input datacube based on good flux data and applies Singular Value Decomposition (SVD).
    This function reduces the dimensionality of the datacube while retaining the most significant components.

    Args:
        datacube (numpy.ndarray): The input datacube with dimensions (Nwave, Noutput, Ncube, Nmod).
        flux_goodData (numpy.ndarray): A boolean mask indicating which data points have good flux.
        Nsingular (int): The number of singular values to retain.

    Returns:
        numpy.ndarray: The filtered datacube with reduced dimensionality.
    """

    Nwave=datacube.shape[0] #100
    Noutput=datacube.shape[1] #38
    Ncube=datacube.shape[2] #10
    Nmod=datacube.shape[3] #625
    datacube=datacube.reshape((Nwave*Noutput,Ncube,Nmod)) #reshape to (3800, 10, 625)

    pos_2_data = datacube[:,flux_goodData] #(3800, 3017) datacube is (3800, 10, 625), flux_good is (10, 625)

    U,s,Vh=linalg.svd(pos_2_data,full_matrices=False)

    #pos_2_singular = Vh[:Nsingular]*s[:Nsingular,None]
    singular_2_data = U[:,:Nsingular] #(3800, 57)
    pos_2_singular = singular_2_data.T @ datacube.reshape((Nwave*Noutput,Ncube*Nmod)) #(57, 6250)
    datacube_filtered = singular_2_data @ pos_2_singular

    datacube_filtered = datacube_filtered.reshape((Nwave,Noutput,Ncube,Nmod))
    datacube = datacube.reshape((Nwave,Noutput,Ncube,Nmod))

    return datacube_filtered

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
    Nmod=datacube.shape[3] #625
    datacube=datacube.reshape((Nwave*Noutput,Ncube,Nmod)) #reshape to (3800, 10, 625)

    pos_2_data = datacube[:,flux_goodData] #(3800, 3017) datacube is (3800, 10, 625), flux_good is (10, 625)

    U,s,Vh=linalg.svd(pos_2_data,full_matrices=False)

    #pos_2_singular = Vh[:Nsingular]*s[:Nsingular,None]
    singular_2_data = U[:,:Nsingular] #(3800, 57)
    pos_2_singular = singular_2_data.T @ datacube.reshape((Nwave*Noutput,Ncube*Nmod)) #(57, 6250)

    singular_values = s #(3017,)
    pos_2_singular = pos_2_singular.reshape((Nsingular,Ncube,Nmod)) #reshape to (57, 10, 625)
    singular_2_data = singular_2_data.reshape((Nwave,Noutput,Nsingular))

    return pos_2_singular,singular_values,singular_2_data

def get_fluxtiptilt_matrices(singular_2_data, pos_2_singular_mean, triangles):
    """
    Computes the flux and tip-tilt matrix from the projected data.

    This function calculates matrices for converting between projected data and flux/tip-tilt values.

    Returns:
        tuple: A tuple containing:
            - flux_2_data (numpy.ndarray): Matrix to convert flux to data.
            - data_2_flux (numpy.ndarray): Matrix to convert data to flux.
            - fluxtiptilt_2_data (numpy.ndarray): Matrix to convert flux and tip-tilt to data.
            - data_2_fluxtiptilt (numpy.ndarray): Matrix to convert data to flux and tip-tilt.
    """

    
    Nsingular=pos_2_singular_mean.shape[0]
    Nmod=pos_2_singular_mean.shape[1]
    Nwave=singular_2_data.shape[0]
    Noutput=singular_2_data.shape[1]
    # Ntriangles=len(triangles)

    masque_positions=~np.isnan(pos_2_singular_mean[0])
    masque_triangles=(masque_positions[triangles].sum(axis=1) ==3)
    Npositions=np.sum(masque_positions)
    Ntriangles=np.sum(masque_triangles)


    flux_2_data_tmp = singular_2_data.reshape((Nwave*Noutput,Nsingular)) @ pos_2_singular_mean
    flux_2_data_tmp = flux_2_data_tmp.reshape((Nwave,Noutput,Nmod))
    flux_2_data = flux_2_data_tmp[:,:,masque_positions]
    flux_norm_wave = flux_2_data.sum(axis=(1,2), keepdims=True)
    flux_2_data /= flux_norm_wave

    data_2_flux = np.zeros((Nwave,Npositions,Noutput))
    print("Inverting flux_2_data to data_2_flux for each wavelength:")
    for w in tqdm(range(Nwave)):
        data_2_flux[w]=pinv(flux_2_data[w])

    fluxtiptilt_2_data = flux_2_data_tmp[:,:,triangles[masque_triangles]].transpose((2,0,1,3)).copy()
    data_2_fluxtiptilt = np.zeros((Ntriangles,Nwave,3,Noutput))
    print("Inverting fluxtiptilt_2_data to data_2_fluxtiptilt:")
    for w in tqdm(range(Nwave)):
        for t in range(Ntriangles):
            data_2_fluxtiptilt[t,w]=pinv(fluxtiptilt_2_data[t,w])


    return flux_2_data,data_2_flux,fluxtiptilt_2_data,data_2_fluxtiptilt,masque_positions,masque_triangles



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
                                wavelength_smooth = 20,
                                wavelength_bin = 15,
                                Nsingular=19*3):
    """
    Used in lancementserie.py for global generation
    
    """
    
    plt.close("all")

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
    triangles = datalist[0].get_triangle()

    # select data only above a threshold based on flux
    flux_threshold=np.percentile(datacube.mean(axis=(0,1)),80)/5
    flux_goodData=datacube.mean(axis=(0,1)) > flux_threshold
    # plt.imshow(flux_goodData)
    if np.sum(flux_goodData)<57:
        #too little good data, we need to lower the bar
        flux_goodData=datacube.mean(axis=(0,1)) > flux_threshold/2
        print("Not enough good data, lowering the threshold to ",flux_threshold/2)

    # get the Nsingulat highest singular values and the projection vectors into that space 
    #VSD
    #datacube : (100, 38, 10, 625)
    #flux_gooddata : (10, 625)
    #Nsingular : 57
    pos_2_singular,singular_values,singular_2_data=get_projection_matrice(datacube,flux_goodData,Nsingular)

    # average all the datacubes, do not includes the bad frames
    pos_2_singular[:,~flux_goodData]=np.nan
    pos_2_singular_mean = np.nanmean(pos_2_singular,axis=1)

    # compute the matrices to go from the projected data to the flux and tip tilt (and inverse)
    flux_2_data,data_2_flux,fluxtiptilt_2_data,data_2_fluxtiptilt,masque_positions,masque_triangles = get_fluxtiptilt_matrices(singular_2_data, pos_2_singular_mean, triangles)

    #use flux tip tilt matrice to check if the observations are point like
    # To do so, fits the vector model and check if the chi2 decrease resonably
    chi2_min,chi2_max,arg_triangle=runlib_i.get_chi2_maps(datacube,fluxtiptilt_2_data,data_2_fluxtiptilt)
    chi2_delta=chi2_min/chi2_max
    percents=np.nanpercentile(chi2_delta[flux_goodData],[16,50,84])
    chi2_threshold=percents[1]+(percents[2]-percents[0])*3/2
    chi2_goodData = (chi2_delta < chi2_threshold)&flux_goodData

    #redo most of the work above but with flagged datasets
    pos_2_singular,singular_values,singular_2_data=get_projection_matrice(datacube,chi2_goodData,Nsingular)
    pos_2_singular[:,~chi2_goodData]=np.nan
    pos_2_singular_mean = np.nanmean(pos_2_singular,axis=1)
    flux_2_data,data_2_flux,fluxtiptilt_2_data,data_2_fluxtiptilt,masque_positions,masque_triangles = get_fluxtiptilt_matrices(singular_2_data, pos_2_singular_mean, triangles)
    # Save arrays into a FITS file

    # Create a primary HDU with no data, just the header
    hdu_primary = fits.PrimaryHDU()

    # Create HDUs for each array
    hdu_1 = fits.ImageHDU(data=flux_2_data, name='F2DATA')
    hdu_2 = fits.ImageHDU(data=data_2_flux, name='DATA2F')
    hdu_3 = fits.ImageHDU(data=fluxtiptilt_2_data, name='FTT2DATA')
    hdu_4 = fits.ImageHDU(data=data_2_fluxtiptilt, name='DATA2FTT')

    # Create columns for xmod and ymod using fits.Column
    x_pos = xmod[masque_positions]
    y_pos = ymod[masque_positions]
    x_triangles = xmod[triangles[masque_triangles]]
    y_triangles = ymod[triangles[masque_triangles]]

    # shifting all positions around the maximum of flux found from gaussian fitting
    fluxes = datacube.mean(axis=(0,1,2))
    popt = runlib_i.fit_gaussian_on_flux(fluxes, xmod, ymod)
    x_fit=popt[1]
    y_fit=popt[2]
    x_fit = x_pos[((x_fit-x_pos)**2).argmin()] 
    y_fit = y_pos[((y_fit-y_pos)**2).argmin()] 

    x_triangles -= x_fit
    y_triangles -= y_fit
    x_pos -= x_fit
    y_pos -= y_fit

    col_xmod = fits.Column(name='X_POS', format='E', array=x_pos, unit='mas')
    col_ymod = fits.Column(name='Y_POS', format='E', array=y_pos, unit='mas')

    col_xtriangles = fits.Column(name='X_TRI', format='3E', array=x_triangles, unit='mas')
    col_ytriangles = fits.Column(name='Y_TRI', format='3E', array=y_triangles, unit='mas')

    # Create a table HDU for xmod and ymod
    hdu_table_mod = fits.BinTableHDU.from_columns([col_xmod, col_ymod], name='POSITIONS')
    hdu_table_triangle = fits.BinTableHDU.from_columns([col_xtriangles, col_ytriangles], name='TRIANGLES')

    modulation_hdu = fits.open(datalist[-1].filename)['MODULATION']

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
    header['WLSMOOTH'] = wavelength_smooth  # Add wavelength smoothing factor
    header['WL_BIN'] = wavelength_bin
    header['NSINGUL'] = Nsingular  # Add number of singular values
    header['FLUXTHR'] = flux_threshold  # Add flux threshold
    header['CHI2THR'] = chi2_threshold  # Add chi2 threshold

    # Créer les dossiers "output" et "pixel" s'ils n'existent pas déjà
    os.makedirs(output_dir, exist_ok=True)

    hdu_primary.header.extend(header, strip=True)

    # Combine all HDUs into an HDUList
    hdul = fits.HDUList([hdu_primary, hdu_1, hdu_2, hdu_3, hdu_4,hdu_table_mod,hdu_table_triangle,modulation_hdu])

    output_filename = os.path.join(output_dir, runlib.create_output_filename(header))

    # Write to a FITS file
    print(f"Saving data to {output_filename}")
    hdul.writeto(output_filename, overwrite=True)


    runlib_i.generate_plots(datacube, xmod, ymod, masque_positions, flux_2_data, singular_values, Nsingular, chi2_delta, flux_goodData, chi2_goodData, flux_threshold, chi2_threshold, output_dir)


if __name__ == "__main__":
    parser = OptionParser(usage)


    # Default values
    wavelength_smooth = 20
    wavelength_bin = 15
    make_movie = False
    Nsingular=19*3 #for cmap=7, 57 is too high (34, 19 for plots is max for novemeber data in cmap=7)

    # Add options for these values
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
            file_patterns = "/Users/slacour/DATA/LANTERNE/Mai/preproc2/firstpl_2025-05-06T09:52:?1_BETUMA.fits"
        if getpass.getuser() == "jsarrazin":
            file_patterns = "/home/jsarrazin/Bureau/PLDATA/moreTest/2024-11-21_13-48-32_science_copie/preproc"
            file_patterns = "/home/jsarrazin/Bureau/PLDATA/novembre/les_preproc"
        #file_patterns = "/home/jsarrazin/Bureau/PLDATA/2025_03_14"
        #file_patterns = "/home/jsarrazin/Bureau/PLDATA/selection_prises_15_mars"
    else:
        # Parse the options
        (options, args) = parser.parse_args()

        # Pass the parsed options to the function
        Nsingular=options.Nsingular
        wavelength_smooth=options.wavelength_smooth
        make_movie=options.make_movie
        wavelength_bin=options.wavelength_bin
        file_patterns=args if args else ['./preproc/*.fits']

    print(file_patterns)
    filelist = runlib.get_filelist(file_patterns)
    print(filelist)
    files_with_dark = filter_filelist(filelist)

    run_create_coupling_maps(files_with_dark, 
                                wavelength_smooth = wavelength_smooth,
                                wavelength_bin = wavelength_bin,
                                Nsingular=19*3)


# %%
