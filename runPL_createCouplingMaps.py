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
import runPL_library_io as runlib
import runPL_library_imaging as runlib_i
import runPL_library_basic as basic
from scipy.ndimage import zoom
from astropy.io import fits
import shutil
from scipy.interpolate import interpn
from astropy.table import Table
from scipy.interpolate import griddata

plt.ion()

DEBUG = True

# Add options
usage = """
    usage:  %prog [options] files.fits

    Goal: Create coupling maps from preprocessed photonic lantern data.

    Sumary
    It will get as input a list of files with DPR_CATG=CMAP and DPR_TYPE=PREPROC keywords.
    It will select files based on modulation pattern, modulation scale, and object name if specified.
    The script computes SVD-based coupling maps, saves results to FITS, and generates diagnostic plots.

    Input:
    - Files of type X_FIRTYP=PREPROC in the directory or in the argument pattern.

    Output:
    - Files of type X_FIRTYP=COULPLINGMAP in the directory "../couplingmaps".
    - A pdf report with the plots of the coupling maps and the SVD analysis.

    Options:
    --wavelength_smooth: Smoothing factor for wavelength (default: 20)
    --wavelength_bin: Binning factor for wavelength (default: 15)
    --object_name: Selection of the data by the Object name (default: NONE)
    --modID: Selection of the modulation pattern by user [0 == first in the list] (default: 0)
    --modScale: Selection of the modulation scale by user [0 == first in the list] (default: 0)
    --Nsingular: Number of singular values to use (default: 57)

    Example:
    runPL_createCouplingMaps.py  *.fits
"""

def get_filelist(file_patterns, dark_patterns, flat_patterns, modID, modScale, object_name, wollaston):

        fits_keywords = {'X_FIRTYP': ['PREPROC'],
                        'DATA-TYP': ['OBJECT','OJECT','TEST'],
                        'X_FIRTRG': ['EXT'],
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
        fits_keywords['X_FIRWOL'] = [wollaston]

        print("----------------")
        print(f"Selected object='{object_name}' with modScale={modScale} and modID={modID}")
        print("----------------")

        filelist = runlib.get_filelist(file_patterns, fits_keywords)

        # finding darks files
        fits_keywords = {'X_FIRTYP': ['PREPROC'],
                        'DATA-TYP': ['DARK'],
                        'X_FIRWOL': [wollaston],
                        }    
    
        try:
            filelist_dark = runlib.get_filelist(dark_patterns, fits_keywords,  name_search="dark")
        except FileNotFoundError as e:
            print(f"WARNING!!! {e}")
            filelist_dark = []

        # finding flats files
        fits_keywords = {'X_FIRTYP': ['PREPROC'],
                        'DATA-TYP': ['FLAT'],
                        'X_FIRWOL': [wollaston],
                        }    
        try:
            filelist_flat = runlib.get_filelist(flat_patterns, fits_keywords,  name_search="flat")
        except FileNotFoundError as e:
            print(f"WARNING!!! {e}")
            filelist_flat = filelist

        files_with_dark = runlib.associate_dark(filelist, filelist_dark)
        flats_with_dark = runlib.associate_dark(filelist_flat, filelist_dark)

        return files_with_dark, flats_with_dark


def compute_flat(flat_with_dark):
    
    datalist=runlib_i.extract_datacube(flat_with_dark)
    flats=[d.data.sum(axis=(0,1)) for d in datalist]
    flat=np.sum(flats,axis=0)
    flat/=np.mean(flat,axis=0)

    return flat

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
    Nedges=len(triangles[0])

    masque_positions=~np.isnan(pos_2_singular_mean[0])
    masque_triangles=(masque_positions[triangles].sum(axis=1) > 0.79*Nedges)
    Npositions=np.sum(masque_positions)
    Ntriangles=np.sum(masque_triangles)

    flux_2_data_tmp = singular_2_data.reshape((Nwave*Noutput,Nsingular)) @ pos_2_singular_mean
    flux_2_data_tmp = flux_2_data_tmp.reshape((Nwave,Noutput,Nmod))
    flux_2_data_tmp[:,:,~masque_positions] = 0.0
    flux_2_data = flux_2_data_tmp[:,:,masque_positions]
    flux_norm_wave = flux_2_data.sum(axis=(1,2), keepdims=True)
    flux_2_data /= flux_norm_wave
    flux_2_data_tmp /= flux_norm_wave

    data_2_flux = np.zeros((Nwave,Npositions,Noutput))
    print("Inverting flux_2_data to data_2_flux for each wavelength:")
    for w in tqdm(range(Nwave)):
        data_2_flux[w]=pinv(flux_2_data[w])

    fluxtiptilt_2_data = flux_2_data_tmp[:,:,triangles[masque_triangles]].transpose((2,0,1,3)).copy()
    data_2_fluxtiptilt = np.zeros((Ntriangles,Nwave,Nedges,Noutput))
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

def fluxmap_interpolation(fluxmaps, xmod, ymod, gridsize=500):
    if len(fluxmaps.shape) == 1:
        fluxmaps=fluxmaps[np.newaxis]
    # Define the grid for interpolation
    grid_x, grid_y = np.mgrid[np.min(xmod):np.max(xmod):gridsize*1j, 
                              np.min(ymod):np.max(ymod):gridsize*1j]  # 500x500 grid
    # Interpolate the fluxes onto the grid
    fluxmap_interp= np.zeros((len(fluxmaps), gridsize, gridsize))
    for i,fm in enumerate(fluxmaps):
        fluxmap_interp[i] = griddata((xmod, ymod), fm, (grid_x, grid_y), method='cubic').T

    return fluxmap_interp
    


if __name__ == "__main__":
    parser = OptionParser(usage)

    # Default values
    wavelength_smooth = 20
    wavelength_bin = 10
    Nsingular=19*6 

    # Add options for these values

    # Add options for these values
    parser.add_option("--object_name", type="string", 
                    help="Selection of the data by the Object name (default: first target the list)")
    parser.add_option("--dark_files", type="string", 
                    help="Select one or more specific dark(s) files to use")
    parser.add_option("--flat_files", type="string", 
                    help="Select a specific flat file to use (default: use the flat files or if not the ones used to create the coupling maps)")
    parser.add_option("--wavelength_smooth", type="int", default=wavelength_smooth,
                    help="smoothing factor for wavelength (default: %default)")
    parser.add_option("--wavelength_bin", type="int", default=wavelength_bin,
                    help="binning factor for wavelength (default: %default)")
    parser.add_option("--Nsingular", type="int", default=Nsingular,
                      help="Number of singular values to use (default: %default)")
    parser.add_option("--modID", type="int", 
                      help="Selection of the modulation pattern by user (default: first in the list)")
    parser.add_option("--modScale", type="int", 
                      help="Selection of the modulation pattern by user (default: first in the list)")
    parser.add_option("--wollaston", type="string", 
                      help="Wollaston status. Use IN for internal or OUT for no wollaston (default: first in the list)")
    
    if ("VSCODE_PID" in os.environ or os.environ.get('TERM_PROGRAM') == 'vscode' or os.environ.get('SPYDER_DEBUG_FILE')):
        print("Running in compiler")
        flat_patterns = None
        dark_patterns = None
        modID = None
        modScale = None
        object_name = None
        wollaston = None
        if getpass.getuser() == "slacour":
            file_patterns = "/Users/slacour/DATA/LANTERNE/20250514/preproc/firstpl_2025-05-14T11?3*fits"
            file_patterns = "/Users/slacour/DATA/LANTERNE/20250808/preproc/firstpl_2025-08-08T07:16:??_HIP84212_P.fits"
        if getpass.getuser() == "jsarrazin":
            file_patterns = "/home/jsarrazin/Bureau/PLDATA/moreTest/2024-11-21_13-48-32_science_copie/preproc"
            file_patterns = "/home/jsarrazin/Bureau/PLDATA/novembre/les_preproc"
        if getpass.getuser() == "ehuby":
            file_patterns = "/home/ehuby/WORK/DATA/FIRST-PL/2025-05-10/preproc/"
    else:
        # Parse the options
        (options, args) = parser.parse_args()
        file_patterns=args if args else ['*.fits','./preproc/*.fits']

        # Pass the parsed options to the function
        modID=options.modID
        modScale=options.modScale
        object_name = options.object_name
        wollaston = options.wollaston
        Nsingular=options.Nsingular
        wavelength_smooth=options.wavelength_smooth
        wavelength_bin=options.wavelength_bin
        flat_patterns = options.flat_files
        dark_patterns = options.dark_files

    # If the user specifies a coupling map, use it, otherwise look into the arguments
    if flat_patterns is None:
        flat_patterns = file_patterns
    # If the user specify a dark, use it. Otherwise, use the science file pattern
    if dark_patterns is None:
        dark_patterns = file_patterns

    files_with_dark, flats_with_dark = get_filelist(file_patterns, dark_patterns, flat_patterns, modID, modScale, object_name, wollaston)

    flat = compute_flat(flats_with_dark)


    ### run_create_coupling_maps function
    
    plt.close("all")

    #Input preproc
    #clean and sum all data
    datalist=runlib_i.extract_datacube(files_with_dark,Nsmooth=wavelength_smooth,Nbin=wavelength_bin, flat =flat, normalize=True)

    filenames= [d.filename for d in datalist]
    
    datacube=np.concatenate([d.data for d in datalist])
    datacube_var=np.concatenate([d.variance for d in datalist])


    datacube=datacube.transpose((3,2,0,1))

    xmod=datalist[0].xmod
    ymod=datalist[0].ymod
    triangles = datalist[0].get_triangle()
    crosses     = datalist[0].get_crosses()

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
    flux_2_data,data_2_flux,fluxtiptiltderiv_2_data,data_2_fluxtiptiltderiv,masque_positions,masque_triangles = get_fluxtiptilt_matrices(singular_2_data, pos_2_singular_mean, crosses)

    #use flux tip tilt matrice to check if the observations are point like
    # To do so, fits the vector model and check if the chi2 decrease resonably
    chi2_min,chi2_max,arg_triangle=runlib_i.get_chi2_maps(datacube,fluxtiptiltderiv_2_data,data_2_fluxtiptiltderiv)
    chi2_delta=chi2_min/chi2_max
    percents=np.nanpercentile(chi2_delta[flux_goodData],[16,50,84])
    chi2_threshold=percents[1]+(percents[2]-percents[0])*3/2
    chi2_goodData = (chi2_delta < chi2_threshold)&flux_goodData

    #redo most of the work above but with flagged datasets
    pos_2_singular,singular_values,singular_2_data=get_projection_matrice(datacube,chi2_goodData,Nsingular)
    pos_2_singular[:,~chi2_goodData]=np.nan
    pos_2_singular_mean = np.nanmean(pos_2_singular,axis=1)
    flux_2_data,data_2_flux,fluxtiptilt_2_data,data_2_fluxtiptilt,masque_positions, masque_triangles  = get_fluxtiptilt_matrices(singular_2_data, pos_2_singular_mean, triangles)
    flux_2_data,data_2_flux,fluxtiptiltderiv_2_data,data_2_fluxtiptiltderiv,masque_positions,masque_crosses = get_fluxtiptilt_matrices(singular_2_data, pos_2_singular_mean, crosses)
    
    # Flux maps for inspection
    fluxmaps = np.mean(datacube, axis=(0,1))
    fluxmap_interp = fluxmap_interpolation(fluxmaps, xmod, ymod, gridsize=50)
    
    # 2D-interpolation of the modes for inspection
    gridsize=50
    Ncube=datacube.shape[2]
    modes_rect = np.zeros((Nsingular,Ncube,gridsize,gridsize))
    modes_mean = np.zeros((Nsingular, gridsize, gridsize))
    for s in range(Nsingular):
        modes_rect[s] = fluxmap_interpolation(np.nan_to_num(np.array(pos_2_singular[s])), xmod, ymod, gridsize=gridsize)
        modes_mean[s] = fluxmap_interpolation(np.nan_to_num(np.array(pos_2_singular_mean[s])), xmod, ymod, gridsize=gridsize)
        
    # Save arrays into a FITS file

    # Create a primary HDU with no data, just the header
    hdu_primary = fits.PrimaryHDU()

    # Create HDUs for each array
    hdu_1 = fits.ImageHDU(data=flux_2_data, name='F2DATA')
    hdu_2 = fits.ImageHDU(data=data_2_flux, name='DATA2F')
    hdu_3 = fits.ImageHDU(data=fluxtiptilt_2_data, name='FTT2DATA')
    hdu_4 = fits.ImageHDU(data=data_2_fluxtiptilt, name='DATA2FTT')
    hdu_5 = fits.ImageHDU(data=fluxtiptiltderiv_2_data, name='FTTDER2DATA')
    hdu_6 = fits.ImageHDU(data=data_2_fluxtiptiltderiv, name='DATA2FTTDER')
    hdu_fluxmap = fits.ImageHDU(data=fluxmap_interp, name='FLUXMAP')
    hdu_modes2D = fits.ImageHDU(data=modes_rect.reshape((Nsingular,gridsize*Ncube,gridsize)), name='MODES2D')
    hdu_flat = fits.ImageHDU(data=flat, name='FLAT')

    # Create columns for xmod and ymod using fits.Column
    x_pos = xmod[masque_positions]
    y_pos = ymod[masque_positions]
    x_triangles = xmod[triangles[masque_triangles]]
    y_triangles = ymod[triangles[masque_triangles]]
    x_crosses = xmod[crosses[masque_crosses]]
    y_crosses = ymod[crosses[masque_crosses]]

    # shifting all positions around the maximum of flux found from gaussian fitting
    fluxes = datacube.mean(axis=(0,1,2))
    popt = basic.fit_gaussian_on_flux(fluxes, xmod, ymod)
    x_fit=popt[1]
    y_fit=popt[2]
    x_fit = x_pos[((x_fit-x_pos)**2).argmin()] 
    y_fit = y_pos[((y_fit-y_pos)**2).argmin()] 

    x_triangles -= x_fit
    y_triangles -= y_fit
    x_pos -= x_fit
    y_pos -= y_fit
    x_crosses -= x_fit
    y_crosses -= y_fit

    col_xmod = fits.Column(name='X_POS', format='E', array=x_pos, unit='mas')
    col_ymod = fits.Column(name='Y_POS', format='E', array=y_pos, unit='mas')

    col_xtriangles = fits.Column(name='X_TRI', format='3E', array=x_triangles, unit='mas')
    col_ytriangles = fits.Column(name='Y_TRI', format='3E', array=y_triangles, unit='mas')
    # Dynamically set the format based on the size of x_crosses' second dimension
    n_cross = x_crosses.shape[1] if x_crosses.ndim > 1 else 1
    format_str = f'{n_cross}E'
    col_xcrosses = fits.Column(name='X_CROS', format=format_str, array=x_crosses, unit='mas')
    col_ycrosses = fits.Column(name='Y_CROS', format=format_str, array=y_crosses, unit='mas')

    # Create a table HDU for xmod and ymod
    hdu_table_mod = fits.BinTableHDU.from_columns([col_xmod, col_ymod], name='POSITIONS')
    hdu_table_triangle = fits.BinTableHDU.from_columns([col_xtriangles, col_ytriangles], name='TRIANGLES')
    hdu_table_crosses = fits.BinTableHDU.from_columns([col_xcrosses, col_ycrosses], name='CROSSES')

    modulation_hdu = fits.open(datalist[-1].filename)['MODULATION']

    header = datalist[-1].header
    # Définir le chemin complet du sous-dossier "output/couplingmaps"
    folder = datalist[-1].dirname
    output_dir = os.path.join(folder,"../couplingmaps")

    header['X_FIRTYP'] = 'COUPLINGMAP'
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
    header['CM_CHECK'] = np.random.randint(0, 2**32, dtype=np.uint32)
    for i, filename in enumerate(filenames):
        header['FILE_%i' % i] = filename

    # Créer les dossiers "output" et "pixel" s'ils n'existent pas déjà
    os.makedirs(output_dir, exist_ok=True)

    hdu_primary.header.extend(header, strip=True)

    # Combine all HDUs into an HDUList
    hdul = fits.HDUList([hdu_primary, hdu_1, hdu_2, hdu_3, hdu_4,hdu_5, hdu_6,
                         hdu_table_mod,hdu_table_triangle,hdu_table_crosses,modulation_hdu,
                         hdu_fluxmap, hdu_modes2D,hdu_flat])

    output_filename = os.path.join(output_dir, runlib.create_output_filename(header))

    # Write to a FITS file
    print(f"Saving data to {output_filename}")
    hdul.writeto(output_filename, overwrite=True)


    runlib_i.generate_plots(filenames, datacube, xmod, ymod, masque_positions, flux_2_data, 
                            singular_values, Nsingular, chi2_delta, flux_goodData, 
                            modes_rect, modes_mean, 
                            chi2_goodData, flux_threshold, chi2_threshold, output_filename)



# %%
