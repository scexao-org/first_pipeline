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
from scipy.linalg import solve_triangular


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
import subspace_numpy as ss
from scipy import odr
from scipy.optimize import least_squares

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
        wollaston = hd.get('X_FIRWOL', None)
        fits_keywords['OBJECT'] = [object_name]
        fits_keywords['X_FIRMID'] = [modID]
        fits_keywords['X_FIRMSC'] = [modScale]
        if wollaston is not None:
            fits_keywords['X_FIRWOL'] = [wollaston]

        print("----------------")
        print(f"Selected object='{object_name}' with modScale={modScale}, modID={modID}, and wollaston={wollaston}")
        print("----------------")

        filelist = runlib.get_filelist(file_patterns, fits_keywords)

        # finding darks files
        fits_keywords = {'X_FIRTYP': ['PREPROC'],
                        'DATA-TYP': ['DARK'],
                        }
        if wollaston is not None:
            fits_keywords['X_FIRWOL'] = [wollaston]

        try:
            filelist_dark = runlib.get_filelist(dark_patterns, fits_keywords,  name_search="dark")
        except FileNotFoundError as e:
            print(f"WARNING!!! {e}")
            filelist_dark = []

        # finding flats files
        fits_keywords = {'X_FIRTYP': ['PREPROC'],
                        'DATA-TYP': ['FLAT'],
                        }    
        if wollaston is not None:
            fits_keywords['X_FIRWOL'] = [wollaston]
            
        try:
            filelist_flat = runlib.get_filelist(flat_patterns, fits_keywords,  name_search="flat")
        except FileNotFoundError as e:
            print(f"WARNING!!! {e}")
            filelist_flat = filelist

        files_with_dark = runlib.associate_dark(filelist, filelist_dark)
        flats_with_dark = runlib.associate_dark(filelist_flat, filelist_dark)

        return files_with_dark, flats_with_dark


def compute_flat(flats_with_dark):
    
    datalist=runlib_i.extract_datacube(flats_with_dark, center = False)
    flats=[d.data.sum(axis=(0,1)) for d in datalist]
    flat=np.sum(flats,axis=0)
    flat/=np.mean(flat,axis=0)

    Nflat_smooth = 100
    # window = np.ones(Nflat_smooth)/Nflat_smooth
    window = np.hanning(Nflat_smooth)
    window /= window.sum()
    conv_ref = np.convolve(np.ones(len(flat[0])), window, mode='same')
    for f in flat:
        f[:] *= conv_ref / np.convolve(f, window, mode='same') 

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
    pos_2_singular = singular_2_data.T @ datacube.reshape((Nwave*Noutput,Ncube,Nmod)) #(57, 6250)
    datacube_filtered = singular_2_data @ pos_2_singular

    datacube_filtered = datacube_filtered.reshape((Nwave,Noutput,Ncube,Nmod))
    datacube = datacube.reshape((Nwave,Noutput,Ncube,Nmod))

    return datacube_filtered

def fit_QR(data,QT,R):

    def phi_vec(x, y):
        return np.array([1.0, x, y, x*y, x*x, y*y], dtype=float)
    
    def optimal_k(R, b, x, y):
        v = phi_vec(x, y)
        Rv = R @ v
        denom = Rv @ Rv
        if denom < 1e-14:
            return 0.0
        return (b @ Rv) / denom

    def resid(z):
        x, y = z
        k = optimal_k(R, b, x, y)
        return k * (R @ phi_vec(x, y)) - b  # 6-vector

    def jac(z):
        # analytic jacobian of residual wrt x,y
        x, y = z
        # first compute v and its jac
        v = phi_vec(x, y)
        dv_dx = np.array([0, 1, 0, y, 2*x, 0])
        dv_dy = np.array([0, 0, 1, x, 0, 2*y])
        Rv = R @ v
        denom = Rv @ Rv
        if denom < 1e-14:
            return np.zeros((6,2))

        # derivatives of k wrt x and y
        num = b @ Rv
        dRv_dx = R @ dv_dx
        dRv_dy = R @ dv_dy
        dk_dx = (b @ dRv_dx * denom - num * (2*Rv @ dRv_dx)) / (denom**2)
        dk_dy = (b @ dRv_dy * denom - num * (2*Rv @ dRv_dy)) / (denom**2)

        # residual = k Rv - b
        dr_dx = dk_dx * Rv + optimal_k(R,b,x,y) * dRv_dx
        dr_dy = dk_dy * Rv + optimal_k(R,b,x,y) * dRv_dy

        return np.column_stack([dr_dx, dr_dy])

    b = QT @ data # 6-vector

    #init 
    # cs = solve_triangular(R, b, lower=False)
    # x0 = float(cs[1]/cs[0]) if cs[0] != 0 else 0.0
    # y0 = float(cs[2]/cs[0]) if cs[0] != 0 else 0.0
    # init = (x0, y0)
    init = (0.0, 0.0)

    res = least_squares(resid, x0=np.array(init), jac=jac, method="trf")
    x_hat, y_hat = res.x
    k_hat = optimal_k(R, b, x_hat, y_hat)

    return x_hat, y_hat, k_hat, res.cost
    
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
    pos_2_singular = singular_2_data.T @ datacube.reshape((Nwave*Noutput,Ncube,Nmod)) #(57, 6250)

    singular_values = s #(3017,)
    pos_2_singular = pos_2_singular.reshape((Nsingular,Ncube,Nmod)) #reshape to (57, 10, 625)
    singular_2_data = singular_2_data.reshape((Nwave,Noutput,Nsingular))

    return pos_2_singular,singular_values,singular_2_data

def singular_value_filtering(datacube,flux_goodData,Nsingular):
    """
    Applies singular value filtering to the input datacube.

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
    # pos_2_singular = Vh[:Nsingular]*s[:Nsingular,None]
    singular_2_data = U[:,:Nsingular] #(3800, 119)
    pos_2_singular = singular_2_data.T @ datacube.reshape((Nwave*Noutput,Ncube,Nmod)) #(57, 6250)
    datacube_filtered = singular_2_data @ pos_2_singular
    datacube_filtered = datacube_filtered.reshape((Nwave,Noutput,Ncube,Nmod))
    datacube = datacube.reshape((Nwave,Noutput,Ncube,Nmod))
    return datacube_filtered,s


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
    parser.add_option("--compute_position", action="store_true", default=False,
                    help="Compute position of individual DITs (slow) (default: %default)")
    
    if ("VSCODE_PID" in os.environ or os.environ.get('TERM_PROGRAM') == 'vscode' or os.environ.get('SPYDER_DEBUG_FILE')):
        print("Running in compiler")
        flat_patterns = None
        dark_patterns = None
        modID = None
        modScale = None
        object_name = None
        wollaston = None
        compute_position = True
        if getpass.getuser() == "slacour":
            file_patterns = "/Users/slacour/DATA/LANTERNE/20250514/preproc/firstpl_2025-05-14T11?3*fits"
            file_patterns = "/Users/slacour/DATA/LANTERNE/20250808/preproc/firstpl_2025-08-08T06:4?:??_HIP84212_P.fits"
            # file_patterns = "/Users/slacour/DATA/LANTERNE/20250808/preproc/firstpl_2025-08-08T06:4[3-4]:??_HIP84212_P.fits"
            file_patterns = "/Users/slacour/DATA/LANTERNE/20250510/preproc/*10T09?2[0-3]*TETCRB_P.fits"
            file_patterns = "/Users/slacour/DATA/LANTERNE/20250514/preproc/firstpl_2025-05-14T11?3*s"
            dark_patterns = "/Users/slacour/DATA/LANTERNE/20250514/preproc"
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
        compute_position = options.compute_position

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
    datalist=runlib_i.extract_datacube(files_with_dark,Nsmooth=wavelength_smooth,Nbin=wavelength_bin, flat =flat)
    
    flux = np.concatenate([d.flux for d in datalist])
    datacube=np.concatenate([d.data for d in datalist])
    datacube_var=np.concatenate([d.variance for d in datalist])
    xmod=datalist[0].xmod[0]
    ymod=datalist[0].ymod[0]


    basenames = []
    for d in datalist:
        n = d.data.shape[0]  # first dimension of d.data
        basenames.extend([d.basename] * n)

    filenames = [d.filename for d in datalist]

    # datacube=datacube.transpose((3,2,0,1)) #to have (Nwave,Noutput,Ncube,Nmod)

    # select data only above a threshold based on flux
    flux_threshold=np.percentile(flux.mean(axis=(2)),80)/5
    flux_goodData=flux.mean(axis=(2)) > flux_threshold
    # plt.imshow(flux_goodData)
    if np.sum(flux_goodData)<57:
        #too little good data, we need to lower the bar
        flux_goodData=flux.mean(axis=(2,3)) > flux_threshold/2
        print("Not enough good data, lowering the threshold to ",flux_threshold/2)


    runlib_i.plot_couplinng_map(flux.mean(axis=(2))[0], xmod, ymod)
    # Remove mean per wavelength/output
    Nwave = datacube.shape[3]
    Noutput = datacube.shape[2]
    Ncube = datacube.shape[0]
    Nmod = datacube.shape[1]

    datacube_flux_goodData = datacube[flux_goodData]
    datacube_flux_goodData = datacube_flux_goodData.reshape((datacube_flux_goodData.shape[0], -1))
    res = ss.robust_subspace(datacube_flux_goodData, k=Nsingular, center=False, k_sigma=2.5, max_refit=1,verbose=True)
    singular_values = res["model"]["S"][:-1]
    data_svdfiltered, residuals, errors = ss.project(datacube.reshape((datacube.shape[0]*datacube.shape[1], -1)), res["model"])
    data_svdfiltered = data_svdfiltered.reshape(datacube.shape)
    fit_goodData = errors.reshape((datacube.shape[0], -1)) < res["threshold"]
    goodData = flux_goodData & fit_goodData
    spectra = flux[goodData].mean(axis=0)

    
    index_triangles, center_triangles = datalist[0].get_pyramids()
    goodTriangles = (goodData[:,index_triangles].mean(axis=0) > 0.3).sum(axis=1)  > 4
    goodPositions = goodData.mean(axis=0) > 0.3


    vectors_all_triangles = []
    center_all_triangles = []
    singular_all_triangles = []
    distance_xy_fit=[]

    # Ncubes = datacube.shape[0]

    for i in tqdm(np.arange(len(index_triangles))[goodTriangles][:], desc="Computing triangles"):
        t = index_triangles[i]
        center = center_triangles[i]
        center_all_triangles.append(center)

        good_data_triangle=goodData[:,t]
        data_triangle = data_svdfiltered[:,t][good_data_triangle]
        data_triangle = data_triangle.reshape((data_triangle.shape[0], -1))
        xmod_triangle = np.broadcast_to(xmod[t], good_data_triangle.shape)[good_data_triangle] - center[0]
        ymod_triangle = np.broadcast_to(ymod[t], good_data_triangle.shape)[good_data_triangle] - center[1]
        xymod_triangle = np.array([xmod_triangle, ymod_triangle])

        svd_res = ss.robust_subspace(data_triangle, k=6, center=False, k_sigma=3.5, max_refit=1)
        V = svd_res["model"]["V"]
        D = data_triangle.T

        def phi(xy):
            Xv, Yv = xy[0], xy[1]
            return np.vstack([np.ones_like(Xv), Xv, Yv, Xv*Yv, Xv**2, Yv**2 ])  # (6,)

        ## on a la relation D = V.M.P
        ## que l'on peut ecrire B = M.P
        ## avec B = VT.D
        B = V.T @ D
        ## et avec P la matrice des positions (x,y,xy,x^2,y^2)
        P = phi(xymod_triangle)

        ## V la matrice des vecteurs singuliers
        ## On cherche M (6,6) 
        ## On pose B = VT.Y
        B = V.T @ D

        xy_new = xymod_triangle.copy()
        xy_old = xymod_triangle.copy()

        M = B @ np.linalg.pinv(P)

        # Errors-in-Variables alternating minimization.
        # B       : (6,n) observations 
        # xymod_triangle : (n,) mesures bruitées des entrées
        # sigma : écart-types des erreurs sur X,Y

        sigma = 1
        max_iter = 5
        for it in range(max_iter):
            for i in range(len(B[0])):
                def resid(z):
                    r_model=B[:,i] - M @ phi(z)[:,0]
                    r_prior=(xy_old[:,i]-z)/sigma

                    return np.concatenate([r_model,r_prior])    
                z = least_squares(resid, x0=xy_new[:,i])
                if z.success:
                    # print("success")
                    xy_new[:,i] = z.x
            P = phi(xy_new)
            M = B @ np.linalg.pinv(P)

        Vectors_triangle = (V @ M) #(n,6)
        vectors_all_triangles.append(Vectors_triangle)

        singular_values = svd_res["model"]["S"]
        singular_values /= np.linalg.norm(singular_values)
        Nvalues = len(t)*len(data_svdfiltered)
        if len(singular_values) < Nvalues:
            singular_values = np.concatenate([singular_values, np.full(Nvalues - len(singular_values), np.nan)])
        singular_all_triangles.append(singular_values)

        Q, R = np.linalg.qr(Vectors_triangle, mode="reduced")


        error_xy = np.zeros((Ncube,len(t),2))
        for c in range(Ncube):
            for ti in range(len(t)): 

                data=datacube[c,t[ti]].ravel()
                x_theory = xmod[t[ti]]
                y_theory = ymod[t[ti]]


                x_hat, y_hat, k_hat, chi2 = fit_QR(data, Q.T, R)

                X_measured = x_hat + center[0]
                Y_measured = y_hat + center[1]

                error_xy[c,ti] = ((X_measured - x_theory) , (Y_measured - y_theory))
        distance_xy_fit.append(error_xy)
        
    distance_xy_fit = np.array(distance_xy_fit)
    singular_all_triangles = np.array(singular_all_triangles)
    center_all_triangles = np.array(center_all_triangles)
    vectors_all_triangles = np.array(vectors_all_triangles)
    Ntriangles = vectors_all_triangles.shape[0]
    vectors_all_triangles = vectors_all_triangles.reshape((Ntriangles, Noutput, Nwave,6))
    vectors_normalisation = np.linalg.norm(vectors_all_triangles[:,:,:,0],axis=1,keepdims=True)
    # vectors_all_triangles_normed = vectors_all_triangles/vectors_normalisation[:,:,:,None]
    vectors_all_triangles_normed = vectors_all_triangles/spectra[:,None]
    flux_2_data = vectors_all_triangles_normed[:,:,:,0]
    flux_2_data = flux_2_data.transpose((2,1,0))
    data_2_flux = np.zeros((Nwave, Ntriangles, Noutput))
    for w in tqdm(range(Nwave), desc="Inverting flux_2_data to data_2_flux for each wavelength"):
        data_2_flux[w] = pinv(flux_2_data[w])


    fluxtiptilt_2_data = vectors_all_triangles_normed
    fluxtiptilt_2_data = fluxtiptilt_2_data.transpose((0,2,1,3))
    QT_fluxtiptilt_2_data = np.zeros((Ntriangles,Nwave,6,Noutput))
    R_fluxtiptilt_2_data = np.zeros((Ntriangles,Nwave,6,6))

    for w in tqdm(range(Nwave), desc = "Calculating QR matrices for fluxtiptilt_2_data"):
        for p in range(Ntriangles):
            Q, R = np.linalg.qr(fluxtiptilt_2_data[p,w], mode="reduced")
            QT_fluxtiptilt_2_data[p,w] = Q.T
            R_fluxtiptilt_2_data[p,w] = R


    ############### Save results ####################
    # Save arrays into a FITS file

    # Create a primary HDU with no data, just the header
    hdu_primary = fits.PrimaryHDU()

    # Create HDUs for each array
    hdu_1 = fits.ImageHDU(data=flux_2_data, name='F2DATA')
    hdu_2 = fits.ImageHDU(data=data_2_flux, name='DATA2F')
    hdu_3 = fits.ImageHDU(data=fluxtiptilt_2_data, name='FTT2DATA')
    hdu_4 = fits.ImageHDU(data=QT_fluxtiptilt_2_data, name='QT_FTT2DATA')
    hdu_5 = fits.ImageHDU(data=R_fluxtiptilt_2_data, name='R_FTT2DATA')
    hdu_6 = fits.ImageHDU(data=center_all_triangles, name='XY_POS')
    hdu_7 = fits.ImageHDU(data=flat, name='FLAT')

    modulation_hdu = fits.open(datalist[-1].filename)['MODULATION']

    header = datalist[-1].header
    # Définir le chemin complet du sous-dossier "output/couplingmaps"
    folder = datalist[-1].dirname
    output_dir = os.path.join(folder,"../couplingmaps")

    header['X_FIRTYP'] = 'COUPLINGMAP'
    header['X_FIRWOL'] = header.get('X_FIRWOL', 'IN')

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
    # header['CHI2THR'] = chi2_threshold  # Add chi2 threshold
    header['CM_CHECK'] = np.random.randint(0, 2**32, dtype=np.uint32)
    for i, filename in enumerate(filenames):
        header['FILE_%i' % i] = filename

    # Créer les dossiers "output" et "pixel" s'ils n'existent pas déjà
    os.makedirs(output_dir, exist_ok=True)

    hdu_primary.header.extend(header, strip=True)

    # Combine all HDUs into an HDUList
    hdul = fits.HDUList([hdu_primary, hdu_1, hdu_2, hdu_3, hdu_4,  hdu_5,  hdu_6, hdu_7])

    output_filename = os.path.join(output_dir, runlib.create_output_filename(header))

    # Write to a FITS file
    print(f"Saving data to {output_filename}")
    hdul.writeto(output_filename, overwrite=True)



    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    ###############################################
    # Diagnostic plots
    ###############################################
    fig_resid, axs_resid = plt.subplots(1, Ncube, figsize=(4*Ncube, 6), num="Residuals in xy of each dataset", clear=True)
    for c in range(Ncube):
        ax = axs_resid[c] if Ncube > 1 else axs_resid
        dx = distance_xy_fit[:, c, :, 0].ravel()
        dy = distance_xy_fit[:, c, :, 1].ravel()
        bins = np.linspace(-20, 20, 41)
        counts_dx, _, _ = ax.hist(dx, bins=bins, alpha=0.7, color='tab:blue', edgecolor='black')
        counts_dy, _, _ = ax.hist(dy, bins=bins, alpha=0.7, color='tab:orange', edgecolor='black')
        # Median values
        median_dx = np.nanmedian(dx)
        median_dy = np.nanmedian(dy)
        ax.axvline(median_dx, color='tab:blue', linestyle='--', linewidth=2, label=f'Median dx: {median_dx:.2f}')
        ax.axvline(median_dy, color='tab:orange', linestyle='--', linewidth=2, label=f'Median dy: {median_dy:.2f}')
        # 1 sigma percentiles
        dx_p16, dx_p84 = np.nanpercentile(dx, [16, 84])
        dy_p16, dy_p84 = np.nanpercentile(dy, [16, 84])
        ax.axvline(dx_p16, color='tab:blue', linestyle=':', linewidth=2, label=f'dx 1σ: {dx_p16:.2f}')
        ax.axvline(dx_p84, color='tab:blue', linestyle=':', linewidth=2)
        ax.axvline(dy_p16, color='tab:orange', linestyle=':', linewidth=2, label=f'dy 1σ: {dy_p16:.2f}')
        ax.axvline(dy_p84, color='tab:orange', linestyle=':', linewidth=2)
        ax.set_title(f"{basenames[c][8:]}")
        ax.set_xlabel("Error [mas]")
        ax.set_ylabel("Count")
        ax.grid(True, linestyle=':', alpha=0.5)
        ax.legend()
    plt.tight_layout()
    plt.show()

    plt.figure("Singular Values per Triangle", figsize=(8, 6), clear=True)
    for i, sv in enumerate(singular_all_triangles):
        plt.plot(np.arange(1, len(sv) + 1), sv, alpha=0.3)
    mean_singular = np.nanmean(singular_all_triangles, axis=0)
    plt.plot(np.arange(1, len(mean_singular) + 1), mean_singular, '-o', color='k', linewidth=2, label='Mean Singular Values')
    plt.yscale('log')
    plt.xlim(1, len(mean_singular))
    plt.xlabel('Singular Value Index')
    plt.ylabel('Singular Value (log scale)')
    plt.title('Singular Values for All Triangles')
    plt.legend()
    plt.tight_layout()
    plt.show()

    fig_flat, ax_flat = plt.subplots(num="Flat Field", figsize=(12, 6), clear=True)
    im_flat = ax_flat.imshow(flat, aspect='auto', origin='lower', cmap='viridis', interpolation='none', rasterized=True)
    ax_flat.set_title("Flat Field ")
    ax_flat.set_xlabel("Wavelength Index")
    ax_flat.set_ylabel("Output Index")
    plt.colorbar(im_flat, ax=ax_flat, label="Flat Value")
    plt.tight_layout()
    plt.show()

    fig, axs = plt.subplots(2, 2, figsize=(12, 10), num="Flux/GoodData Selection", clear=True)

    # The data used to make them: mean flux per (wavelength, output)
    mean_flux = flux.mean(axis=(2))
    axs[0, 0].imshow(mean_flux, aspect='auto', origin='lower', cmap='viridis', interpolation='none', rasterized=True)
    axs[0, 0].set_title("Mean Flux (per wavelength/output)")
    axs[0, 0].set_xlabel("Output")
    axs[0, 0].set_ylabel("files")
    # Show the threshold as a horizontal line (if 1D), else as a contour
    # axs[0, 0].contour(flux_goodData, levels=[0.5], colors='r', linewidths=1, linestyles='--')

    # flux_goodData mask
    axs[0, 1].imshow(flux_goodData, aspect='auto', origin='lower', cmap='Greens', interpolation='none', rasterized=True, vmin=0, vmax=1)
    axs[0, 1].set_title("From flux, good Dataset (mask)")
    axs[0, 1].set_xlabel("Output")
    axs[0, 1].set_ylabel("Wavelength")

    # The data used to make them: mean flux per (wavelength, output)
    error_norm = errors.reshape((datacube.shape[0], -1))
    axs[1, 0].imshow(error_norm, aspect='auto', origin='lower', cmap='viridis', interpolation='none', rasterized=True)
    axs[1, 0].set_title("Amplitude of residuals after SVD filtering")
    axs[1, 0].set_xlabel("Output")
    axs[1, 0].set_ylabel("files")
    # Show the threshold as a horizontal line (if 1D), else as a contour
    # axs[1, 0].contour(fit_goodData, levels=[0.5], colors='r', linewidths=1, linestyles='--')

    # flux_goodData mask
    axs[1, 1].imshow(fit_goodData, aspect='auto', origin='lower', cmap='Greens', interpolation='none', rasterized=True, vmin=0, vmax=1)
    axs[1, 1].set_title("From SVD fits, good Dataset (mask)")
    axs[1, 1].set_xlabel("Output")
    axs[1, 1].set_ylabel("Wavelength")


    plt.tight_layout()
    plt.show()

    fig, axs = plt.subplots(1, 2, num=" Positions fiber and of triangles" , figsize=(18, 6), sharex=True, sharey=True, clear=True)

    # 1. Plot positions (xmod, ymod) for all triangles
    axs[0].set_title("Positions of Fiber")
    axs[0].scatter(xmod, ymod, c='k', marker='.')
    axs[0].scatter(xmod[goodPositions], ymod[goodPositions], facecolors='g', marker='o', edgecolor='k', label='Good Positions')
    axs[0].set_xlabel("x [mas]")
    axs[0].set_ylabel("y [mas]")
    axs[0].set_aspect('equal')
    axs[0].legend()

    # 1. Plot positions (xmod, ymod) for all triangles
    axs[1].set_title("Positions of Triangles")
    axs[1].scatter(center_triangles[:, 0], center_triangles[:, 1], c='k', marker='.')
    axs[1].scatter(center_triangles[goodTriangles, 0], center_triangles[goodTriangles, 1], facecolors='g', marker='o', edgecolor='k', label='Good Triangles')
    axs[1].set_xlabel("x [mas]")
    axs[1].set_ylabel("y [mas]")
    axs[1].set_aspect('equal')
    axs[1].legend()


    ###############################################
    # Covariance and correlation matrix plot
    ###############################################

    cov_matrix = np.cov(flux_2_data.reshape((Nwave*Noutput,Ntriangles)).T)
    cor_matrix = np.corrcoef(flux_2_data.reshape((Nwave*Noutput,Ntriangles)).T)

    fig, ax = plt.subplots(1, 2, num='Covariance and Correlation Matrix', figsize=(12, 6), clear=True)
    cax0 = ax[0].matshow(cov_matrix, cmap='viridis')
    fig.colorbar(cax0, ax=ax[0])
    cax1 = ax[1].matshow(cor_matrix, cmap='viridis')
    fig.colorbar(cax1, ax=ax[1])
    ax[0].set_title('Covariance Matrix of Singular Vector Models')
    ax[1].set_title('Correlation Matrix of Singular Vector Models')
    fig.tight_layout()


    if compute_position:
        datacube=np.concatenate([d.data for d in datalist])

        datacube_T=datacube.transpose((3,2,0,1))
        datacube_T=datacube_T.reshape((datacube_T.shape[0], datacube_T.shape[1], -1))
        chi2_max = np.sum(datacube_T**2, axis=(0,1))

        chi2_map = np.zeros((Ntriangles,Ncube * Nmod))
        chi2_map = np.zeros((Ntriangles, Ncube * Nmod))
        chi2_map[:] =  chi2_max
        # gram_fluxtiptilt_inv =np.zeros_like(gram_fluxtiptilt)
        # for t in range(Ntriangles):
        #     for w in range(Nwave):
        #         gram_fluxtiptilt_inv[t,w] = linalg.pinv(gram_fluxtiptilt[t,w])
        for t in tqdm(range(Ntriangles), desc="Computing chi2 map"):
            k= QT_fluxtiptilt_2_data[t] @ datacube_T
            chi2_map[t,:] -= np.sum(k ** 2, axis=(0,1))

        chi2_argmin = chi2_map.argmin(axis=0)
        residuals = datacube_T.copy()

        distances = np.linalg.norm((xmod- center_all_triangles[:,0,None], ymod- center_all_triangles[:,1,None]),axis=0)
        # chi2_argmin = np.argmin(distances,axis=0)

        Xpos = np.zeros(Ncube * Nmod)
        Ypos = np.zeros(Ncube * Nmod)
        Xcen = np.zeros(Ncube * Nmod)
        Ycen = np.zeros(Ncube * Nmod)
        OK =[]
        for i in tqdm(range(Ncube * Nmod), desc="Computing XY positions"):
            t = chi2_argmin[i]
            center = center_all_triangles[t]
            data = datacube_T[:,:,i]
            QT = QT_fluxtiptilt_2_data[t]
            R = R_fluxtiptilt_2_data[t]

            res = []
            for w in range(Nwave):
                data_w = data[w]
                QT_w = QT[w]
                R_w = R[w]
                x_hat, y_hat, k_hat, chi2 = fit_QR(data_w, QT_w, R_w)
                res.append((x_hat, y_hat, k_hat, chi2))
            res = np.array(res)


            Xpos[i] = np.median(res[:,0])
            Ypos[i] = np.median(res[:,1])

            Xcen[i] = center[0]
            Ycen[i] = center[1]
        
        Xpos = Xpos.reshape((Ncube, Nmod))
        Ypos = Ypos.reshape((Ncube, Nmod))
        Xcen = Xcen.reshape((Ncube, Nmod))
        Ycen = Ycen.reshape((Ncube, Nmod))
        
        fig, axs = plt.subplots(1, Ncube, num="XY position", clear=True,sharex=True, sharey=True, figsize=(7*Ncube,6), squeeze=False)
        axs=axs[0]
        for i in range(Ncube):
            axs[i].plot(Xcen[i],Ycen[i],'.',label='Center of pyramids')
            axs[i].set_ylim(axs[i].get_ylim()[0], axs[i].get_ylim()[1])
            axs[i].set_xlim(axs[i].get_xlim()[0], axs[i].get_xlim()[1])
            axs[i].plot((Xcen+Xpos)[i],(Ycen+Ypos)[i],'.-',label='Detected position')
            axs[i].plot((Xcen[i],(Xcen+Xpos)[i]),(Ycen[i],(Ycen+Ypos)[i]),'-k',alpha=0.3,linewidth=0.5)
            axs[i].set_title(basenames[i][8:])
            axs[i].set_xlabel("X [mas]")
            axs[i].set_ylabel("Y [mas]")
            axs[i].legend()
            for ax in axs:
                ax.set_aspect('equal')
        plt.tight_layout()



    # Save all open figures to a PDF
    from matplotlib.backends.backend_pdf import PdfPages
    pdf_filename = os.path.splitext(output_filename)[0] + ".pdf"
    with PdfPages(pdf_filename) as pdf:
        for i in plt.get_fignums():
            fig = plt.figure(i)
            pdf.savefig(fig)
    print(f"All figures saved to {pdf_filename}")

# %%
