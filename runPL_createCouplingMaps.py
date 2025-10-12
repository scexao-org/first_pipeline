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
from scipy.ndimage import uniform_filter1d

from scipy.spatial.distance import pdist, squareform
from scipy.optimize import curve_fit
from fit_QR import compute_broadband_QR,solve_QR_2,solve_QR_3,fit_QR_6

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

        filelist = runlib.get_filelist(file_patterns, fits_keywords)

        print(f"Found {len(filelist)} files matching criteria.")
        print("----------------")

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

def singular_vector_basis(data_svdfiltered,goodData,indexes, centers, xmod, ymod):

    vectors_all_triangles = []
    center_all_triangles = []
    Ntriangles,Nqr = indexes.shape
    if Nqr == 3:
        description = "Computing triangles singular vectors"
    else:
        description = "Computing pyramids singular vectors"

    for i in tqdm(np.arange(len(indexes)), desc=description):

        # as a first step 
        # extract the singular vectors for each triangle or pyramid
        t = indexes[i]
        center = centers[i]
        center_all_triangles.append(center)

        good_data_triangle=goodData[:,t]
        data_triangle = data_svdfiltered[:,t][good_data_triangle]
        data_triangle = data_triangle.reshape((data_triangle.shape[0], -1))
        xmod_triangle = xmod[:,t][good_data_triangle] - center[0]
        ymod_triangle = ymod[:,t][good_data_triangle] - center[1]
        xymod_triangle = np.array([xmod_triangle, ymod_triangle])

        svd_res = ss.robust_subspace(data_triangle, k=Nqr, center=False, k_sigma=3.5, max_refit=1)
        V = svd_res["model"]["V"]

        # as a second step
        # now that we have the basis V of singular vectors, we want to fit the polynomial model
        # to get the coefficients of the polynomial for each singular vector


        ############# Errors-in-Variables fitting #############
        # We want to fit B = M.P where B = VT.D and P is the polynomial basis
        # We have noisy measurements of B and P, so we use an alternating minim
        # imization to estimate the true P and M
        # see https://arxiv.org/abs/2305.17180 for details (reference from chatGPT)
        

        def phi(xy):
            Xv, Yv = xy[0], xy[1]
            if Nqr == 6:
                return np.vstack([np.ones_like(Xv), Xv, Yv, Xv*Yv, Xv**2, Yv**2 ])  # (6,)
            elif Nqr == 3:
                # return np.identity(3)
                return np.vstack([np.ones_like(Xv), Xv, Yv])  # (3,)
            else:
                return None

        ## on a la relation D = V.M.P
        ## que l'on peut ecrire B = M.P
        ## avec B = VT.D
        D = data_triangle.T
        B = V.T @ D
        ## et avec P la matrice des positions (x,y,xy,x^2,y^2)

        xy_new = xymod_triangle.copy()

        # Errors-in-Variables alternating minimization.
        # B       : (6,n) observations 
        # xymod_triangle : (n,) mesures bruitées des entrées
        # sigma : écart-types des erreurs sur X,Y

        # initial estimate of M and P
        P = phi(xymod_triangle)
        M = B @ np.linalg.pinv(P)

        # initial estimate of noise levels
        sigma_B = (B-M @ P).std(axis=1) + 1
        sigma_B = np.sqrt(np.mean(sigma_B)**2+sigma_B**2)/np.sqrt(2)
        sigma_pos = np.linalg.norm(xymod_triangle-xymod_triangle.mean(axis=1)[:,None],axis=0).mean()/10
        max_iter = 10
        for it in range(max_iter):
            for i in range(len(B[0])):   
                def resid(z):
                    r_model=(B[:,i] - M @ phi(z)[:,0])/sigma_B
                    r_prior=(xymod_triangle[:,i]-z)/sigma_pos
                    return np.concatenate([r_model,r_prior]) 
                z = least_squares(resid, x0=xy_new[:,i])
                if z.success:
                    # print("success")
                    xy_new[:,i] = z.x
            P = phi(xy_new)
            M = B @ np.linalg.pinv(P)
            sigma_B = (B-M @ P).std(axis=1) + 1
            sigma_B = np.sqrt(np.mean(sigma_B)**2+sigma_B**2)/np.sqrt(2)


        Vectors_triangle = (V @ M) #(n,6)
        vectors_all_triangles.append(Vectors_triangle)

    Noutput, Nwave = data_svdfiltered.shape[2:]

    center_all_triangles = np.array(center_all_triangles)
    vectors_all_triangles = np.array(vectors_all_triangles).reshape((Ntriangles, Noutput, Nwave, Nqr))

    return vectors_all_triangles, center_all_triangles


def flux_matrices(singular_vectors):

    Ntriangles = singular_vectors.shape[0]
    Noutput = singular_vectors.shape[1]
    Nwave = singular_vectors.shape[2]

    flux_2_data = singular_vectors[:,:,:,0]
    flux_2_data = flux_2_data.transpose((2,1,0))
    data_2_flux = np.zeros((Nwave, Ntriangles, Noutput))
    data_2_flux = np.linalg.pinv(flux_2_data)

    return flux_2_data,data_2_flux


def Q_and_R_matrices(singular_vectors):

    Ntriangles = singular_vectors.shape[0]
    Noutput = singular_vectors.shape[1]
    Nwave = singular_vectors.shape[2]
    Nqr = singular_vectors.shape[3]

    singular_vectors = singular_vectors.transpose((0,2,1,3))
    QT_singular_vectors = np.zeros((Ntriangles,Nwave,Nqr,Noutput))
    R_singular_vectors = np.zeros((Ntriangles,Nwave,Nqr,Nqr))

    if Nqr == 3:
        description = "Calculating QR matrices for triangles"
    else:
        description = "Calculating QR matrices for pyramids"

    for p in tqdm(range(Ntriangles), desc = description):
        for w in range(Nwave):
            Q, R = np.linalg.qr(singular_vectors[p,w], mode="reduced")
            QT_singular_vectors[p,w] = Q.T
            R_singular_vectors[p,w] = R

    return QT_singular_vectors,R_singular_vectors

def save_allfig_pdf(output_filename):
    # Save all open figures to a PDF
    from matplotlib.backends.backend_pdf import PdfPages
    pdf_filename = os.path.splitext(output_filename)[0] + ".pdf"
    with PdfPages(pdf_filename) as pdf:
        for i in plt.get_fignums():
            fig = plt.figure(i)
            pdf.savefig(fig)
    print(f"All figures saved to {pdf_filename}")

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


if __name__ == "__main__":
    parser = OptionParser(usage)

    # Default values
    wavelength_smooth = 20
    wavelength_bin = 1
    Nsingular=19*6 


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
            # file_patterns = "/Users/slacour/DATA/LANTERNE/20250510/preproc/*10T09?21*TETCRB_P.fits"
            # file_patterns = "/Users/slacour/DATA/LANTERNE/20250514/preproc/firstpl_2025-05-14T11?3*s"
            # dark_patterns = "/Users/slacour/DATA/LANTERNE/20250514/preproc"
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
    xmod=np.concatenate([d.xmod for d in datalist])
    ymod=np.concatenate([d.ymod for d in datalist])


    basenames = []
    for d in datalist:
        n = d.data.shape[0]  # first dimension of d.data
        basenames.extend([d.basename] * n)

    filenames = [d.filename for d in datalist]


    def flux_filtering(flux):
        
        # select data only above a threshold based on flux
        flux_threshold=np.percentile(flux.mean(axis=(2)),80)/5
        flux_goodData=flux.mean(axis=(2)) > flux_threshold
        # plt.imshow(flux_goodData)
        if np.sum(flux_goodData)<57:
            #too little good data, we need to lower the bar
            flux_goodData=flux.mean(axis=(2,3)) > flux_threshold/2
            print("Not enough good data, lowering the threshold to ",flux_threshold/2)
            flux_goodData=flux.mean(axis=(2)) > flux_threshold

        return flux_goodData,flux_threshold

    flux_goodData,flux_threshold = flux_filtering(flux)

    def svd_filtering(datacube,flux_goodData,Nsingular):

        datacube_flux_goodData = datacube[flux_goodData]
        datacube_flux_goodData = datacube_flux_goodData.reshape((datacube_flux_goodData.shape[0], -1))
        res = ss.robust_subspace(datacube_flux_goodData, k=Nsingular, center=False, k_sigma=2.5, max_refit=1,verbose=True)
        singular_values = res["model"]["S"][:-1]
        data_svdfiltered, residuals, errors = ss.project(datacube.reshape((datacube.shape[0]*datacube.shape[1], -1)), res["model"])
        data_svdfiltered = data_svdfiltered.reshape(datacube.shape)
        fit_goodData = errors.reshape((datacube.shape[0], -1)) < res["threshold"]
        goodData = flux_goodData & fit_goodData
        

        return data_svdfiltered,goodData,errors

    data_svdfiltered,goodData,errors = svd_filtering(datacube,flux_goodData,Nsingular)

    runlib_i.plot_couplinng_map(flux.mean(axis=(2))[0], xmod[0], ymod[0])

    goodPositions = goodData.mean(axis=0) > 0.3
    index_triangles , center_triangles = datalist[0].get_triangles()
    index_pyramids, center_pyramids = datalist[0].get_pyramids()

    # Select only triangles with good data
    goodTriangles = goodPositions[index_triangles].mean(axis=1)  == 1
    index_triangles=index_triangles[goodTriangles]
    center_triangles=center_triangles[goodTriangles]
    # Select only pyramids with good data
    goodPyramids = goodPositions[index_pyramids].mean(axis=1)  == 1
    index_pyramids=index_pyramids[goodPyramids]
    center_pyramids=center_pyramids[goodPyramids]

    indexes = index_triangles
    centers = center_triangles

    vectors_all_triangles, center_all_triangles = singular_vector_basis(data_svdfiltered,goodData,index_triangles,center_triangles, xmod, ymod)
    vectors_all_pyramids, center_all_pyramids = singular_vector_basis(data_svdfiltered,goodData,index_pyramids,center_pyramids, xmod, ymod)


    # Ntriangles = vectors_all_triangles.shape[0]
    # vectors_all_triangles = vectors_all_triangles.reshape((Ntriangles, Noutput, Nwave,6))
    spectra = flux[goodData].mean(axis=0)
    vectors_all_triangles = vectors_all_triangles/spectra[:,None]
    vectors_all_pyramids = vectors_all_pyramids/spectra[:,None]

    #getting the flux 2 data matrices
    flux_2_data_triangles,data_2_flux_triangles = flux_matrices(vectors_all_triangles)
    flux_2_data_pyramids,data_2_flux_pyramids = flux_matrices(vectors_all_pyramids)

    #getting the Q and R matrices
    QT_triangles,R_triangles = Q_and_R_matrices(vectors_all_triangles)
    QT_pyramids,R_pyramids = Q_and_R_matrices(vectors_all_pyramids)

    ############### Save results ####################
    # Save arrays into a FITS file

    # Create a primary HDU with no data, just the header
    hdu_primary = fits.PrimaryHDU()

    # Create HDUs for each array
    hdu = [fits.ImageHDU(data=flux_2_data_triangles, name='F2DATA_T')]
    hdu += [fits.ImageHDU(data=data_2_flux_triangles, name='DATA2F_T')]
    hdu += [fits.ImageHDU(data=QT_triangles, name='QT_T')]
    hdu += [fits.ImageHDU(data=R_triangles, name='R_T')]
    hdu += [fits.ImageHDU(data=center_triangles, name='XY_T')]
    hdu += [fits.ImageHDU(data=flux_2_data_pyramids, name='F2DATA_P')]
    hdu += [fits.ImageHDU(data=data_2_flux_pyramids, name='DATA2F_P')]
    hdu += [fits.ImageHDU(data=QT_pyramids, name='QT_P')]
    hdu += [fits.ImageHDU(data=R_pyramids, name='R_P')]
    hdu += [fits.ImageHDU(data=center_pyramids, name='XY_P')]
    hdu += [fits.ImageHDU(data=flat, name='FLAT')]
    hdu += [fits.ImageHDU(data=spectra, name='SPECTRA')]

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
    header['P_CMWSMO'] = wavelength_smooth  # Add wavelength smoothing factor
    header['P_CMWBIN'] = wavelength_bin
    header['P_CMSING'] = Nsingular  # Add number of singular values
    header['P_CM_FT'] = flux_threshold  # Add flux threshold
    # header['CHI2THR'] = chi2_threshold  # Add chi2 threshold
    header['P_CM_CK'] = np.random.randint(0, 2**32, dtype=np.uint32)
    for i, filename in enumerate(filenames):
        header['P_CM_F%i' % i] = filename

    header['P_CMNAME'] = runlib.create_output_filename(header)

    # Créer les dossiers "output" et "pixel" s'ils n'existent pas déjà
    os.makedirs(output_dir, exist_ok=True)

    hdu_primary.header.extend(header, strip=True)

    # Combine all HDUs into an HDUList
    hdul = fits.HDUList([hdu_primary, *hdu, modulation_hdu])

    output_filename = os.path.join(output_dir, runlib.create_output_filename(header))

    # Write to a FITS file
    print(f"Saving data to {output_filename}")
    hdul.writeto(output_filename, overwrite=True)


    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    ###############################################
    # Diagnostic plots
    ###############################################

    fig_flat, ax_flat = plt.subplots(num="Flat Field", figsize=(12, 6), clear=True)
    im_flat = ax_flat.imshow(flat, aspect='auto', origin='lower', cmap='viridis', interpolation='none', rasterized=True)
    ax_flat.set_title("Flat Field ")
    ax_flat.set_xlabel("Wavelength Index")
    ax_flat.set_ylabel("Output Index")
    plt.colorbar(im_flat, ax=ax_flat, label="Flat Value")
    plt.tight_layout()
    plt.show()

    dark = np.array([d.dark for d in datalist]).mean(axis=0)
    if dark.ndim == 0 or dark.shape == ():
        dark = np.full_like(flat, dark)
    elif dark.ndim == 1:
        dark = np.tile(dark, (flat.shape[0], 1))

    fig_dark, ax_dark = plt.subplots(num="Dark Field", figsize=(12, 6), clear=True)
    im_dark = ax_dark.imshow(dark, aspect='auto', origin='lower', cmap='viridis', interpolation='none', rasterized=True)
    ax_dark.set_title("Dark Field ")
    ax_dark.set_xlabel("Wavelength Index")
    ax_dark.set_ylabel("Output Index")
    plt.colorbar(im_dark, ax=ax_dark, label="Dark Value")
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
    axs[1, 1].imshow(goodData, aspect='auto', origin='lower', cmap='Greens', interpolation='none', rasterized=True, vmin=0, vmax=1)
    axs[1, 1].set_title("From SVD fits, good Dataset (mask)")
    axs[1, 1].set_xlabel("Output")
    axs[1, 1].set_ylabel("Wavelength")


    plt.tight_layout()
    plt.show()

    fig, axs = plt.subplots(1, 2, num=" Positions fiber and of triangles" , figsize=(18, 6), sharex=True, sharey=True, clear=True)

    # 1. Plot positions (xmod, ymod) for all triangles
    axs[0].set_title("Positions of Fiber")
    axs[0].scatter(xmod, ymod, c='k', marker='.')
    axs[0].scatter(xmod[0,goodPositions], ymod[0,goodPositions], facecolors='g', marker='o', edgecolor='k', label='Good Positions')
    axs[0].set_xlabel("x [mas]")
    axs[0].set_ylabel("y [mas]")
    axs[0].set_aspect('equal')
    axs[0].legend()

    # 1. Plot positions (xmod, ymod) for all triangles
    axs[1].set_title("Positions of Triangles")
    axs[1].scatter(center_triangles[:, 0], center_triangles[:, 1], c='k', marker='.')
    axs[1].scatter(center_triangles[:, 0], center_triangles[:, 1], facecolors='g', marker='o', edgecolor='k', label='Good Triangles')
    axs[1].set_xlabel("x [mas]")
    axs[1].set_ylabel("y [mas]")
    axs[1].set_aspect('equal')
    axs[1].legend()


    ###############################################
    # Covariance and correlation matrix plot
    ###############################################


    def plot_covariance(flux_2_data_triangles,centers,name):
        Ntriangles = flux_2_data_triangles.shape[2]
        cov_matrix = np.cov(flux_2_data_triangles.reshape((-1,Ntriangles)).T)
        cor_matrix = np.corrcoef(flux_2_data_triangles.reshape((-1,Ntriangles)).T)

        fig, ax = plt.subplots(1, 3, num='Covariance and Correlation Matrix for '+name, figsize=(16, 6), clear=True)
        fig.suptitle(name)
        cax0 = ax[0].matshow(cov_matrix, cmap='viridis')
        fig.colorbar(cax0, ax=ax[0])
        cax1 = ax[1].matshow(cor_matrix, cmap='viridis')
        fig.colorbar(cax1, ax=ax[1])
        ax[0].set_title('Covariance Matrix of Singular Vector Models')
        ax[1].set_title('Correlation Matrix of Singular Vector Models')
        # Compute pairwise distances between centers
        distances = np.linalg.norm((centers-centers[:,None]),axis=2).ravel()
        correlations = cor_matrix.ravel()

        # Scatter plot: correlation vs distance
        ax[2].plot(distances, correlations,'.', alpha=0.1,  label='Pairs')
        ax[2].set_xlabel('Distance between vectors')
        ax[2].set_ylabel('Correlation')
        ax[2].set_title('Correlation vs Distance')
        # Smooth correlation vs distance using a moving average

        # Sort by distance for smoothing
        sort_idx = np.argsort(distances)
        distances_sorted = distances[sort_idx]
        correlations_sorted = correlations[sort_idx]

        window_size = max(10, len(cor_matrix))
        if window_size % 2 == 0:
            window_size += 1  # Ensure odd window size

        # Moving average smoothing
        correlations_smoothed = uniform_filter1d(correlations_sorted, size=window_size, mode='nearest')

        ax[2].plot(distances_sorted, correlations_smoothed, 'r-', label=f'Moving avg (window={window_size})')

        ax[2].legend()
        fig.tight_layout()

    plot_covariance(flux_2_data_triangles,center_triangles,"Triangles")
    plot_covariance(flux_2_data_pyramids,center_pyramids,"Pyramids")


    ###############################################
    save_allfig_pdf(output_filename)

    print("Coupling Map stored. You can quit by ctrlC")
    print("Computing now additional health check plots.")

    for coupling in ["triangles","pyramids"]:

        if coupling == "triangles": 
            couplingMap = basic.CouplingMap(output_filename,pyramids = True)
        else:
            couplingMap = basic.CouplingMap(output_filename,pyramids = False)

        QT= couplingMap.QT
        R= couplingMap.R * spectra[:,None,None]
        centers = couplingMap.position

        # QT= QT_pyramids
        # R= R_pyramids * spectra[:,None,None]
        # centers = center_pyramids   

        wmin = QT.shape[1] // 4
        wmax = 3 * QT.shape[1] // 4
        QT_broadband, R_broadband = compute_broadband_QR(R, wmin, wmax)
        

        datacube=np.concatenate([d.data for d in datalist])


        datacube_T=datacube.transpose((3,2,0,1))
        # datacube_T=data_svdfiltered.transpose((3,2,0,1))
        Nwave, Noutput, Ncube, Nmod = datacube_T.shape
        Ntriangles = QT.shape[0]

        datacube_T=datacube_T.reshape((datacube_T.shape[0], datacube_T.shape[1], -1))
        chi2_max = np.sum(datacube_T**2, axis=(0,1))

        chi2_map = np.zeros((Ntriangles,Ncube * Nmod))
        chi2_map = np.zeros((Ntriangles, Ncube * Nmod))
        chi2_map[:] =  chi2_max
        # Here, the computation of the chi2 is simplified by the fact that QT is orthonormal
        # chi2 = ||data - Q @ Q.T @ data||^2 = ||data||^2 - ||Q.T @ data||^2
        for t in tqdm(range(Ntriangles), desc="Computing chi2 map"):
            k= QT[t] @ datacube_T
            chi2_map[t,:] -= np.sum(k ** 2, axis=(0,1))

        chi2_argmin = chi2_map.argmin(axis=0)
        # chi2_argmin[300] = 395  # manual fix for a weird outlier
        # chi2_argmin[300] = 412  # manual fix for a weird outlier

        QTdata = np.zeros((QT.shape[1],QT.shape[2],datacube_T.shape[2]))
        for i in tqdm(range(Ncube * Nmod), desc="Projection onto QT space"):
            t = chi2_argmin[i]
            data = datacube_T[:,:,i]
            QTdata[:,:,i] = (QT[t] @ data[:,:,None])[:,:,0]


        Xpos = np.zeros((Ncube , Nmod))
        Ypos = np.zeros((Ncube , Nmod))
        Xcen = np.zeros((Ncube , Nmod))
        Ycen = np.zeros((Ncube , Nmod))
        Xdiff = np.zeros((Ncube , Nmod))
        Ydiff = np.zeros((Ncube , Nmod))

        X_wave = np.zeros((Nwave, Ncube * Nmod))
        Y_wave = np.zeros((Nwave, Ncube * Nmod))
        Z_wave = np.zeros((Nwave, Ncube * Nmod))
        QTdata_dxy = np.zeros_like(QTdata)
        Nqr = R.shape[2]
        R_dxy = np.zeros((Nwave, Nqr, Ncube * Nmod, 2))

            
        for i in tqdm(range(Ncube * Nmod), desc="Computing XY positions"):
            t = chi2_argmin[i]
            center = centers[t]

            QTdata_broadband = QT_broadband[t] @ QTdata[wmin:wmax,:,i].ravel()
            
            if Nqr == 6:
                x_hat_broadband, y_hat_broadband, k_hat_broadband, chi2_broadband, _ = fit_QR_6(QTdata_broadband, R_broadband[t])
            else:
                x_hat_broadband, y_hat_broadband, k_hat_broadband, chi2_broadband, _ = solve_QR_3(QTdata_broadband, R_broadband[t])

            if Nqr == 6:
                v = np.array([1.0, x_hat_broadband, y_hat_broadband, x_hat_broadband*y_hat_broadband, x_hat_broadband**2, y_hat_broadband**2])
                dv_dx = np.array([0.0, 1.0, 0.0, y_hat_broadband, 2.0*x_hat_broadband, 0.0])
                dv_dy = np.array([0.0, 0.0, 1.0, x_hat_broadband, 0.0, 2.0*y_hat_broadband])
            else:
                v = np.array([1.0, x_hat_broadband, y_hat_broadband])
                dv_dx = np.array([0.0, 1.0, 0.0])
                dv_dy = np.array([0.0, 0.0, 1.0])

            r = R[t] @ v
            Kernel_v = np.identity(len(v)) - (r[:,:,None] @ r[:,None]) / (r[:,None] @ r[:,:,None])
            QTdata_dxy[:,:,i] = (Kernel_v @ QTdata[:,:,i,None])[...,0]

            dev_phi = np.array((dv_dx,dv_dy)).T
            R_dxy[:,:,i] = Kernel_v @ (R[t] @ dev_phi)

            xy_dev = (np.linalg.pinv(R_dxy[:,:,i]) @ QTdata_dxy[:,:,i,None])[...,0]

            X_wave[:,i] = xy_dev[:,0]
            Y_wave[:,i] = xy_dev[:,1]

            Xpos.ravel()[i] = x_hat_broadband
            Ypos.ravel()[i] = y_hat_broadband

            Xcen.ravel()[i] = center[0]
            Ycen.ravel()[i] = center[1]

            Xdiff.ravel()[i] = x_hat_broadband + center[0] - xmod.ravel()[i]
            Ydiff.ravel()[i] = y_hat_broadband + center[1] - ymod.ravel()[i]

        xy_dev = np.linalg.pinv(R_dxy.reshape((Nwave,-1,2))) @ QTdata_dxy.reshape((Nwave,-1,1))
        xy_dev = xy_dev[...,0]

        
        fig, axs = plt.subplots(2, Ncube, num="XY position -- using "+coupling, clear=True, figsize=(7*Ncube,10), squeeze=False)
        for i in range(Ncube):
            axs[0,i].plot(Xcen[i],Ycen[i],'.',label='Center of pyramids')
            axs[0,i].set_ylim(axs[0,i].get_ylim()[0], axs[0,i].get_ylim()[1])
            axs[0,i].set_xlim(axs[0,i].get_xlim()[0], axs[0,i].get_xlim()[1])
            axs[0,i].plot((Xcen+Xpos)[i],(Ycen+Ypos)[i],'.-',label='Detected position')
            axs[0,i].plot((Xcen[i],(Xcen+Xpos)[i]),(Ycen[i],(Ycen+Ypos)[i]),'-k',alpha=0.3,linewidth=0.5)
            axs[0,i].set_title(basenames[i][8:])
            axs[0,i].set_xlabel("X [mas]")
            axs[0,i].set_ylabel("Y [mas]")
            axs[0,i].legend()
        for ax in axs[0]:
            ax.set_aspect('equal')
        for i in range(Ncube):
            x_median = np.median(Xdiff[i])
            y_median = np.median(Ydiff[i])
            x_1sigma = np.percentile(Xdiff[i], [16, 84])
            y_1sigma = np.percentile(Ydiff[i], [16, 84])
            range_max = np.max((np.abs(x_1sigma), np.abs(y_1sigma))) * 2 +10
            axs[1,i].hist(Xdiff[i], bins=51, alpha=0.5, color='b', label='Xdiff', range=(-range_max, range_max))
            axs[1,i].hist(Ydiff[i], bins=51, alpha=0.5, color='r', label='Ydiff', range=(-range_max, range_max))
            x_median = np.median(Xdiff[i])
            y_median = np.median(Ydiff[i])
            x_1sigma = np.percentile(Xdiff[i], [16, 84])
            y_1sigma = np.percentile(Ydiff[i], [16, 84])
            axs[1,i].axvline(x_median, color='b', linestyle='--', label=f'X median: {x_median:.2f}')
            axs[1,i].axvline(y_median, color='r', linestyle='--', label=f'Y median: {y_median:.2f}')
            # axs[1,i].axvspan(x_1sigma[0], x_1sigma[1], color='b', alpha=0.2, label=f'X 1σ: [{x_1sigma[0]:.2f}, {x_1sigma[1]:.2f}]')
            # axs[1,i].axvspan(y_1sigma[0], y_1sigma[1], color='r', alpha=0.2, label=f'Y 1σ: [{y_1sigma[0]:.2f}, {y_1sigma[1]:.2f}]')
            axs[1,i].set_xlabel('Difference [mas]')
            axs[1,i].set_ylabel('Count')
            axs[1,i].legend()
        
        plt.tight_layout()

        save_allfig_pdf(output_filename)


# %%
