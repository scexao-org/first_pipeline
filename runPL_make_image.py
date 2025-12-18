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

from typing import List
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
from classes.runPL_class_flatMap import FlatMap
from classes.runPL_class_waveMap import WaveMap
from classes.runPL_class_fileList import FileList
from classes.runPL_class_dataCube import DataCube 
from classes.runPL_class_couplingMap import CouplingMap 

import libraries.runPL_library_basic as runlib_basic
import libraries.runPL_library_io as runlib_io
import libraries.runPL_library_plots as runlib_plots
import libraries.runPL_library_linalg as runlib_linalg

from astropy.io import fits
from astroplan import Observer
from astropy.time import Time
import math

subaru = Observer.at_site("Subaru")
now_time = Time.now()
if subaru.is_night(now_time):
    print("It's night time at Subaru Observatory.")
else:
    print("It's day time at Subaru Observatory.")

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


def get_filelist(file_patterns, dark_patterns, cmap_patterns, wollaston):

        fits_keywords = {'X_FIRTYP': ['PREPROC'],
                        'DATA-TYP': ['OBJECT','OJECT','TEST'],
                        }    
        
        # Adding other constraints if asked by user
        if wollaston is not None:
            fits_keywords['X_FIRWOL'] = [wollaston]
        
        print(file_patterns)
        filelist = runlib_io.get_filelist(file_patterns, fits_keywords)

        # Adding new constraints if not asked by user
        hd=fits.getheader(filelist[0])
        wollaston = hd.get('X_FIRWOL', None)
        if wollaston is not None:
            fits_keywords['X_FIRWOL'] = [wollaston]

        print("----------------")
        print(f"Selected object with wollaston = {wollaston}")
        print("----------------")

        filelist = runlib_io.get_filelist(file_patterns, fits_keywords)

        fits_keywords = {'X_FIRTYP': ['PREPROC'],
                        'DATA-TYP': ['DARK'],
                        }    
        if wollaston is not None:
            fits_keywords['X_FIRWOL'] = [wollaston]
        
        try:
            filelist_dark = runlib_io.get_filelist(dark_patterns, fits_keywords, name_search="dark")
        except FileNotFoundError as e:
            print(f"NO DARKS: {e}")
            filelist_dark = []

        if len(filelist_dark) == 0:
            print("WARNING: No dark files found with the specified patterns and keywords.")

        fits_keywords = {'X_FIRTYP': ['COUPLINGMAP'],
                        }    
        if wollaston is not None:
            fits_keywords['X_FIRWOL'] = [wollaston]

        filelist_cmap = runlib_io.get_filelist(cmap_patterns, fits_keywords, name_search="coupling map")
        filelist_cmap  = filter_couplingmapfile(filelist_cmap)

        files_with_dark = runlib_io.associate_dark(filelist, filelist_dark)

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
    # files_with_dark = {cmap: runlib_io.find_closest_dark(cmap, filelist_dark) for cmap in filelist_data}

    return filelist_cmap


def quick_fits(data, title=""):
    if DEBUG:
        #For debugging purpose
        now = datetime.now()
        date_time_str = now.strftime("%Y_%m_%d_%H_%M_%S")
        if getpass.getuser() == "jsarrazin":
            runlib_io.save_fits_file(data, "/home/jsarrazin/Bureau/test zone/coupling_maps/"+title+"_"+date_time_str+".fits")
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
    parser = argparse.ArgumentParser(
        description="Reconstruct images from FIRST Photonic Lantern data using specified coupling maps and options.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Summary:
    This script processes preprocessed FITS files, applies coupling maps, reconstructs images, and saves the results.
    It supports selection by object name, modulation pattern, and smoothing options. The script can save individual frames,
    wavelength slices, and residuals, and allows explicit selection of coupling map files.

Input files:
    - Preprocessed data FITS files (e.g., with DPR_CATG=OBJECT and DPR_TYPE=PREPROC)
    - Coupling map FITS files (e.g., with X_FIRTYP=COUPLINGMAP)

Output files:
    - Reconstructed image FITS files, including summed images, residuals, and optionally individual frames and wavelength slices.
        """
    )

    # needed to work in VLC:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--f", help=argparse.SUPPRESS)

    # Add positional argument for files
    parser.add_argument('files', nargs='*', default=['*.fits'],
                       help='FITS files to process (supports wildcards)')

    # Add optional arguments
    parser.add_argument("--object_name", 
                       help="Selection of the data by the Object name (default: first target in the list)")
    parser.add_argument("--dark_files", 
                       help="Select one or more specific dark(s) files to use")
    parser.add_argument("--coupling_map", 
                       help="Force to select which coupling map file to use (default: the one in the directory)")
    parser.add_argument("--wavelength_smooth", type=int, default=20,
                       help="Smoothing factor for wavelength (default: %(default)s)")
    parser.add_argument("--modID", type=int, 
                       help="Selection of the modulation pattern by user (default: first in the list)")
    parser.add_argument("--modScale", type=int, 
                       help="Selection of the modulation scale by user (default: first in the list)")
    parser.add_argument("--wollaston", 
                       help="Wollaston status. Use IN for internal or OUT for no wollaston (default: first in the list)")
    parser.add_argument("--save_individual_frames", action="store_true", default=True,
                       help="Save individual frames (default: %(default)s)")
    parser.add_argument("--save_individual_wavelength", action="store_true", default=False,
                       help="Save individual wavelength (default: %(default)s)")
    

    # Parse the arguments
    args = parser.parse_args()
    file_patterns = args.files if args.files else ['*.fits','./preproc/*.fits']

    # Extract the parsed arguments
    modID = args.modID
    modScale = args.modScale
    object_name = args.object_name
    wollaston = args.wollaston
    wavelength_smooth = args.wavelength_smooth
    dark_patterns = args.dark_files
    flat_patterns = args.flatMap
    wave_patterns = args.waveMap
    save_individual_frames = args.save_individual_frames
    save_individual_wavelength = args.save_individual_wavelength
    cmap_patterns = args.coupling_map
    
    if (("VSCODE_PID" in os.environ or os.environ.get('TERM_PROGRAM') == 'vscode') or os.environ.get('SPYDER_DEBUG_FILEfile =')):
        print("Running in compiler")
        wollaston = None
        dark_patterns = None

        if getpass.getuser() == "slacour":
            # file_patterns = "/Users/slacour/DATA/LANTERNE/20250614/preproc/firstpl_2025-06-14T01:38*fits"
            file_patterns = "/Users/slacour/DATA/LANTERNE/20250510/preproc/*10T05?5*BETACMI_P.fits"
            file_patterns = "/Users/slacour/DATA/LANTERNE/20250510/preproc/*10T05?53*BETACMI_P.fits"
            file_patterns = "/Users/slacour/DATA/LANTERNE/20250510/preproc/*10T09?2[0-2]*TETCRB_P.fits"
            cmap_patterns = "/Users/slacour/DATA/LANTERNE/20250510/preproc/../couplingmaps/firstpl_2025-05-10T09:23:36_COUPLINGMAP.fits"
            dark_patterns = "/Users/slacour/DATA/LANTERNE/20250514/preproc"
            file_patterns = "/Users/slacour/DATA/LANTERNE/20250514/preproc/firstpl_2025-05-14T11?3*s"
            # Mathias Binary
            cmap_patterns = "/Users/slacour/DATA/LANTERNE/20250514/preproc/../couplingmaps/firstpl_2025-05-14T11:39:58_HIP85819_CM.fits"
            file_patterns = "/Users/slacour/DATA/LANTERNE/20250514/preproc/firstpl_2025-05-14T10:10?*s"
            file_patterns = "/Users/slacour/DATA/LANTERNE/20250514/preproc/firstpl_2025-05-14T10:06*s" # large FOV
            dark_patterns = "/Users/slacour/DATA/LANTERNE/20250514/preproc"

            # file_patterns = "/Users/slacour/DATA/LANTERNE/20250510/preproc/*10T05?53*BETACMI_P.fits"
            # file_patterns = "/Users/slacour/DATA/LANTERNE/20250510/preproc/*10T09?21*TETCRB_P.fits"
            # file_patterns = "/Users/slacour/DATA/LANTERNE/20250510/preproc/firstpl_2025-05-10T09:4[4-9]*_HIP81126_P.fits"
            # file_patterns = "/Users/slacour/DATA/LANTERNE/20250510/preproc/firstpl_2025-05-10T07:58*DELVIR_P.fits"

            # cmap_patterns = "/Users/slacour/DATA/LANTERNE/20250614/preproc/../couplingmaps/firstpl_2025-06-14T01:48:57_HIP105966_CM.fits"
            # file_patterns = "/Users/slacour/DATA/LANTERNE/20250614/preproc/firstpl_2025-06-14T01:50*fits"
        if getpass.getuser() == "ehuby" :
            file_patterns = "/home/ehuby/WORK/DATA/FIRST-PL/2025-05-10/preproc/firstpl_*.fits"
            cmap_patterns = "/home/ehuby/WORK/DATA/FIRST-PL/2025-05-10/preproc/couplingmaps_TETCRB/"
        file_patterns = [file_patterns] if isinstance(file_patterns, str) else file_patterns
        cmap_patterns = [cmap_patterns] if isinstance(cmap_patterns, str) else cmap_patterns

    # If the user specify a dark, use it. Otherwise, use the science file pattern
    if dark_patterns is None:
        dark_patterns = file_patterns
    # If the user specifies a specific map, use it, otherwise look into the arguments + default directories
    if cmap_patterns is None:
        folder = os.path.dirname(file_patterns[0])
        flat_patterns = file_patterns + [os.path.join(folder,"../couplingmaps")]


    fileList = FileList(file_patterns, data_type= "OBJECT", first_type='PREPROC', wollaston=wollaston, object_name=object_name, modID=modID, modScale=modScale)

    # Adding constraints to make sure the dataset is coherent:
    object_name = fileList.fits_keywords.get('OBJECT', [None])[0] if object_name is None else object_name
    wollaston = fileList.fits_keywords.get('X_FIRWOL', [None])[0] if wollaston is None else wollaston
    modID = fileList.fits_keywords.get('X_FIRMID', [0])[0] if modID is None else modID
    modScale = fileList.fits_keywords.get('X_FIRMSC', [0])[0] if modScale is None else modScale

    fileList = FileList(file_patterns, data_type= "OBJECT", first_type='PREPROC', wollaston=wollaston, object_name=object_name, modID=modID, modScale=modScale)

    fileList.make_association(darks_pattern=dark_patterns)

    # reading all the calibration files that should be appended to the cmap files (including wavelength and flat)
    file_flat = fileList.get_flatmap_file(cmap_patterns)
    file_wave = fileList.get_wavemap_file(cmap_patterns)
    file_coup = fileList.get_couplingmap_file(cmap_patterns)

    flatMap =  FlatMap(file_flat) if file_flat is not None else None
    waveMap =  WaveMap(file_wave) if file_wave is not None else None

    datalist : List[DataCube] = fileList.extract_data_from_list(flatMap = flatMap, waveMap = waveMap, center = False)

    couplingMap = CouplingMap(file_coup,pyramids = True)


    Npos = couplingMap.Npositions
    Npixels = 150


    for i,d in enumerate(datalist[:]):

        flux = d.flux
        datacube= d.data 
        datacube_var= d.variance 
        ra_dec = d.compute_xy_sky(couplingMap) 
        # xmod=datalist[0].xmod
        # ymod=datalist[0].ymod
        Ncube = ra_dec.shape[0]  # number of cubes
        Nmod = ra_dec.shape[1]  # number of modulation positions
        Npos = ra_dec.shape[2]  # Positions on sky
        Nwave = datacube.shape[3]  # number of wavelength channels
        Noutput = datacube.shape[2]  # number of outputs
        Nimages = Ncube * Nmod

        ra_dec = ra_dec.reshape((-1, *ra_dec.shape[2:]))
        datacube = datacube.reshape((-1, *datacube.shape[2:]))
        flux = flux.reshape((-1, *flux.shape[2:]))
        datacube_var = datacube_var.reshape((-1, *datacube_var.shape[2:]))

        filename = d.filename
        print( f"---->  Filename : {filename}")


        datacube_T=datacube.transpose((2,1,0))
        datacube_var_T=datacube_var.transpose((2,1,0))
        star_detected, star_index, star_radec, chi2 = couplingMap.chi2_filtering(datacube_T, ra_dec)
        print(f"* Percentage of data with star detected: {np.sum(star_detected)/len(star_detected)*100:.1f} % (flux, svd and chi2 threshold)")

        residuals = datacube_T.copy()
        for i in tqdm(range(residuals.shape[2]), desc="Calculating residuals of the 3D image"):
            if star_detected[i]:
                t = star_index[i]
                k = couplingMap.QT[t] @ residuals[:,:,i,None]
                residuals[:,:,i] -=  (couplingMap.QT[t].transpose((0,2,1)) @ k)[:,:,0]

        fluxes = np.matmul(couplingMap.data_2_flux, datacube_T)/d.dit*d.gain
        fluxes_residuals = np.matmul(couplingMap.data_2_flux, residuals)/d.dit*d.gain
        fluxes_variance = fluxes.mean(axis=0,keepdims=True)
        fluxes_variance[:] = fluxes_variance.std(axis=1,keepdims=True)**2*19

        # Define the grid for interpolation
        # calcul de la grille de l'image que l'on souhaite reconstruire
        # if it is for a quick look of the real time display, use xmod=ymod=0

        flux_maps = runlib_plots.make_image_using_grid(ra_dec, fluxes, desc="Creating flux maps", Npixels=Npixels)
        flux_maps_residuals = runlib_plots.make_image_using_grid(ra_dec, fluxes_residuals, desc="Creating flux residuals", Npixels=Npixels)
        flux_maps_variance = runlib_plots.make_image_using_grid(ra_dec, fluxes_variance, desc="Creating flux variance", Npixels=Npixels)
        flux_maps_sum = np.nanmean(flux_maps, axis=0)
        flux_maps_residuals_sum = np.nanmean(flux_maps_residuals, axis=0)
        flux_maps_variance = np.nanmean(flux_maps_variance, axis=0)
            
        flux_maps_snr = flux_maps_sum / np.sqrt(flux_maps_variance)
        flux_maps_contrast = np.sqrt(flux_maps_variance)/np.nanmax(flux_maps)

        header = d.header
        header['X_FIRTYP'] = 'IMAGE'

        list_of_hdus = []
        # Create a primary HDU with the data
        hdu_primary = fits.PrimaryHDU(flux_maps_sum)
        hdu_residual = fits.ImageHDU(flux_maps_residuals_sum, name="RESIDUAL")
        list_of_hdus += [hdu_primary, hdu_residual]

        # Create a primary HDU with no data, just the header
        if save_individual_frames:
            hdu_frame = fits.ImageHDU(flux_maps, name="FRAMES")
            hdu_frame_residual = fits.ImageHDU(flux_maps_residuals, name="FRAMES_RESIDUAL")
            hdu_snr = fits.ImageHDU(flux_maps_snr, name="SNR")
            hdu_contrast = fits.ImageHDU(flux_maps_contrast, name="CONTRAST")
            list_of_hdus += [hdu_frame, hdu_frame_residual, hdu_snr, hdu_contrast]

        if save_individual_wavelength:
            flux_maps_wave = []
            residuals_maps_wave = []

            for w in tqdm(range(Nwave), desc="Creating wavelength slices"):
                flux_maps_tmp = runlib_plots.make_image_using_grid(ra_dec, fluxes[w,None])
                flux_maps_residuals_tmp = runlib_plots.make_image_using_grid(ra_dec, fluxes_residuals[w,None])
                flux_maps_sum = np.nanmean(flux_maps_tmp, axis=0)
                flux_maps_residuals_sum = np.nanmean(flux_maps_residuals_tmp, axis=0)

                flux_maps_wave.append(flux_maps_sum)
                residuals_maps_wave.append(flux_maps_residuals_sum)

            hdu_wave = fits.ImageHDU(flux_maps_wave, name="3D_IMAGE")
            hdu_wave_residual = fits.ImageHDU(residuals_maps_wave, name="3D_IMAGE_RESIDUAL")
            list_of_hdus += [hdu_wave, hdu_wave_residual]
            header['X_FIRTYP'] = 'WDIMAGE'


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

        output_filename = os.path.join(output_dir, runlib_io.create_output_filename(header))

        # Write to a FITS file
        hdul.writeto(output_filename, overwrite=True)
        print(f"Image saved to {output_filename}")

        grid_x, grid_y = runlib_plots.make_image_grid(ra_dec, Npixels=Npixels)

        # Plot flux_maps_sum and flux_maps_residuals_sum side by side
        fig, axes = plt.subplots(2, 2, num="Flux Maps", figsize=(15, 15.6), clear=True)
        im0 = axes[0,0].imshow(flux_maps_sum[::-1].T, origin='lower', aspect='auto',
                     extent=[grid_x.max(), grid_x.min(), grid_y.min(), grid_y.max()],vmin=0)
        axes[0,0].set_title('Flux Map')
        axes[0,0].set_xlabel('RA (mas)')
        axes[0,0].set_ylabel('Dec (mas)')
        fig.colorbar(im0, ax=axes[0,0], orientation='vertical', label='Flux (e-/s)')

        im1 = axes[0,1].imshow(flux_maps_residuals_sum[::-1].T, origin='lower', aspect='auto',
                     extent=[grid_x.max(), grid_x.min(), grid_y.min(), grid_y.max()],vmin=0)
        axes[0,1].set_title('Flux Residuals')
        axes[0,1].set_xlabel('RA (mas)')
        axes[0,1].set_ylabel('Dec (mas)')
        fig.colorbar(im1, ax=axes[0,1], orientation='vertical', label='Residual Flux (e-/s)')

        im2 = axes[1,0].imshow(flux_maps_snr[::-1].T, origin='lower', aspect='auto',
                        extent=[grid_x.max(), grid_x.min(), grid_y.min(), grid_y.max()],
                        vmin=0, vmax=np.nanmax(flux_maps_snr))
        axes[1,0].set_title('SNR Map')
        axes[1,0].set_xlabel('RA (mas)')
        axes[1,0].set_ylabel('Dec (mas)')
        fig.colorbar(im2, ax=axes[1,0], orientation='vertical', label='SNR')

        im3 = axes[1,1].imshow(5*flux_maps_contrast[::-1].T, origin='lower', aspect='auto',
                     extent=[grid_x.max(), grid_x.min(), grid_y.min(), grid_y.max()],vmin=0)
        axes[1,1].set_title('5 sigma Contrast ratio')
        axes[1,1].set_xlabel('RA (mas)')
        axes[1,1].set_ylabel('Dec (mas)')


        # Mask invalid values for contouring
        contrast_disp = np.ma.masked_invalid(5.0 * flux_maps_contrast)
        disp_levels = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0]
        CS = axes[1,1].contour(grid_x, grid_y, contrast_disp, levels=disp_levels, colors='white', linewidths=1)

        fmt = {}
        strs = ['1e-5', '1e-4', '1e-3', '1e-2', '1e-1', '1']
        for l, s in zip(CS.levels, strs):
            fmt[l] = s  
        # Add labels on the contours
        axes[1,1].clabel(CS, fmt=fmt, inline=True, fontsize=10, colors='white')

        fig.colorbar(im3, ax=axes[1,1], orientation='vertical', label='5 sigma Contrast ratio')

        # Overlay decade contours for contrast (0.1, 0.01, 0.001, ...)

        xmod = d.xmod
        ymod = d.ymod

        N_middle=np.linalg.norm((xmod,ymod),axis=0).argmin()
        for i in range(4):
            axes.ravel()[i].plot(star_radec[0], star_radec[1], 'rx', markersize=10, label='Star Position')
            axes.ravel()[i].plot(d.x_object, d.y_object, 'kx', markersize=5, label='PL center position',alpha=0.5)

            fov_large = np.isnan(flux_maps_sum)
            fov_small = np.isnan(flux_maps[N_middle])
            # Overlay contours for the two FOVs
            axes.ravel()[i].contour(grid_x, grid_y, fov_large, levels=[0.5], colors='k', linewidths=1)
            axes.ravel()[i].contour(grid_x, grid_y, fov_small, levels=[0.5], colors='w', linewidths=0.1, linestyles='solid', label='FOV')
            axes.ravel()[i].set_aspect('equal')

        axes.ravel()[i].legend()
        fig.suptitle(f"{d.basename} : {Ncube} x {Nmod} x {d.dit:.3f}s - RA = {d.x_object:.2f}, Dec = {d.y_object:.2f}", fontsize=16)
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        png_filename = os.path.join(output_dir, runlib_io.create_output_filename(header).replace('.fits', '.png'))
        plt.savefig(png_filename, dpi=150)
        print(f"PNG image saved to {png_filename}")
        # plt.close(fig)


#         # Plot contrast vs separation
        # contrast = flux_maps_contrast.ravel() * 5
        # separation = np.sqrt(grid_x**2 + grid_y**2).ravel()
        # # Remove NaN values
        # valid = ~np.isnan(contrast) & ~np.isnan(separation) & (grid_y.ravel() > 0)
        # contrast = contrast[valid]
        # separation = separation[valid]
        # sorted_indices = np.argsort(separation)
        # contrast = contrast[sorted_indices]
        # separation = separation[sorted_indices]
        # fig2, ax2 = plt.subplots(num="Contrast Curve", figsize=(8, 6), clear=True)
        # ax2.plot(separation, contrast, 'b-')
        # ax2.set_yscale('log')
        # ax2.set_xlabel('Separation (mas)')
        # ax2.set_ylabel('5 Sigma Contrast Ratio')
        # ax2.set_title(f'Contrast Curve for HIP81126')
        # ax2.grid(True, which="both", ls="--")
        # ax2.set_xlim(25, np.max(separation))
        # plt.savefig('HIP81126_contrast_curve.png', dpi=150)



# %%
