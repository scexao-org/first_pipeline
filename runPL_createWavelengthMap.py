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
import peakutils

import getpass
import matplotlib
if "VSCODE_PID" in os.environ:
    matplotlib.use('Qt5Agg')
else:
    matplotlib.use('Agg')
     
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot,hist,clf,figure,legend,imshow
from datetime import datetime
from tqdm import tqdm
import runPL_library_io as runlib
import shutil
from collections import defaultdict
from scipy import linalg
from matplotlib import animation
from itertools import product
#from scipy.linalg import pinvfind_closest_dark
from runPL_calibrateNeon import its_a_match, run_trials_for_all_combination_of_waves

#plt.ion()

# Add options
usage = """
    usage:  %prog [options] files.fits

    Goal: Create a wavelength map from the provided FITS files.

    It will get as input a list of files with DATA-CAT=PREPROC and DATA-TYP=WAVE keywords. 
    It will also find the corresponding dark files with DATA-CAT=PREPROC and DATA-TYP=DARK keywords.
    It will read the wave files and subtract the median of the dark files from them.
    Then, it will find the highest N peaks in the flux and fit a polynomial to create a wavelength map.
    The value N is the number of wavelength provided in th wave_list

    Example:
    runPL_createWavelengthMap.py --wave_list="[753.6, 748.9, 743.9, 724.5, 717.4, 703.2, 693, 671.7, 667.8, 659.9, 653.3, 650.7, 640.2, 638.2, 633.4, 630.5, 626.7, 621.7, 616.4]" *.fits

    Options:
    --wave_list: Comma-separated list of emission lines (default: [753.6, 748.9, 743.9, 724.5, 717.4, 703.2, 693, 671.7, 667.8, 659.9, 653.3, 650.7, 640.2, 638.2, 633.4, 630.5, 626.7, 621.7, 616.4])
"""



""" if "VSCODE_PID" in os.environ:
    filelist = glob("/home/jsarrazin/Bureau/PLDATA/InitData/preproc/*fits")
    wave_list_string = wave_list_string_default
    filelist.sort()  # process the files in alphabetical order
else:
    (options, args) = parser.parse_args()
    wave_list_string = options.wave_list
    filelist = []
    #DEBUGGING ON
    ##args = glob("/home/jsarrazin/Bureau/PLDATA/InitData/Neon2/preproc/*fits")
    if len(args) > 0:
        for f in args:
            if os.path.isdir(f):
                filelist += glob(os.path.join(f, "*.fit*"))
            else:
                filelist += glob(f)
    # Processing of the full current directory
    else:
        for file in os.listdir("."):
            if file.endswith(".fits"):
                filelist.append(file)

 """

def shift_array_left(arr):
    """
    Shifts the array 100 values to the left, discarding the first 100 values,
    and fills the right with ones.

    Parameters:
    - arr (np.ndarray): Input array of length 1500.

    Returns:
    - np.ndarray: Modified array with the shift applied.

    this will be used to simulate another neon file
    """
    if len(arr) != 1500:
        raise ValueError("Input array must have exactly 1500 elements.")
    
    shifted = np.empty_like(arr)
    shifted[:-100] = arr[100:]  # Shift left by 100
    shifted[-100:] = 1          # Fill the rightmost 100 values with 1
    return shifted

def widen_array(arr, factor):
    """
    Widens the input array by repeating each element a specified number of times.

    Parameters:
    - arr (np.ndarray): Input array to widen.
    - factor (int): The number of times each element should be repeated.

    Returns:
    - np.ndarray: Widened array.
    """
    if factor <= 0:
        raise ValueError("Factor must be a positive integer.")
    
    widened = np.repeat(arr, factor)
    return widened


def prep_data(filelist, star=False):

    # Keys to keep only the RAW files
    fits_keywords = {'DATA-CAT': ['PREPROC'], 
                    'DATA-TYP': ['DARK']}

    # Use the function to clean the filelist
    filelist_dark = runlib.clean_filelist(fits_keywords, filelist)

    # Keys to keep only the WAVE files
    fits_keywords = {'DATA-CAT': ['PREPROC'], 
                    'DATA-TYP': ['WAVE']}

    if star:
        fits_keywords = {'DATA-CAT': ['PREPROC'], 
                        'DATA-TYP': ['OBJECT']}

    # Use the function to clean the filelist
    filelist_wave = runlib.clean_filelist(fits_keywords, filelist)

    # raise an error if data is not suitable
    if len(filelist_wave) == 0:
        raise ValueError("No Neon file to reduce -- are they preprocessed?")
    if len(filelist_wave) > 1:
        raise ValueError("Too many Neon file to reduce -- which one shall I use?")
    if len(filelist_dark) == 0:
        raise ValueError("No darks")

    data_file = filelist_wave[0]

    closest_dark_files = runlib.find_closest_dark(data_file, filelist_dark, filter_by_directory = True)
    header=fits.getheader(data_file)
    data=np.double(fits.getdata(data_file))
    data_dark=fits.getdata(closest_dark_files)
    data-=np.median(data_dark,axis=0)


    flux = data.mean(axis=(0,1))
    flux [flux<1] =1 

    #flux = shift_array_left(flux)

    return flux, header, filelist_wave

def findPeaks(flux, wave_list_string):

    threshold_array=np.linspace(0.01,0.3,100)
    wavelength_list = [float(w) for w in wave_list_string.strip('[]').split(',')]

    #######

    found = False
    detectedWavePeaks_solo = peakutils.peak.indexes(flux,threshold_array[0], min_dist=3)
    detectedWavePeaks_solo_list = detectedWavePeaks_solo.tolist()
    peak_weight = [1 for detectedPeak in detectedWavePeaks_solo_list]


    for t in threshold_array:
        detectedWavePeaks_tmp = peakutils.peak.indexes(flux,thres=t, min_dist=3)

        for detectedPeak in detectedWavePeaks_tmp:
            index = detectedWavePeaks_solo_list.index(detectedPeak)
            peak_weight[index]+=1 ##each detection will make the peak worth more

    temp = run_trials_for_all_combination_of_waves(detectedWavePeaks_solo_list, peak_weight, wavelength_list,1,5)
    its_a_match_peaks = temp[0]
    its_a_match_waves = temp[1]


    # Définir le chemin complet du sous-dossier "output/wave"
    output_dir = os.path.join("output", "wave")

    return its_a_match_peaks, its_a_match_waves, output_dir, peak_weight, detectedWavePeaks_solo_list, wavelength_list

def runFigure(flux, detectedWavePeaks_solo, its_a_match_peaks,its_a_match_waves, header, filelist_wave, peak_weight, saveWhere):

    WavePolyBest=np.polyfit(its_a_match_peaks,its_a_match_waves,2)
    WavefitBest=np.poly1d(WavePolyBest)
    pixels=np.arange(flux.shape[0])
    pix_to_wavelength_map_best=WavefitBest(pixels)
    os.makedirs(saveWhere, exist_ok=True)
    output_dir = saveWhere#os.path.join("output", "wave")

    

    figure2(flux, its_a_match_peaks, its_a_match_waves, output_dir, header)
    figure3(flux, detectedWavePeaks_solo, peak_weight, output_dir)
    figure4(flux, pix_to_wavelength_map_best,its_a_match_peaks,its_a_match_waves, pixels, output_dir)

    return WavePolyBest


def figure2(flux, its_a_match_peaks, its_a_match_waves, output_dir, header):
    """
    Match pixels and peak
    """
    fig2,axs2=plt.subplots(3,num="wavelength position",clear=True,figsize=(13,7))

    axs2[0].plot(flux)
    axs2[0].plot(np.arange(len(flux))[its_a_match_peaks],flux[its_a_match_peaks],'o')
    axs2[0].set_title("Peak detected new match")
    axs2[0].set_xlabel("Pixel number")
    axs2[0].set_ylabel("Flux (ADU)")
    axs2[0].set_yscale("linear")


    WavePolyBest=np.polyfit(its_a_match_peaks,its_a_match_waves,2)
    WavefitBest=np.poly1d(WavePolyBest)
    pixels=np.arange(flux.shape[0])

    pix_to_wavelength_map_best=WavefitBest(pixels)

    axs2[1].plot(pixels,pix_to_wavelength_map_best,label='Polynomial fit (deg={})'.format(2))
    axs2[1].plot(its_a_match_peaks,its_a_match_waves,'o',label='Detected peaks')
    axs2[1].set_title("Wavelength vrs Pixels new")
    axs2[1].set_xlabel("Pixel number")
    axs2[1].set_ylabel("Wavelength (nm)")
    axs2[1].legend()

    # Convert pixel numbers to wavelengths for the top plot
    wavelengths = pix_to_wavelength_map_best
    detected_wavelengths = pix_to_wavelength_map_best[its_a_match_peaks]

    # Top plot: Flux vs. Wavelength
    axs2[2].plot(wavelengths, flux)  # Use wavelengths instead of pixel numbers
    axs2[2].plot(detected_wavelengths, flux[its_a_match_peaks], 'o')  # Detected peaks in wavelength space

    # Annotate each detected peak with its wavelength value
    for wavelength, flux_value in zip(detected_wavelengths, flux[its_a_match_peaks]):
        axs2[2].annotate(
            f'{wavelength:.1f} nm',  # Label text showing wavelength rounded to 1 decimal
            xy=(wavelength, flux_value),  # Position of the annotation
            xytext=(0, 10),  # Offset position (0 pixels right, 10 pixels up)
            textcoords='offset points',  # Relative to the point
            fontsize=9,  # Font size of the annotation
            ha='center',  # Horizontal alignment
            color='black'
        )

    axs2[2].set_title("Peak detected new match")
    axs2[2].set_xlabel("Wavelength (nm)")  # Update label
    axs2[2].set_ylabel("Flux (ADU)")
    axs2[2].set_yscale("linear")

    fig2.tight_layout()

    # Créer les dossiers "output" et "wave" s'ils n'existent pas déjà
    os.makedirs(output_dir, exist_ok=True)
    # Save fits file with traces_loc inside
    hdu = fits.PrimaryHDU(pix_to_wavelength_map_best)
    header['DATA-CAT'] = 'WAVEMAP'
    # Add date and time to the header
    current_time = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
    header['DATE-PRO'] = current_time


    hdu.header.extend(header, strip=True)
    hdul = fits.HDUList([hdu])
    output_filename = os.path.join(output_dir, runlib.create_output_filename(header))
    hdul.writeto(output_filename, overwrite=True)

    # %%

    fig2.savefig(output_filename[:-4]+"png",dpi=300)
    print("Saved : "+output_filename[:-4]+"png")

def figure3(flux, detectedWavePeaks_solo, peak_weight, output_dir):
    """
    All detected peaks and their weight
    """
    fig3, axs3 = plt.subplots(1, num="wavelength position", clear=True, figsize=(13, 7))

    # First subplot: Plot flux with peak detection markers
    scatter = axs3.scatter(
        np.arange(len(flux))[detectedWavePeaks_solo],  # x-coordinates
        flux[detectedWavePeaks_solo],                 # y-coordinates
        c=peak_weight,                                # Colors based on weight
        cmap='viridis',                               # Colormap for weights
        edgecolor='black',                            # Marker edge color
        s=50                                          # Marker size
    )
    axs3.plot(flux, label="Flux")
    axs3.set_title("Peak detected new match")
    axs3.set_xlabel("Pixel number")
    axs3.set_ylabel("Flux (ADU)")
    axs3.set_yscale("linear")

    # Add a color bar to represent the weight values
    colorbar = fig3.colorbar(scatter, ax=axs3, orientation="vertical", pad=0.02)
    colorbar.set_label("Peak Weight")

    fig3.tight_layout()

    # Save the figure
    fig3.savefig(output_dir + "_ALL.png", dpi=300)
    print("Saved : "+output_dir + "_ALL.png")

######

def figure4(flux, pix_to_wavelength_map_best,its_a_match_peaks,its_a_match_waves, pixels, output_dir):
    fig4, axs4 = plt.subplots(1, num="wavelength position", clear=True, figsize=(13, 7))

    # Convert pixel numbers to wavelengths for the top plot
    wavelengths = pix_to_wavelength_map_best
    detected_wavelengths = pix_to_wavelength_map_best[its_a_match_peaks]

    # Top plot: Flux vs. Wavelength
    axs4.plot(wavelengths, flux)  # Use wavelengths instead of pixel numbers
    axs4.plot(detected_wavelengths, flux[its_a_match_peaks], 'o')  # Detected peaks in wavelength space

    # Annotate each detected peak with its wavelength value
    for wavelength, flux_value in zip(detected_wavelengths, flux[its_a_match_peaks]):
        axs4.annotate(
            f'{wavelength:.1f} nm',  # Label text showing wavelength rounded to 1 decimal
            xy=(wavelength, flux_value),  # Position of the annotation
            xytext=(0, 10),  # Offset position (0 pixels right, 10 pixels up)
            textcoords='offset points',  # Relative to the point
            fontsize=9,  # Font size of the annotation
            ha='center',  # Horizontal alignment
            color='black'
        )

    axs4.set_title("Peak detected new match")
    axs4.set_xlabel("Wavelength (nm)")  # Update label
    axs4.set_ylabel("Flux (ADU)")
    axs4.set_yscale("linear")

    fig4.tight_layout()

    # Save the figure
    fig4.savefig(output_dir + "_wave_version.png", dpi=300)
    print("Saved : "+output_dir + "_wave_version.png")

##########




# %%

def runCreateWavelengthMap(filepath, wave_list_string, saveWhere = ""):
    filelist=[]
    for file in os.listdir(filepath):
        if file.endswith(".fits"):
            filelist.append(os.path.join(filepath, file))
    if saveWhere=="":
        saveWhere=filepath

    flux, header, filelist_wave = prep_data(filelist)
    its_a_match_peaks, its_a_match_waves, output_dir, peak_weight, detectedWavePeaks_solo_list, wavelength_list=findPeaks(flux, wave_list_string)

    wavePolyBest = runFigure(flux, detectedWavePeaks_solo_list, its_a_match_peaks,its_a_match_waves, header, filelist_wave, peak_weight, saveWhere)
    # Add input parameters to the header
    for i in range(len(wavelength_list)):
        header['WAVE%i'%i] = wavelength_list[i]
    
    return wavePolyBest


def starFigure(flux, waveBest, output_dir):

    fig5, axs5 = plt.subplots(1, num="wavelength position", clear=True, figsize=(13, 7))

    # Convert pixel numbers to wavelengths for the top plot
    pixels=np.arange(flux.shape[0])

    # Top plot: Flux vs. Wavelength
    axs5.plot(pixels, flux)  # Use wavelengths instead of pixel numbers
    axs5.set_title("Spectra")
    axs5.set_xlabel("Wavelength (nm)")  # Update label
    axs5.set_ylabel("Flux (ADU)")
    axs5.set_yscale("linear")

    fig5.tight_layout()

    # Save the figure
    os.makedirs(output_dir , exist_ok=True)
    fig5.savefig(output_dir + "pixel_version.png", dpi=300)
    print("Saved : "+output_dir + "pixel_version.png")

    fig4, axs4 = plt.subplots(1, num="wavelength position", clear=True, figsize=(13, 7))

    # Convert pixel numbers to wavelengths for the top plot
    pixels=np.arange(flux.shape[0])
    wavelengths = waveBest(pixels)

    # Top plot: Flux vs. Wavelength
    axs4.plot(wavelengths, flux)  # Use wavelengths instead of pixel numbers
    axs4.set_title("Spectra")
    axs4.set_xlabel("Wavelength (nm)")  # Update label
    axs4.set_ylabel("Flux (ADU)")
    axs4.set_yscale("linear")

    fig4.tight_layout()

    # Save the figure
    os.makedirs(output_dir , exist_ok=True)
    fig4.savefig(output_dir + "wave_version.png", dpi=300)
    print("Saved : "+output_dir + "wave_version.png")
    return

def runForStar(filepath, bestWavesFit, output_dir=""):
    filelist=[]
    for file in os.listdir(filepath):
        if file.endswith(".fits"):
            filelist.append(os.path.join(filepath, file))
    if output_dir=="":
        output_dir=filepath

    flux = prep_data(filelist, star=True)[0]
    bestFit = np.poly1d(bestWavesFit)
    starFigure(flux, bestFit, output_dir)
    return





if __name__ == "__main__":
    '''
    run for neon only, call functions for star

    to change the parameters to skip wavelenght or consider more peaks, change value in function
    findPeaks directly, in the instance of "run_trials_for_all_combination_of_waves"
    '''
    parser = OptionParser(usage)
    


    # Default values
    wave_list_string_default = "[748.9, 743.9, 724.5, 717.4, 703.2, 693, 671.7, 667.8, 659.9, 653.3, 650.7, 640.2, 638.2, 633.4, 630.5, 626.7, 621.7, 616.4]"
    #wave_list_string_default = "[753.6, 748.9, 743.9, 724.5, 717.4, 703.2, 693, 671.7, 667.8, 659.9, 653.3, 650.7, 640.2, 638.2, 633.4, 630.5, 626.7, 621.7, 616.4]"

    filelist = "."
    # Add options for these values
    parser.add_option("--wave_list", type="string", default=wave_list_string_default,
        help="comma-separated list of emmission lines (default: %s)"%wave_list_string_default)
    parser.add_option("--filelist", type="string", default=filelist,
        help="folder in which the preprocess files can be found (default: .)")
    
    options, args = parser.parse_args()

    runCreateWavelengthMap(options.filelist, options.wave_list)
