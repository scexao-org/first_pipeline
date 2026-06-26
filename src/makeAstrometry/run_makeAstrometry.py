#%%
"""
FIRST Pipeline - Astrometric Analysis Core Algorithms

Core functions for performing precise astrometric measurements from preprocessed FIRST data.
Separated from CLI interface to enable interactive use in VS Code and notebooks.

Created on Wed May 21 22:56:25 2025
@author: slacour
"""

import sys
import os
# Add src directory to path for imports to work in both interactive and package contexts
if os.path.join(os.path.dirname(__file__), '..') not in sys.path:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from typing import List, Tuple

import getpass
import matplotlib
if "VSCODE_PID" in os.environ:
    matplotlib.use('macosx')
elif os.environ.get('SPYDER_DEBUG_FILE'):
    print("Running in Spyder")
else:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, hist, clf, figure, legend, imshow
plt.ion()

from tqdm import tqdm
from astroplan import Observer
from astropy.time import Time

from first_pipeline_shared.classes.runPL_class_flatMap import FlatMap  
from first_pipeline_shared.classes.runPL_class_waveMap import WaveMap
from first_pipeline_shared.classes.runPL_class_fileList import FileList
from first_pipeline_shared.classes.runPL_class_dataCube import DataCube
from first_pipeline_shared.classes.runPL_class_couplingMap import CouplingMap

from first_pipeline_shared.libraries import runPL_library_io as runlib_io
from first_pipeline_shared.libraries import runPL_library_plots as runlib_plots
from first_pipeline_shared.libraries import runPL_library_linalg as runlib_linalg


# Subaru Observatory instance for timing
subaru = Observer.at_site("Subaru")


def get_filelist_astrometry(file_patterns, dark_patterns=None, flat_patterns=None, 
                         wave_patterns=None, object_name=None, modID=None, 
                         modScale=None, wollaston=None):
    """
    Create file list for astrometry analysis with calibration associations.
    
    Parameters
    ----------
    file_patterns : list
        List of file patterns to search for OBJECT data
    dark_patterns : list, optional
        List of patterns for dark files
    flat_patterns : list, optional
        List of patterns for flat field files
    wave_patterns : list, optional
        List of patterns for wavelength map files
    object_name : str, optional
        Filter by object name
    modID : int or list, optional
        Modulation pattern ID(s)
    modScale : int, optional
        Modulation scale
    wollaston : str, optional
        Wollaston polarizer status
        
    Returns
    -------
    fileList : FileList
        Configured file list object
    flatMap : FlatMap or None
        Flat field map object
    waveMap : WaveMap or None
        Wavelength map object
    """
    # Set default modID if not provided
    if modID is None:
        modID = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

    # Create initial file list
    # fileList = FileList(file_patterns, data_type="OBJECT", first_type='PREPROC', 
    #                    wollaston=wollaston, object_name=object_name, 
    #                    modID=modID, modScale=modScale)
    fileList = FileList(file_patterns, data_type="OBJECT", first_type='PREPROC', 
                       wollaston=wollaston, object_name=object_name, 
                       modID=modID, modScale=modScale)

    # Get constraints from the dataset
    object_name = fileList.header.get('OBJECT', None)
    wollaston = fileList.header.get('X_FIRWOL', None)
    modID = fileList.header.get('X_FIRMID', None)
    modScale = fileList.header.get('X_FIRMSC', None)

    # Recreate with constraints
    fileList = FileList(file_patterns, data_type="OBJECT", first_type='PREPROC',
                       wollaston=wollaston, object_name=object_name, 
                       modID=modID, modScale=modScale)

    # Set up associations and maps
    fileList.make_association(dark_patterns=dark_patterns)
    file_flat = fileList.get_flatmap_file(flat_patterns)
    file_wave = fileList.get_wavemap_file(wave_patterns)

    flatMap = FlatMap(file_flat) if file_flat is not None else None
    waveMap = WaveMap(file_wave) if file_wave is not None else None

    return fileList, flatMap, waveMap

def check_observatory_status():
    """
    Check if it's currently night at Subaru Observatory.
    
    Returns
    -------
    str
        Status message about observatory conditions
    """
    now_time = Time.now()
    if subaru.is_night(now_time):
        return "It's night at Subaru Observatory."
    else:
        return "It's day at Subaru Observatory."


if __name__ == "__main__":
    """
    Run astrometric analysis with development defaults.
    Perfect for testing and direct execution of core functionality.
    """
    print("Running createCouplingMap core with development defaults...")
    

    # Development/interactive mode handling
    print("Running in compiler")
    if getpass.getuser() == "slacour":
        object_name = None
        dark_patterns = None
        flat_patterns = None
        wave_patterns = None
        wavelength_smooth = 1
        wavelength_bin = 1
        Nsingular = 19*6
        modID = None
        modScale = None
        wollaston = None
        center_data = False

        file_patterns = ["/Users/slacour/DATA/LANTERNE/tmp/firstpl_13:0*.fits"]
        file_patterns = ["/Users/slacour/DATA/LANTERNE/20251230/preproc/*T12?2*.fits"]
        file_patterns = ["/Users/slacour/DATA/FIRST/20260608/preproc/firstpl_2026-06-08T10h[1-2]*_RASALHAGUE_P.fits"]
        wave_patterns = ["/Users/slacour/DATA/FIRST/20260608/wavemaps/"]
        # flat_patterns = wave_patterns


        # file_patterns = ["/Users/slacour/DATA/LANTERNE/20260114/preproc/*14T20h56*.fits"]
        # file_patterns = ["/Users/slacour/DATA/LANTERNE/20260114/preproc/*14T21h10*.fits"]
        # wave_patterns = ["/Users/slacour/DATA/LANTERNE/20251231/wavemaps/"]
        
        # #PDS70
        # file_patterns = ["/Users/slacour/DATA/LANTERNE/20260306/preproc/firstpl*_PDS70_P.fits"]
        # wave_patterns = ["/Users/slacour/DATA/LANTERNE/20260307/wavemaps/"]
        # flat_patterns = ["/Users/slacour/DATA/LANTERNE/20260114/flatmaps/"]

        # object_name = "HD163296"
        # modID = 2
        # file_patterns = ["/Users/slacour/DATA/LANTERNE/20260306/preproc/firstpl**.fits"]
        # wave_patterns = ["/Users/slacour/DATA/LANTERNE/20260307/wavemaps/"]

        
    print(f"Development override: wavelength_smooth={wavelength_smooth}, wavelength_bin={wavelength_bin}, Nsingular={Nsingular}")
    print(f"Development file patterns: {file_patterns}")

    def process_astrometric_data(
        file_patterns, object_name=None, dark_patterns=None, flat_patterns=None, wave_patterns=None, modID=None, modScale=None, wollaston=None,
        wavelength_smooth=1, wavelength_bin=1, Nsingular=19*6, center_data=False):

        # Set up default patterns
        if dark_patterns is None:
            dark_patterns = file_patterns
        if flat_patterns is None and file_patterns:
            folder = os.path.dirname(file_patterns[0])
            flat_patterns = file_patterns + [os.path.join(folder, "../flatmaps")] + [os.path.join(folder, "flatmaps")]
        if wave_patterns is None and file_patterns:
            folder = os.path.dirname(file_patterns[0])
            wave_patterns = file_patterns + [os.path.join(folder, "../wavemaps")] + [os.path.join(folder, "wavemaps")]

        # Get file list and calibration maps
        fileList, flatMap, waveMap = get_filelist_astrometry(
            file_patterns, dark_patterns, flat_patterns, wave_patterns,
            object_name, modID, modScale, wollaston
        )

        # Extract data
        datalist: List[DataCube] = fileList.extract_data_from_list(
            Nsmooth=wavelength_smooth, Nbin=wavelength_bin, flatMap=flatMap,
            waveMap=waveMap, center=center_data
        )

        # Concatenate data arrays
        flux = np.concatenate([d.flux for d in datalist])
        datacube = np.concatenate([d.data for d in datalist])
        datacube_var = np.concatenate([d.variance for d in datalist])
        wave = datalist[0].wave  # Assuming all have the same wavelength grid
        xmod = np.concatenate([d.xmod for d in datalist])
        ymod = np.concatenate([d.ymod for d in datalist])
        ra_dec = np.concatenate([d.compute_xy_sky() for d in datalist])

        # Create filename associations
        basenames = []
        for d in datalist:
            n = d.data.shape[0]
            basenames.extend([d.basename] * n)
        filenames = [d.filename for d in datalist]

        # Data quality filtering
        flux_goodData, flux_threshold = runlib_linalg.flux_filtering(flux)
        print(f"* Percentage of good data: {np.sum(flux_goodData)/len(flux_goodData.ravel())*100:.1f} % (flux threshold)")

        # SVD filtering
        data_svdfiltered, fit_goodData, errors = runlib_linalg.svd_filtering(datacube, flux_goodData, Nsingular)
        goodData = flux_goodData & fit_goodData
        print(f"* Percentage of good data: {np.sum(goodData)/len(goodData.ravel())*100:.1f} % (flux and svd threshold)")

        # Plot flux map
        runlib_plots.plot_flux_map(flux.mean(axis=(2))[0], xmod[0], ymod[0])

        data_normalized = data_svdfiltered / np.nanmean(flux, axis=(0,1))

        Nhanning = 20
        Nzeros = 11
        Nwave = len(wave)
        Ncube = data_normalized.shape[0]
        hanning_window = np.hanning(Nhanning)
        hanning_window = np.append( np.append(hanning_window, np.zeros(Nzeros)), hanning_window)
        hanning_window /= hanning_window.sum()  # Normalize the window
        data_hanning = data_normalized.copy()
        norm = np.convolve(np.ones(Nwave), hanning_window, mode='same')
        for i in tqdm(range(Ncube)):
            for j in range(data_normalized.shape[1]):
                for k in range(data_normalized.shape[2]):
                    data_hanning[i, j, k] = np.convolve(data_normalized[i, j, k], hanning_window, mode='same')/norm

        # ra_dec = np.stack([xmod,ymod],axis=-1)
        u = ra_dec[:,2:] - ra_dec[:,1:-1]
        v=  ra_dec[:,:-2] - ra_dec[:,1:-1]

        M_k = np.stack([u, v], axis=-1)
        M_k_inv = np.linalg.pinv(M_k)

        det_M = np.linalg.det(M_k)
        good_M = np.abs(det_M) > np.max(np.abs(det_M)) * 1e-2
        good_M &=  goodData[:,2:] & goodData[:,:-2] & goodData[:,1:-1]

        data_u = data_normalized[:,2:] - data_normalized[:,1:-1]
        data_v = data_normalized[:,:-2] - data_normalized[:,1:-1]
        data_w = data_normalized[:,1:-1] - data_hanning[:,1:-1]

        J_blocks = []
        C_blocks = []
        D_blocks = []
        W_blocks = []

        for i in tqdm(range(Ncube)):
            for j in range(data_u.shape[1]):
                    
                if good_M[i,j] == False:
                    continue

                # Y_k = [A_k  B_k], shape (19,2)
                Y_k = np.stack([data_u[i,j], data_v[i,j]], axis=-1)

                # J_k = Y_k @ inv(M_k)
                # Using solve is numerically better than explicit inverse:
                # J_k.T = solve(M_k.T, Y_k.T)
                J_k = Y_k @ M_k_inv[i,j]

                w = (np.linalg.pinv(J_k.transpose((1,0,2))) @ data_w[i,j].T[:,:,None])[:,:,0]

                J_blocks.append(J_k)
                C_blocks.append(data_w[i,j])
                D_blocks.append(data_normalized[i,j+1])
                W_blocks.append(w)

        J_tilde = np.vstack(J_blocks).transpose((1,0,2))
        C_tilde = np.concatenate(C_blocks).T
        D_tilde = np.concatenate(D_blocks).T
        W_tilde = np.array(W_blocks)

        w_hat = (np.linalg.pinv(J_tilde) @ C_tilde[:,:,None])[:,:,0]
        figure("astromet",clear=True)
        plt.title(object_name)
        plot(wave, w_hat[:,0],label="RA")
        plot(wave, w_hat[:,1],label="DEC")
        flux_scaled = np.nanmean(flux, axis=(0,1)) / np.nanmax(np.nanmean(flux, axis=(0,1)))*0.1
        # flux_scaled = f / f[:,1000:].max(axis=1)[:,None] *0.1
        plot(wave, flux_scaled.T,'k',label="Flux (scaled)")
        plt.ylabel("Astrometric signal (mas)")
        plt.xlabel("Wavelength")
        plt.legend()

        plt.savefig("astrometry_result.png", dpi=300)

        # Speed of light in km/s
        c = 299792.458
        # Rest wavelength of H-alpha in nm
        lambda0 = 656.28

        To_plot = (wave > 655.5) & (wave < 657.2)
        astrometry_xy = w_hat[To_plot]
        astrometry_wave = wave[To_plot]
        # Doppler velocity (km/s)
        velocity = c * (astrometry_wave - lambda0) / lambda0




        fig, ax = plt.subplots(1, 1, figsize=(8, 6), num="astrometry_scatter", clear=True)
        flux_scaled_filtered = flux_scaled[To_plot]
        scatter = ax.scatter(astrometry_xy[:, 0], astrometry_xy[:, 1], c=velocity, s=flux_scaled_filtered*1000, cmap='viridis', alpha=0.6)
        ax.plot(astrometry_xy[:, 0], astrometry_xy[:, 1], 'k-', alpha=0.3, linewidth=1)
        ax.set_xlabel("RA (mas)")
        ax.set_ylabel("DEC (mas)")
        ax.set_aspect('equal')
        lim = np.max(np.abs(ax.get_xlim() + ax.get_ylim()))
        ax.set_xlim(lim, -lim)
        ax.set_ylim(-lim, lim)
        fig.colorbar(scatter, ax=ax, label="Velocity (km/s)")
        ax.set_title(f"{object_name} - Astrometry vs Velocity")
        fig.savefig("astrometry_scatter.png", dpi=300)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(plt.MultipleLocator(0.1))
        ax.yaxis.set_major_locator(plt.MultipleLocator(0.1))

        PA=132*np.pi/180
        y = np.linspace(-lim,lim,100)
        x = np.tan(PA)*y
        ax.plot(x,y,'k--',label="PA=132°") 
        ax.legend() 

        fig.savefig("astrometry_scatter_PA.png", dpi=300)


        C_tilde_2 = J_tilde @ w_hat[:,:,None]
        residuals = C_tilde - C_tilde_2[:,:,0]
        halpha_index = 1099

        y = C_tilde[halpha_index] 
        y_fit = C_tilde_2[halpha_index,:,0]

        x=np.arange(len(y))

        residuals = y - y_fit

        fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=False)
        fig.suptitle("fit residuals")

        # Top: data + fit
        ax = axes[0]
        ax.plot(x, y, 'o', label='data', alpha=0.8)
        order = np.argsort(x)
        ax.plot(np.array(x)[order], np.array(y_fit)[order], '-', label='fit')
        ax.set_ylabel("y")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Middle: residuals vs x
        ax = axes[1]
        ax.plot(x, residuals, 'o', alpha=0.8)
        ax.set_ylabel("Residuals")
        ax.axhline(0, linestyle='--')
        ax.grid(True, alpha=0.3)

        # Bottom: histogram of residuals
        ax = axes[2]
        ax.hist(residuals, bins=20, alpha=0.8)
        ax.set_xlabel("Residuals")
        ax.set_ylabel("Count")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        fig, ax = plt.subplots(1, 2, figsize=(14, 6), num="residual histogram", clear=True)

        res = residuals
        values = y
        
        # Plot residuals histogram
        ax[0].hist(res, bins=30, edgecolor='black', alpha=0.7)
        ax[0].set_xlabel("Residuals")
        ax[0].set_ylabel("Frequency")
        ax[0].set_title(f"Residual Distribution at H-alpha (λ={wave[halpha_index]:.2f} nm)")
        ax[0].grid(True, alpha=0.3)

        # Plot values histogram
        ax[1].hist(values**2-residuals**2, bins=100, edgecolor='black', alpha=0.7, color='orange')
        ax[1].set_xlabel("Delta chi2 values")
        ax[1].set_ylabel("Frequency")
        ax[1].set_title(f"Data Values Distribution at H-alpha (λ={wave[halpha_index]:.2f} nm)")
        ax[1].grid(True, alpha=0.3)

        # Calculate significance
        mean_res = np.mean(res)
        std_res = np.std(res)
        mean_val = np.mean(values)
        std_val = np.std(values)
        significance = np.abs(mean_res) / std_res if std_res > 0 else 0
        
        # Compare residuals to values
        residual_to_data_ratio = std_res / std_val if std_val > 0 else np.inf
        
        stats_text = (f"Residuals - Mean: {mean_res:.4f}, Std: {std_res:.4f}\n"
                    f"Values - Mean: {mean_val:.4f}, Std: {std_val:.4f}\n"
                    f"Significance: {significance:.2f}σ\n"
                    f"Residual/Data ratio: {residual_to_data_ratio:.4f}")
        
        ax[0].text(0.98, 0.97, stats_text, 
            transform=ax[0].transAxes, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), fontsize=9)

        fig.tight_layout()
        fig.savefig("residual_histogram.png", dpi=300)




    process_astrometric_data(
        file_patterns=file_patterns,
        object_name=object_name,
        dark_patterns=dark_patterns,
        flat_patterns=flat_patterns,
        wave_patterns=wave_patterns,
        modID=modID,
        modScale=modScale,
        wollaston=wollaston,
        wavelength_smooth=wavelength_smooth,
        wavelength_bin=wavelength_bin,
        Nsingular=Nsingular,
        center_data=center_data)
        # save_individual_frames=save_individual_frames,)
# %%
