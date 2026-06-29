#%%

"""
FIRST Pipeline - Image Reconstruction Core Algorithms

Core functions for reconstructing images from preprocessed FIRST Visible Photonic Lantern data.
Separated from CLI interface to enable interactive use in VS Code and notebooks.

Created on Wed May 21 22:56:25 2025
@author: slacour
"""

import os
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

import numpy as np
from typing import List, Tuple, Optional
from scipy.interpolate import griddata
from astropy.io import fits
from datetime import datetime

     
import matplotlib.pyplot as plt
from tqdm import tqdm

from first_pipeline_shared.classes.runPL_class_flatMap import FlatMap
from first_pipeline_shared.classes.runPL_class_waveMap import WaveMap
from first_pipeline_shared.classes.runPL_class_fileList import FileList
from first_pipeline_shared.classes.runPL_class_dataCube import DataCube
from first_pipeline_shared.classes.runPL_class_couplingMap import CouplingMap

from first_pipeline_shared.libraries import runPL_library_io as runlib_io
from first_pipeline_shared.libraries import runPL_library_plots as runlib_plots



def get_filelist_image(file_patterns, dark_patterns=None, cmap_patterns=None,
                      object_name=None, modID=None, modScale=None, wollaston=None):
    """
    Create file list for image reconstruction with coupling map association.
    
    Parameters
    ----------
    file_patterns : list
        List of file patterns to search for OBJECT data
    dark_patterns : list, optional
        List of patterns for dark files
    cmap_patterns : list, optional
        List of patterns for coupling map files
    object_name : str, optional
        Filter by object name
    modID : int, optional
        Modulation pattern ID
    modScale : int, optional
        Modulation scale
    wollaston : str, optional
        Wollaston polarizer status
        
    Returns
    -------
    fileList : FileList
        Configured file list object
    couplingMap : CouplingMap
        Coupling map object for image reconstruction
    flatMap : FlatMap or None
        Flat field map if available in coupling map
    waveMap : WaveMap or None
        Wavelength map if available in coupling map
    """
    # Set default patterns
    if dark_patterns is None:
        dark_patterns = file_patterns
    if cmap_patterns is None and file_patterns:
        folder = os.path.dirname(file_patterns[0])
        cmap_patterns = file_patterns + [os.path.join(folder, "../couplingmaps")] + [os.path.join(folder, "couplingmaps")]

    # Create file list
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

    fileList.make_association(dark_patterns=dark_patterns)

    # Load coupling map and associated calibrations
    print("Coupling map patterns: ", cmap_patterns)
    file_coup = fileList.get_couplingmap_file(cmap_patterns)
    couplingMap = CouplingMap(file_coup)

    # Check for embedded calibrations in coupling map file
    with fits.open(file_coup) as hdul:
        extension_names = [hdu.name for hdu in hdul]
        print(f"Extensions in coupling map file: {extension_names}")

    flatMap = FlatMap(file_coup) if 'FLAT' in extension_names else None
    waveMap = WaveMap(file_coup) if 'WAVELENGTH' in extension_names else None

    return fileList, couplingMap, flatMap, waveMap


def compute_star_positions(datacube_T, couplingMap, ra_dec):
    """
    Detect star positions using coupling map chi-squared filtering.
    
    Parameters
    ----------
    datacube_T : numpy.ndarray
        Transposed data cube with shape (Nwave, Noutput, Nimages)
    couplingMap : CouplingMap
        Coupling map object for position detection
    ra_dec : numpy.ndarray
        Sky coordinate positions for each image
        
    Returns
    -------
    star_detected : numpy.ndarray
        Boolean array indicating frames with detected stars
    star_index : numpy.ndarray
        Triangle indices for detected stars
    star_radec : numpy.ndarray
        Sky coordinates of detected stars
    chi2 : numpy.ndarray
        Chi-squared values for star detection
    """

    QT, R = couplingMap.QT_and_R_matrices()

    position = couplingMap.position
    Ntriangles = QT.shape[0]
    Nqr = QT.shape[2]

    Nwave, Noutput, Nimages = datacube_T.shape

    datacube_T=datacube_T.reshape((datacube_T.shape[0], datacube_T.shape[1], -1))
    chi2_max = np.sum(datacube_T**2, axis=(0,1))

    chi2_map = np.zeros((Ntriangles,  Nimages))
    chi2_map[:] = chi2_max
    # Here, the computation of the chi2 is simplified by the fact that QT is orthonormal
    # chi2 = ||data - Q @ Q.T @ data||^2 = ||data||^2 - ||Q.T @ data||^2
    for t in tqdm(range(Ntriangles), desc="Computing chi2 map"):
        k = QT[t] @ datacube_T
        chi2_map[t, :] -= np.sum(k ** 2, axis=(0, 1))

    # Replace inf/nan values with a large number before finding argmin
    chi2_map_nonan = np.nan_to_num(chi2_map, nan=np.inf, posinf=np.inf, neginf=np.inf)
    chi2_argmin = np.argmin(chi2_map_nonan, axis=0)
    figure(num="Chi2 map arg min", figsize=(16, 6), clear=True)
    plot(position[chi2_argmin])

    Npixel = 150
    grid_x, grid_y = runlib_plots.make_image_grid(ra_dec, Npixel)

    chi2_images = []
    for i in tqdm(range(Nimages), desc="Calculating chi2 images"):
            # Interpolate the fluxes onto the grid
        # chi2_image = griddata((ra_dec[i,:,0],ra_dec[i,:,1]), chi2_map[:,i], (grid_x, grid_y), method='nearest')
        chi2_image = griddata((ra_dec[i,:,0],ra_dec[i,:,1]), chi2_map[:,i], (grid_x, grid_y), method='cubic')
        chi2_images.append(chi2_image)

    chi2_images = np.array(chi2_images)

    for i in range(Nimages):
        point_nan = np.isnan(chi2_images[i])
        chi2_images[i,point_nan]=np.nanmax(chi2_images[i])

    chi2_images_argmin = np.nansum(chi2_images,axis=0).argmin()
    star_radec = np.array((grid_x.ravel()[chi2_images_argmin], grid_y.ravel()[chi2_images_argmin]))
    star_indices = np.linalg.norm(ra_dec - star_radec, axis=-1) < 10
    chi2_map[~star_indices.T] = np.nan
    chi2_map_argmin = np.zeros(Nimages, dtype=int)
    chi2_goodData = np.zeros(Nimages, dtype=bool)
    for i in range(Nimages):
        try:
            chi2_map_argmin[i] = np.nanargmin(chi2_map[:,i], axis=0)
            chi2_goodData[i] = True
        except:
            chi2_goodData[i] = False

    # Instead of np.diag, use advanced indexing for clarity and correctness:
    chi2_min = chi2_map[chi2_map_argmin, np.arange(Nimages)]
    chi2_ratio=chi2_min/chi2_max

    # Filter out outliers based on chi2 and chi2_ratio thresholds
    chi2_min_threshold = np.nanmedian(chi2_min[chi2_goodData]) * 5
    chi2_ratio_threshold = np.nanmedian(chi2_ratio[chi2_goodData]) * 3.5

    chi2_goodData &= (chi2_min < chi2_min_threshold) & (chi2_ratio < chi2_ratio_threshold)

    star_detected = chi2_goodData
    star_index = chi2_map_argmin

    print(f"* Percentage of data with star detected: {np.sum(star_detected)/len(star_detected)*100:.1f} % (flux, svd and chi2 threshold)")
    
    return star_detected, star_index, star_radec, chi2_images


def compute_residuals(datacube_T, couplingMap, star_detected, star_index):
    """
    Compute residuals after removing best-fit stellar signal using coupling maps.
    
    Parameters
    ----------
    datacube_T : numpy.ndarray
        Transposed data cube with shape (Nwave, Noutput, Nimages)
    couplingMap : CouplingMap
        Coupling map object for residual calculation
    star_detected : numpy.ndarray
        Boolean array indicating frames with stars
    star_index : numpy.ndarray
        Triangle indices for each star detection
        
    Returns
    -------
    residuals : numpy.ndarray
        Residual data cube after star removal
    """
    residuals = datacube_T.copy()

    QT, R = couplingMap.QT_and_R_matrices()
    
    for i in tqdm(range(residuals.shape[2]), desc="Calculating residuals of the 3D image"):
        if star_detected[i]:
            t = star_index[i]
            k = QT[t] @ residuals[:, :, i, None]
            residuals[:, :, i] -= (QT[t].transpose((0, 2, 1)) @ k)[:, :, 0]

    return residuals


def reconstruct_fluxes(datacube_T, residuals, couplingMap, dit, gain):
    """
    Reconstruct flux maps from data cube using coupling map transformation.
    
    Parameters
    ----------
    datacube_T : numpy.ndarray
        Transposed data cube with shape (Nwave, Noutput, Nimages)
    residuals : numpy.ndarray
        Residual data cube after stellar removal
    couplingMap : CouplingMap
        Coupling map for flux reconstruction
    dit : float
        Detector integration time
    gain : float
        Detector gain factor
        
    Returns
    -------
    fluxes : numpy.ndarray
        Reconstructed flux maps
    fluxes_residuals : numpy.ndarray
        Residual flux maps
    fluxes_variance : numpy.ndarray
        Variance estimates for flux maps
    """
    # Transform data to flux using coupling map
    fluxes = np.matmul(couplingMap.data_2_flux, datacube_T) / dit * gain
    fluxes_residuals = np.matmul(couplingMap.data_2_flux, residuals) / dit * gain
    
    # Estimate variance from flux statistics
    fluxes_variance = fluxes.mean(axis=0, keepdims=True)
    fluxes_variance[:] = fluxes_variance.std(axis=1, keepdims=True)**2 * 19

    return fluxes, fluxes_residuals, fluxes_variance


def create_image_maps(ra_dec, fluxes, fluxes_residuals, fluxes_variance, Npixels=75):
    """
    Create 2D image maps from flux data using sky coordinate interpolation.
    
    Parameters
    ----------
    ra_dec : numpy.ndarray
        Sky coordinates for each measurement
    fluxes : numpy.ndarray
        Flux measurements
    fluxes_residuals : numpy.ndarray
        Residual flux measurements
    fluxes_variance : numpy.ndarray
        Variance estimates
    Npixels : int, optional
        Number of pixels for image grid (default: 75)
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'flux_maps_sum': Combined flux image
        - 'flux_maps_residuals_sum': Combined residual image
        - 'flux_maps_variance': Variance map
        - 'flux_maps_snr': Signal-to-noise ratio map
        - 'flux_maps_contrast': Contrast map
        - 'flux_maps': Individual frame flux maps (if requested)
        - 'flux_maps_residuals': Individual frame residual maps (if requested)
    """
    # Create interpolated image maps
    flux_maps = runlib_plots.make_image_using_grid(ra_dec, fluxes, desc="Creating flux maps", Npixels=Npixels)
    flux_maps_residuals = runlib_plots.make_image_using_grid(ra_dec, fluxes_residuals, desc="Creating flux residuals", Npixels=Npixels)
    flux_maps_variance = runlib_plots.make_image_using_grid(ra_dec, fluxes_variance, desc="Creating flux variance", Npixels=Npixels)
    
    # Combine maps
    flux_maps_sum = np.nanmean(flux_maps, axis=0)
    flux_maps_residuals_sum = np.nanmean(flux_maps_residuals, axis=0)
    flux_maps_variance = np.nanmean(flux_maps_variance, axis=0)
    
    # Calculate derived quantities
    flux_maps_snr = flux_maps_sum / np.sqrt(flux_maps_variance)
    flux_maps_contrast = np.sqrt(flux_maps_variance) / np.nanmax(flux_maps)

    return {
        'flux_maps_sum': flux_maps_sum,
        'flux_maps_residuals_sum': flux_maps_residuals_sum,
        'flux_maps_variance': flux_maps_variance,
        'flux_maps_snr': flux_maps_snr,
        'flux_maps_contrast': flux_maps_contrast,
        'flux_maps': flux_maps,
        'flux_maps_residuals': flux_maps_residuals
    }


def create_wavelength_slices(ra_dec, fluxes, fluxes_residuals, Nwave):
    """
    Create wavelength-resolved image cubes for spectral analysis.
    
    Parameters
    ----------
    ra_dec : numpy.ndarray
        Sky coordinates for each measurement
    fluxes : numpy.ndarray
        Flux measurements with shape (Nwave, Ntriangles, Nimages)
    fluxes_residuals : numpy.ndarray
        Residual flux measurements
    Nwave : int
        Number of wavelength channels
        
    Returns
    -------
    flux_maps_wave : numpy.ndarray
        3D image cube with wavelength dimension
    residuals_maps_wave : numpy.ndarray
        3D residual cube with wavelength dimension
    """
    flux_maps_wave = []
    residuals_maps_wave = []

    for w in tqdm(range(Nwave), desc="Creating wavelength slices"):
        flux_maps_tmp = runlib_plots.make_image_using_grid(ra_dec, fluxes[w, None])
        flux_maps_residuals_tmp = runlib_plots.make_image_using_grid(ra_dec, fluxes_residuals[w, None])
        
        flux_maps_sum = np.nanmean(flux_maps_tmp, axis=0)
        flux_maps_residuals_sum = np.nanmean(flux_maps_residuals_tmp, axis=0)

        flux_maps_wave.append(flux_maps_sum)
        residuals_maps_wave.append(flux_maps_residuals_sum)

    return np.array(flux_maps_wave), np.array(residuals_maps_wave)


def save_reconstructed_image(image_data, header, output_dir, 
                           save_individual_frames=True, save_individual_wavelength=False,
                           wavelength_smooth=1):
    """
    Save reconstructed images to FITS file with optional extensions.
    
    Parameters
    ----------
    image_data : dict
        Dictionary containing image arrays and metadata
    header : astropy.io.fits.Header
        FITS header to be updated
    output_dir : str
        Output directory path
    save_individual_frames : bool, optional
        Include individual frame data (default: True)
    save_individual_wavelength : bool, optional
        Include wavelength-resolved cubes (default: False)
    wavelength_smooth : int, optional
        Wavelength smoothing parameter for header
        
    Returns
    -------
    str
        Path to saved image file
    """
    # Update header
    header = header.copy()
    header['X_FIRTYP'] = 'IMAGE'
    current_time = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
    header['DATE-PRO'] = current_time
    header['WLSMOOTH'] = wavelength_smooth

    # Create HDU list
    list_of_hdus = []
    
    # Primary HDU with summed image
    hdu_primary = fits.PrimaryHDU(image_data['flux_maps_sum'])
    
    # Residual map
    hdu_residual = fits.ImageHDU(image_data['flux_maps_residuals_sum'], name="RESIDUAL")
    list_of_hdus += [hdu_primary, hdu_residual]

    # Individual frames if requested
    if save_individual_frames:
        hdu_frame = fits.ImageHDU(image_data['flux_maps'], name="FRAMES")
        hdu_frame_residual = fits.ImageHDU(image_data['flux_maps_residuals'], name="FRAMES_RESIDUAL")
        hdu_snr = fits.ImageHDU(image_data['flux_maps_snr'], name="SNR")
        hdu_contrast = fits.ImageHDU(image_data['flux_maps_contrast'], name="CONTRAST")
        list_of_hdus += [hdu_frame, hdu_frame_residual, hdu_snr, hdu_contrast]

    # Wavelength slices if requested
    if save_individual_wavelength and 'flux_maps_wave' in image_data:
        hdu_wave = fits.ImageHDU(image_data['flux_maps_wave'], name="3D_IMAGE")
        hdu_wave_residual = fits.ImageHDU(image_data['residuals_maps_wave'], name="3D_IMAGE_RESIDUAL")
        list_of_hdus += [hdu_wave, hdu_wave_residual]
        header['X_FIRTYP'] = 'WDIMAGE'

    # Create output directory and save
    os.makedirs(output_dir, exist_ok=True)
    
    # Add header to primary HDU
    hdu_primary.header.extend(header, strip=True)
    
    # Create and save FITS file
    hdul = fits.HDUList(list_of_hdus)
    output_filename = os.path.join(output_dir, runlib_io.create_basename(header))
    hdul.writeto(output_filename, overwrite=True)
    
    return output_filename


def create_diagnostic_plots(image_data, ra_dec, Npixels=75):
    """
    Generate diagnostic plots for image reconstruction quality assessment.
    
    Parameters
    ----------
    image_data : dict
        Dictionary containing reconstructed image data
    ra_dec : numpy.ndarray
        Sky coordinates for plotting grid
    Npixels : int, optional
        Number of pixels for image grid
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure showing reconstruction diagnostics
    """
    # Create coordinate grid for plotting
    grid_x, grid_y = runlib_plots.make_image_grid(ra_dec, Npixels=Npixels)

    # Create diagnostic plots
    fig, axes = plt.subplots(2, 2, num="Flux Maps", figsize=(15, 15.6), clear=True)
    
    # Flux map
    im0 = axes[0,0].imshow(image_data['flux_maps_sum'][::-1].T, origin='lower', aspect='auto',
                          extent=[grid_x.max(), grid_x.min(), grid_y.min(), grid_y.max()], vmin=0)
    axes[0,0].set_title('Flux Map')
    axes[0,0].set_xlabel('RA (mas)')
    axes[0,0].set_ylabel('Dec (mas)')
    fig.colorbar(im0, ax=axes[0,0], orientation='vertical', label='Flux (e-/s)')

    # Residual map
    im1 = axes[0,1].imshow(image_data['flux_maps_residuals_sum'][::-1].T, origin='lower', aspect='auto',
                          extent=[grid_x.max(), grid_x.min(), grid_y.min(), grid_y.max()], vmin=0)
    axes[0,1].set_title('Flux Residuals')
    axes[0,1].set_xlabel('RA (mas)')
    axes[0,1].set_ylabel('Dec (mas)')
    fig.colorbar(im1, ax=axes[0,1], orientation='vertical', label='Residual Flux (e-/s)')

    # SNR map
    im2 = axes[1,0].imshow(image_data['flux_maps_snr'][::-1].T, origin='lower', aspect='auto',
                          extent=[grid_x.max(), grid_x.min(), grid_y.min(), grid_y.max()])
    axes[1,0].set_title('Signal-to-Noise Ratio')
    axes[1,0].set_xlabel('RA (mas)')
    axes[1,0].set_ylabel('Dec (mas)')
    fig.colorbar(im2, ax=axes[1,0], orientation='vertical', label='SNR')

    # Contrast map
    im3 = axes[1,1].imshow(image_data['flux_maps_contrast'][::-1].T, origin='lower', aspect='auto',
                          extent=[grid_x.max(), grid_x.min(), grid_y.min(), grid_y.max()])
    axes[1,1].set_title('Contrast')
    axes[1,1].set_xlabel('RA (mas)')
    axes[1,1].set_ylabel('Dec (mas)')
    fig.colorbar(im3, ax=axes[1,1], orientation='vertical', label='Contrast')

    plt.tight_layout()
    
    return fig


def process_image_reconstruction_data(file_patterns=None, object_name=None, dark_patterns=None,
                                    cmap_patterns=None, wavelength_smooth=None, modID=None, 
                                    modScale=None, wollaston=None, save_individual_frames=None,
                                    save_individual_wavelength=None, Npixels=75):
    """
    Complete workflow for image reconstruction from coupling maps and preprocessed data.
    
    This is the main processing function that orchestrates the entire image
    reconstruction workflow from file loading through final image creation.
    
    Parameters
    ----------
    file_patterns : list, optional
        List of file patterns to search for OBJECT data files.
        If None, uses development defaults.
    object_name : str, optional
        Filter by object name
    dark_patterns : list, optional
        List of patterns for dark files
    coupling_map : str, optional
        Specific coupling map file to use
    wavelength_smooth : int, optional
        Wavelength smoothing factor. If None, uses development defaults.
    modID : int, optional
        Modulation pattern ID
    modScale : int, optional
        Modulation scale
    wollaston : str, optional
        Wollaston polarizer status
    save_individual_frames : bool, optional
        Include individual frame data. If None, uses development defaults.
    save_individual_wavelength : bool, optional
        Include wavelength cubes. If None, uses development defaults.
    Npixels : int, optional
        Number of pixels for reconstructed images (default: 75)
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'output_filename': path to saved image file
        - 'image_data': reconstructed image arrays
        - 'star_detected': star detection results
        - 'figures': list of diagnostic figures
    """


    # Get file list and coupling map
    fileList, couplingMap, flatMap, waveMap = get_filelist_image(
        file_patterns, dark_patterns, cmap_patterns, object_name, modID, modScale, wollaston
    )

    # Extract data with coupling map wavelength binning
    datalist: List[DataCube] = fileList.extract_data_from_list(
        Nsmooth=wavelength_smooth, Nbin=couplingMap.wavelength_bin,
        flatMap=flatMap, waveMap=waveMap, center=False
    )



    results = []
    figures = []

    for i, d in enumerate(datalist[:]):
        print(f"---->  Processing file {i+1}/{len(datalist)}: {d.filename}")

        # Extract data arrays
        flux = d.flux
        datacube = d.data
        datacube_var = d.variance
        ra_dec = d.compute_xy_sky(couplingMap)

        # Get dimensions
        Ncube = ra_dec.shape[0]
        Nmod = ra_dec.shape[1]
        Npos = ra_dec.shape[2]
        Nwave = datacube.shape[3]
        Noutput = datacube.shape[2]
        Nimages = Ncube * Nmod

        # Reshape for processing
        ra_dec = ra_dec.reshape((-1, *ra_dec.shape[2:]))
        datacube = datacube.reshape((-1, *datacube.shape[2:]))
        flux = flux.reshape((-1, *flux.shape[2:]))
        datacube_var = datacube_var.reshape((-1, *datacube_var.shape[2:]))

        # Transpose for coupling map operations
        datacube_T = datacube.transpose((2, 1, 0))
        datacube_var_T = datacube_var.transpose((2, 1, 0))

        # Detect stars and compute positions
        star_detected, star_index, star_radec, chi2 = compute_star_positions(datacube_T, couplingMap, ra_dec)

        # Compute residuals after star removal
        residuals = compute_residuals(datacube_T, couplingMap, star_detected, star_index)

        # Reconstruct fluxes
        fluxes, fluxes_residuals, fluxes_variance = reconstruct_fluxes(
            datacube_T, residuals, couplingMap, d.dit, d.gain
        )

        # Create image maps
        image_data = create_image_maps(ra_dec, fluxes, fluxes_residuals, fluxes_variance, Npixels)

        # Create wavelength slices if requested
        if save_individual_wavelength:
            flux_maps_wave, residuals_maps_wave = create_wavelength_slices(
                ra_dec, fluxes, fluxes_residuals, Nwave
            )
            image_data['flux_maps_wave'] = flux_maps_wave
            image_data['residuals_maps_wave'] = residuals_maps_wave

        # Set up output directory
        output_dir = os.path.join(d.dirname, "../images")

        # Save reconstructed image
        output_filename = save_reconstructed_image(
            image_data, d.header, output_dir, save_individual_frames, 
            save_individual_wavelength, wavelength_smooth
        )

        print(f"Image saved to {output_filename}")

        # Create diagnostic plots
        fig = create_diagnostic_plots(image_data, ra_dec, Npixels)
        figures.append(fig)

        # Save plots
        runlib_plots.save_pdf_in_file(output_filename)

        results.append({
            'output_filename': output_filename,
            'image_data': image_data,
            'star_detected': star_detected,
            'star_radec': star_radec,
            'chi2': chi2
        })

    return {
        'results': results,
        'figures': figures,
        'couplingMap': couplingMap
    }


if __name__ == "__main__":
    """
    Run image reconstruction with development defaults.
    Perfect for testing and direct execution of core functionality.
    """
    print("Running makeImage core with development defaults...")
    

    # Development/interactive mode handling
    print("Running in compiler")
    if getpass.getuser() == "slacour":

        object_name = None
        dark_patterns = None
        cmap_patterns = None
        wavelength_smooth = 1
        modID = None
        modScale = None
        wollaston = None
        save_individual_frames = False
        save_individual_wavelength = False
        Npixels = 114

        # HIP81126
        file_patterns = ["/Users/slacour/DATA/LANTERNE/20250510/preproc/firstpl_2025-05-14T09h59*fits"]
        
        # ALFDEL
        file_patterns = ["/Users/slacour/DATA/FIRST/20260625/preproc/firstpl_2026-06-25T14h18*_ALFDEL_P.fits", "/Users/slacour/DATA/FIRST/20260625/preproc/firstpl_2026-06-25T14h17m56s_ALFDEL_P.fits"]

        

    # Note: Development environment detection and default paths
    # are handled autonomously in run_makeImage()

    # Process image reconstruction data
    result = process_image_reconstruction_data(
        file_patterns=file_patterns,
        object_name=object_name,
        dark_patterns=dark_patterns,
        cmap_patterns=cmap_patterns,
        wavelength_smooth=wavelength_smooth,
        modID=modID,
        modScale=modScale,
        wollaston=wollaston,
        save_individual_frames=save_individual_frames,
        save_individual_wavelength=save_individual_wavelength,
        Npixels=Npixels
    )
    
# %%
