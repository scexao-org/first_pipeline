#! /usr/bin/env python3
# -*- coding: iso-8859-15 -*-
#%%
"""
Created on Sun May 24 22:56:25 2015

@author: slacour
"""
import os
import numpy as np
from scipy import linalg
from astropy.io import fits
from scipy.ndimage import uniform_filter1d
from scipy.spatial import Delaunay
from scipy.interpolate import griddata
from scipy.optimize import curve_fit
from scipy.spatial import cKDTree

import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
from astroplan import Observer
from astropy.time import Time, TimeDelta
from astropy.coordinates import SkyCoord
import astropy.units as u
subaru = Observer.at_site("Subaru")
plt.ion()

def plot_wavefit_coeffs(peaks_all_sub, peaks_all_sub_good, aberations, aberations_fit):

    # Plot peaks_all_sub and peaks_all_sub_good using imshow
    fig, axes = plt.subplots(2, 3, figsize=(18, 8))
    fig.suptitle('Peak Analysis and Aberration Fitting')

    # Plot all peaks
    im1 = axes[0, 0].imshow(peaks_all_sub, aspect='auto', interpolation='none', rasterized=True, cmap='viridis')
    fig.colorbar(im1, ax=axes[0, 0], label='Peak Position (x pixels)')
    axes[0, 0].set_title('All Detected Peaks')
    axes[0, 0].set_xlabel('Peak Index')
    axes[0, 0].set_ylabel('PL output')

    # Plot only good peaks
    im2 = axes[0, 1].imshow(peaks_all_sub_good, aspect='auto', interpolation='none', rasterized=True, cmap='viridis')
    fig.colorbar(im2, ax=axes[0, 1], label='Peak Position (x pixels)')
    axes[0, 1].set_title('Good Peaks Only')
    axes[0, 1].set_xlabel('Good Peak Index')
    axes[0, 1].set_ylabel('PL output')

    # Plot aberrations
    im3 = axes[0, 2].imshow(aberations, aspect='auto', interpolation='none', rasterized=True, cmap='RdBu')
    fig.colorbar(im3, ax=axes[0, 2], label='Aberration (delta x pixels)')
    axes[0, 2].set_title('Aberrations')
    axes[0, 2].set_xlabel('Good Peak Index')
    axes[0, 2].set_ylabel('PL output')

    # Plot all peaks with median removed
    peaks_all_sub_centered = peaks_all_sub - np.median(peaks_all_sub, axis=0)[None, :]
    im4 = axes[1, 0].imshow(peaks_all_sub_centered, aspect='auto', interpolation='none', rasterized=True, cmap='viridis')
    fig.colorbar(im4, ax=axes[1, 0], label='Peak Position Deviation (delta x pixels)')
    axes[1, 0].set_title('All Peaks (Median Removed)')
    axes[1, 0].set_xlabel('Peak Index')
    axes[1, 0].set_ylabel('PL output')

    # Plot good peaks with median removed
    peaks_all_sub_good_centered = peaks_all_sub_good - np.median(peaks_all_sub_good, axis=0)[None, :]
    im5 = axes[1, 1].imshow(peaks_all_sub_good_centered, aspect='auto', interpolation='none', rasterized=True, cmap='viridis')
    fig.colorbar(im5, ax=axes[1, 1], label='Peak Position Deviation (delta x pixels)')
    axes[1, 1].set_title('Good Peaks (Median Removed)')
    axes[1, 1].set_xlabel('Good Peak Index')
    axes[1, 1].set_ylabel('PL output')

    # Plot residual aberrations (observed - fitted)
    aberations_residual = aberations - aberations_fit
    im6 = axes[1, 2].imshow(aberations_residual, aspect='auto', interpolation='none', rasterized=True, cmap='RdBu')
    fig.colorbar(im6, ax=axes[1, 2], label='Residual Aberration (delta x pixels)')
    axes[1, 2].set_title('Residual Aberrations (obs - fit)')
    axes[1, 2].set_xlabel('Good Peak Index')
    axes[1, 2].set_ylabel('PL output')

    fig.tight_layout()

    return fig

def plot_results_of_line_identification(spectrum, ref_pixels_lines, neon_wavelengths, best_idx, best_valid_idx, coeffs_poly, Nexclude):

    p2w = np.poly1d(coeffs_poly)
    pixels = np.arange(0, len(spectrum))
    wave =  p2w(pixels)

    ref_pixels_lines_int = np.clip(np.round(ref_pixels_lines).astype(int), 0, len(spectrum)-1)
    ref_pixels_neon_int = np.abs(wave-neon_wavelengths[:,None]).argmin(axis=1)

    fig,axs=plt.subplots(3,1,num="line fitting results, Nexclude="+str(Nexclude),figsize=(18, 8), sharex = True, clear=True)
    axs[0].plot(p2w(pixels),spectrum,label="cumulative spectrum (all outputs)")
    axs[0].plot(p2w(ref_pixels_lines), spectrum[ref_pixels_lines_int], "o", markersize=10, markerfacecolor='none', markeredgecolor='g', markeredgewidth=2,label="discarded peak")
    axs[0].plot(p2w(ref_pixels_lines)[best_valid_idx], spectrum[ref_pixels_lines_int][best_valid_idx], "o", markersize=10, markerfacecolor='none', markeredgecolor='r', markeredgewidth=2,label="used peak")
    axs[0].plot(neon_wavelengths, spectrum[ref_pixels_neon_int], "s", markersize=10, markerfacecolor='none', markeredgecolor='b', markeredgewidth=2, label="catalog Neon lines",zorder=-12)
    # axs[0].plot(neon_wavelengths, neon_intensity*2e5, "g^", markersize=10, markerfacecolor='none', markeredgecolor='g', markeredgewidth=2)
    axs[0].set_title("Detected Peaks in Neon Spectrum, Nexclude="+str(Nexclude))
    axs[0].set_xlabel("Wavelength (nm)")
    axs[0].set_ylabel("Intensity")
    axs[0].legend()

    axs[1].plot(wave,wave,'k')
    axs[1].plot(p2w(ref_pixels_lines),neon_wavelengths[best_idx], "o", markersize=10, markerfacecolor='none', markeredgecolor='g', markeredgewidth=2)
    axs[1].plot(p2w(ref_pixels_lines)[best_valid_idx],neon_wavelengths[best_idx][best_valid_idx], "o", markersize=10, markerfacecolor='none', markeredgecolor='r', markeredgewidth=2)
    axs[1].set_title("Wavelengths for each peak")
    axs[1].set_xlabel("Wavelength (nm)")
    axs[1].set_ylabel("Wavelength (nm)")

    axs[2].plot(wave,wave-wave,'k')
    # axs[2].plot(p2w(ref_pixels_lines),p2w(ref_pixels_lines)-neon_wavelengths[best_idx], "o", markersize=10, markerfacecolor='none', markeredgecolor='g', markeredgewidth=2)
    axs[2].plot(p2w(ref_pixels_lines)[best_valid_idx],p2w(ref_pixels_lines)[best_valid_idx]-neon_wavelengths[best_idx][best_valid_idx], "o", markersize=10, markerfacecolor='none', markeredgecolor='r', markeredgewidth=2)
    axs[2].set_title("Wavelengths error for each peak")
    axs[2].set_xlabel("Wavelength (nm)")
    axs[2].set_ylabel("Wavelength error (nm)")
    plt.tight_layout()

    return fig

def plot_flat_fit_quality(poly_coeffs, fit_quality, desc = "Flat Fit Quality"):
    fig, axes = plt.subplots(2, 3, num=desc, figsize=(18, 10), constrained_layout=True, sharex=True, sharey=True, clear=True)
    
    # Top row: Fit coefficients
    # Plot coefficient a (slope)
    im1 = axes[0, 0].imshow(poly_coeffs[:, :, 0], aspect='auto', cmap='viridis', interpolation='none', rasterized=True)
    axes[0, 0].set_title('Coefficient a (slope)')
    axes[0, 0].set_xlabel('Pixel')
    axes[0, 0].set_ylabel('Output')
    plt.colorbar(im1, ax=axes[0, 0])

    # Plot coefficient b (intercept)
    im2 = axes[0, 1].imshow(poly_coeffs[:, :, 1], aspect='auto', cmap='viridis', interpolation='none', rasterized=True)
    axes[0, 1].set_title('Coefficient b (intercept)')
    axes[0, 1].set_xlabel('Pixel')
    axes[0, 1].set_ylabel('Output')
    plt.colorbar(im2, ax=axes[0, 1])

    # Combined fit success metric (R^2 weighted by inverse chi^2)
    success_metric = fit_quality[:, :, 1] / (1 + fit_quality[:, :, 0])  # R^2 / (1 + chi^2)
    im3 = axes[0, 2].imshow(success_metric, aspect='auto', cmap='RdYlGn', interpolation='none', rasterized=True)
    axes[0, 2].set_title(r'Fit Success Metric: $R^2/(1+\chi^2)$ (higher=better)')
    axes[0, 2].set_xlabel('Pixel')
    axes[0, 2].set_ylabel('Output')
    plt.colorbar(im3, ax=axes[0, 2])

    # Bottom row: Detailed quality metrics
    # Plot reduced chi-squared (should be 1 for good fit)
    im4 = axes[1, 0].imshow(fit_quality[:, :, 0], aspect='auto', cmap='coolwarm', interpolation='none', 
                        rasterized=True)
    axes[1, 0].set_title(r'Reduced $\chi^2$ (ideal $\approx$ 1)')
    axes[1, 0].set_xlabel('Pixel')
    axes[1, 0].set_ylabel('Output')
    plt.colorbar(im4, ax=axes[1, 0])

    # Plot R-squared (coefficient of determination, 1=perfect fit)
    im5 = axes[1, 1].imshow(fit_quality[:, :, 1], aspect='auto', cmap='viridis', interpolation='none', 
                        rasterized=True)
    axes[1, 1].set_title(r'$R^2$ Coefficient (higher=better)')
    axes[1, 1].set_xlabel('Pixel')
    axes[1, 1].set_ylabel('Output')
    plt.colorbar(im5, ax=axes[1, 1])

    # Plot weighted RMSE (normalized, lower=better)
    im6 = axes[1, 2].imshow(fit_quality[:, :, 2], aspect='auto', cmap='plasma_r', interpolation='none', 
                        rasterized=True, vmin=0, vmax=np.percentile(fit_quality[:, :, 2], 95))
    axes[1, 2].set_title('Weighted RMSE (lower=better)')
    axes[1, 2].set_xlabel('Pixel')
    axes[1, 2].set_ylabel('Output')
    plt.colorbar(im6, ax=axes[1, 2])

    plt.suptitle('Flatfield Fit Quality Analysis', fontsize=16)
    
    return fig

def make_image_grid(ra_dec, Npixels):
    """
    Generate a grid for image reconstruction based on the coupling map positions.
    This function creates a 2D grid for interpolation, which can be used to reconstruct
    an image from the coupling map data. The grid is defined based on the x and y 
    positions of the coupling map, with optional modifications using `xmod` and `ymod`.
    Parameters:
        ra_dec (numpy.ndarray): Array of shape (..., 2) containing x and y positions.
        Npixels (int): The number of pixels along each dimension of the grid.
    Returns:
        tuple: A tuple containing two 2D arrays (`grid_x`, `grid_y`) representing the 
                x and y coordinates of the grid.
    """


    xmin, xmax   = np.min(ra_dec[..., 0]), np.max(ra_dec[..., 0])
    ymin, ymax   = np.min(ra_dec[..., 1]), np.max(ra_dec[..., 1])
    grid_x, grid_y = np.mgrid[xmin:xmax:Npixels*1j, ymin:ymax:Npixels*1j]

    return grid_x, grid_y


# Define a 2D Gaussian function
def gaussian_2d(xy, amplitude, xo, yo, sigma, offset):
    x, y = xy
    xo = float(xo)
    yo = float(yo)
    w = 1/(sigma**2)
    g = offset + amplitude * np.exp(-(w*((x-xo)**2) + w*((y-yo)**2)))
    return g.ravel()
    
def fit_gaussian_on_flux(fluxes, xmod, ymod):
    """
    Fit a 2D Gaussian to the flux data.
    """
    # Interpolate the fluxes onto a grid
    # Create a grid of points for interpolation
    # Use the mean fluxes for the grid
    
    # Prepare data for fitting
    z_is_nan = np.isnan(fluxes)
    z = fluxes[~z_is_nan]
    x = xmod[~z_is_nan]
    y = ymod[~z_is_nan]
    amplitude_0=np.nanmax(fluxes)-np.nanmin(fluxes)
    x_0= x[np.nanargmax(fluxes)]
    y_0= y[np.nanargmax(fluxes)]
    sigma_0 = (x.max()-x.min())/4
    offset_0=np.nanmin(fluxes)

    # Initial guess for the parameters
    initial_guess = (amplitude_0,x_0,y_0,sigma_0,offset_0)

    # Fit the Gaussian
    try:
        popt, _ = curve_fit(gaussian_2d, (x, y), z, p0=initial_guess)
    except RuntimeError:
        print("Error: Gaussian fit failed")
        popt = np.array(initial_guess)
        popt[1] = 0.0
        popt[2] = 0.0

    return popt

def save_pdf_in_file(output_filename):
    # Save all open figures to a PDF
    from matplotlib.backends.backend_pdf import PdfPages
    pdf_filename = os.path.splitext(output_filename)[0] + ".pdf"
    with PdfPages(pdf_filename) as pdf:
        for i in plt.get_fignums():
            fig = plt.figure(i)
            pdf.savefig(fig,dpi=300)
    print(f"All figures saved to {pdf_filename}")


def save_all_as_PDF(output_dir = "/home/jsarrazin/Bureau/test zone/coupling_maps/"):
    # Save all plots to a PDF
    now = datetime.now()
    date_time_str = now.strftime("%Y_%m_%d_%H_%M_%S")
    pdf_filename = os.path.join(output_dir, f"plots_summary_{date_time_str}.pdf")
    with PdfPages(pdf_filename) as pdf:
        for i in plt.get_fignums():
            fig = plt.figure(i)
            pdf.savefig(fig)

    print(f"All plots saved to {pdf_filename}")
    return 1

def plot_flux_map(fluxes, xmod, ymod, desc = "Flux Map"):

    Ndit = len(fluxes)
    Nmod = len(xmod)
    Ncube = Ndit//Nmod

    if (Ncube*Nmod)!=Ndit:
        print(f"WARNING, CUBE not multiple of modulation pattern (Ncube={Ncube}, Nmod={Nmod}, Ndit={Ndit})")
        print("filling with zeros")
        Ncube += 1

    size_new = (Ncube,Nmod)
    size_old = Ndit

    flux_padded=np.zeros(np.prod(size_new))
    flux_padded[np.prod(size_new)-size_old:]=fluxes
    flux_padded=flux_padded.reshape(size_new)

    if plt.fignum_exists(desc):
        plt.close(desc)
    fig,axs = plt.subplots(Ncube, 1, num=desc, figsize=(8, 1+5*Ncube), clear=True,squeeze=False)
    
    for c in range(Ncube):
        fluxes = flux_padded[c]
        popt = fit_gaussian_on_flux(fluxes, xmod, ymod)
        x_fit=popt[1]
        y_fit=popt[2]

        xmin, xmax   = np.min(xmod), np.max(xmod)
        ymin, ymax   = np.min(ymod), np.max(ymod)

        # Define the grid for interpolation
        grid_x, grid_y = np.mgrid[xmin:xmax:500j, ymin:ymax:500j]  # 500x500 grid

        # Interpolate the fluxes onto the grid
        flux_map = griddata((xmod, ymod), fluxes, (grid_x, grid_y), method='nearest')


        # Generate the fitted Gaussian for plotting
        fitted_gaussian = gaussian_2d((grid_x, grid_y), *popt).reshape(grid_x.shape)

        # Plot the contours of the fitted Gaussian on top of the image
        # Plot the interpolated 2D image

        axs[c,0].imshow(flux_map.T, extent=(xmin, xmax, ymin, ymax), origin="lower", aspect='auto')
        axs[c,0].scatter(xmod, ymod, c='red', s=1, label='Data Points')
        fig.colorbar(axs[c,0].images[-1], ax=axs[c,0], label="Mean Flux per pixel")
        axs[c,0].set_xlabel("X")
        axs[c,0].set_ylabel("Y")
        axs[c,0].set_title("(Xmod,Ymod) maximum position: (%.3f,%.3f)"%(x_fit,y_fit))
        axs[c,0].contour(grid_x, grid_y, fitted_gaussian, levels=10, colors='red', linewidths=0.8)
        axs[c,0].set_aspect('equal')
    
    return fig


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
    ax[2].plot(distances[::len(cor_matrix)], correlations[::len(cor_matrix)],'.', alpha=0.3,  label='Pairs')
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

def plot_R_amplitude(R_triangles,name="triangles"):

    R_amplitude = np.linalg.norm(R_triangles,axis=2)
    Nxy =  R_amplitude.shape[2] 

    label = ["1","x","y","xy","x2","y2"]

    fig,axs = plt.subplots(Nxy,figsize=(12,6),num="R matrix amplitude "+name,clear=True)
    fig.suptitle(fig.get_label())
    for i in range(Nxy):
        ax=axs[i]
        im=ax.imshow(R_amplitude[:,:,i],aspect='auto',origin='lower',cmap='viridis',interpolation='none',rasterized=True)
        if i == Nxy//2:
            ax.set_ylabel(name+ " number")
        ax.set_xlabel("Wavelength")
        fig.colorbar(im,ax=ax,label=label[i])


def plot_detector_field(flat, title="Flat Field"):
    fig, ax_flat = plt.subplots(num=title, figsize=(12, 6), clear=True)
    im_flat = ax_flat.imshow(flat, aspect='auto', origin='lower', cmap='viridis', interpolation='none', rasterized=True)
    ax_flat.set_title(title)
    ax_flat.set_xlabel("Wavelength Index (pixels)")
    ax_flat.set_ylabel("Output Index (pixels)")
    fig.colorbar(im_flat, ax=ax_flat, label="Field Value")
    fig.tight_layout()


def make_image_using_grid(ra_dec, fluxes, Npixels=150, desc = "Making image using grid", 
                          sumwave = True):

    Nimages = fluxes.shape[2]
    Nwave   = fluxes.shape[0]
    grid_x, grid_y = make_image_grid(ra_dec, Npixels)

    if sumwave is True : 
        flux_maps = []
        for i in tqdm(range(Nimages), desc=desc):
            # Interpolate the fluxes onto the grid
            flux_map = griddata((ra_dec[i,:,0],ra_dec[i,:,1]), fluxes[:,:,i].sum(axis=0), 
                                (grid_x, grid_y), method='cubic')
            flux_maps += [flux_map]
        flux_maps = np.array(flux_maps)
    else:
        flux_maps = np.zeros((Nimages, Nwave, Npixels, Npixels))

        for i in tqdm(range(Nimages), desc=desc):
            # Interpolate the fluxes onto the grid
            for l in range(Nwave):
                flux_map = griddata((ra_dec[i,:,0],ra_dec[i,:,1]), fluxes[l,:,i], 
                                    (grid_x, grid_y), method='cubic')
                flux_maps[i, l] = flux_map

    return flux_maps

def plot_star_fit_position(cmap_style, Xmod, Ymod, 
                           Xpos, Ypos, Xcen, Ycen, Xdiff, Ydiff):
    
    fig, axs = plt.subplots(2,1, num="XY position -- using "+cmap_style, 
                            clear=True, figsize=(7,12), squeeze=False)
    axs[0,0].plot(Xcen,Ycen,'.',label='Center of '+cmap_style)
    axs[0,0].set_ylim(axs[0,0].get_ylim()[0], axs[0,0].get_ylim()[1])
    axs[0,0].set_xlim(axs[0,0].get_xlim()[0], axs[0,0].get_xlim()[1])
    axs[0,0].plot((Xpos),(Ypos),'.-',label='Detected position')
    axs[0,0].plot((Xcen,(Xpos)),(Ycen,(Ypos)),'-k',alpha=0.3,linewidth=0.5)
    
    # axs[0,0].plot((d.xmod[0],(Xpos)),(d.ymod[0],(Ypos)),'-c',alpha=0.3,linewidth=0.5)
    # axs[0,0].plot((d.xmod[0],(Xcen)),(d.ymod[0],(Ycen)),'-m',alpha=0.3,linewidth=0.5)
    axs[0,0].plot(Xmod,Ymod,'om',alpha=0.3,linewidth=0.5)
    
    # axs[0].set_title(basenames[i][8:])
    axs[0,0].set_xlabel("X [mas]")
    axs[0,0].set_ylabel("Y [mas]")
    axs[0,0].legend()
    axs[0,0].set_aspect('equal')

    x_median = np.median(Xdiff)
    y_median = np.median(Ydiff)
    x_1sigma = np.percentile(Xdiff, [16, 84])
    y_1sigma = np.percentile(Ydiff, [16, 84])
    range_max = np.max((np.abs(x_1sigma), np.abs(y_1sigma))) * 2 +10
    axs[1,0].hist(Xdiff, bins=51, alpha=0.5, color='b', label='Xdiff', range=(-range_max, range_max))
    axs[1,0].hist(Ydiff, bins=51, alpha=0.5, color='r', label='Ydiff', range=(-range_max, range_max))
    x_median = np.median(Xdiff)
    y_median = np.median(Ydiff)
    x_1sigma = np.percentile(Xdiff, [16, 84])
    y_1sigma = np.percentile(Ydiff, [16, 84])
    axs[1,0].axvline(x_median, color='b', linestyle='--', label=f'X median: {x_median:.2f}')
    axs[1,0].axvline(y_median, color='r', linestyle='--', label=f'Y median: {y_median:.2f}')
   # axs[1,0].axvspan(x_1sigma[0], x_1sigma[1], color='b', alpha=0.2, label=f'X 1σ: [{x_1sigma[0]:.2f}, {x_1sigma[1]:.2f}]')
    # axs[1,0].axvspan(y_1sigma[0], y_1sigma[1], color='r', alpha=0.2, label=f'Y 1σ: [{y_1sigma[0]:.2f}, {y_1sigma[1]:.2f}]')
    axs[1,0].set_xlabel('Difference [mas]')
    axs[1,0].set_ylabel('Count')
    axs[1,0].legend()
    
    plt.tight_layout()     

# %%
