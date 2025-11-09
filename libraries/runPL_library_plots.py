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
    z = fluxes
    x = xmod
    y = ymod
    amplitude_0=np.max(fluxes)-np.min(fluxes)
    x_0= x[fluxes.argmax()]
    y_0= y[fluxes.argmax()]
    sigma_0 = (x.max()-x.min())/4
    offset_0=np.min(fluxes)

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
            pdf.savefig(fig)
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

def plot_flux_map(fluxes, xmod, ymod):

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
    flux_padded[:size_old]=fluxes
    flux_padded=flux_padded.reshape(size_new)

    plt.close("Coupling Map")
    fig,axs = plt.subplots(Ncube, num="Coupling Map", figsize=(8, 6*Ncube), clear=True,squeeze=False)
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


def make_image_using_grid(ra_dec, fluxes, Npixels=150, desc = None):

    Nimages = fluxes.shape[2]
    grid_x, grid_y = make_image_grid(ra_dec, Npixels)

    flux_maps = []
    if desc is None:
        for i in range(Nimages):
                # Interpolate the fluxes onto the grid
                flux_map = griddata((ra_dec[i,:,0],ra_dec[i,:,1]), fluxes[:,:,i].sum(axis=0), (grid_x, grid_y), method='cubic')
                flux_maps += [flux_map]
    else:
        for i in tqdm(range(Nimages), desc=desc):
                # Interpolate the fluxes onto the grid
                flux_map = griddata((ra_dec[i,:,0],ra_dec[i,:,1]), fluxes[:,:,i].sum(axis=0), (grid_x, grid_y), method='cubic')
                flux_maps += [flux_map]
    flux_maps = np.array(flux_maps)

    return flux_maps
        