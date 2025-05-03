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
from tqdm import tqdm
from astropy.io import fits
from scipy.ndimage import uniform_filter1d

import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime

def create_movie_cross(datacube):

    Nwave,Noutput,Ncube,Npos=datacube.shape
    all_flux=datacube.transpose((2,0,1,3)).reshape((Ncube,Nwave*Noutput,Npos))

    all_Minv=[]
    all_Ut=[]
    Nsingular=76
    # Nsingular=19

    for flux in tqdm(all_flux):
        U,s,Vh=linalg.svd(flux,full_matrices=False)
        s_inv=1/s[:Nsingular]
        Ut=U[:,:Nsingular].T
        Minv=np.dot(Vh[:Nsingular].T*s_inv,Ut)
        all_Minv+=[Minv]
        all_Ut+=[Ut]


    all_Minv=np.array(all_Minv)
    all_Ut=np.array(all_Ut)

    mp=[]
    images=[]
    fit_flux=[]
    for Ut in all_Ut:
        mp+=[np.matmul(Ut,all_flux)]
    for Minv in all_Minv:
        images+=[np.matmul(Minv,all_flux)]
        fit_flux+=[np.matmul(linalg.pinv(Minv),images[-1])]
    mp=np.array(mp)
    images=np.array(images)
    fit_flux=np.array(fit_flux)

    residuals=fit_flux-all_flux
    residuals_std=np.std(residuals,axis=2)

    Npts=np.sqrt(images.shape[-1]).astype(int)
    Ncmap=images.shape[0]
    images=images.reshape((Ncmap*Ncmap,Npts,Npts,-1))
    images/=images.max(axis=(1,2,3))[:,None,None,None]

    print("Making movie ... ")

    def make_image(images,i):
            return images[:,:,i]

    Image=make_image(images[0],0)

    fig, axs = plt.subplots(Ncmap, Ncmap, num=15, figsize=(9.25, 9.25), clear=True)
    plt.subplots_adjust(wspace=0.025, hspace=0.025, top=0.99, bottom=0.01, left=0.01, right=0.99)

    ims=[ax.imshow(Image,vmax=0.2,vmin=-0.1) for ax in axs.ravel()]
    for ax in axs.ravel():
            ax.set_axis_off()


    def init():
        for im in ims:
                im.set_array(make_image(images[0],0))
        return ims

    def animate(i):
        for k,im in enumerate(ims):
            im.set_array(make_image(images[k],i))
        return ims

    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=Npts*Npts, interval=20, blit=True)

    FFwriter = animation.FFMpegWriter(fps=30, extra_args=['-vcodec', 'libx264'])
    anim.save('firtpl_CMAP_MOVIE.mp4', writer=FFwriter)



def reconstruct_images(projected_data,projected_data_2_image,masque,dither_x,dither_y, sumPos = True):


    Npos=len(dither_x)
    Ncube = np.prod(projected_data.shape[1:]) // Npos

    image = projected_data_2_image @ projected_data.reshape((len(projected_data),-1))
    image_2d = np.zeros((Ncube*Npos,*masque.shape))
    image_2d[:,masque] = image.T

    image_2d_bigger= resize_and_shift(image_2d, dither_x, dither_y, sumPos)

    return image_2d_bigger

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

def generate_plots(singular_values, chi2_delta, flux_goodData, chi2_goodData, chi2_threshold, cross_correlated_projected_data, shifted_projected_data, fluxtiptilt_2_data, output_dir):
    # Singular values plot

    Nsingular = cross_correlated_projected_data.shape[0]
    Ncube = cross_correlated_projected_data.shape[1]
    cmap_size = cross_correlated_projected_data.shape[-1]

    energy_estimation = (singular_values)**2 / np.sum(singular_values**2)
    reverse_cumulative_energy = np.cumsum(energy_estimation[::-1])[::-1]

    plt.figure("Singular values", clear=True)
    plt.plot(1+np.arange(len(energy_estimation)), energy_estimation**.5, marker='o', label='All Singular Values')
    plt.plot(1+np.arange(Nsingular), energy_estimation[:Nsingular]**.5, marker='o', label='Selected Singular Values')
    plt.plot(1 + np.arange(len(reverse_cumulative_energy)), reverse_cumulative_energy**.5, marker='D', label='Reverse Cumulative Energy', alpha=0.5)
    plt.plot(1+np.arange(Nsingular), reverse_cumulative_energy[:Nsingular]**.5, marker='D', alpha=0.5)

    plt.legend()
    plt.xlabel('Singular Vector Index')
    plt.ylabel('Energy Estimation')
    plt.title('Amplitude of Singular Values')
    plt.yscale('log')
    plt.xscale('log')
    plt.grid(True)

    # Chi2 maps plots
    fig, axs = plt.subplots(5, 1, num="reduced chi23", clear=True, figsize=(15, 20))

    chi2_delta = chi2_delta.reshape((Ncube, -1))

    axs[0].imshow(chi2_delta, aspect="auto", interpolation='none')
    axs[0].set_ylabel('Chi2')
    axs[0].set_title('Chi2 Delta')
    axs[0].set_rasterized(True)

    axs[1].imshow(flux_goodData.reshape((Ncube, -1)), aspect="auto", interpolation='none')
    axs[1].set_ylabel('N cube')
    axs[1].set_title('Masque on flux')
    axs[1].set_rasterized(True)

    axs[2].imshow(chi2_goodData.reshape((Ncube, -1)), aspect="auto", interpolation='none')
    axs[2].set_ylabel('N cube')
    axs[2].set_title('Masque on chi2')
    axs[2].set_rasterized(True)

    axs[3].plot(chi2_delta.T)
    axs[3].plot(np.ones(cmap_size * cmap_size) * chi2_threshold, 'r')
    axs[3].set_ylabel('N cube')
    axs[3].set_title('Chi2 Delta Plot')
    axs[3].set_xlim((0, cmap_size * cmap_size))

    axs[4].set_axis_off()

    axs_last = [fig.add_subplot(5, 3, 13), fig.add_subplot(5, 3, 14), fig.add_subplot(5, 3, 15)]

    max_chi2 = np.nanmax(chi2_delta.ravel())
    axs_last[0].hist(chi2_delta.ravel(), bins=30, range=(0, max_chi2),alpha=0.2, label='All data')
    axs_last[0].hist(chi2_delta[flux_goodData], bins=30, range=(0, max_chi2), label='flux_goodData')
    axs_last[0].hist(chi2_delta[chi2_goodData], bins=30, range=(0, max_chi2), label='chi2_goodData')
    axs_last[0].legend()
    axs_last[0].set_title('Chi2 Delta Histogram')

    axs_last[1].imshow(np.nansum(chi2_delta.reshape((Ncube, cmap_size, cmap_size)), axis=0), interpolation='none', vmin=0, vmax=max_chi2)
    axs_last[1].set_title('Chi2 Delta Sum')
    axs_last[1].set_rasterized(True)

    axs_last[2].imshow(chi2_goodData.reshape((Ncube, cmap_size, cmap_size)).sum(axis=0), interpolation='none')
    axs_last[2].set_title('Chi2 Good Data Sum')
    axs_last[2].set_rasterized(True)

    plt.tight_layout()

    # Cross-correlation matrix plot
    fig, axs = plt.subplots(Ncube, Ncube, num='cross_correlation_singular_vector', clear=True, figsize=(5, 5), squeeze=False)
    plt.subplots_adjust(wspace=0, hspace=0)
    fig.suptitle('Cross-correlation of singular vectors')

    for j in np.arange(Ncube):
        for k in range(Ncube):
            if j == 0:
                axs[j, k].set_title(f"{k}")
            if k == 0:
                axs[j, k].set_ylabel(f"{j}")
            axs[j, k].imshow(cross_correlated_projected_data[:, j, k].mean(axis=0), interpolation=None)
            if j != Ncube - 1:
                axs[j, k].set_xticks([])
            axs[j, k].set_yticks([])

    # Shifted singular vectors plot
    per_row = 19
    n_plots = int(np.ceil(Nsingular / per_row))
    for p in range(n_plots):
        fig, axs = plt.subplots(Ncube, per_row, num='Position map of singular vector %i to %i' % (1 + p * per_row, (1 + p) * per_row), figsize=(14 + 1, 1 + 14 * Ncube / per_row), clear=True, squeeze=False)
        plt.subplots_adjust(wspace=0, hspace=0, top=0.99, bottom=0.01, left=0.01, right=0.99)

        for i in range(p * per_row, (1 + p) * per_row):
            position_map = shifted_projected_data[i]
            for j in range(Ncube):
                vmax = np.nanmax(np.abs(position_map[j]))
                vmin = -vmax
                ax = axs[j, i % per_row]
                im = ax.imshow(position_map[j], origin='lower', cmap='viridis', vmax=vmax, vmin=vmin)
                ax.set_xticks([])
                ax.set_yticks([])
            ax.text(0.5, 0.05, f'#{i + 1}', transform=ax.transAxes, ha='center')

    # Covariance and correlation matrix plot
    Nmodel = fluxtiptilt_2_data.shape[0]
    Nwave = fluxtiptilt_2_data.shape[1]
    Noutput = fluxtiptilt_2_data.shape[2]

    F2PM = fluxtiptilt_2_data[:, :, :, 0]
    cov_matrix = np.cov(F2PM.reshape((Nmodel,Nwave*Noutput)))
    cor_matrix = np.corrcoef(F2PM.reshape((Nmodel,Nwave*Noutput)))

    fig, ax = plt.subplots(1, 2, num='Covariance and Correlation Matrix', figsize=(12, 6), clear=True)
    cax0 = ax[0].matshow(cov_matrix, cmap='viridis')
    fig.colorbar(cax0, ax=ax[0])
    cax1 = ax[1].matshow(cor_matrix, cmap='viridis')
    fig.colorbar(cax1, ax=ax[1])
    ax[0].set_title('Covariance Matrix of Singular Vector Models')
    ax[1].set_title('Correlation Matrix of Singular Vector Models')
    fig.tight_layout()

    # Save all plots to a PDF
    now = datetime.now()
    date_time_str = now.strftime("%Y_%m_%d_%H_%M_%S")
    pdf_filename = os.path.join(output_dir, f"plots_summary_{date_time_str}_cmap{cmap_size}.pdf")
    with PdfPages(pdf_filename) as pdf:
        for i in plt.get_fignums():
            fig = plt.figure(i)
            pdf.savefig(fig)

    print(f"All plots saved to {pdf_filename}")

class DataCube:
    """
    A class to represent a data cube.
    Attributes:
        data (numpy.ndarray): The data cube.
        variance (numpy.ndarray): The variance of the data cube.
        header (astropy.io.fits.Header): The header information.
    """

    def __init__(self, data, variance, filename, header):
        self.data = data
        self.variance = variance
        self.dirname = os.path.dirname(filename)
        self.filename = filename
        self.header = header
        self.Npos = data.shape[0]
        self.Noutput = data.shape[1]
        self.Nwave = data.shape[2]


def extract_datacube(closest_dark_files,Nsmooth = 1,Nbin = 1):
    """
    Extracts and processes data cubes from the input files.
    Subtracts dark files, applies wavelength smoothing, and calculates variance.
    Returns the processed data cubes, variance cubes, and a header to save.
    If Nsmooth > 1, the data is smoothed along its wavelength dimension by Nsmooth values.
    If Nbin > 1, the data is binned along its wavelength dimension by Nbin values.
    """

    datalist=[]

    for data_file,dark_file  in closest_dark_files.items():
        header=fits.getheader(data_file)
        data_dark=fits.getdata(dark_file)
        if len(data_dark)==1:
            data_dark=data_dark[0]
            data_dark_std=data_dark[0]*0+12
        else:
            data_dark=data_dark.mean(axis=0)
            data_dark_std=data_dark.std(axis=0)
        data=np.double(fits.getdata(data_file))
        data-=data_dark
        gain=header['GAIN']
        data_var=data_dark_std**2+gain*np.abs(data)

        Npos=data.shape[0]
        Noutput=data.shape[1]
        Nwave=data.shape[2]

        if Nsmooth > 1:
            # Smooth data along its third dimension by Nsmooth values using uniform_filter1d
            data = uniform_filter1d(data, size=Nsmooth, axis=2, mode='nearest')
            data_var = uniform_filter1d(data_var, size=Nsmooth, axis=2, mode='nearest')

        if Nbin > 1:
            data=data[:,:,:(Nwave//Nbin)*Nbin]
            data_var=data_var[:,:,:(Nwave//Nbin)*Nbin]

            data=data.reshape((Npos,Noutput,Nwave//Nbin,Nbin)).sum(axis=-1)
            data_var=data_var.reshape((Npos,Noutput,Nwave//Nbin,Nbin)).sum(axis=-1)

        datalist += [DataCube(data, data_var, data_file, header)]

    return datalist


def resize_and_shift(flux, masque, dither_x, dither_y):
    """
    Resize and shift a 2D or 3D flux map based on dither offsets and a mask.
    This function processes a flux map by resizing it and applying shifts 
    determined by the dither offsets in the x and y directions. The output 
    is a larger image cube that accommodates the shifts while preserving 
    the original flux data within the specified mask.
    Args:
        flux (numpy.ndarray): A 3D or 4D array representing the flux data. 
            The shape is expected to be (Npos, Nmodel, Ncube[, Nwave]), 
            where Npos is the number of positions, Nmodel is the number of 
            models, Ncube is the cube size, and Nwave is the number of 
            wavelengths (optional).
        masque (numpy.ndarray): A 2D boolean array of of size Npos*Npos,
            indicating which elements of the flux map are valid.
        dither_x (numpy.ndarray): A 1D array of length Npos containing 
            the dither offsets in the x direction.
        dither_y (numpy.ndarray): A 1D array of length Npos containing 
            the dither offsets in the y direction.
    Returns:
        numpy.ndarray: A resized and shifted 4D or 5D array of shape 
            (Npos, cmap_size2, cmap_size2, Ncube[, Nwave]), where cmap_size2 
            is the adjusted size to accommodate the maximum dither offsets.
    Raises:
        ValueError: If the sum of the positive elements of `masque` does not equal Nmodel.
        ValueError: If Npos does not match the length of `dither_x` or `dither_y`.
    Notes:
        - The function calculates the required size of the output array 
          (`cmap_size2`) based on the maximum dither offsets in both 
          x and y directions.
        - The input flux data is placed into the larger output array 
          at positions determined by the dither offsets.
    """

    Npos= flux.shape[0]
    Nmodel = flux.shape[1]
    Ncube = flux.shape[2]
    cmap_size = masque.shape[0]
    if len(flux.shape) == 4:
        Nwave= flux.shape[3]
    else:
        Nwave=1

    if np.sum(masque) != Nmodel:
        raise ValueError(f"The sum of masque ({np.sum(masque)}) is not equal to Nmodel ({Nmodel}).")
    if Npos != len(dither_x):
        raise ValueError(f"Npos ({Npos}) is not equal to the length of the third axis of flux ({flux.shape[3]}).")
    
    delta_x = dither_x.max()-dither_x.min()
    delta_y = dither_y.max()-dither_y.min()
    cmap_size2 = cmap_size + max(delta_x, delta_y)
    if Nwave > 1:
        image_2d_bigger = np.zeros((Npos, cmap_size2, cmap_size2, Ncube, Nwave ))
    else:
        image_2d_bigger = np.zeros((Npos, cmap_size2, cmap_size2, Ncube ))

    for i in tqdm(range(Npos)):
        x2 = -dither_x.min()-dither_x[i]
        y2 = -dither_y.min()-dither_y[i]
        image_2d_bigger[i,x2:x2+cmap_size, y2:y2+cmap_size][masque]  = flux[i]

    return image_2d_bigger