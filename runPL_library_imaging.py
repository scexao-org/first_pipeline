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
from scipy.spatial import Delaunay
from scipy.interpolate import griddata
from scipy.optimize import curve_fit

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

def generate_plots(datacube, xmod, ymod, masque_positions, flux_2_data, singular_values, Nsingular, chi2_delta, flux_goodData, chi2_goodData, flux_threshold, chi2_threshold, output_dir):

    fluxes = datacube.mean(axis=(0,1,2))
    popt = fit_gaussian_on_flux(fluxes, xmod, ymod)
    x_fit=popt[1]
    y_fit=popt[2]

    xmin, xmax   = np.min(xmod), np.max(xmod)
    ymin, ymax   = np.min(ymod), np.max(ymod)

    # Define the grid for interpolation
    grid_x, grid_y = np.mgrid[xmin:xmax:500j, ymin:ymax:500j]  # 500x500 grid

    # Interpolate the fluxes onto the grid
    flux_map = griddata((xmod, ymod), fluxes, (grid_x, grid_y), method='cubic')

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
    popt, _ = curve_fit(gaussian_2d, (x, y), z, p0=initial_guess)
    x_fit=popt[1]
    y_fit=popt[2]

    # Generate the fitted Gaussian for plotting
    fitted_gaussian = gaussian_2d((grid_x, grid_y), *popt).reshape(grid_x.shape)

    # Plot the contours of the fitted Gaussian on top of the image
    # Plot the interpolated 2D image
    import matplotlib.pyplot as plt
    plt.ion()

    plt.figure("Interpolated Flux",clear=True)
    plt.imshow(flux_map.T, extent=(xmin, xmax, ymin, ymax), origin="lower", aspect='auto')
    plt.scatter(xmod, ymod, c='red', s=1, label='Data Points')
    plt.scatter(xmod[masque_positions], ymod[masque_positions], c='w', s=3, label='Data Points')
    plt.colorbar(label="Flux")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("(Xmod,Ymod) maximum position: (%.3f,%.3f)"%(x_fit,y_fit))
    plt.contour(grid_x, grid_y, fitted_gaussian, levels=10, colors='red', linewidths=0.8)


    # Singular values plot

    Ncube = flux_goodData.shape[0]
    Nmod = flux_goodData.shape[1]
    # cmap_size = cross_correlated_projected_data.shape[-1]

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
    fig, axs = plt.subplots(7, 1, num="reduced chi23", clear=True, figsize=(12, 16))

    chi2_delta = chi2_delta.reshape((Ncube, -1))
    flux = datacube.mean(axis=(0,1))

    axs[0].imshow(flux, aspect="auto", interpolation='none')
    axs[0].set_title('Flux')

    axs[1].imshow(chi2_delta, aspect="auto", interpolation='none')
    axs[1].set_title('Normalised Chi2')

    axs[2].imshow(flux_goodData.reshape((Ncube, -1)), aspect="auto", interpolation='none')
    axs[2].set_title('Masque on flux')
    axs[2].set_rasterized(True)

    axs[3].imshow(chi2_goodData.reshape((Ncube, -1)), aspect="auto", interpolation='none')
    axs[3].set_ylabel('N cube')
    axs[3].set_title('Masque on chi2')

    axs[4].plot(flux.T)
    axs[4].plot(np.ones(Nmod) * flux_threshold, 'r')
    axs[4].set_title('Flux Plot')

    axs[5].plot(chi2_delta.T)
    axs[5].plot(np.ones(Nmod) * chi2_threshold, 'r')
    axs[5].set_title('Chi2 Delta Plot')
    axs[5].set_xlim((0, Nmod))

    for i in range(len(axs)-3):
        axs[i].set_rasterized(True)
    for i in range(len(axs)-1):
        axs[i].set_ylabel('N cube')

    max_chi2 = np.nanmax(chi2_delta.ravel())
    axs[-1].hist(chi2_delta.ravel(), bins=100, range=(0, max_chi2),alpha=0.8, label='All data')
    axs[-1].hist(chi2_delta[flux_goodData], bins=100, range=(0, max_chi2), label='flux_goodData')
    axs[-1].hist(chi2_delta[chi2_goodData], bins=100, range=(0, max_chi2), label='chi2_goodData')
    axs[-1].legend()
    axs[-1].set_title('Chi2 Delta Histogram')


    plt.tight_layout()

    # Covariance and correlation matrix plot
    Nwave = flux_2_data.shape[0]
    Noutput = flux_2_data.shape[1]
    Nmodel = flux_2_data.shape[2]

    cov_matrix = np.cov(flux_2_data.reshape((Nwave*Noutput,Nmodel)).T)
    cor_matrix = np.corrcoef(flux_2_data.reshape((Nwave*Noutput,Nmodel)).T)

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
    pdf_filename = os.path.join(output_dir, f"plots_summary_{date_time_str}_cmap.pdf")
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
        self.Ndit = data.shape[0]
        self.Noutput = data.shape[1]
        self.Nwave = data.shape[2]

    def add_modulation(self, xmod, ymod):
        self.xmod = xmod
        self.ymod = ymod
        self.Nmod = len(xmod)
        self.Ncube = self.Ndit//self.Nmod
        if (self.Ncube*self.Nmod)!=self.Ndit:
            print("WARNING, CUBE not multiple of modulation pattern")
            print("filling with zeros")
            self.Ncube += 1

        size_new = (self.Ncube,self.Nmod,self.Noutput,self.Nwave)
        size_old = np.prod((self.Ndit,self.Noutput,self.Nwave))

        data_padded=np.zeros(np.prod(size_new))
        data_padded[:size_old]=self.data.ravel()[:size_old]
        self.data=data_padded.reshape(size_new)

        variance_padded=np.zeros(np.prod(size_new))
        variance_padded[:size_old]=self.variance.ravel()[:size_old]
        self.variance=variance_padded.reshape(size_new)

    def get_triangle(self):
    
        xmod=self.xmod
        ymod=self.ymod

        # Combine xmod and ymod into a 2D array of points
        points = np.array([xmod, ymod]).T

        # Perform Delaunay triangulation
        delaunay_triangles = Delaunay(points)

        # Extract the triangles
        triangles = delaunay_triangles.simplices
        # Filter triangles to keep only equatorial ones
        equatorial_triangles = []
        for triangle in triangles:
            # Get the y-coordinates of the vertices
            x_coords = points[triangle, 0]
            y_coords = points[triangle, 1]
            l1=np.sqrt((x_coords[0]-x_coords[1])**2+(y_coords[0]-y_coords[1])**2)
            l2=np.sqrt((x_coords[1]-x_coords[2])**2+(y_coords[1]-y_coords[2])**2)
            l3=np.sqrt((x_coords[2]-x_coords[0])**2+(y_coords[2]-y_coords[0])**2)
            # Check if the triangle is equilateral within a tolerance
            tolerance = 1e-3  # Adjust tolerance as needed
            if abs(l1 - l2) < tolerance and abs(l2 - l3) < tolerance and abs(l1 - l3) < tolerance:
                equatorial_triangles.append(triangle)

        equatorial_triangles = np.array(equatorial_triangles)
        print(f"Computed {len(triangles)} triangles for the given positions.")
        print(f"Computed {len(equatorial_triangles)} equatorial triangles.")

        return equatorial_triangles

def extract_datacube(files_with_dark,Nsmooth = 1,Nbin = 1):
    """
    Extracts and processes data cubes from the input files.
    Subtracts dark files, applies wavelength smoothing, and calculates variance.
    Returns the processed data cubes, variance cubes, and a header to save.
    If Nsmooth > 1, the data is smoothed along its wavelength dimension by Nsmooth values.
    If Nbin > 1, the data is binned along its wavelength dimension by Nbin values.
    """

    datalist=[]

    for data_file,dark_file  in files_with_dark.items():

        # reading header data
        header=fits.getheader(data_file)
        # important to cast the data in double!
        data=np.double(fits.getdata(data_file))
        # reading modulation data
        xmod=np.double(fits.getdata(data_file,'MODULATION').field('xmod'))
        ymod=np.double(fits.getdata(data_file,'MODULATION').field('ymod'))

        if dark_file is not None:
            data_dark=fits.getdata(dark_file)
            if len(data_dark)==1:
                data_dark=data_dark[0]
                data_dark_std=data_dark[0]*0+12
            else:
                data_dark=data_dark.mean(axis=0)
                data_dark_std=data_dark.std(axis=0)
        else:
            # using default values if we do not know the dark
            data_dark=header["DETBIAS"]*(1+2*header["PIX_WIDE"])
            data_dark_std=20
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
        datalist[-1].add_modulation(xmod,ymod)

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
    popt, _ = curve_fit(gaussian_2d, (x, y), z, p0=initial_guess)

    return popt