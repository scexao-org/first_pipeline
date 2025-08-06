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

import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime
from tqdm import tqdm
import runPL_library_basic as basic
import matplotlib.pyplot as plt
plt.ion()

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

def plot_couplinng_map(fluxes, xmod, ymod):

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
        popt = basic.fit_gaussian_on_flux(fluxes, xmod, ymod)
        x_fit=popt[1]
        y_fit=popt[2]

        xmin, xmax   = np.min(xmod), np.max(xmod)
        ymin, ymax   = np.min(ymod), np.max(ymod)

        # Define the grid for interpolation
        grid_x, grid_y = np.mgrid[xmin:xmax:500j, ymin:ymax:500j]  # 500x500 grid

        # Interpolate the fluxes onto the grid
        flux_map = griddata((xmod, ymod), fluxes, (grid_x, grid_y), method='nearest')


        # Generate the fitted Gaussian for plotting
        fitted_gaussian = basic.gaussian_2d((grid_x, grid_y), *popt).reshape(grid_x.shape)

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


def generate_plots(filenames, datacube, xmod, ymod, masque_positions, flux_2_data, singular_values, 
                   Nsingular, chi2_delta, flux_goodData, modes_rect, modes_mean, 
                   chi2_goodData, flux_threshold, chi2_threshold, output_filename):


    fluxes = datacube.mean(axis=(0,1,2))
    if len(xmod) >1:
        plot_couplinng_map(fluxes, xmod, ymod)
    plt.scatter(xmod[masque_positions], ymod[masque_positions], c='w', s=3, label='Data Points')
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

    # Plot the list of filenames in the middle of the plot
    plt.gca().text(0.5, 0.5, "\n".join(filenames), fontsize=10, ha='center', va='center', wrap=True, transform=plt.gca().transAxes)


    if Ncube < 6:
        plt.figure(num="modes", figsize=(15,12))
        plt.clf()
        Nsingular0 = int(Nsingular/19)
        gridsize=modes_rect.shape[2]
        fig, ax = plt.subplots(nrows=(Ncube+1)*Nsingular0, ncols=19, num="modes", sharex=True, sharey=True)
        for s0 in range(Nsingular0):
            for s in range(19):
                ax[s0*(Ncube+1),s].set_title(f"m{s0*19+s}")
                for i in range(Ncube) :
                    ax[s0*(Ncube+1)+i,s].imshow(modes_rect[s0*19+s,i])
                    ax[s0*(Ncube+1)+i,s].axis('off')
                    if s==0:
                        ax[s0*(Ncube+1)+i,s].text(-gridsize/2, 2*gridsize/3, f"Cube {s0+i+1}", rotation="vertical")
                ax[s0*(Ncube+1)+i+1,s].imshow(modes_mean[s0*19+s])
                ax[s0*(Ncube+1)+i+1,s].axis('off')
                if s==0:
                    ax[s0*(Ncube+1)+i+1,s].text(-gridsize/2, 2*gridsize/3, f"MEAN", rotation="vertical")


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
    pdf_filename = output_filename[:-5]+".pdf"
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
        self.modID = int(header.get('X_FIRMID', 0))
        self.modScale = int(header.get('X_FIRMSC', 1))
        self.object_name = header.get('OBJECT', 'Unknown')
        self.wollaston = header.get('X_FIRWOL', 'IN')
        self.add_modulation()

    def add_modulation(self):
        """ 
        Adds modulation information to the data cube.
        Reads the 'MODULATION' extension from the FITS file and extracts xmod and ymod arrays.
        If the extension does not exist, initializes xmod and ymod to zeros.
        """

        # Check if 'MODULATION' extension exists in the FITS file
        with fits.open(self.filename) as hdul:
            if 'MODULATION' not in hdul:
                print(f"WARNING: 'MODULATION' extension not found in {self.filename}")
                xmod = np.zeros(1)
                ymod = np.zeros(1)
            elif hdul[0].header.get('X_FIRMID', -1) < 0:
                xmod = np.zeros(1)
                ymod = np.zeros(1)
            else:
                # reading modulation data
                modulation_data = hdul['MODULATION'].data
                xmod = np.double(modulation_data['xmod'])
                ymod = np.double(modulation_data['ymod'])
                # Ensure xmod and ymod are arrays, even if they are scalars
                if np.isscalar(xmod):
                    xmod = np.array([xmod])
                if np.isscalar(ymod):
                    ymod = np.array([ymod])

        self.xmod = xmod
        self.ymod = ymod
        self.Nmod = len(xmod)
        self.Ncube = self.Ndit//self.Nmod
        if (self.Ncube*self.Nmod)!=self.Ndit:
            print(f"WARNING, CUBE not multiple of modulation pattern (Ncube={self.Ncube}, Nmod={self.Nmod}, Ndit={self.Ndit})")
            print("filling with zeros file: ",self.filename)
            self.Ncube += 1

        size_new = (self.Ncube,self.Nmod,self.Noutput,self.Nwave)
        size_old = np.prod((self.Ndit,self.Noutput,self.Nwave))

        if np.prod(size_new) != size_old:
            data_padded=np.zeros(np.prod(size_new))
            data_padded[:size_old]=self.data.ravel()[:size_old]
            self.data=data_padded.reshape(size_new)

            variance_padded=np.zeros(np.prod(size_new))
            variance_padded[:size_old]=self.variance.ravel()[:size_old]
            self.variance=variance_padded.reshape(size_new)
        else:
            self.data = self.data.reshape(size_new)
            self.variance = self.variance.reshape(size_new)

    def normalize_with_flat(self, flat):
        """
        Normalize the data cube by a flat field.
        Args:
            flat (numpy.ndarray): The flat field to normalize the data cube.
        """
        self.data /= flat
        self.variance /= flat**2

    def normalize_with_spectra(self):
        """
        Normalize the extracted data.
        """
        inv_spectra = 1/self.data.mean(axis=(0, 1, 2))

        self.data *= inv_spectra
        self.variance *= inv_spectra**2  # Update variance based on the normalized spectra
        
    def smooth(self, Nsmooth):
        """
        Smooth the data cube.
        """

        self.data = uniform_filter1d(self.data, size=Nsmooth, axis=-1, mode='nearest')
        self.variance = uniform_filter1d(self.variance, size=Nsmooth, axis=-1, mode='nearest')

    def bin(self, Nbin):
        """
        Bin the data cube.
        """
        Nwave = self.data.shape[3]

        self.data = self.data[:, :, :, :(Nwave // Nbin) * Nbin]
        self.variance = self.variance[:, :, :, :(Nwave // Nbin) * Nbin]

        self.data = self.data.reshape((self.Ncube, self.Nmod, self.Noutput, Nwave // Nbin, Nbin)).sum(axis=-1)
        self.variance = self.variance.reshape((self.Ncube, self.Nmod, self.Noutput, Nwave // Nbin, Nbin)).sum(axis=-1)
        
        self.Nwave = self.data.shape[3]

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
        good_triangles = []
        for triangle in triangles:
            # Get the y-coordinates of the vertices
            x_coords = points[triangle, 0]
            y_coords = points[triangle, 1]
            l1=np.sqrt((x_coords[0]-x_coords[1])**2+(y_coords[0]-y_coords[1])**2)
            l2=np.sqrt((x_coords[1]-x_coords[2])**2+(y_coords[1]-y_coords[2])**2)
            l3=np.sqrt((x_coords[2]-x_coords[0])**2+(y_coords[2]-y_coords[0])**2)
            # Check if the triangle is equilateral within a tolerance
            lenghts_triangle = np.array([l1, l2, l3])
            l_max = np.max(lenghts_triangle)
            l_min = np.min(lenghts_triangle)

            # good only if l_max/l_min < (1+1.5**2)**.5
            # to avoid edge triangles
            if l_max/l_min < 1.8:
                good_triangles.append(triangle)

        good_triangles = np.array(good_triangles)
        print(f"Computed {len(triangles)} triangles for the given positions.")
        print(f"Computed {len(good_triangles)} good triangles.")

        return good_triangles

def extract_datacube(files_with_dark, Nsmooth = 1, Nbin = 1, flat = None, normalize = False):
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
            data_dark_std=12*np.sqrt(1+2*header["PIX_WIDE"])

        data-=data_dark
        gain=header['GAIN']
        data_var=data_dark_std**2+gain*np.abs(data)

        dataCube = DataCube(data, data_var, data_file, header)

        # Normalize the data cube by the flat field if provided
        if flat is not None:
            dataCube.normalize_with_flat(flat)

        # If smoothing and binning is required
        if Nsmooth > 1:
            dataCube.smooth(Nsmooth)
        if Nbin > 1:
            dataCube.bin(Nbin)

        # If normalization with spectra is required
        if normalize == True:
            dataCube.normalize_with_spectra()

        datalist += [dataCube]

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

def get_chi2_maps(datacube,fluxtiptilt_2_data,data_2_fluxtiptilt):
    """
    Calculates chi-squared maps to evaluate the fit of the data to the model.
    Returns the minimum chi-squared, maximum chi-squared, and the chi-squared map.
    """

    print("Computing chi2 of observations for each triangle :")
    Nwave=datacube.shape[0]
    Noutput=datacube.shape[1]
    Ncube=datacube.shape[2]
    Nmod=datacube.shape[3]
    Ntriangles=data_2_fluxtiptilt.shape[0]
    # Nmodel = postiptilt_2_data.shape[0]

    chi2=np.zeros((Ntriangles,Ncube*Nmod))
    b=datacube.reshape(Nwave,Noutput,Ncube*Nmod)
    for t in tqdm(range(Ntriangles)):
        a=data_2_fluxtiptilt[t]
        c=fluxtiptilt_2_data[t]
        ftt=np.matmul(a,b)
        residual = (b-np.matmul(c,ftt))**2
        chi2[t]= residual.sum(axis=(0,1))

    arg_triangle=chi2.argmin(axis=0)
    # best_ftt = np.array([ftt[best_model[n],:,:,n] for n in range(Ncube*Nmod)])

    chi2_min=chi2.min(axis=0).reshape((Ncube,Nmod))
    chi2_max=chi2.max(axis=0).reshape((Ncube,Nmod))
    arg_triangle=arg_triangle.reshape((Ncube,Nmod))

    return chi2_min,chi2_max,arg_triangle

def chi2_cleaning(datacube,couplingMap):

    fluxtiptilt_2_data = couplingMap.fluxtiptilt_2_data
    data_2_fluxtiptilt = couplingMap.data_2_fluxtiptilt

    chi2_min,chi2_max,arg_triangle=get_chi2_maps(datacube,fluxtiptilt_2_data,data_2_fluxtiptilt)

    flux_thresold=np.percentile(datacube.mean(axis=(0,1)),80)/5
    flux_goodData=datacube.mean(axis=(0,1)) > flux_thresold
    chi2_delta=chi2_min/chi2_max
    percents=np.nanpercentile(chi2_delta[flux_goodData],[16,50,84])
    chi2_threshold=percents[1]+(percents[2]-percents[0])*3/2

    chi2_goodData = (chi2_delta < chi2_threshold)&flux_goodData

    datacube_cleaned = datacube.copy()
    datacube_cleaned[:,:,~chi2_goodData]=0

    return datacube_cleaned,arg_triangle
    