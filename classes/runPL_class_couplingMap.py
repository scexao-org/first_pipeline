from astropy.io import fits
import numpy as np
from tqdm import tqdm
from scipy.interpolate import griddata
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import libraries.runPL_library_plots as runlib_plots

class CouplingMap:
    def __init__(self, file, pyramids = False):
        cmap_file = fits.open(file)
        header = cmap_file[0].header
        self.wavelength_bin = header['Q_CMWBIN']
        if pyramids:
            add_key = "_P"
        else:
            add_key =  "_T"

        self.flux_2_data = cmap_file['F2DATA'+add_key].data
        self.data_2_flux = cmap_file['DATA2F'+add_key].data
        self.QT = cmap_file['QT'+add_key].data
        self.R = cmap_file['R'+add_key].data
        self.position = cmap_file['XY'+add_key].data
        self.flat = cmap_file["FLAT"].data if "FLAT" in cmap_file else None 
        self.ref_spectra = cmap_file['SPECTRA'].data

        self.Npositions = self.position.shape[0]
        self.Nqr = self.R.shape[2]
        self.Nwave = self.R.shape[1]
        self.Ntriangles = self.R.shape[0]
        self.Noutput = self.QT.shape[3]

        cmap_file.close()

    def compute_broadband_QR(self, wmin, wmax, spectra):
        """
        Compute broadband QR matrices over a wavelength range.
        typically for Nqr=6
        Parameters
        ----------
        wmin : int
            Start index for wavelength range
        wmax : int
            End index for wavelength range
        spectra : np.ndarray
            Shape (Nwave)

        Returns
        -------
        QT_broadband : np.ndarray
            Shape (Ntriangles, Nqr, wmax-wmin, Nqr)
        R_broadband : np.ndarray
            Shape (Ntriangles, Nqr, Nqr)
        """
        Ntriangles, Nwave, Nqr, _ = self.R.shape
        if wmin < 0 or wmax > Nwave or wmin >= wmax:
            raise ValueError("Invalid wavelength range")

        QT_broadband = np.zeros((Ntriangles, Nqr, (wmax-wmin) * Nqr))
        R_broadband = np.zeros((Ntriangles, Nqr, Nqr))

        for t in tqdm(range(Ntriangles),desc="commputing broad band QT"):
            R_scaled = self.R[t,wmin:wmax] * spectra[wmin:wmax, None, None]
            R_stack = np.vstack(R_scaled)        # (num_wave*6, 6)
            Q_new, R_new = np.linalg.qr(R_stack, mode='reduced')     # Q_intermediate: (num_wave*6,6), R_new: (6,6)

            R_broadband[t] = R_new
            QT_broadband[t]= Q_new.T

        return QT_broadband, R_broadband
    
    def QT_dot_data(self, index , data):
                
        Nimages = data.shape[2]
        QTdata = np.zeros((self.Nwave,self.Nqr,Nimages))
        for i in range(Nimages):
            t = index[i]
            QTdata[:,:,i] = (self.QT[t] @ data[:,:,i,None])[...,0]

        return QTdata

    def Q_dot_QTdata(self, index , QTdata):

        Nimages = QTdata.shape[2]
        data = np.zeros((self.Nwave,self.Noutput,Nimages))
        for i in range(Nimages):
            t = index[i]
            data[:,:,i] = (self.QT[t].transpose((0,2,1)) @ QTdata[:,:,i,None])[...,0]

        return data


    def chi2_filtering(self, datacube_T, ra_dec = None, nx_min=0,nx_max=10000):

        if ra_dec is None:
            ra_dec = np.zeros((datacube_T.shape[2],*self.position.shape))
            ra_dec[:] = self.position
        chi2_max = np.sum(datacube_T[nx_min:nx_max]**2, axis=(0,1))
        Npos = self.Npositions
        Nimages = datacube_T.shape[2]

        chi2_map = np.zeros((Npos, Nimages))
        chi2_map[:] =  chi2_max

        for t in tqdm(range(Npos), desc="Computing chi2 map"):
            k= self.QT[t, nx_min:nx_max] @ datacube_T[nx_min:nx_max]
            chi2_map[t,:] -= np.sum(k ** 2, axis=(0,1))


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
        return star_detected, star_index, star_radec, chi2_images
