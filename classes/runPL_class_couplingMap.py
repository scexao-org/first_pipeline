from astropy.io import fits
import numpy as np
from tqdm import tqdm
from scipy.interpolate import griddata
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import libraries.runPL_library_plots as runlib_plots
import libraries.runPL_library_linalg as runlib_linalg


class CouplingMap:
    """
    A class to handle coupling maps for the FIRST Visible Photonic Lantern.
    
    Attributes:
        file (str): Path to the FITS file containing the coupling map
        basename (str): Base name of the file
        pyramids (bool): Whether using pyramid or triangle data mode
        wavelength_bin (int): Wavelength binning factor
        flux_2_data (numpy.ndarray): Flux to data transformation matrix
        data_2_flux (numpy.ndarray): Data to flux transformation matrix
        QT (numpy.ndarray): QT transformation matrices
        R (numpy.ndarray): R matrices for coupling analysis
        position (numpy.ndarray): Position data for coupling points
        ref_spectra (numpy.ndarray): Reference spectra data
        Npositions (int): Number of positions
        Nqr (int): Number of QR components
        Nwave (int): Number of wavelength channels
        Ntriangles (int): Number of triangles
        Noutput (int): Number of output channels
    """
    def __init__(self, file=None, pyramids=False):
        self.pyramids = pyramids
        self.file = file
        self.basename = os.path.basename(file) if file else None
        
        # Initialize all attributes to None
        self.wavelength_bin = None
        self.flux_2_data = None
        self.data_2_flux = None
        self.QT = None
        self.R = None
        self.position = None
        self.ref_spectra = None
        self.Npositions = None
        self.Nqr = None
        self.Nwave = None
        self.Ntriangles = None
        self.Noutput = None
        
        if file is not None:
            self.load(file, pyramids)

    def load(self, file, pyramids=False):
        """
        Load the coupling map from a FITS file.
        
        Args:
            file (str): Path to the FITS file containing the coupling map
            pyramids (bool): Whether to load pyramid data (True) or triangle data (False)
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            KeyError: If required FITS extensions are missing
        """
        if not os.path.exists(file):
            raise FileNotFoundError(f"Coupling map file not found: {file}")
            
        self.file = file
        self.basename = os.path.basename(file)
        self.pyramids = pyramids
        
        cmap_file = fits.open(file)
        header = cmap_file[0].header
        
        if 'Q_CMWBIN' not in header:
            raise KeyError("FITS header missing required 'Q_CMWBIN' keyword")
            
        self.wavelength_bin = header['Q_CMWBIN']
        
        if pyramids:
            add_key = "_P"
        else:
            add_key = "_T"

        required_extensions = [f'F2DATA{add_key}', f'DATA2F{add_key}', f'QT{add_key}', f'R{add_key}', f'XY{add_key}', 'SPECTRA']
        available_extensions = [hdu.name for hdu in cmap_file]
        missing_extensions = [ext for ext in required_extensions if ext not in available_extensions]
        
        if missing_extensions:
            cmap_file.close()
            raise KeyError(f"FITS file missing required extensions: {missing_extensions}")

        self.flux_2_data = cmap_file['F2DATA'+add_key].data
        self.data_2_flux = cmap_file['DATA2F'+add_key].data
        self.QT = cmap_file['QT'+add_key].data
        self.R = cmap_file['R'+add_key].data
        self.position = cmap_file['XY'+add_key].data
        self.ref_spectra = cmap_file['SPECTRA'].data

        if self.QT is None or self.QT.size == 0:
            cmap_file.close()
            raise ValueError("Coupling map data is empty or invalid")
            
        self.Npositions = self.position.shape[0]
        self.Nqr = self.R.shape[2]
        self.Nwave = self.R.shape[1]
        self.Ntriangles = self.R.shape[0]
        self.Noutput = self.QT.shape[3]

        cmap_file.close()

    def create_from_data(self, flux_2_data_triangles, data_2_flux_triangles, QT_triangles, R_triangles, 
                        center_triangles, flux_2_data_pyramids, data_2_flux_pyramids, QT_pyramids, 
                        R_pyramids, center_pyramids, spectra, wavelength_bin, filename=None):
        """
        Create a coupling map from data arrays.
        Args:
            flux_2_data_triangles, data_2_flux_triangles, QT_triangles, R_triangles, center_triangles: Triangle data arrays
            flux_2_data_pyramids, data_2_flux_pyramids, QT_pyramids, R_pyramids, center_pyramids: Pyramid data arrays  
            spectra: Reference spectra array
            wavelength_bin: Wavelength binning factor
            filename (str, optional): Optional filename to associate with this coupling map.
        """
        # Store all the data arrays
        self.flux_2_data_triangles = flux_2_data_triangles
        self.data_2_flux_triangles = data_2_flux_triangles 
        self.QT_triangles = QT_triangles
        self.R_triangles = R_triangles
        self.center_triangles = center_triangles
        self.flux_2_data_pyramids = flux_2_data_pyramids
        self.data_2_flux_pyramids = data_2_flux_pyramids
        self.QT_pyramids = QT_pyramids
        self.R_pyramids = R_pyramids
        self.center_pyramids = center_pyramids
        self.ref_spectra = spectra
        self.wavelength_bin = wavelength_bin
        
        # Set default to triangles
        self.pyramids = False
        self._set_active_data()
        
        if filename:
            self.file = filename
            self.basename = os.path.basename(filename)
        else:
            self.file = None
            self.basename = None
    
    def _set_active_data(self):
        """Set the active data based on pyramids flag"""
        if self.pyramids:
            self.flux_2_data = self.flux_2_data_pyramids
            self.data_2_flux = self.data_2_flux_pyramids
            self.QT = self.QT_pyramids
            self.R = self.R_pyramids
            self.position = self.center_pyramids
        else:
            self.flux_2_data = self.flux_2_data_triangles
            self.data_2_flux = self.data_2_flux_triangles
            self.QT = self.QT_triangles
            self.R = self.R_triangles
            self.position = self.center_triangles
            
        if self.position is not None:
            self.Npositions = self.position.shape[0]
        if self.R is not None:
            self.Nqr = self.R.shape[2]
            self.Nwave = self.R.shape[1]
            self.Ntriangles = self.R.shape[0]
        if self.QT is not None:
            self.Noutput = self.QT.shape[3]

    def set_pyramids(self, pyramids):
        """
        Switch between triangle and pyramid data.
        Args:
            pyramids (bool): True for pyramid data, False for triangle data.
        """
        self.pyramids = pyramids
        if hasattr(self, 'flux_2_data_triangles'):  # Check if created from data
            self._set_active_data()
        elif self.file is not None:  # Reload from file
            self.load(self.file, pyramids)

    def save(self, output_filename, header=None, flat_map=None, wave_map=None, modulation_hdu=None):
        """
        Save the coupling map to a FITS file.
        
        Args:
            output_filename (str): Path for the output FITS file
            header (astropy.io.fits.Header, optional): Additional header information
            flat_map (FlatMap, optional): FlatMap object to include in the FITS file
            wave_map (WaveMap, optional): WaveMap object to include in the FITS file
            modulation_hdu (fits.HDU, optional): Modulation HDU to include
            
        Raises:
            ValueError: If no coupling map data is available to save
        """
        from datetime import datetime
        
        # Check if we have the required data
        if hasattr(self, 'flux_2_data_triangles'):
            # Created from data - use stored arrays
            flux_2_data_triangles = self.flux_2_data_triangles
            data_2_flux_triangles = self.data_2_flux_triangles
            QT_triangles = self.QT_triangles
            R_triangles = self.R_triangles
            center_triangles = self.center_triangles
            flux_2_data_pyramids = self.flux_2_data_pyramids
            data_2_flux_pyramids = self.data_2_flux_pyramids
            QT_pyramids = self.QT_pyramids
            R_pyramids = self.R_pyramids
            center_pyramids = self.center_pyramids
            spectra = self.ref_spectra
        elif self.file is not None:
            # Loaded from file - need to reload both triangle and pyramid data
            temp_pyramids = self.pyramids
            # Load triangles
            self.load(self.file, pyramids=False)
            flux_2_data_triangles = self.flux_2_data
            data_2_flux_triangles = self.data_2_flux
            QT_triangles = self.QT
            R_triangles = self.R
            center_triangles = self.position
            # Load pyramids
            self.load(self.file, pyramids=True)
            flux_2_data_pyramids = self.flux_2_data
            data_2_flux_pyramids = self.data_2_flux
            QT_pyramids = self.QT
            R_pyramids = self.R
            center_pyramids = self.position
            spectra = self.ref_spectra
            # Restore original mode
            self.load(self.file, pyramids=temp_pyramids)
        else:
            raise ValueError("No coupling map data to save. Load or create coupling map data first.")
            
        # Create a primary HDU with no data, just the header
        hdu_primary = fits.PrimaryHDU()

        # Create HDUs for coupling map data
        hdu = [fits.ImageHDU(data=flux_2_data_triangles, name='F2DATA_T')]
        hdu += [fits.ImageHDU(data=data_2_flux_triangles, name='DATA2F_T')]
        hdu += [fits.ImageHDU(data=QT_triangles, name='QT_T')]
        hdu += [fits.ImageHDU(data=R_triangles, name='R_T')]
        hdu += [fits.ImageHDU(data=center_triangles, name='XY_T')]
        hdu += [fits.ImageHDU(data=flux_2_data_pyramids, name='F2DATA_P')]
        hdu += [fits.ImageHDU(data=data_2_flux_pyramids, name='DATA2F_P')]
        hdu += [fits.ImageHDU(data=QT_pyramids, name='QT_P')]
        hdu += [fits.ImageHDU(data=R_pyramids, name='R_P')]
        hdu += [fits.ImageHDU(data=center_pyramids, name='XY_P')]
        hdu += [fits.ImageHDU(data=spectra, name='SPECTRA')]
        
        if modulation_hdu is not None:
            hdu += [modulation_hdu]
        
        # Add flat and wave maps if provided
        if flat_map is not None:
            hdu += flat_map.return_hdu_list()
        if wave_map is not None:
            hdu += wave_map.return_hdu_list()

        if header is not None:
            new_header = header.copy()
            # Add date and time to the header if not present
            if 'DATE-PRO' not in new_header:
                current_time = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
                new_header['DATE-PRO'] = current_time
            
            if 'X_FIRTYP' not in new_header:
                new_header['X_FIRTYP'] = 'COUPLINGMAP'
                
            # Add flat and wave headers if available
            if flat_map is not None:
                flat_header = flat_map.return_header()
                new_header.extend(flat_header, strip=True)
            if wave_map is not None:
                wave_header = wave_map.return_header()
                new_header.extend(wave_header, strip=True)
                
            hdu_primary.header.extend(new_header, strip=True)

        # Combine all HDUs into an HDUList
        hdul = fits.HDUList([hdu_primary, *hdu])

        # Write to a FITS file
        print(f"Saving coupling map to {output_filename}")
        hdul.writeto(output_filename, overwrite=True)

    def return_hdu_list(self):
        """
        Return a list of FITS HDUs representing the coupling map.
        
        Returns:
            list: List of FITS HDUs containing coupling map data
            
        Raises:
            ValueError: If no coupling map data is available
        """
        if hasattr(self, 'flux_2_data_triangles'):
            # Created from data
            hdu = [fits.ImageHDU(data=self.flux_2_data_triangles, name='F2DATA_T')]
            hdu += [fits.ImageHDU(data=self.data_2_flux_triangles, name='DATA2F_T')]
            hdu += [fits.ImageHDU(data=self.QT_triangles, name='QT_T')]
            hdu += [fits.ImageHDU(data=self.R_triangles, name='R_T')]
            hdu += [fits.ImageHDU(data=self.center_triangles, name='XY_T')]
            hdu += [fits.ImageHDU(data=self.flux_2_data_pyramids, name='F2DATA_P')]
            hdu += [fits.ImageHDU(data=self.data_2_flux_pyramids, name='DATA2F_P')]
            hdu += [fits.ImageHDU(data=self.QT_pyramids, name='QT_P')]
            hdu += [fits.ImageHDU(data=self.R_pyramids, name='R_P')]
            hdu += [fits.ImageHDU(data=self.center_pyramids, name='XY_P')]
            hdu += [fits.ImageHDU(data=self.ref_spectra, name='SPECTRA')]
        else:
            # Use current active data only
            add_key = "_P" if self.pyramids else "_T"
            hdu = [fits.ImageHDU(data=self.flux_2_data, name='F2DATA'+add_key)]
            hdu += [fits.ImageHDU(data=self.data_2_flux, name='DATA2F'+add_key)]
            hdu += [fits.ImageHDU(data=self.QT, name='QT'+add_key)]
            hdu += [fits.ImageHDU(data=self.R, name='R'+add_key)]
            hdu += [fits.ImageHDU(data=self.position, name='XY'+add_key)]
            hdu += [fits.ImageHDU(data=self.ref_spectra, name='SPECTRA')]
        return hdu
    
    def return_header(self):
        """
        Return the header of the FITS file.
        
        Returns:
            astropy.io.fits.Header: The header of the FITS file or empty header if none available
        """
        if self.file is not None:
            with fits.open(self.file) as hdul:
                header = hdul[0].header
            return header
        else:
            # Return empty header if no file is loaded
            return fits.Header()

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

    def get_star_position(self, datacube_T, star_index, Xmod, Ymod):
        """
        Estimate star positions from a spectrally resolved data cube using
        QR-based coupling maps.
    
        The data are projected onto precomputed QR bases, a broadband QR fit
        is used to estimate global (x, y) position offsets, and the result is
        combined with the triangle reference centers. Both 3- and 6-parameter
        QR models are supported.
    
        Parameters
        ----------
        datacube_T : np.ndarray
            Transposed data cube of shape (Nwave, Noutputs, Nmod).
    
        star_index : np.ndarray
            Triangle index associated with each modeled source (shape: Nmod).
    
        Xmod, Ymod : np.ndarray
            Model x- and y-positions of the sources.
    
        Returns
        -------
        Xpos : np.ndarray
            Estimated x-positions of the sources (shape: Nmod).
    
        Ypos : np.ndarray
            Estimated y-positions of the sources (shape: Nmod).
    
        Xcen, Ycen : np.ndarray
            Coordinates of the reference triangle centers.
    
        Xdiff, Ydiff : np.ndarray
            Differences between estimated and model positions.
        """
        
        Nwave, Noutputs, Nmod = datacube_T.shape
        
        QT= self.QT
        spectra = self.ref_spectra
        R= self.R * spectra[:,None,None]
        centers = self.position
        
        wmin = QT.shape[1] // 4
        wmax = 3 * QT.shape[1] // 4
        QT_broadband, R_broadband = self.compute_broadband_QR(wmin, wmax, spectra)

        QTdata = np.zeros((QT.shape[1],QT.shape[2],datacube_T.shape[2]))
        for i in tqdm(range(Nmod), desc="Projection onto QT space"):
            t = star_index[i]
            data = datacube_T[:,:,i]
            QTdata[:,:,i] = (QT[t] @ data[:,:,None])[:,:,0]

        Xpos = np.zeros((Nmod))
        Ypos = np.zeros((Nmod))
        Xcen = np.zeros((Nmod))
        Ycen = np.zeros((Nmod))
        Xdiff = np.zeros((Nmod))
        Ydiff = np.zeros((Nmod))

        X_wave = np.zeros((Nwave, Nmod))
        Y_wave = np.zeros((Nwave, Nmod))
        Z_wave = np.zeros((Nwave, Nmod))
        QTdata_dxy = np.zeros_like(QTdata)
        Nqr = R.shape[2]
        R_dxy = np.zeros((Nwave, Nqr, Nmod, 2))
        
        for i in tqdm(range(Nmod), desc="Computing XY positions"):
            t = star_index[i]
            center = centers[t]

            QTdata_broadband = QT_broadband[t] @ QTdata[wmin:wmax,:,i].ravel()
            
            if Nqr == 6:
                x_hat_broadband, y_hat_broadband, k_hat_broadband, chi2_broadband, _ = runlib_linalg.fit_QR_6(QTdata_broadband, R_broadband[t])
            else:
                x_hat_broadband, y_hat_broadband, k_hat_broadband, chi2_broadband, _ = runlib_linalg.solve_QR_3(QTdata_broadband, R_broadband[t])

            # Add NaN checks and set to zero if needed
            if np.isnan(x_hat_broadband):
                x_hat_broadband = 0.0
            if np.isnan(y_hat_broadband):
                y_hat_broadband = 0.0
            if np.isnan(k_hat_broadband):
                k_hat_broadband = 0.0

            if Nqr == 6:
                v = np.array([1.0, x_hat_broadband, y_hat_broadband, x_hat_broadband*y_hat_broadband, x_hat_broadband**2, y_hat_broadband**2])
                dv_dx = np.array([0.0, 1.0, 0.0, y_hat_broadband, 2.0*x_hat_broadband, 0.0])
                dv_dy = np.array([0.0, 0.0, 1.0, x_hat_broadband, 0.0, 2.0*y_hat_broadband])
            else:
                v = np.array([1.0, x_hat_broadband, y_hat_broadband])
                dv_dx = np.array([0.0, 1.0, 0.0])
                dv_dy = np.array([0.0, 0.0, 1.0])

            r = R[t] @ v
            Kernel_v = np.identity(len(v)) - (r[:,:,None] @ r[:,None]) / (r[:,None] @ r[:,:,None])
            QTdata_dxy[:,:,i] = (Kernel_v @ QTdata[:,:,i,None])[...,0]

            dev_phi = np.array((dv_dx,dv_dy)).T
            R_dxy[:,:,i] = Kernel_v @ (R[t] @ dev_phi)

            # xy_dev = (np.linalg.pinv(R_dxy[:,:,i]) @ QTdata_dxy[:,:,i,None])[...,0]

            # X_wave[:,i] = xy_dev[:,0]
            # Y_wave[:,i] = xy_dev[:,1]

            xd = x_hat_broadband + center[0] - Xmod.ravel()[i]
            yd = y_hat_broadband + center[1] - Ymod.ravel()[i]

            xmodmax, ymodmax = np.max(np.abs(Xmod)), np.max(np.abs(Ymod))
        
            Xpos.ravel()[i] = center[0]+x_hat_broadband if (np.abs(xd) < xmodmax) else Xmod[0,i]
            Ypos.ravel()[i] = center[1]+y_hat_broadband if (np.abs(yd) < ymodmax) else Ymod[0,i]

            Xcen.ravel()[i] = center[0]
            Ycen.ravel()[i] = center[1]

            Xdiff.ravel()[i] = xd
            Ydiff.ravel()[i] = yd

        # xy_dev = np.linalg.pinv(R_dxy.reshape((Nwave,-1,2))) @ QTdata_dxy.reshape((Nwave,-1,1))
        # xy_dev = xy_dev[...,0]
        
        return Xpos, Ypos, Xcen, Ycen, Xdiff, Ydiff
