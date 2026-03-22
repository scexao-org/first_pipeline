from astropy.io import fits
import numpy as np
from tqdm import tqdm
from scipy.interpolate import griddata
import os
from ..libraries import runPL_library_plots as runlib_plots
from ..libraries import runPL_library_linalg as runlib_linalg


class CouplingMap:
    """
    A class to handle coupling maps for the FIRST Visible Photonic Lantern.
    
    Attributes:
        filename (str): Path to the FITS file containing the coupling map
        basename (str): Base name of the file
        wavelength_bin (int): Wavelength binning factor
        flux_2_data (numpy.ndarray): Flux to data transformation matrix
        data_2_flux (numpy.ndarray): Data to flux transformation matrix
        vectors_all_triangles (numpy.ndarray): Singular vectors for all valid triangles/pyramids
        position (numpy.ndarray): Position data for coupling points
        ref_spectra (numpy.ndarray): Reference spectra data
        Npositions (int): Number of positions
        Nqr (int): Number of QR components
        Nwave (int): Number of wavelength channels
        Ntriangles (int): Number of triangles
        Noutput (int): Number of output channels
        is_loaded (bool): Whether the coupling map data has been loaded
    """
    def __init__(self, filename=None):
        self.filename = filename
        self.basename = os.path.basename(filename) if filename else None
        
        # Initialize all attributes to None
        self.wavelength_bin = None
        self.flux_2_data = None
        self.data_2_flux = None
        self.vectors_all_triangles = None
        self.position = None
        self.ref_spectra = None
        self.Npositions = None
        self.Nqr = None
        self.Nwave = None
        self.Ntriangles = None
        self.Noutput = None
        self.is_loaded = False

        self.QT = None
        self.R = None
        
        if filename is not None:
            self.load(filename)

    def load(self, filename):
        """
        Load the coupling map from a FITS file.
        
        Args:
            filename (str): Path to the FITS file containing the coupling map
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            KeyError: If required FITS extensions are missing
        """
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Coupling map file not found: {filename}")
            
        self.filename = filename
        self.basename = os.path.basename(filename)
        
        cmap_file = fits.open(filename)
        header = cmap_file[0].header
        
        if 'Q_CMWBIN' not in header:
            raise KeyError("FITS header missing required 'Q_CMWBIN' keyword")
            
        self.wavelength_bin = header['Q_CMWBIN']
        
        required_extensions = ['F2DATA', 'DATA2F', 'VECTORS', 'XY', 'SPECTRA']
        available_extensions = [hdu.name for hdu in cmap_file]
        missing_extensions = [ext for ext in required_extensions if ext not in available_extensions]
        
        if missing_extensions:
            cmap_file.close()
            raise KeyError(f"FITS file missing required extensions: {missing_extensions}")

        self.flux_2_data = cmap_file['F2DATA'].data
        self.data_2_flux = cmap_file['DATA2F'].data
        self.vectors_all_triangles = cmap_file['VECTORS'].data
        self.position = cmap_file['XY'].data
        self.ref_spectra = cmap_file['SPECTRA'].data

        if self.vectors_all_triangles is None or self.vectors_all_triangles.size == 0:
            cmap_file.close()
            raise ValueError("Coupling map data is empty or invalid")
            
        self.Npositions = self.position.shape[0]
        self.Nqr = self.vectors_all_triangles.shape[3]
        self.Nwave = self.vectors_all_triangles.shape[2]
        self.Ntriangles = self.vectors_all_triangles.shape[0]
        self.Noutput = self.vectors_all_triangles.shape[1]

        cmap_file.close()
        self.is_loaded = True

    def create_from_data(self, flux_2_data, data_2_flux, vectors_all_triangles, position, spectra, wavelength_bin, filename=None):
        """
        Create a coupling map from data arrays.
        Args:
            flux_2_data: Flux to data transformation matrix
            data_2_flux: Data to flux transformation matrix  
            vectors_all_triangles: Singular vectors for all valid triangles/pyramids
            position: Position data for coupling points
            spectra: Reference spectra array
            wavelength_bin: Wavelength binning factor
            filename (str, optional): Optional filename to associate with this coupling map.
        """
        # Store the data arrays directly
        self.flux_2_data = flux_2_data
        self.data_2_flux = data_2_flux 
        self.vectors_all_triangles = vectors_all_triangles
        self.position = position
        self.ref_spectra = spectra
        self.wavelength_bin = wavelength_bin
        
        # Set dimensions
        if self.position is not None:
            self.Npositions = self.position.shape[0]
        if self.vectors_all_triangles is not None:
            self.Nqr = self.vectors_all_triangles.shape[3]
            self.Nwave = self.vectors_all_triangles.shape[2]
            self.Ntriangles = self.vectors_all_triangles.shape[0]
            self.Noutput = self.vectors_all_triangles.shape[1]
        
        if filename:
            self.filename = filename
            self.basename = os.path.basename(filename)
        else:
            self.filename = None
            self.basename = None
            
        self.is_loaded = True

    def _check_loaded(self):
        """Check if the coupling map is properly loaded."""
        if not self.is_loaded:
            raise ValueError("Coupling map not loaded. Use load() or create_from_data() first.")
    
    def _set_active_data(self):
        """Set the active data (no longer needed since we only have one dataset)"""
        # This method is kept for compatibility but does nothing since we no longer
        # distinguish between triangle and pyramid data
        if self.position is not None:
            self.Npositions = self.position.shape[0]
        if self.vectors_all_triangles is not None:
            self.Nqr = self.vectors_all_triangles.shape[3]
            self.Nwave = self.vectors_all_triangles.shape[2]
            self.Ntriangles = self.vectors_all_triangles.shape[0]
            self.Noutput = self.vectors_all_triangles.shape[1]

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
        
        self._check_loaded()
        
        # Create a primary HDU with no data, just the header
        hdu_primary = fits.PrimaryHDU()

        # Create HDUs for coupling map data
        hdu = [fits.ImageHDU(data=self.flux_2_data, name='F2DATA')]
        hdu += [fits.ImageHDU(data=self.data_2_flux, name='DATA2F')]
        hdu += [fits.ImageHDU(data=self.vectors_all_triangles, name='VECTORS')]
        hdu += [fits.ImageHDU(data=self.position, name='XY')]
        hdu += [fits.ImageHDU(data=self.ref_spectra, name='SPECTRA')]
        
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

        self.basename = os.path.basename(output_filename)
        self.filename = output_filename
        self.is_loaded = True

    def return_hdu_list(self):
        """
        Return a list of FITS HDUs representing the coupling map.
        
        Returns:
            list: List of FITS HDUs containing coupling map data
            
        Raises:
            ValueError: If no coupling map data is available
        """
        # Use current active data
        hdu = [fits.ImageHDU(data=self.flux_2_data, name='F2DATA')]
        hdu += [fits.ImageHDU(data=self.data_2_flux, name='DATA2F')]
        hdu += [fits.ImageHDU(data=self.vectors_all_triangles, name='VECTORS')]
        hdu += [fits.ImageHDU(data=self.position, name='XY')]
        hdu += [fits.ImageHDU(data=self.ref_spectra, name='SPECTRA')]
        return hdu
    
    def return_header(self):
        """
        Return the header of the FITS file.
        
        Returns:
            astropy.io.fits.Header: The header of the FITS file or empty header if none available
        """
        if self.filename is not None:
            with fits.open(self.filename) as hdul:
                header = hdul[0].header
            return header
        else:
            # Return empty header if no file is loaded
            return fits.Header()

    def QT_and_R_matrices(self):
        """
        Compute QR decomposition matrices for singular vectors.
        
        Parameters
        ----------
        singular_vectors : numpy.ndarray
            Singular vectors array with shape (Ntriangles, Noutput, Nwave, Nqr)
            
        Returns
        -------
        QT_singular_vectors : numpy.ndarray
            Transpose of Q matrices from QR decomposition
        R_singular_vectors : numpy.ndarray
            R matrices from QR decomposition
        """

        if self.QT is not None:
            return self.QT,self.R
        else:
            singular_vectors = self.vectors_all_triangles
            Ntriangles = singular_vectors.shape[0]
            Noutput = singular_vectors.shape[1]
            Nwave = singular_vectors.shape[2]
            Nqr = singular_vectors.shape[3]

            QT_singular_vectors = np.zeros((Ntriangles, Nwave, Nqr, Noutput))
            R_singular_vectors = np.zeros((Ntriangles, Nwave, Nqr, Nqr))

            if Nqr == 3:
                description = "Calculating QR matrices for triangles"
            else:
                description = "Calculating QR matrices for pyramids"

            for t in tqdm(range(Ntriangles), desc=description):
                for w in range(Nwave):
                    Q, R = np.linalg.qr(singular_vectors[t, :, w], mode="reduced")
                    QT_singular_vectors[t, w] = Q.T
                    R_singular_vectors[t, w] = R

            self.QT = QT_singular_vectors
            self.R = R_singular_vectors

            return QT_singular_vectors, R_singular_vectors

    def QT_dot_data(self, index , data):
        self._check_loaded()
                
        Nimages = data.shape[2]
        QTdata = np.zeros((self.Nwave,self.Nqr,Nimages))
        for i in range(Nimages):
            t = index[i]
            QTdata[:,:,i] = (self.QT[t] @ data[:,:,i,None])[...,0]

        return QTdata

    def Q_dot_QTdata(self, index , QTdata):
        self._check_loaded()

        Nimages = QTdata.shape[2]
        data = np.zeros((self.Nwave,self.Noutput,Nimages))
        for i in range(Nimages):
            t = index[i]
            data[:,:,i] = (self.QT[t].transpose((0,2,1)) @ QTdata[:,:,i,None])[...,0]

        return data
