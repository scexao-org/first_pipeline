import numpy as np
from astropy.io import fits
from classes.runPL_class_dataCube import DataCube
import os

class WaveMap:
    """
    A class to handle wavelength maps for the FIRST Visible Photonic Lantern.
    
    Attributes:
        file (str): Path to the FITS file containing the wavelength map
        basename (str): Base name of the file
        Nwave (int): Number of wavelength channels
        wave (numpy.ndarray): Wavelength data array
        index (numpy.ndarray): Index data array for interpolation
        weights (numpy.ndarray): Weights data array for interpolation
        wave_label (str): Label for wavelength units
    """
    def __init__(self, file=None):

        self.file = file
        self.basename = os.path.basename(file) if file else None
        self.Nwave = None
        self.wave = None
        self.index = None
        self.weights = None
        self.wave_label = "Wavelength (nm)"
        
        if file is not None:
            self.load(file)

    def load(self, file):
        """
        Load the wavelength map from a FITS file.
        
        Args:
            file (str): Path to the FITS file containing the wavelength map
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            KeyError: If required FITS extensions are missing
        """
        if not os.path.exists(file):
            raise FileNotFoundError(f"Wavelength map file not found: {file}")
            
        self.file = file
        self.basename = os.path.basename(file)
        
        # Read the FITS file
        with fits.open(file) as hdul:
            required_extensions = ['WAVELENGTH', 'INDEX', 'WEIGHT']
            available_extensions = [hdu.name for hdu in hdul]
            missing_extensions = [ext for ext in required_extensions if ext not in available_extensions]
            
            if missing_extensions:
                raise KeyError(f"FITS file missing required extensions: {missing_extensions}")
                
            self.Nwave = hdul['WAVELENGTH'].data.shape[0]
            self.wave = hdul['WAVELENGTH'].data
            self.index = hdul['INDEX'].data
            self.weights = hdul['WEIGHT'].data
            
        if self.wave is None or self.wave.size == 0:
            raise ValueError("Wavelength map data is empty or invalid")

    def create_from_data(self, wave, index, weights, filename=None):
        """
        Create a wavelength map from data arrays.
        Args:
            wave (numpy.ndarray): The wavelength data array.
            index (numpy.ndarray): The index data array.
            weights (numpy.ndarray): The weights data array.
            filename (str, optional): Optional filename to associate with this wavelength map.
        """
        self.wave = wave
        self.index = index
        self.weights = weights
        self.Nwave = wave.shape[0] if wave is not None else None
        
        if filename:
            self.file = filename
            self.basename = os.path.basename(filename)
        else:
            self.file = None
            self.basename = None
            
    def _validate_data(self):
        """
        Validate that the wavelength map data is properly loaded and consistent.
        
        Raises:
            ValueError: If data is invalid or inconsistent
        """
        if self.wave is None or self.index is None or self.weights is None:
            raise ValueError("Incomplete wavelength map data")
        if self.wave.size == 0 or self.index.size == 0 or self.weights.size == 0:
            raise ValueError("Wavelength map data is empty")
        if self.Nwave != self.wave.shape[0]:
            raise ValueError("Inconsistent wavelength data dimensions")

    def save(self, output_filename, header=None):
        """
        Save the wavelength map to a FITS file.
        
        Args:
            output_filename (str): Path for the output FITS file
            header (astropy.io.fits.Header, optional): Additional header information
            
        Raises:
            ValueError: If no wavelength map data is available to save
        """
        from datetime import datetime
        
        if self.wave is None or self.index is None or self.weights is None:
            raise ValueError("No wavelength map data to save. Load or create wavelength map data first.")
            
        # Create a primary HDU with no data, just the header
        hdu_primary = fits.PrimaryHDU()

        # Create HDUs for the wavelength map data
        hdu = [fits.ImageHDU(data=self.wave, name='WAVELENGTH')]
        hdu += [fits.ImageHDU(data=self.index, name='INDEX')]
        hdu += [fits.ImageHDU(data=self.weights, name='WEIGHT')]

        if header is not None:
            # Add date and time to the header if not present
            if 'DATE-PRO' not in header:
                current_time = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
                header['DATE-PRO'] = current_time

            hdu_primary.header.extend(header, strip=True)

        hdu_primary.header['X_FIRTYP'] = 'WAVEMAP'
        # Combine all HDUs into an HDUList
        hdul = fits.HDUList([hdu_primary, *hdu])

        # Write to a FITS file
        print(f"Saving wavelength map to {output_filename}")
        hdul.writeto(output_filename, overwrite=True)

    def interpolate_data(self, dataCube):
        """
        Apply the wavelength index to interpolate the data cube.
        Args:
                dataCube (DataCube): The data cube to apply the wavelength map to.
        """
        data = dataCube.data
        variance = dataCube.variance
        Ncube, Nmod, Noutput, Nwave = data.shape
        Nwave_new = self.Nwave
        new_data = np.zeros((Ncube, Nmod, Noutput, Nwave_new)) 
        new_variance = np.zeros((Ncube, Nmod, Noutput, Nwave_new)) 

        for o in range(Noutput):
            new_data[:,:,o,:] = (data[:,:,o,self.index[:,o]] * self.weights[:,o]).sum(axis=2)
            new_variance[:,:,o,:] = (variance[:,:,o,self.index[:,o]] * self.weights[:,o]).sum(axis=2)

        dataCube.data = new_data
        dataCube.variance = new_variance
        
    def return_hdu_list(self):
        """
        Return a list of FITS HDUs representing the wavelength map.
        
        Returns:
            list: List of FITS HDUs containing wavelength map data
            
        Raises:
            ValueError: If no wavelength map data is available
        """
        if self.wave is None or self.index is None or self.weights is None:
            raise ValueError("No wavelength map data available")
        hdu = [fits.ImageHDU(data=self.wave, name='WAVELENGTH')]
        hdu += [fits.ImageHDU(data=self.index, name='INDEX')]
        hdu += [fits.ImageHDU(data=self.weights, name='WEIGHT')]
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
    
