import numpy as np
from astropy.io import fits
from classes.runPL_class_dataCube import DataCube
import os

class WaveMap:
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
            file (str): Path to the FITS file containing the wavelength map.
        """
        self.file = file
        self.basename = os.path.basename(file)
        
        # Read the FITS file
        with fits.open(file) as hdul:
            self.Nwave = hdul['WAVELENGTH'].data.shape[0]
            self.wave = hdul['WAVELENGTH'].data
            self.index = hdul['INDEX'].data
            self.weights = hdul['WEIGHT'].data

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

    def save(self, output_filename, header=None):
        """
        Save the wavelength map to a FITS file.
        Args:
            output_filename (str): Path for the output FITS file.
            header (astropy.io.fits.Header, optional): Header for the FITS file.
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
            
            if 'X_FIRTYP' not in header:
                header['X_FIRTYP'] = 'WAVEMAP'
                
            hdu_primary.header.extend(header, strip=True)

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
                list: List of FITS HDUs.
        """
        hdu = [fits.ImageHDU(data=self.wave, name='WAVELENGTH')]
        hdu += [fits.ImageHDU(data=self.index, name='INDEX')]
        hdu += [fits.ImageHDU(data=self.weights, name='WEIGHT')]
        return hdu
    
    def return_header(self):
        """
        Return the header of the FITS file.
        Returns:
                astropy.io.fits.Header: The header of the FITS file.
        """
        if self.file is not None:
            with fits.open(self.file) as hdul:
                header = hdul[0].header
            return header
        else:
            # Return empty header if no file is loaded
            return fits.Header()
    
