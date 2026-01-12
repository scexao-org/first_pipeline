import numpy as np
from astropy.io import fits
import os

class FlatMap:
    def __init__(self, file=None):

        self.file = file
        self.basename = os.path.basename(file) if file else None
        self.flat = None
        self.inv_flat = None
        self.inv_flat_squared = None
        
        if file is not None:
            self.load(file)

    def load(self, file):
        """
        Load the flat field map from a FITS file.
        Args:
            file (str): Path to the FITS file containing the flat field map.
        """
        self.file = file
        self.basename = os.path.basename(file)
        
        with fits.open(file) as hdul:
            self.flat = hdul['FLAT'].data

        self.inv_flat = 1.0 / self.flat
        self.inv_flat_squared = 1.0 / self.flat**2

    def create_from_data(self, flat_data, filename=None):
        """
        Create a flat field map from data array.
        Args:
            flat_data (numpy.ndarray): The flat field data array.
            filename (str, optional): Optional filename to associate with this flat map.
        """
        self.flat = flat_data
        self.inv_flat = 1.0 / self.flat
        self.inv_flat_squared = 1.0 / self.flat**2
        
        if filename:
            self.file = filename
            self.basename = os.path.basename(filename)
        else:
            self.file = None
            self.basename = None

    def save(self, output_filename, header=None):
        """
        Save the flat field map to a FITS file.
        Args:
            output_filename (str): Path for the output FITS file.
            header (astropy.io.fits.Header, optional): Header for the FITS file.
        """
        from datetime import datetime
        
        if self.flat is None:
            raise ValueError("No flat field data to save. Load or create flat field data first.")
            
        # Create a primary HDU with no data, just the header
        hdu_primary = fits.PrimaryHDU()

        # Create HDUs for the flat field
        hdu = [fits.ImageHDU(data=self.flat, name='FLAT')]

        if header is not None:
            # Add date and time to the header if not present
            if 'DATE-PRO' not in header:
                current_time = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
                header['DATE-PRO'] = current_time
            
            if 'X_FIRTYP' not in header:
                header['X_FIRTYP'] = 'FLATMAP'
                
            hdu_primary.header.extend(header, strip=True)

        # Combine all HDUs into an HDUList
        hdul = fits.HDUList([hdu_primary, *hdu])

        # Write to a FITS file
        print(f"Saving flat field map to {output_filename}")
        hdul.writeto(output_filename, overwrite=True)


    def normalize_with_flat(self, dataCube):
        """
        Normalize the data cube by a flat field.
        Args:
                dataCube (DataCube): The data cube to normalize.
        """
        dataCube.data *= self.inv_flat
        dataCube.variance *= self.inv_flat_squared
    
    def return_hdu_list(self):
        """
        Return a list of FITS HDUs representing the wavelength map.
        Returns:
                list: List of FITS HDUs.
        """
        hdu = [fits.ImageHDU(data=self.flat, name='FLAT')]
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