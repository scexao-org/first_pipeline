import numpy as np
from astropy.io import fits

class FlatMap:
    def __init__(self, file):

        self.file = file

        with fits.open(file) as hdul:
            self.flat = hdul['FLAT'].data

        self.inv_flat = 1.0 / self.flat
        self.inv_flat_squared = 1.0 / self.flat**2


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
        with fits.open(self.file) as hdul:
            header = hdul[0].header
        return header