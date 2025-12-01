import numpy as np
from astropy.io import fits

class FlatMap:
    def __init__(self, file):

        # Read the FITS file
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
        