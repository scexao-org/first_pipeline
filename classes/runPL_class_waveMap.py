import numpy as np
from astropy.io import fits
from classes.runPL_class_dataCube import DataCube

class WaveMap:
    def __init__(self, file):

        self.file = file

            # Read the FITS file
        with fits.open(file) as hdul:
            self.Nwave = hdul['WAVELENGTH'].data.shape[0]
            self.wave = hdul['WAVELENGTH'].data
            self.index = hdul['INDEX'].data
            self.weights = hdul['WEIGHT'].data
            self.wave_label = "Wavelength (nm)"



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
        with fits.open(self.file) as hdul:
            header = hdul[0].header
        return header
    
