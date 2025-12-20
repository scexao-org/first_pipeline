from astropy.io import fits
import numpy as np
import os

class PixelMap:
    def __init__(self, file):
        self.file = file
        self.basename = os.path.basename(file)
        self.header = fits.getheader(file)
        self.traces_loc = fits.getdata(file)
        # Check for required header keywords and raise error if not found
        required_keys = ['Q_PMXMIN', 'Q_PMXMAX', 'Q_PMWIDE', 'Q_PMCHAN', 'Q_PM_CK']
        missing_keys = [key for key in required_keys if key not in self.header]
        if missing_keys:
            raise KeyError(f"FITS header keywords missing in Pixel Map: {missing_keys}")
        
        self.pixel_min = self.header['Q_PMXMIN']
        self.pixel_max = self.header['Q_PMXMAX']
        self.pixel_wide = self.header['Q_PMWIDE']
        self.output_channels = self.header['Q_PMCHAN']
        self.pm_check = self.header['Q_PM_CK']

    def preprocess_cutData(self, data, dark_calculation=False):
        """
        Preprocesses and extracts specific pixel data from the input data array based on the pixel map.
        """
        pixel_min = self.pixel_min
        pixel_max = self.pixel_max
        pixel_wide = self.pixel_wide
        output_channels = self.output_channels
        traces_loc = self.traces_loc

        Nwave = pixel_max - pixel_min
        window_size = (pixel_wide * 2 + 1)

        add_dimension_for_cubelike_data = False
        if len(data.shape) == 2:
            add_dimension_for_cubelike_data = True
            data = data[None]

        Nimages = data.shape[0] 

        data_cut_pixels = np.zeros((Nimages, output_channels, Nwave, window_size), dtype='uint16')
        data_dark_pixels = np.zeros((Nimages, output_channels - 1, Nwave), dtype='uint16')
        for x in range(Nwave):
            for i in range(output_channels):
                for w in range(pixel_wide*2+1):
                    t=traces_loc[x + pixel_min, i]+w-pixel_wide
                    if t<0:
                        t=0
                    if t>=data.shape[1]:
                        t=data.shape[1]-1
                    data_cut_pixels[:,i,x,w] = data[:, t, x + pixel_min]
                if (i > 0)&(dark_calculation):
                    t=(traces_loc[x + pixel_min, i-1]+traces_loc[x + pixel_min, i])//2+w-pixel_wide
                    data_dark_pixels[:,i-1,x] = data[:, t, x + pixel_min]
        
        if add_dimension_for_cubelike_data:
            data_cut_pixels = data_cut_pixels[0]
            data_dark_pixels = data_dark_pixels[0]

        return data_cut_pixels, data_dark_pixels
