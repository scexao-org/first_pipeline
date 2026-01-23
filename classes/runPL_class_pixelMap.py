from astropy.io import fits
import numpy as np
import os
from typing import Optional, Tuple, List
from numpy.typing import NDArray

class PixelMap:
    """
    A class to handle pixel maps for the FIRST Visible Photonic Lantern.
    
    Attributes:
        filename (str): Path to the FITS file containing the pixel map
        basename (str): Base name of the file
        header (astropy.io.fits.Header): FITS header information
        traces_loc (numpy.ndarray): Trace locations data array
        pixel_min (int): Minimum pixel value
        pixel_max (int): Maximum pixel value
        pixel_wide (int): Pixel width parameter
        output_channels (int): Number of output channels
        pm_check (int): Pixel map checksum value
        is_loaded (bool): Whether the pixel map data has been loaded
    """
    def __init__(self, filename: Optional[str] = None) -> None:
        self.filename = filename
        self.basename = os.path.basename(filename) if filename else None
        self.header = None
        self.traces_loc = None
        self.edge_coef = None
        self.pixel_min = None
        self.pixel_max = None
        self.pixel_wide = None
        self.output_channels = None
        self.pm_check = None
        self.is_loaded = False
        
        if filename is not None:
            self.load(filename)

    def load(self, filename: str) -> None:
        """
        Load the pixel map from a FITS file.
        
        Args:
            filename (str): Path to the FITS file containing the pixel map
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            KeyError: If required FITS header keywords are missing
        """
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Pixel map file not found: {filename}")
            
        self.filename = filename
        self.basename = os.path.basename(filename)
        self.header = fits.getheader(filename)
        self.traces_loc = fits.getdata(filename)
        
        # Check for required header keywords and raise error if not found
        required_keys = ['Q_PMXMIN', 'Q_PMXMAX', 'Q_PMWIDE', 'Q_PMCHAN', 'Q_PM_CK']
        missing_keys = [key for key in required_keys if key not in self.header]
        if missing_keys:
            raise KeyError(f"FITS header keywords missing in Pixel Map: {missing_keys}")
        
        if self.traces_loc is None or self.traces_loc.size == 0:
            raise ValueError("Pixel map data is empty or invalid")
        
        # Check if there are additional HDUs containing edge coefficients
        with fits.open(filename) as hdul:
            if len(hdul) > 1:
                for hdu in hdul[1:]:
                    if hdu.name == 'EDGE_COEF' and hdu.data is not None:
                        self.edge_coef = hdu.data
                        break


        self.pixel_min = self.header['Q_PMXMIN']
        self.pixel_max = self.header['Q_PMXMAX']
        self.pixel_wide = self.header['Q_PMWIDE']
        self.output_channels = self.header['Q_PMCHAN']
        self.pm_check = self.header['Q_PM_CK']
        self.is_loaded = True

    def _check_loaded(self) -> None:
        """Check if the pixel map is properly loaded."""
        if not self.is_loaded:
            raise ValueError("Pixel map not loaded. Use load() or create_from_data() first.")

    def create_from_data(self, traces_loc: NDArray[np.integer], edge_coef: NDArray[np.floating], pixel_min: int, pixel_max: int, pixel_wide: int, output_channels: int, pm_check: Optional[int] = None, filename: Optional[str] = None) -> None:
        """
        Create a pixel map from data arrays and parameters.
        Args:
            traces_loc (numpy.ndarray): The traces location data array.
            edge_coef (numpy.ndarray): The edge coefficient data array.
            pixel_min (int): Minimum pixel value.
            pixel_max (int): Maximum pixel value.
            pixel_wide (int): Pixel width.
            output_channels (int): Number of output channels.
            pm_check (int, optional): Pixel map check value. If None, a random value is generated.
            filename (str, optional): Optional filename to associate with this pixel map.
        """

        if traces_loc is None or traces_loc.size == 0:
            return

        self.edge_coef = edge_coef
        self.traces_loc = traces_loc
        self.pixel_min = pixel_min
        self.pixel_max = pixel_max
        self.pixel_wide = pixel_wide
        self.output_channels = output_channels
        self.pm_check = pm_check if pm_check is not None else np.random.randint(0, 2**32, dtype=np.uint32)
        
        # Create a basic header
        self.header = fits.Header()
        self.header['Q_PMXMIN'] = pixel_min
        self.header['Q_PMXMAX'] = pixel_max
        self.header['Q_PMWIDE'] = pixel_wide
        self.header['Q_PMCHAN'] = output_channels
        self.header['Q_PM_CK'] = self.pm_check
        
        if filename is not None:
            self.filename = filename
            self.basename = os.path.basename(filename)
        else:
            self.filename = None
            self.basename = None
        
        self.is_loaded = True

    def save(self, output_filename: str, header: Optional[fits.Header] = None) -> None:
        """
        Save the pixel map to a FITS file.
        
        Args:
            output_filename (str): Path for the output FITS file
            header (astropy.io.fits.Header, optional): Additional header information
            
        Raises:
            ValueError: If no pixel map data is available to save
        """
        from datetime import datetime
        
        self._check_loaded()
        if self.traces_loc is None:
            raise ValueError("No pixel map data to save. Load or create pixel map data first.")
            
        # Handle case when traces_loc is None (failed pixelmap generation)
        traces_loc_data = self.traces_loc.copy()
        edge_coef = self.edge_coef
        
        # Create HDU
        hdu = fits.PrimaryHDU(traces_loc_data)
        
        # Use provided header or create from internal data
        if header is not None:
            save_header = header.copy()
        elif self.header is not None:
            save_header = self.header.copy()
        else:
            save_header = fits.Header()
            
        # Add required keywords
        save_header['X_FIRTYP'] = 'PIXELMAP'
            
        # Add date and time to the header if not present
        if 'DATE-PRO' not in save_header:
            current_time = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
            save_header['DATE-PRO'] = current_time
            
        if 'DATE' not in save_header:
            save_header['DATE'] = save_header.get('DATE-PRO', datetime.now().strftime('%Y-%m-%dT%H:%M:%S'))

        # Add pixel map parameters
        save_header['Q_PMXMIN'] = self.pixel_min
        save_header['Q_PMXMAX'] = self.pixel_max
        save_header['Q_PMWIDE'] = self.pixel_wide
        save_header['Q_PMCHAN'] = self.output_channels
        save_header['Q_PM_CK'] = self.pm_check
        
        # Add filename if not present
        if 'Q_PMNAME' not in save_header:
            import libraries.runPL_library_io as runlib_io
            basename = runlib_io.create_basename(save_header)
            save_header['Q_PMNAME'] = basename
            
        hdu.header.extend(save_header, strip=True)
        hdul = fits.HDUList([hdu])

        if edge_coef is not None:
            edge_hdu = fits.ImageHDU(data=edge_coef, name='EDGE_COEF')
            hdul.append(edge_hdu)
        
        # Write to a FITS file
        print(f"Saving pixel map to {output_filename}")
        hdul.writeto(output_filename, overwrite=True)
        hdul.close()

    def return_hdu_list(self) -> List[fits.ImageHDU]:
        """
        Return a list of FITS HDUs representing the pixel map.
        
        Returns:
            list: List of FITS HDUs containing pixel map data
            
        Raises:
            ValueError: If no pixel map data is available
        """
        if self.traces_loc is None:
            raise ValueError("No pixel map data available")
        self._check_loaded()
        hdu = [fits.ImageHDU(data=self.traces_loc, name='PIXELMAP')]
        return hdu
    
    def return_header(self) -> fits.Header:
        """
        Return the header of the FITS file.
        
        Returns:
            astropy.io.fits.Header: The header of the FITS file or empty header if none available
        """
        if self.header is not None:
            return self.header.copy()
        elif self.filename is not None:
            with fits.open(self.filename) as hdul:
                header = hdul[0].header
            return header
        else:
            # Return empty header if no file is loaded
            return fits.Header()

    def preprocess_cutData(self, data: NDArray[np.integer], dark_calculation: bool = False) -> Tuple[NDArray[np.uint16], NDArray[np.uint16], NDArray[np.floating]]:
        """
        Preprocesses and extracts specific pixel data from the input data array based on the pixel map.
        """
        self._check_loaded()
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

        data_edge_pixels = np.zeros((Nimages, output_channels, Nwave))
        if self.edge_coef is not None:
            edge_pixel_below = traces_loc.copy() - pixel_wide - 1
            edge_pixel_above = traces_loc.copy() + pixel_wide + 1
            edge_pixel_below[edge_pixel_below < 0] = 0
            edge_pixel_above[edge_pixel_above >= data.shape[1]] = data.shape[1] - 1

            edge_coef_above = 0.5+self.edge_coef
            edge_coef_below = 0.5-self.edge_coef

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
                if self.edge_coef is not None:
                    t=edge_pixel_below[x + pixel_min, i]
                    data_edge_pixels[:,i,x] = data[:, t, x + pixel_min] * edge_coef_below[x + pixel_min, i]
                    t=edge_pixel_above[x + pixel_min, i]
                    data_edge_pixels[:,i,x] += data[:, t, x + pixel_min] * edge_coef_above[x + pixel_min, i]

        if add_dimension_for_cubelike_data:
            data_cut_pixels = data_cut_pixels[0]
            data_dark_pixels = data_dark_pixels[0]
            data_edge_pixels = data_edge_pixels[0]

        return data_cut_pixels, data_dark_pixels, data_edge_pixels