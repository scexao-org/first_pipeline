import numpy as np
from astropy.io import fits
import os

class FlatMap:
    """
    A class to handle flat field maps for the FIRST Visible Photonic Lantern.
    
    Attributes:
        filename (str): Path to the FITS file containing the flat field map
        basename (str): Base name of the file
        flat (numpy.ndarray): The flat field data array
        inv_flat (numpy.ndarray): Inverse of the flat field for normalization
        inv_flat_squared (numpy.ndarray): Squared inverse for variance normalization
        is_loaded (bool): Whether the flat field map data has been loaded
    """
    def __init__(self, filename=None):

        self.filename = filename
        self.basename = os.path.basename(filename) if filename else None
        self.flat = None
        self.inv_flat = None
        self.inv_flat_squared = None
        self.is_loaded = False
        
        if filename is not None:
            self.load(filename)

    def load(self, filename):
        """
        Load the flat field map from a FITS file.
        
        Args:
            filename (str): Path to the FITS file containing the flat field map
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            KeyError: If required FITS extensions are missing
        """
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Flat field map file not found: {filename}")
            
        self.filename = filename
        self.basename = os.path.basename(filename)
        
        with fits.open(filename) as hdul:
            if 'FLAT' not in [hdu.name for hdu in hdul]:
                raise KeyError("FITS file missing required 'FLAT' extension")
            self.flat = hdul['FLAT'].data

        if self.flat is None or self.flat.size == 0:
            raise ValueError("Flat field data is empty or invalid")
            
        self.inv_flat = 1.0 / self.flat
        self.inv_flat_squared = 1.0 / self.flat**2
        self.is_loaded = True

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
            self.filename = filename
            self.basename = os.path.basename(filename)
        else:
            self.filename = None
            self.basename = None

        self.is_loaded = True
            
    def _check_loaded(self):
        """Check if the flat field map is properly loaded."""
        if not self.is_loaded:
            raise ValueError("Flat field map not loaded. Use load() or create_from_data() first.")
        
    def _validate_data(self):
        """
        Validate that the flat field data is properly loaded and consistent.
        
        Raises:
            ValueError: If data is invalid or inconsistent
        """
        if self.flat is None:
            raise ValueError("No flat field data loaded")
        if self.flat.size == 0:
            raise ValueError("Flat field data is empty")
        if np.any(self.flat <= 0):
            raise ValueError("Flat field contains zero or negative values")

    def save(self, output_filename, header=None):
        """
        Save the flat field map to a FITS file.
        
        Args:
            output_filename (str): Path for the output FITS file
            header (astropy.io.fits.Header, optional): Additional header information
            
        Raises:
            ValueError: If no flat field data is available to save
        """
        from datetime import datetime
        
        self._check_loaded()
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
                
            hdu_primary.header.extend(header, strip=True)

        hdu_primary.header['X_FIRTYP'] = 'FLATMAP'
        # Combine all HDUs into an HDUList
        hdul = fits.HDUList([hdu_primary, *hdu])

        # Write to a FITS file
        print(f"Saving flat field map to {output_filename}")
        hdul.writeto(output_filename, overwrite=True)

        self.filename = output_filename
        self.basename = os.path.basename(output_filename)
        self.is_loaded = True


    def normalize_with_flat(self, dataCube):
        """
        Normalize the data cube by a flat field.
        Args:
                dataCube (DataCube): The data cube to normalize.
        Raises:
            ValueError: If the flat field trailing dimensions are not
                broadcast-compatible with the data cube.
        """
        self._check_loaded()
        flat_shape = self.flat.shape
        data_shape = dataCube.data.shape[-self.flat.ndim:]
        if not all(f in (1, d) for f, d in zip(flat_shape, data_shape)):
            raise ValueError(
                f"Flat field shape {flat_shape} is not broadcast-compatible "
                f"with data cube trailing dimensions {data_shape}"
            )
        dataCube.data *= self.inv_flat
        dataCube.variance *= self.inv_flat_squared
    
    def return_hdu_list(self):
        """
        Return a list of FITS HDUs representing the flat field map.
        
        Returns:
            list: List of FITS HDUs containing flat field data
            
        Raises:
            ValueError: If no flat field data is available
        """
        if self.flat is None:
            raise ValueError("No flat field data available")
        self._check_loaded()
        hdu = [fits.ImageHDU(data=self.flat, name='FLAT')]
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