import numpy as np
from astropy.io import fits
from .runPL_class_dataCube import DataCube
import os

class WaveMap:
    """
    A class to handle wavelength maps for the FIRST Visible Photonic Lantern.
    
    Attributes:
        filename (str): Path to the FITS file containing the wavelength map
        basename (str): Base name of the file
        Nwave (int): Number of wavelength channels
        wave (numpy.ndarray): Wavelength data array
        index (numpy.ndarray): Index data array for interpolation
        weights (numpy.ndarray): Weights data array for interpolation
        wave_label (str): Label for wavelength units (states air or vacuum)
        medium (str): 'air' (standard air) or 'vacuum' 
        is_loaded (bool): Whether the wavelength map data has been loaded
    """
    def __init__(self, filename=None):

        self.filename = filename
        self.basename = os.path.basename(filename) if filename else None
        self.Nwave = None
        self.wave = None
        self.index = None
        self.weights = None
        self.medium = 'air'
        self.npixel = None          # number of raw pixels the map applies to
        self.interpolation = None   # 'linear' or 'lanczos3'
        self.wave_label = self._make_label(self.medium)
        self.is_loaded = False
        
        if filename is not None:
            self.load(filename)

    # FITS spectral-axis codes: AWAV = wavelength in air, WAVE = in vacuum
    _MEDIUM_TO_CODE = {'air': 'AWAV', 'vacuum': 'WAVE'}
    _CODE_TO_MEDIUM = {'AWAV': 'air', 'WAVE': 'vacuum'}

    @staticmethod
    def _make_label(medium):
        return f"Wavelength in {medium} (nm)"

    def _set_medium(self, medium):
        if medium not in self._MEDIUM_TO_CODE:
            raise ValueError(f"medium must be 'air' or 'vacuum', got {medium!r}")
        self.medium = medium
        self.wave_label = self._make_label(medium)

    def load(self, filename):
        """
        Load the wavelength map from a FITS file.
        
        Args:
            filename (str): Path to the FITS file containing the wavelength map
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            KeyError: If required FITS extensions are missing
        """
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Wavelength map file not found: {filename}")
            
        self.filename = filename
        self.basename = os.path.basename(filename)
        
        # Read the FITS file
        with fits.open(filename) as hdul:
            required_extensions = ['WAVELENGTH', 'INDEX', 'WEIGHT']
            available_extensions = [hdu.name for hdu in hdul]
            missing_extensions = [ext for ext in required_extensions if ext not in available_extensions]
            
            if missing_extensions:
                raise KeyError(f"FITS file missing required extensions: {missing_extensions}")
                
            self.Nwave = hdul['WAVELENGTH'].data.shape[0]
            self.wave = hdul['WAVELENGTH'].data
            self.index = hdul['INDEX'].data
            self.weights = hdul['WEIGHT'].data
            # Air/vacuum flag: read from the WAVELENGTH extension first, so it
            # also works when the map is embedded in a coupling map file.
            # Maps made before this keyword existed used the standard-air
            # Neon catalogue, hence the 'AWAV' default.
            code = hdul['WAVELENGTH'].header.get('Q_WMSYS',
                                                  hdul[0].header.get('Q_WMSYS', 'AWAV'))
            self._set_medium(self._CODE_TO_MEDIUM.get(code, 'air'))
            ext_header = hdul['WAVELENGTH'].header
            self.npixel = ext_header.get('Q_WMNPIX', None)
            self.interpolation = ext_header.get('Q_WMINTP',
                                                'linear' if self.index.shape[0] == 2 else None)
            
        if self.wave is None or self.wave.size == 0:
            raise ValueError("Wavelength map data is empty or invalid")
        
        self.is_loaded = True

    def create_from_data(self, wave, index, weights, filename=None, medium='air',
                         npixel=None, interpolation=None):
        """
        Create a wavelength map from data arrays.
        Args:
            wave (numpy.ndarray): The wavelength data array.
            index (numpy.ndarray): The index data array.
            weights (numpy.ndarray): The weights data array.
            filename (str, optional): Optional filename to associate with this wavelength map.
            medium (str, optional): 'air' (default, standard air) or 'vacuum'.
            npixel (int, optional): number of raw pixels along the spectrum.
            interpolation (str, optional): kernel used for index/weights.
        """
        self._set_medium(medium)
        self.npixel = npixel
        self.interpolation = interpolation
        self.wave = wave
        self.index = index
        self.weights = weights
        self.Nwave = wave.shape[0] if wave is not None else None
        
        if filename:
            self.filename = filename
            self.basename = os.path.basename(filename)
        else:
            self.filename = None
            self.basename = None

        self.is_loaded = True
            
    def _check_loaded(self):
        """Check if the wavelength map is properly loaded."""
        if not self.is_loaded:
            raise ValueError("Wavelength map not loaded. Use load() or create_from_data() first.")
    
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
        
        self._check_loaded()
        if self.wave is None or self.index is None or self.weights is None:
            raise ValueError("No wavelength map data to save. Load or create wavelength map data first.")
            
        # Create a primary HDU with no data, just the header
        hdu_primary = fits.PrimaryHDU()

        # Create HDUs for the wavelength map data
        hdu = [self._wavelength_hdu()]
        hdu += [fits.ImageHDU(data=self.index, name='INDEX')]
        hdu += [fits.ImageHDU(data=self.weights, name='WEIGHT')]

        if header is not None:
            # Processing date of THIS product (always reset: the input header may carry
            # the DATE-PRO of the preprocessed file, and the most recent product is
            # selected downstream by DATE-PRO)
            current_time = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
            header['DATE-PRO'] = current_time

            hdu_primary.header.extend(header, strip=True)

        hdu_primary.header['X_FIRTYP'] = 'WAVEMAP'
        hdu_primary.header['Q_WMSYS'] = (self._MEDIUM_TO_CODE[self.medium],
                                         'AWAV = standard air, WAVE = vacuum')
        # Combine all HDUs into an HDUList
        hdul = fits.HDUList([hdu_primary, *hdu])

        # Write to a FITS file
        print(f"Saving wavelength map to {output_filename}")
        from first_pipeline_shared.version import add_version_keywords
        add_version_keywords(hdul[0].header)   # Q_PIPVER / Q_PIPGIT
        hdul.writeto(output_filename, overwrite=True)

        self.basename = os.path.basename(output_filename)
        self.filename = output_filename


    def interpolate_data(self, dataCube):
        """
        Resample data onto the common wavelength grid of the map.

        Each output o is interpolated with the precomputed pixel indices and
        weights, arrays of shape (Ntap, Noutput, Nwave) (Ntap = 2 for linear,
        6 for Lanczos-3 interpolation):
        new[k] = sum_j w[j,o,k] * data[index[j,o,k]].
        The variance is propagated as sum_j w[j,o,k]**2 * var[index[j,o,k]]
        (raw pixels assumed independent; note that the interpolation itself
        correlates adjacent output channels, which is not tracked).
        Pixels entering with a zero weight are ignored, so a NaN pixel only
        affects the channels that actually use it.

        Args:
            dataCube (DataCube or numpy.ndarray): data with the wavelength
                (pixel) axis last and the output axis just before it, i.e.
                shape (..., Noutput, Npixel).
                - DataCube: its data and variance are resampled in place and
                  its wave, Nwave and wave_label attributes are updated.
                  Nothing is returned.
                - numpy.ndarray: a new resampled array of shape
                  (..., Noutput, Nwave) is returned; the input is unchanged.
        """
        self._check_loaded()

        is_dataCube = isinstance(dataCube, DataCube)
        data = dataCube.data if is_dataCube else np.asarray(dataCube)
        variance = dataCube.variance if is_dataCube else None

        if data.ndim < 2:
            raise ValueError(f"Data must have at least 2 dimensions (..., Noutput, Npixel), got shape {data.shape}")
        Noutput, Npixel = data.shape[-2:]
        if self.npixel is not None:
            if Npixel != self.npixel:
                raise ValueError(f"Wavelength map size mismatch: expected {self.npixel} pixels, got {Npixel}")
        elif self.index.max() != Npixel - 1:   # older maps without Q_WMNPIX
            raise ValueError(f"Wavelength map size mismatch: expected {self.index.max() + 1} pixels, got {Npixel}")
        if self.index.shape[1] != Noutput:
            raise ValueError(f"Wavelength map has {self.index.shape[1]} outputs, data has {Noutput}")

        def resample(array, power):
            out = np.zeros(array.shape[:-2] + (Noutput, self.Nwave),
                           dtype=np.result_type(array.dtype, self.weights.dtype))
            for o in range(Noutput):
                for j in range(self.index.shape[0]):
                    w = self.weights[j, o, :]
                    used = w != 0
                    contribution = array[..., o, self.index[j, o, :]] * w**power
                    # zero-weight pixels must not propagate NaNs
                    out[..., o, :] += np.where(used, contribution, 0.0)
            return out

        new_data = resample(data, 1)

        if not is_dataCube:
            return new_data

        dataCube.data = new_data
        if variance is not None:
            dataCube.variance = resample(variance, 2)
        dataCube.wave_label = self.wave_label
        dataCube.Nwave = self.Nwave
        dataCube.wave = self.wave
        
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
        self._check_loaded()
        hdu = [self._wavelength_hdu()]
        hdu += [fits.ImageHDU(data=self.index, name='INDEX')]
        hdu += [fits.ImageHDU(data=self.weights, name='WEIGHT')]
        return hdu
    
    def _wavelength_hdu(self):
        """WAVELENGTH extension, tagged with its unit and medium."""
        hdu = fits.ImageHDU(data=self.wave, name='WAVELENGTH')
        hdu.header['BUNIT'] = ('nm', 'wavelength unit')
        if self.npixel is not None:
            hdu.header['Q_WMNPIX'] = (int(self.npixel), 'number of raw pixels along the spectrum')
        if self.interpolation is not None:
            hdu.header['Q_WMINTP'] = (self.interpolation, 'interpolation kernel of INDEX/WEIGHT')
        hdu.header['Q_WMSYS'] = (self._MEDIUM_TO_CODE[self.medium],
                                 'AWAV = standard air, WAVE = vacuum')
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
    
