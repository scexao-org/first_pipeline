import numpy as np
from astropy.io import fits
import os
from datetime import datetime
from .runPL_class_pixelMap import PixelMap
from .runPL_class_dataCube import DataCube
from ..libraries import runPL_library_io as runlib_io
from ..libraries import runPL_library_plots as runlib_plots
from astroplan import Observer
from astropy.time import Time


class Preproc:
    """
    A class to handle preprocessed data for the FIRST Visible Photonic Lantern.
    
    Attributes:
        filename (str): Path to the FITS file containing the preprocessed data
        basename (str): Base name of the file
        data (numpy.ndarray): The preprocessed data array
        header (astropy.io.fits.Header): FITS header with processing metadata
        modulation_data (numpy.ndarray): Modulation data if available
        quality_metrics (dict): Quality control metrics
        pixel_map_info (dict): Information about the pixel map used
        is_loaded (bool): Whether the preprocessed data has been loaded
    """
    def __init__(self, filename=None):
        
        self.filename = filename
        self.basename = os.path.basename(filename) if filename else None
        self.data = None
        self.header = None
        self.modulation_hdu = None
        self.modulation_data = None
        self.telemetry_hdu = None
        self.telemetry_data = None
        self.quality_metrics = {}
        self.pixel_map_info = {}
        self.raw_data = None
        self.is_loaded = False
        
        if filename is not None:
            self.load(filename)

    def load(self, filename):
        """
        Load preprocessed data from a FITS file.
        
        Args:
            filename (str): Path to the FITS file containing the preprocessed data
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            KeyError: If required FITS extensions are missing
        """
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Preprocessed data file not found: {filename}")
            
        self.filename = filename
        self.basename = os.path.basename(filename)
        
        with fits.open(filename) as hdul:
            # Load primary data
            self.data = hdul[0].data
            self.header = hdul[0].header.copy()
            
            # Load modulation data if present
            if 'MODULATION' in [hdu.name for hdu in hdul]:
                self.modulation_hdu = hdul['MODULATION'].copy()
            else:   
                self.modulation_hdu = None

            if 'TELEMETRY' in [hdu.name for hdu in hdul]:
                self.telemetry_hdu = hdul['TELEMETRY'].copy()
            else:
                self.telemetry_hdu = None
                
        self.modulation_data = None
        # Handle modulation data
        if self.modulation_hdu is not None:
            self.modulation_data = self.modulation_hdu.data
            self.header['MOD_LEN'] = self.modulation_hdu.header['NAXIS2']

        self.telemetry_data = None
        if self.telemetry_hdu is not None:
            self.telemetry_data = self.telemetry_hdu.data
            self.header['TEL_LEN'] = self.telemetry_hdu.header.get('NAXIS2', 0)


        # Extract quality metrics from header
        self._extract_quality_metrics()
        
        # Extract pixel map information from header
        self._extract_pixel_map_info()
        self.is_loaded = True

    def _check_loaded(self):
        """Check if the preprocessed data is properly loaded."""
        if not self.is_loaded:
            raise ValueError("Preprocessed data not loaded. Use load() or create_from_raw() first.")

    def detect_frame_shifts(self, tolerance=0.01):
        """
        Detect the frame shift between the modulation pattern and the acquired
        frames using the metrology glitch recorded in the telemetry timing.

        The metrology system introduces a deliberate timing glitch (an extra
        delay) on a known modulation frame (X_FIRGFR). By locating that glitch in
        the telemetry log timestamps (ABS_LOG) and comparing the frame index at
        which it is detected, modulo the modulation length, to the expected
        frame, the offset between the modulation pattern and the data frames is
        recovered.

        Args:
            tolerance (float): Half-width, in seconds, used to match the measured
                inter-frame delay to the expected glitch delay. Default 0.01 s.

        Returns:
            dict or None: ``{'frame_shifts': list, 'median_frame_shift': float}``
                if the glitch is enabled and detected; ``None`` if telemetry or
                modulation data is missing, the glitch is off, or no glitch is
                found within the tolerance.
        """
        self._check_loaded()

        # Need telemetry timing and a modulation pattern to compute a frame shift
        if self.telemetry_data is None:
            print(f"WARNING: no telemetry data available in {self.basename}")
            return None
        if self.modulation_data is None:
            print(f"WARNING: no modulation data available in {self.basename}")
            return None

        # Nmod is the length of the modulation array
        Nmod = len(self.modulation_data)

        # getting parameters of the metrology glitch
        glitch_on = self.header.get('X_FIRGON', 0)
        glitch_frame = self.header.get('X_FIRGFR', 0)
        glitch_delay = self.header.get('X_FIRGEX', 0) / 1000  # ms -> s

        if not glitch_on:
            return None

        frame_idx = self.telemetry_data['FRAME_IDX']
        abs_log = self.telemetry_data['ABS_LOG']

        # inter-frame delay with the nominal cadence removed
        dif_minus_median = np.diff(abs_log) - np.median(np.diff(abs_log))

        # Find where the inter-frame delay matches the expected glitch delay
        glitch_mask = np.abs(dif_minus_median - glitch_delay) <= tolerance
        glitch_indices = np.where(glitch_mask)[0]

        if len(glitch_indices) == 0:
            print(f"No glitch detected within +/-{tolerance} s of "
                  f"{glitch_delay} s in {self.basename}")
            return None

        frame_shifts = []
        for glitch_idx in glitch_indices:
            # +1 because diff shifts the index by one
            glitch_detected_frame = frame_idx[glitch_idx + 1] % Nmod
            frame_shifts.append(glitch_detected_frame - glitch_frame)

        median_frame_shift = np.median(frame_shifts)

        return {'frame_shifts': frame_shifts,
                'median_frame_shift': median_frame_shift}

    def _record_frame_shift(self):
        """
        Compute the modulation-to-frame shift from the metrology glitch and store
        its median value in the header under the ``X_FIRGSH`` keyword.

        Does nothing if no shift can be measured (missing telemetry/modulation,
        glitch off, or no glitch detected).
        """
        result = self.detect_frame_shifts()
        if result is not None:
            self.header['X_FIRGSH'] = (
                float(result['median_frame_shift']),
                'median modulation frame shift (frames)'
            )

    def _extract_quality_metrics(self):
        """Extract quality control metrics from the FITS header."""
        self.quality_metrics = {}
        
        # Extract all Q_P_* keywords
        qc_keys = ['Q_P_CENT', 'Q_P_BACK', 'Q_P_BACN', 'Q_P_FLUX', 'Q_P_NAME']
        for key in qc_keys:
            if key in self.header:
                self.quality_metrics[key] = self.header[key]

    def _extract_pixel_map_info(self):
        """Extract pixel map information from the FITS header."""
        self.pixel_map_info = {}
        
        # Extract pixel map related keywords
        pm_keys = ['PIX_MIN', 'PIX_MAX', 'PIX_WIDE', 'OUT_CHAN', 'PM_FILE', 'PM_CHECK']
        for key in pm_keys:
            if key in self.header:
                self.pixel_map_info[key] = self.header[key]
                
        # Extract P_PM_* keywords from pixel map
        for key in self.header.keys():
            if key.startswith('P_PM'):
                self.pixel_map_info[key] = self.header[key]

    def create_from_raw(self, raw_file, pixelMap, output_dir=None, check_if_exist= True, telemetry_txt_file=None):
        """
        Create preprocessed data from a raw FITS file using a pixel map.
        
        Args:
            raw_file (str): Path to the raw FITS file
            pixelMap (PixelMap): PixelMap object
            output_dir (str, optional): Output directory for preprocessed file
            check_if_exist (bool, optional): Whether to check if the preprocessed file already exists and skip processing if so. Default is True.

            
        Returns:
            bool: True if preprocessing was successful, False otherwise
        """
        # Load pixel map
        # Verify pixelMap is of correct class
        if not isinstance(pixelMap, PixelMap):
            raise TypeError(f"pixelMap must be an instance of PixelMap class, got {type(pixelMap)}")
        if not pixelMap.is_loaded:
            raise ValueError("PixelMap must be loaded before use")
        
        
        # Read raw file header and data
        with fits.open(raw_file) as hdul:
            raw_header = hdul[0].header.copy()
        
            # Check for modulation data
            if 'MODULATION' in [hdu.name for hdu in hdul]:
                modID = raw_header.get('X_FIRMID', 0)
                if isinstance(modID, str):
                    modID = int(modID)
                    raw_header['X_FIRMID'] = modID
                if modID > 0:
                    self.modulation_hdu = hdul['MODULATION'].copy()
        
            # Set up Subaru observer for day/night detection
            subaru = Observer.at_site("Subaru")
                
            # Create preprocessed header
            self.header = self._create_preproc_header(raw_header, pixelMap, raw_file)
            
            # Handle day/night time observation classification
            date = self.header.get('DATE', None)
            if date is None:
                date_preproc = datetime.fromtimestamp(os.path.getctime(raw_file)).strftime('%Y-%m-%dT%H:%M:%S')
                self.header['DATE'] = date_preproc
            else:
                obs_time = Time(date)
                # If data taken during daytime, override the OBJECT keyword
                if not subaru.is_night(obs_time):
                    self.header['OBJECT'] = "DAY"
            
            # Generate output filename
            self.basename = runlib_io.create_basename(self.header)
            self.filename = os.path.join(output_dir, self.basename)
            
            if check_if_exist:              
                # Check if file already exists with same PM_CHECK
                if self._should_skip_processing(self.filename, pixelMap.pm_check, self.modulation_hdu):
                    print(f"Skipping {raw_file} - already processed with same pixel map")
                    return False
            
            if (self.modulation_hdu is None) & (self.header.get('X_FIRMID', 0) > 1):
                print(f"Skipping {raw_file} without modulation data...")
                return False
                
            raw_data = hdul[0].data
            # Process data dimensions
            if len(raw_data.shape) == 2:
                raw_data = raw_data[None]

            # Process data using pixel map
            self.data, self.quality_metrics = self._process_raw_data(raw_data, pixelMap)

            # Store raw image for diagnostics (sum over all dimensions except last two)
            self.raw_image = np.sum(raw_data, axis=tuple(range(len(raw_data.shape)-2)))
            
            # Add quality metrics to header
            self._add_quality_metrics_to_header()
            
            # Add pixel map information to header
            self._add_pixel_map_info_to_header(pixelMap)
            
            # Handle modulation data
            if self.modulation_hdu is not None:
                self.modulation_data = self.modulation_hdu.data
                self.header['MOD_LEN'] = self.modulation_hdu.header['NAXIS2']

            self.telemetry_hdu = self._build_telemetry_hdu_from_txt(telemetry_txt_file)
            if self.telemetry_hdu is not None:
                self.telemetry_data = self.telemetry_hdu.data
                self.header['TEL_FILE'] = os.path.basename(telemetry_txt_file)
                self.header['TEL_LEN'] = self.telemetry_hdu.header.get('NAXIS2', 0)
            else:
                self.telemetry_data = None

            self.is_loaded = True

            # Detect and record the modulation-to-frame shift from the metrology glitch
            self._record_frame_shift()

        return self.is_loaded

    def _build_telemetry_hdu_from_txt(self, telemetry_txt_file):
        """Read telemetry timing txt file and return a TELEMETRY binary table HDU."""
        if telemetry_txt_file is None:
            return None
        if not os.path.exists(telemetry_txt_file):
            return None

        rows = []
        with open(telemetry_txt_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                if len(parts) < 7:
                    continue
                try:
                    rows.append((
                        int(parts[0]),
                        int(parts[1]),
                        float(parts[2]),
                        float(parts[3]),
                        float(parts[4]),
                        int(parts[5]),
                        int(parts[6]),
                    ))
                except ValueError:
                    continue

        if len(rows) == 0:
            return None

        telemetry_dtype = np.dtype([
            ('FRAME_IDX', np.int32),
            ('MAIN_IDX', np.int32),
            ('DT_ORIGIN_LOG', np.float64),
            ('ABS_LOG', np.float64),
            ('ABS_ACQ', np.float64),
            ('CNT0_IDX', np.int32),
            ('CNT1_IDX', np.int32),
        ])
        telemetry_data = np.array(rows, dtype=telemetry_dtype)

        hdu = fits.BinTableHDU(data=telemetry_data, name='TELEMETRY')
        hdu.header['TLMTXT'] = (os.path.basename(telemetry_txt_file), 'source telemetry txt file')
        hdu.header['TLMNROW'] = (len(telemetry_data), 'number of telemetry rows')
        return hdu

    def _create_preproc_header(self, raw_header, pixelMap, raw_file):
        """Create the header for the preprocessed file."""
        header = raw_header.copy()
        
        header['X_FIRTYP'] = "PREPROC"
        header['X_FIRWOL'] = raw_header.get('X_FIRWOL', 'OUT')
        header['X_FIRMID'] = int(raw_header['X_FIRMID'])  # for old data reduction
        header['PM_CHECK'] = pixelMap.pm_check

        # Add processing timestamp
        current_time = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
        header['DATE-PRO'] = current_time

        # Add file provenance information
        header['ORG_NAME'] = os.path.basename(raw_file)
        
        return header

    def _should_skip_processing(self, output_file, pm_check, modulation_hdu):
        """Check if file should be skipped based on existing output."""
        if not os.path.exists(output_file):
            return False
            
        with fits.open(output_file) as existing_hdul:
            existing_header = existing_hdul[0].header
            
            # Check if PM_CHECK matches
            if existing_header.get('PM_CHECK') != pm_check:
                return False
                
            # Check modulation consistency
            has_mod_original = modulation_hdu is not None
            has_mod_existing = 'MODULATION' in [hdu.name for hdu in existing_hdul]
            
            if has_mod_original != has_mod_existing:
                return False
                
        return True

    def _process_raw_data(self, raw_data, pixelMap: PixelMap):
        """Process raw data using the pixel map."""
        # Extract data using pixel map
        data_cut_pixels, data_dark_pixels, data_edge_pixels = pixelMap.preprocess_cutData(raw_data, True)
        
        # Calculate background statistics
        perc_background = np.percentile(data_dark_pixels.ravel(), [50-34.1, 50, 50+34.1], axis=0)
        data_mean = np.percentile(np.mean(data_cut_pixels, axis=(1,2)), 90, axis=0)
        
        # Sum the cut data
        data_cut = np.sum(data_cut_pixels, axis=-1, dtype='uint32')

        # add edge pixel
        data_cut += data_edge_pixels.astype('uint32')
        
        # Calculate quality metrics
        flux_mean = np.mean(data_cut, axis=(0,1,2)) - perc_background[1] * (pixelMap.pixel_wide * 2 + 1)
        centered = data_mean.argmax() - pixelMap.pixel_wide
        
        quality_metrics = {
            'Q_P_CENT': centered,
            'Q_P_BACK': perc_background[1],
            'Q_P_BACN': (perc_background[2] - perc_background[0]) / 2 * np.sqrt(2),
            'Q_P_FLUX': flux_mean,
            'Q_P_NAME': self.basename
        }
        
        return data_cut, quality_metrics

    def _add_quality_metrics_to_header(self):
        """Add quality control metrics to the header."""
        self.header['Q_P_CENT'] = (self.quality_metrics['Q_P_CENT'], 'center of extracted window (pixel index)')
        self.header['Q_P_BACK'] = (self.quality_metrics['Q_P_BACK'], 'average background detected')
        self.header['Q_P_BACN'] = (self.quality_metrics['Q_P_BACN'], 'background noise estimate')
        self.header['Q_P_FLUX'] = (self.quality_metrics['Q_P_FLUX'], 'mean extracted flux per pixel (background-subtracted)')
        self.header['Q_P_NAME'] = (self.quality_metrics['Q_P_NAME'], 'output filename of preprocessed data')

    def _add_pixel_map_info_to_header(self, pixelMap):
        """Add pixel map information to the header."""
        self.header['PIX_MIN'] = pixelMap.pixel_min
        self.header['PIX_MAX'] = pixelMap.pixel_max
        self.header['PIX_WIDE'] = pixelMap.pixel_wide
        self.header['OUT_CHAN'] = pixelMap.output_channels
        self.header['PM_FILE'] = pixelMap.basename
        self.header['PM_CHECK'] = pixelMap.pm_check
        
        # Add all P_PM_* keywords from pixel map header
        for key in pixelMap.header.keys():
            if key.startswith('P_PM'):
                self.header[key] = pixelMap.header[key]

    def generate_diagnostic_figures(self, pixelMap):

        self._check_loaded()
        """Generate diagnostic figures for the preprocessed data."""
        # Create trace overlay figure
        fig, ax = runlib_io.make_figure_of_trace(self.raw_image, pixelMap.traces_loc, 
                                                 pixelMap.pixel_wide, pixelMap.pixel_min, 
                                                 pixelMap.pixel_max)
        fig.savefig(self.filename[:-5] + "_1.png", dpi=250)
        
        # Create coupling map figure if modulation data is available
        if (self.modulation_data is not None and 
            self.header.get('X_FIRMID', 0) > 1 and 
            len(self.modulation_data['XMOD']) > 9):
            
            # print("toto",self.data.shape)
            # Recompute cut pixels for coupling map
            fluxes = self.data.mean(axis=(1,2))
            
            xmod = self.modulation_data['XMOD']
            ymod = self.modulation_data['YMOD']
            
            fig = runlib_plots.plot_flux_map(fluxes, xmod, ymod)
            string_title = (f"{self.header['OBJECT']} - {self.header['DATA-TYP']} - "
                          f"{self.header['EXPTIME']}s\n"
                          f"X_FIROBX = {self.header.get('X_FIROBX', 'N/A')}, "
                          f"X_FIROBY = {self.header.get('X_FIROBY', 'N/A')}\n"
                          f"number of DIT to be shifted = {self.header.get('X_FIRGSH', 'N/A')}")
            fig.suptitle(string_title)
            fig.savefig(self.filename[:-5] + "_2.png", dpi=250)

    def save(self, header=None):
        self._check_loaded()
        """
        Save the preprocessed data to a FITS file.
        
        Args:
            output_filename (str): Path for the output FITS file
            header (astropy.io.fits.Header, optional): Additional header information
            
        Raises:
            ValueError: If no preprocessed data is available to save
        """
        if self.data is None:
            raise ValueError("No preprocessed data to save. Load or create preprocessed data first.")

        # Use the instance header as base
        file_header = self.header.copy() if self.header is not None else fits.Header()
        
        # Merge additional header if provided
        if header is not None:
            file_header.extend(header, strip=True)

        file_header['X_FIRTYP'] = "PREPROC"
        file_header['X_FIRWOL'] = file_header.get('X_FIRWOL', 'UNKNOWN')
            
        # Create primary HDU with preprocessed data
        primary_hdu = fits.PrimaryHDU(data=self.data, header=file_header)
        hdu_list = [primary_hdu]
        
        # Add modulation data if available
        if self.modulation_hdu is not None:
            hdu_list.append(self.modulation_hdu)

        if self.telemetry_hdu is not None:
            hdu_list.append(self.telemetry_hdu)
        
        # Create HDU list
        hdul = fits.HDUList(hdu_list)
        
        # Write to file
        print(f"Saving preprocessed data to {self.filename}")
        hdul.writeto(self.filename, overwrite=True, output_verify='fix', checksum=True)

    def return_hdu_list(self):
        self._check_loaded()
        """
        Return a list of FITS HDUs representing the preprocessed data.
        
        Returns:
            list: List of FITS HDUs containing preprocessed data
            
        Raises:
            ValueError: If no preprocessed data is available
        """
        if self.data is None:
            raise ValueError("No preprocessed data available")
            
        hdu_list = [fits.PrimaryHDU(data=self.data, header=self.header)]
        
        if self.modulation_data is not None:
            modulation_hdu = fits.BinTableHDU(data=self.modulation_data, name='MODULATION')
            hdu_list.append(modulation_hdu)

        if self.telemetry_data is not None:
            telemetry_hdu = fits.BinTableHDU(data=self.telemetry_data, name='TELEMETRY')
            hdu_list.append(telemetry_hdu)
            
        return hdu_list

    def return_header(self):
        """
        Return the header of the FITS file.
        
        Returns:
            astropy.io.fits.Header: The header of the FITS file or empty header if none available
        """
        if self.header is not None:
            return self.header.copy()
        else:
            return fits.Header()

    def get_quality_summary(self):
        self._check_loaded()
        """
        Get a summary of quality control metrics.
        
        Returns:
            dict: Dictionary containing quality metrics and their interpretations
        """
        summary = {
            'quality_metrics': self.quality_metrics,
            'pixel_map_info': self.pixel_map_info
        }
        
        if self.quality_metrics:
            summary['interpretations'] = {
                'centroid_shift': f"Extraction window center at pixel {self.quality_metrics.get('Q_P_CENT', 'N/A')}",
                'background_level': f"Background: {self.quality_metrics.get('Q_P_BACK', 'N/A'):.2f}",
                'background_noise': f"Background noise: {self.quality_metrics.get('Q_P_BACN', 'N/A'):.2f}",
                'flux_level': f"Mean flux: {self.quality_metrics.get('Q_P_FLUX', 'N/A'):.2f}"
            }
            
        return summary

    def has_modulation_data(self):
        self._check_loaded()
        """
        Check if the preprocessed data includes modulation information.
        
        Returns:
            bool: True if modulation data is available, False otherwise
        """
        return self.modulation_data is not None