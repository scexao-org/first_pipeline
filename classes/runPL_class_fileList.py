import os
import numpy as np
from astropy.io import fits
from glob import glob
from datetime import datetime
from classes.runPL_class_dataCube import DataCube
from classes.runPL_class_flatMap import FlatMap
from classes.runPL_class_waveMap import WaveMap

def clean_filelist(fits_keywords, filelist):
    filelist_cleaned = []
    if isinstance(filelist, str):
        filelist = [filelist]
    for filename in filelist:
        try:
            first_file = fits.open(filename, memmap=True)
        except:
            continue
        header = first_file[0].header.copy()
        first_file.close()
        del first_file
        
        # Data files with the correct keywords only
        key_names = list(fits_keywords.keys())
        type_ok = True
        for strname in key_names:
            type_ok *= (strname in header)

        if not type_ok:
            continue

        keys_ok = True
        for name in key_names:
            keys_ok *= (header[name] in fits_keywords[name])

        if not keys_ok:
            continue

        filelist_cleaned.append(filename)
    
    filelist_cleaned = np.array(filelist_cleaned)
    return np.sort(filelist_cleaned)

def get_filelist(file_patterns, fits_keywords, name_search=None):
    """
    Find files based on the given parameters.

    Args:
        file_patterns (list): List of file patterns to process (e.g., ["*.fits"]).
        fits_keywords (dict): Dictionary of FITS keywords to filter files.

    Returns:
        list: A list of files to process.

        Note: If no files are found, a FileNotFoundError is raised.
    """

    if name_search is not None:
        str_search = "for " + str(name_search).upper()
    else:
        str_search=""

    filelist = []

    if isinstance(file_patterns, str):
        file_patterns = [file_patterns]

    print('file_patterns',file_patterns)
    # If file patterns are provided, use glob to find matching files
    for pattern in file_patterns:

        if os.path.isdir(pattern):
            file_path = os.path.join(pattern, '*.fits')
        else:
            file_path = pattern

        filelist += glob(file_path)

    # Filter out non-fits files
    filelist = [file for file in filelist if file.endswith('.fits')]
    
    if len(filelist) == 0:
        raise FileNotFoundError("No fits files found "+str_search+" with the specified patterns")

    if fits_keywords is not None:
        # If fits_keywords is provided, filter the file list based on the keywords
        print("Looking for files with the correct keywords :",fits_keywords)
        filelist_filtered = clean_filelist(fits_keywords, filelist)

        if len(filelist_filtered) == 0:
            for key in list(fits_keywords.keys()):
                test_keywords = fits_keywords.copy()
                test_keywords.pop(key)
                test_filelist = clean_filelist(test_keywords, filelist)
                if len(test_filelist) > 0:
                    continue
            raise FileNotFoundError(f"No fits files found "+str_search+f", Keyword {key}={fits_keywords[key]} may be too restrictive.")

        filelist = filelist_filtered

    # Sort the file list for consistent processing order
    filelist.sort()
    return filelist

def find_closest_pixelmap(raw, filelist_pixelmap):
    """
    Find the closest pixel map file for a given raw data file.
    The closest pixel map is determined by the wollaston status, and date.
    """
    header = fits.getheader(raw)
    raw_wollaston = header.get('X_FIRWOL', 'UNKNOWN')
    raw_dir = os.path.dirname(raw)

    # Filter pixel maps by wollaston status
    pixelmaps_filtered = [pm for pm in filelist_pixelmap if fits.getheader(pm).get('X_FIRWOL', 'UNKNOWN') == raw_wollaston]
    # If no pixel map found, return None
    if not pixelmaps_filtered:
        return None

    # Sort by date and return the most recent one
    pixelmaps_filtered.sort(key=lambda pm: fits.getheader(pm).get('DATE-PRO', '1970-01-01'))
    return pixelmaps_filtered[-1]

def find_closest_in_time_dark(cmap_file, dark_files):
    """
    Finds the closest dark file to a given coupling map file based on the 'DATE' FITS keyword.
    """

    cmap_date = fits.getheader(cmap_file)['DATE']
    
    # find the closest by date
    dark_dates = [(dark, fits.getheader(dark)['DATE']) for dark in dark_files]
    dark_dates.sort(key=lambda x: abs(datetime.strptime(x[1], '%Y-%m-%dT%H:%M:%S') - datetime.strptime(cmap_date, '%Y-%m-%dT%H:%M:%S')))
    
    try:
        return dark_dates[0][0]  # Return the closest dark file by date
    except:
        return None

def find_closest_dark(cmap_file, dark_files):
    """
    Finds the closest dark file to a given coupling map file, prioritizing files in the same directory.
    """

    cmap_dir = os.path.dirname(cmap_file)
    dark_samegain = [dark for dark in dark_files if fits.getheader(dark)['GAIN'] == fits.getheader(cmap_file)['GAIN']]
    
    # Filter dark files by the same directory
    same_dir_darks = [dark for dark in dark_samegain if os.path.dirname(dark) == cmap_dir]
    
    if same_dir_darks:
        return find_closest_in_time_dark(cmap_file, same_dir_darks)  # Return the first match in the same directory    
    else:
        return find_closest_in_time_dark(cmap_file, dark_samegain) 

class FileList:
    """
    Class to handle file listing with constraints for wavelength mapping pipeline.
    """
    
    def __init__(self, file_patterns, data_type= None, first_type= None, 
                 wollaston=None, object_name=None, modID=None, modScale=None):
        """
        Initialize FileList with constraints.
        
        Parameters:
        -----------
        file_patterns : str or list, optional
            File patterns to search for FITS files
        data_type : str
            Expected value for the DATA-TYP FITS header keyword
        first_type : str
            Expected value for the X_FIRTYP FITS header keyword
        dark_patterns : str or list, optional
            File patterns to search for dark files
        flat_patterns : str or list, optional
            File patterns to search for flat files
        wollaston : str, optional
            Wollaston status ('IN' or 'OUT')
        object_name : str, optional
            Object name constraint
        modID : str, optional
            Modulator ID constraint
        modScale : str, optional
            Modulator scale constraint
        """
        self.file_patterns = file_patterns or ['*.fits']

        self.fits_keywords = {}

        if data_type is not None:
            self.fits_keywords['DATA-TYP'] = [data_type]
        if first_type is not None:
            self.fits_keywords['X_FIRTYP'] = [first_type]
        if wollaston is not None:
            self.fits_keywords['X_FIRWOL'] = [wollaston]
        if object_name is not None:
            self.fits_keywords['OBJECT'] = [object_name]
        if modID is not None:
            self.fits_keywords['X_FIRMID'] = [modID]
        if modScale is not None:
            self.fits_keywords['X_FIRMSC'] = [modScale]

        print("----------------")
        # Note : get_filelist will raise FileNotFoundError if no files are found
        # it can be caught by the calling function if needed
        filelist = get_filelist(self.file_patterns, self.fits_keywords)

        if wollaston is not None:
            print(f"Selected wollaston={wollaston}")
        if modID is not None:
            print(f"Selected modID={modID}")
        if modScale is not None:
            print(f"Selected modScale={modScale}")
        if object_name is not None: 
            print(f"Selected object name={object_name}")
        print(f"Found {len(filelist)} files matching criteria.")
        print("----------------")
        self.filelist = filelist
        self.header = fits.getheader(self.filelist[0])

    def get_most_common_dir(self):
        # now get the directory where most of the files are located
        dirs = [os.path.dirname(file) for file in self.filelist]
        most_common_dir = max(set(dirs), key=dirs.count)
        print(f"Most files are located in: {most_common_dir}")

        return most_common_dir

    def get_flatmap_file(self, file_patterns):
        """
        Get flat map file matching constraints.
        
        Returns:
        --------
        str : flat map file path
        """
        fits_keywords= {'X_FIRTYP': ['FLATMAP','COUPLINGMAP']}
        if self.fits_keywords.get('X_FIRWOL') is not None:
            fits_keywords['X_FIRWOL'] = self.fits_keywords['X_FIRWOL']

        try:
            filelist_flatMap = get_filelist(file_patterns, fits_keywords, name_search="flat map")
        except FileNotFoundError as e:
            print(f"WARNING!!! {e}")
            return None

        print(filelist_flatMap)
        # Return the most recent flat map
        filelist_flatMap = sorted(filelist_flatMap, key=lambda pm: fits.getheader(pm).get('DATE-PRO', '1970-01-01'))
        return filelist_flatMap[-1]
    
    def get_couplingmap_file(self, file_patterns):
        """
        Get coupling map file matching constraints.
        
        Returns:
        --------
        str : coupling map file path
        """
        fits_keywords= {'X_FIRTYP': ['COUPLINGMAP']}
        if self.fits_keywords.get('X_FIRWOL') is not None:
            fits_keywords['X_FIRWOL'] = self.fits_keywords['X_FIRWOL']

        filelist_couplingMap = get_filelist(file_patterns, fits_keywords, name_search="coupling map")

        if len(filelist_couplingMap) == 0:
            return None

        # Return the most recent coupling map
        filelist_couplingMap = sorted(filelist_couplingMap, key=lambda pm: fits.getheader(pm).get('DATE-PRO', '1970-01-01'))

        return filelist_couplingMap[-1]
    
    def get_wavemap_file(self, file_patterns):
        """
        Get wavelength map file matching constraints.
        
        Returns:
        --------
        str : wavelength map file path
        """
        fits_keywords= {'X_FIRTYP': ['WAVEMAP','COUPLINGMAP']}
        if self.fits_keywords.get('X_FIRWOL') is not None:
            fits_keywords['X_FIRWOL'] = self.fits_keywords['X_FIRWOL']

        try:
            filelist_waveMap = get_filelist(file_patterns, fits_keywords, name_search="wave map")
        except FileNotFoundError as e:
            print(f"WARNING!!! {e}")
            return None

        # Return the most recent wave map
        filelist_waveMap = sorted(filelist_waveMap, key=lambda pm: fits.getheader(pm).get('DATE-PRO', '1970-01-01'))
        return filelist_waveMap[-1]
    

    def make_association(self, darks_pattern=None, pixelMap=None):
        """
        Get wavelength calibration files matching constraints.
        
        Returns:
        --------
        tuple : (files_with_dark, neon_with_dark)
            Lists of files with associated darks
        """
        filelist_darks = []
        filelist_pixelMap = []

        # Finding dark files (not mandatory)
        if darks_pattern is not None:
            fits_keywords= {'DATA-TYP': ['DARK'],
                       'X_FIRTYP': ['PREPROC']
                       }
            try:
                filelist_darks = get_filelist(darks_pattern, fits_keywords, name_search="dark")
            except FileNotFoundError as e:
                print(f"WARNING!!! {e}")

        # Finding pixel files
        if pixelMap is not None:
            fits_keywords= {'X_FIRTYP': ['PIXELMAP']}
            filelist_pixelMap = get_filelist(pixelMap, fits_keywords, name_search="pixel map")

        self.files_with_associated_files = []

        for file in self.filelist:
            # Find matching dark
            dark = None
            if len(filelist_darks) > 0:
                dark = find_closest_dark(file, filelist_darks)
            pixelMap = None
            if len(filelist_pixelMap) > 0:
                pixelMap = find_closest_pixelmap(file, filelist_pixelMap)

            association = {'file': file, 'dark': dark, 'pixelMap': pixelMap}

            self.files_with_associated_files += [association]

        return self.files_with_associated_files


    def extract_data_from_list(self, Nsmooth = 1, Nbin = 1, flatMap = None, waveMap = None, center = True):
        """
        Extracts and processes data cubes from the input files.
        Subtracts dark files, applies wavelength smoothing, and calculates variance.
        Returns the processed data cubes, variance cubes, and a header to save.
        If Nsmooth > 1, the data is smoothed along its wavelength dimension by Nsmooth values.
        If Nbin > 1, the data is binned along its wavelength dimension by Nbin values.
        """

        datalist=[]

        for association in self.files_with_associated_files:

            data_file = association['file']
            dark_file = association['dark']

            # reading header data
            header=fits.getheader(data_file)
            # important to cast the data in double!
            data=np.double(fits.getdata(data_file))

            if dark_file is not None:
                data_dark=fits.getdata(dark_file)
                if len(data_dark)==1:
                    data_dark=data_dark[0]
                    data_dark_std=data_dark[0]*0+12
                else:
                    data_dark=data_dark.mean(axis=0)
                    data_dark_std=data_dark.std(axis=0)
            else:
                # using default values if we do not know the dark
                data_dark=header["DETBIAS"]*(1+2*header["PIX_WIDE"])
                data_dark_std=12*np.sqrt(1+2*header["PIX_WIDE"])

            data-=data_dark
            gain=header['GAIN']
            data_dark_var=data_dark_std**2
            data_var=data_dark_var+gain*np.abs(data)#+0.05*np.abs(data)**2
            data_var[np.abs(data)>2**16]=np.inf #saturating values

            dataCube = DataCube(data, data_var, data_dark, data_dark_var, data_file, header)
            
            # Normalize the data cube by the flat field if provided
            if flatMap is not None and isinstance(flatMap, FlatMap):
                flatMap.normalize_with_flat(dataCube)

            # Normalize the data cube by the wavelength map if provided
            if waveMap is not None and isinstance(waveMap, WaveMap):
                waveMap.interpolate_data(dataCube)
                dataCube.wave_label = waveMap.wave_label
                dataCube.Nwave = waveMap.Nwave
                dataCube.wave = waveMap.wave

            # If smoothing and binning is required
            if Nsmooth > 1:
                dataCube.smooth(Nsmooth)
            if Nbin > 1:
                dataCube.bin(Nbin)

            # If centering flux is required, do it after smoothing and binning
            dataCube.compute_flux()
            if center == True:
                dataCube.center_flux_outputs()

            datalist += [dataCube]

        return datalist
