# first_pipeline

Pipeline to reduce the FIRST data (using the Visible Photonic Lantern) at SUBARU/SCEXAO.
The scripts are made to run one after the other.
They use the fits keyword to know which files shall be used for what.

## Scripts

### runPL_dfits
Shell script to quickly inspect the key parameters of a FIRST FITS file.  
It requires `dfits`, which can be downloaded from [ESO FITS Tools](https://github.com/granttremblay/eso_fits_tools).

**Usage:**
```bash
./runPL_dfits <path_to_fits_file>
```

---

### runPL_changeKeyword
Python script to rename FITS files by modifying their keywords.  
Useful for temporary keyword classification changes. Should be removed once the classification is finalized.

**Usage:**
```bash
python runPL_changeKeyword.py --DATA-CAT=<category> --DATA-TYP=<type> <file1.fits> <file2.fits>
```

---

### runPL_createPixelMap
Python script to create a Pixel Map for preprocessing raw data.  
This map is essential for aligning and calibrating the data.

**Usage:**
```bash
python runPL_createPixelMap.py --pixel_min=100 --pixel_max=1600 --pixel_wide=2 --output_channels=38 <files.fits>
```

---

### runPL_preprocess
Python script to preprocess raw FIRST data.  
It applies the Pixel Map and performs initial data cleaning and calibration.

**Usage:**
```bash
python runPL_preprocess.py --pixel_map=<path_to_pixel_map.fits> <directory_or_files.fits>
```

---

### runPL_createWavelengthMap
Python script to create a Wavelength Map from preprocessed data.  
It identifies emission lines and maps them to pixel positions.

**Usage:**
```bash
python runPL_createWavelengthMap.py --wave_list="[753.6, 748.9, 743.9, ...]" <files.fits>
```

---

### runPL_createCouplingMaps
Python script to create Coupling Maps from preprocessed data.  
It analyzes the coupling efficiency of the photonic lantern.

**Usage:**
```bash
python runPL_createCouplingMaps.py --cmap_size=25 <files.fits>
```

---

### runPL_calibrateNeon
Python script to calibrate detected peaks in a Neon spectrum.  
It matches detected peaks to known wavelengths.

**Usage:**
```bash
python runPL_calibrateNeon.py --all_peaks=<list_of_peaks> --peaks_weight=<list_of_weights> --wavelength_list=<list_of_wavelengths>
```

---

### runPL_imageReconstruction
Python script to reconstruct images from coupling maps.  
It processes data cubes and generates deconvolved images.

**Usage:**
```bash
python runPL_imageReconstruction.py --coupling_map=<path_to_coupling_map.fits> <files.fits>
```
