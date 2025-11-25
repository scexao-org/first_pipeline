# FIRST Pipeline Reorganization Summary

## Changes Made

### Directory Structure
- Created `/classes/` directory for all class files
- Created `/libraries/` directory for all library files
- Added `__init__.py` files to both directories to make them proper Python packages

### Files Moved

#### Classes → `classes/` directory:
- `runPL_class_couplingMap.py`
- `runPL_class_dataCube.py` 
- `runPL_class_pixelMap.py`

#### Libraries → `libraries/` directory:
- `runPL_library_basic.py`
- `runPL_library_io.py`
- `runPL_library_linalg.py`
- `runPL_library_plots.py`

### Import Updates

#### Updated files in main directory:
1. `runPL_make_image.py` - Updated library and class imports
2. `runPL_create_couplingMaps.py` - Updated library and class imports
3. `runPL_make_astrometry.py` - Updated library and class imports
4. `runPL_make_preproc.py` - Updated library and class imports
5. `read_BETACMI.py` - Updated library and class imports
6. `runPL_dev.py` - Updated library imports
7. `read_faint.py` - Updated library imports
8. `runPL_changeKeyword.py` - Updated library imports
9. `runPL_create_pixelMap.py` - Updated library imports
10. `runPL_create_wavelengthMap.py` - Updated library imports
11. `quickfix.py` - Updated library imports
12. `lancementserie.py` - Updated library imports

#### Updated files in moved directories:
1. `classes/runPL_class_couplingMap.py` - Added path modification to access libraries
2. `libraries/runPL_library_basic.py` - Updated to use relative imports

### Import Pattern Changes

#### Old pattern:
```python
import runPL_library_io as runlib_io
from runPL_class_pixelMap import PixelMap
```

#### New pattern:
```python
import libraries.runPL_library_io as runlib_io
from classes.runPL_class_pixelMap import PixelMap
```

### Package-level Imports Available

You can also use:
```python
from libraries import *  # Import all library functions
from classes import *    # Import all classes
```

## Testing
All imports have been tested and are working correctly.

## Notes
- The reorganization maintains all existing functionality
- All alias names (e.g., `runlib_io`, `runlib_basic`) remain the same
- The change makes the codebase more organized and maintainable
- Future development should follow this directory structure