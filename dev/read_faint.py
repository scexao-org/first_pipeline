#%%

import os
import sys
from astropy.io import fits
from glob import glob
from optparse import OptionParser
import numpy as np
from scipy.signal import correlate
from scipy import linalg

from scipy.interpolate import griddata

import getpass
import matplotlib
if ("VSCODE_PID" in os.environ or os.environ.get('TERM_PROGRAM') == 'vscode'):
    matplotlib.use('Qt5Agg')
elif os.environ.get('SPYDER_DEBUG_FILE'):
    print("Running in Spyder")
else:
    matplotlib.use('Agg')
     
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot,hist,clf,figure,legend,imshow
from datetime import datetime
from tqdm import tqdm
from collections import defaultdict
from scipy import linalg
from matplotlib import animation
from itertools import product
from scipy.linalg import pinv
from scipy.optimize import curve_fit
import libraries.runPL_library_io as runlib
import libraries.runPL_library_plots as runlib_i
import libraries.runPL_library_basic as basic
from scipy.ndimage import zoom
from astropy.io import fits
import shutil
from scipy.interpolate import interpn

plt.ion()

# %%

file="/Users/slacour/DATA/LANTERNE/firstpl_08:56:07.232435815.fits"
file2="/Users/slacour/DATA/LANTERNE/firstpl_09:49:39.097546766.fits"


data=fits.getdata(file)
data_r=np.median(data,axis=0)

# %%
d=data-data[-1]*1.0

# %%

vmin,vmax=np.percentile(d, (5,95))
# vmin,vmax = -200,200

for i in range(data.shape[0]):
    plt.figure()
    plt.imshow(d[i], origin='lower',aspect='auto', vmin=vmin, vmax=vmax)
    plt.title(f'DIT {i}')
    plt.colorbar()
    plt.show()
# %%
