#%%
# 
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('Qt5Agg')

import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import *
from scipy.linalg import pinv, solve_triangular
import astropy.io.fits as fits
from classes.runPL_class_waveMap import WaveMap

plt.ion()

wave_patterns = "/Users/slacour/DATA/LANTERNE/20251231/wavemaps"
wave_file = glob.glob(f"{wave_patterns}/*.fits")[0]
waveMap =  WaveMap(wave_file)

files = glob.glob('/Users/slacour/DATA/LANTERNE/20260103/preproc/firstpl_2026-01-03T12h31*.fits')
# files.sort()
data=[] # shape (n_samples, height, width)
for f in files[1:]:
    data += [fits.getdata(f)]

data = np.array(data,dtype=float) - 1200 # shape (n_samples, height, width)
data_interp = waveMap.interpolate_data(data) # shape (n_samples, height, width, n_wave)
# data = data.reshape( -1, *data.shape[2:]) # reshape to (n_samples, height*width)
data = data_interp.reshape( -1, *data_interp.shape[2:]) # reshape to (n_samples, height*width)


spectra = data.mean(axis=(0,1)) # H alpha 1200 - 1215 pixels

data_resh = data.reshape(data.shape[0], -1) # reshape to (n_samples, n_features)

U, s, Vt = np.linalg.svd(data_resh, full_matrices=False)

Vt = Vt.reshape(-1, data.shape[1], data.shape[2]) # reshape back to (n_components, height, width)
# Extract the 3 main singular vectors
top_3_vectors = Vt[:3]

Vtnorm = (Vt / spectra*19)[:, :, 1000:1400] # Normalize by the mean spectrum in the H alpha region
Vtnorm_2 = (Vt / spectra[1190]*19)[:, :, 1000:1400]

# %%
