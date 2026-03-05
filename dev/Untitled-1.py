#%%
import os
import numpy as np
from astropy.io import fits
import glob
import matplotlib
# if ("VSCODE_PID" in os.environ or os.environ.get('TERM_PROGRAM') == 'vscode'):
matplotlib.use('Qt5Agg')

import matplotlib.pyplot as plt
from matplotlib.pyplot import plot,hist,clf,figure,legend,imshow,clf
plt.ion()

directory="//Users/slacour/DATA/LANTERNE/raw/20260227/firstpl/*fits"

files= glob.glob(directory)
files.sort()

file=files[0]

data = fits.getdata(file)

Nx=data.shape[2]
Ny=data.shape[1]
Nframes=data.shape[0]

# %%


datamean = data.mean(axis=0)
threshold = np.percentile(datamean.ravel(), (1-19/Ny)*100)  # Set threshold at the 99.9th percentile
masque = datamean > threshold

# %%

flux_1=data.mean(axis=(1,2))
flux_2=data[:,masque].mean(axis=(1))
flux_1=flux_1-200
flux_2=flux_2-200

flux_1=flux_1/0.028/20
flux_2=flux_2/0.115/20
# flux_1+=np.random.normal(0,1,size=flux_1.shape)
# flux_2+=np.random.normal(0,1,size=flux_2.shape)

figure(235346)
clf()
plot(flux_1,label="all pixels")
plot(flux_2,label="masked pixels")
legend()
# %%
