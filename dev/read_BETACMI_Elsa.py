#%%

import os
from astropy.io import fits
from glob import glob
import numpy as np

import matplotlib
if ("VSCODE_PID" in os.environ or os.environ.get('TERM_PROGRAM') == 'vscode'):
    matplotlib.use('Qt5Agg')
elif os.environ.get('SPYDER_DEBUG_FILE'):
    print("Running in Spyder")
else:
    matplotlib.use('Agg')
     
import matplotlib.pyplot as plt

plt.ion()

# %%

files_betacmi=glob("/Users/slacour/DATA/LANTERNE/20250510/preproc/*10T05?5*BETACMI_P.fits")

# %%

data=np.array([fits.getdata(f) for f in files_betacmi])
data_betacmi=data.reshape(data.shape[0]*data.shape[1],data.shape[2],data.shape[3])-200

# %%

data_a=data_betacmi.reshape(data_betacmi.shape[0]//1000,1000,*data_betacmi.shape[1:])[:,:,:,1100:1300]
data_s=data_a.mean(axis=0)
data_sm=data_s.mean(axis=0)/data_s.mean(axis=(0,1))

fig,axs=plt.subplots(2,1,figsize=(10,6))
axs[0].imshow(data_sm,aspect='auto')
axs[1].plot(data_sm.T)

# %%
