#%%

import os
import numpy as np
from astropy.io import fits
from glob import glob
import matplotlib
if "VSCODE_PID" in os.environ:
    matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
plt.ion()

from matplotlib.pyplot import *


directory ="/Users/slacour/DATA/LANTERNE/2025-05-12/"
files=glob(os.path.join(directory,"*_13?[2-3]*txt"))
files.sort()

data_list = []
for file in files[:]:
    data = np.loadtxt(file, skiprows=11, usecols=range(7))
    data_list.append(data)
# %%
d=np.array(data_list).reshape((-1,7))

plt.figure(3,figsize=(10, 6),clear=True)
for i, data in enumerate(data_list):
    plt.plot(data[1:, 0], np.diff(data[:, 3]), label=f"File {i+1}")

xsetp = np.median(np.diff(d[:, 3]))
N=len(d)
time = xsetp*np.arange(N)
plt.figure(4,figsize=(10, 6),clear=True)
plt.plot(d[:, 2]-time-d[0, 2], 'o-')
plt.plot(d[:, 3]-time-d[0, 3], 'o-')
plt.plot(d[:, 4]-time-d[0, 4], 'o-')
# %%
