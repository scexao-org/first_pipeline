#%%

import os
import numpy as np
from astropy.io import fits
from glob import glob
import matplotlib
if "VSCODE_PID" in os.environ:
    matplotlib.use('macosx')
import matplotlib.pyplot as plt
plt.ion()

from matplotlib.pyplot import *


directory ="/Users/slacour/DATA/FIRST/20260608/firstpl/"
files=glob(os.path.join(directory,"*_06?[4-5]*fits"))
files.sort()

data_list = []
data_header = []
for file in files[:]:
    file_base = os.path.splitext(file)[0]
    txt_file = file_base + ".txt"
    data = np.loadtxt(txt_file, skiprows=11, usecols=range(7))
    data_list.append(np.array(data))
    header = fits.getheader(file)
    data_header.append(header)



# %%

plt.figure(2,figsize=(10, 6),clear=True)
for i, data in enumerate(data_list):
    plt.plot(data[1:, 0], np.diff(data[:, 3]), label=f"File {i+1}")

plt.figure(3,figsize=(10, 6),clear=True)
for i, data in enumerate(data_list):
    dif_minus_median = np.diff(data[:, 3]) - np.median(np.diff(data[:, 3]))
    plt.plot(data[1:, 0], dif_minus_median, label=f"File {i+1}")
    Nmod =300

    # getting parameters of metrology glitches
    glitch_on=data_header[i].get('X_FIRGON', 0)
    glitch_frame=data_header[i].get('X_FIRGFR', 0)
    glitch_delay=data_header[i].get('X_FIRGEX', 0) / 1000 # in s
    
    if glitch_on:
        # Find where dif_minus_median matches glitch_delay +/- 10 ms
        tolerance = 0.01  # ms
        glitch_mask = np.abs(dif_minus_median - glitch_delay) <= tolerance
        glitch_indices = np.where(glitch_mask)[0]
        
        if len(glitch_indices) > 0:
            frame_shifts = []
            for glitch_idx in glitch_indices:
                glitch_detected_frame = data[glitch_idx + 1, 0] % Nmod # +1 because diff shifts by 1
                frame_shift = glitch_detected_frame - glitch_frame
                frame_shifts.append(frame_shift)
            
            median_frame_shift = np.median(frame_shifts)
            print(f"  Frame shifts: {frame_shifts}")
            print(f"  Median frame shift: {median_frame_shift}")
        else:
            print(f"File {i+1}: No glitch detected within ±{tolerance} ms of {glitch_delay} ms")








# %%
