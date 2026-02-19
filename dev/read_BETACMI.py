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
from scipy.ndimage import zoom
from astropy.io import fits
import shutil
from scipy.interpolate import interpn
import libraries.runPL_library_linalg as runlib_linalg
from classes.runPL_class_couplingMap import CouplingMap

plt.ion()

# %%

directory = "/Users/slacour/DATA/LANTERNE/20250510/preproc/"
files_betacmi=glob(directory+"*10T05?5*BETACMI_P.fits")

file_dark_05 = glob(directory+"firstpl_2025-05-10T10:31*_DARK_P.fits")
file_dark_02 = glob(directory+"firstpl_2025-05-10T10:33*_DARK_P.fits")
file_dark_10 = glob(directory+"firstpl_2025-05-10T10:31*_DARK_P.fits")

files_flat1 = glob(directory+"*10T09?2[0-3]*TETCRB_P.fits")
files_flat2 = glob(directory+"firstpl_2025-05-10T09:4[4-9]*_HIP81126_P.fits")
files_flat3 = glob(directory + "firstpl_2025-05-10T07:5*DELVIR_P.fits")
# %%

n1= 0
n2 = 1580
data_betacmi=np.concatenate([fits.getdata(f) for f in files_betacmi])[:,:,n1:n2]
data_betacmi= data_betacmi.reshape(10,data_betacmi.shape[0]//10,*data_betacmi.shape[1:]).mean(axis=0)
data_flat1=np.concatenate([fits.getdata(f) for f in files_flat1])[:,:,n1:n2]
data_flat2=np.concatenate([fits.getdata(f) for f in files_flat2])[:,:,n1:n2]
data_flat3=np.concatenate([fits.getdata(f) for f in files_flat3])[:,:,n1:n2]

data_dark05=np.concatenate([fits.getdata(f) for f in file_dark_05])[:,:,n1:n2].mean(axis=0)
data_dark10=np.concatenate([fits.getdata(f) for f in file_dark_10])[:,:,n1:n2].mean(axis=0)
data_dark02=np.concatenate([fits.getdata(f) for f in file_dark_02])[:,:,n1:n2].mean(axis=0)

df1=data_flat1.mean(axis=(0)) - data_dark10
df2=data_flat2.mean(axis=(0)) - data_dark05
df3=data_flat3.mean(axis=(0)) - data_dark05

df1 /= df1.mean()
df2 /= df2.mean()
df3 /= df3.mean()

#%%

data_betacmi_dark = (data_betacmi - data_dark02)

dm = data_betacmi_dark.mean(axis=0)
#%%

spectra = data_betacmi_dark.mean(axis=(0,1))
data = data_betacmi_dark #/ spectra
data = data - data.mean(axis=1)[:,None]

# data_svdfiltered,fit_goodData,errors = runlib_basic.svd_filtering(datacube,flux_goodData,Nsingular)

svd_res = runlib_linalg.robust_subspace(data.reshape(-1,38*(n2-n1))[::1], k=6, center=False, k_sigma=3.5, max_refit=1)
data_svdfiltered, residuals, errors = runlib_linalg.project(data.reshape(-1,38*(n2-n1))[::10], svd_res["model"])

data_svdfiltered = data_svdfiltered.reshape(data.shape[0]//10,38,(n2-n1))
V = svd_res['model']['V'].reshape((38, n2-n1,6))#[:,1100:1300]
S = svd_res['model']['S']
U = svd_res['model']['U']

filelist_cmap = "/Users/slacour/DATA/LANTERNE/20250510/preproc/../couplingmaps/firstpl_2025-05-10T09:23:36_TETCRBCM.fits"

couplingMap = CouplingMap(filelist_cmap, pyramids = False)

data_T = data.transpose((2,1,0))
star_detected, star_index, star_radec = couplingMap.chi2_filtering(data_T,nx_min=1100,nx_max=1300)
er = np.median(star_index)
print(er)
# %%
star_index2 = star_index.copy()
# star_index2[:] = er

d = data_T.reshape(-1,data_T.shape[2])
v = V.T.reshape(6,-1)[2:]
dqtR=data_T
dqtR = (v.T @ ( v @ d ) ).reshape((-1, 38, data_T.shape[2]))
# dqtR = dqtR.reshape(data_T.shape)
# dqtR=V.transpose((1,0,2))[:,:,0:1]
# dqtR= data_svdfiltered.reshape(data_svdfiltered.shape[0],38,-1).transpose((2,1,0))
dqtR = dqtR/df2.T[:,:,None]*df2.mean(axis=0)[:,None,None]


Nimages = dqtR.shape[2]
QTdata= couplingMap.QT_dot_data(star_index2,dqtR)


Nwave = QTdata.shape[0]
Nqr = QTdata.shape[1]
Nimages = QTdata.shape[2]
R=couplingMap.R

R_mat = np.zeros([Nwave, Nqr, Nimages,Nqr])
for i in range(Nimages):
    t = star_index2[i] 
    R_mat[:,:,i] = R[t][:,:]
R_mat *= spectra[:,None,None,None]
Rinv=np.linalg.pinv(R_mat.reshape((Nwave,-1,3)))
ZYX= (Rinv @ QTdata.reshape((Nwave,-1,1)))[...,0]

clf()
# plot(ZYX[1100:1300]/ZYX[1100:1300,0,None])
plot(ZYX[1100:1300])


# %%

df1=data_flat1.mean(axis=(0)) - data_dark10
df2=data_flat2.mean(axis=(0)) - data_dark05
df3=data_flat3.mean(axis=(0)) - data_dark05
db = data_betacmi.mean(axis=(0)) - data_dark02 

diff_all=np.array([df1,df2,df3,db])
x1,x2,x3,y = diff_all[...,1:]/diff_all[...,:-1]

figure(123)
clf()
hist((y-x3).ravel(),bins=100,range=(-0.5,0.5))
hist((y-x2).ravel(),bins=100,range=(-0.5,0.5))
hist((x2-x1).ravel(),bins=100,range=(-0.5,0.5),alpha=0.5)

#%%

M1=np.array([x1,np.ones_like(x1)]).T
a=(pinv(M1) @ y.T[:,:,None])[...,0]
M2=np.array([x2,np.ones_like(x2)]).T
s=(pinv(M2) @ y.T[:,:,None])[...,0]

figure(123)
clf()
plot(a.ravel(),'.',label='a')

#%%
x1 = data_a[:,:,1:].reshape((100,-1))
y1 = data_a[:,:,:-1].reshape((100,-1))
M1=np.array([x1,np.ones_like(x1)]).T
a=(pinv(M1) @ y1.T[:,:,None])[...,0]

x2= data_s[:,:,1:].reshape((100,-1))
y2 = data_s[:,:,:-1].reshape((100,-1))
M2=np.array([x2,np.ones_like(x2)]).T
s=(pinv(M2) @ y2.T[:,:,None])[...,0]

figure(123)
clf()
out_p=0
plot(*diff_all[0:3:2,out_p],'.')
plot(*diff_all[0:2,out_p],'.')
plot(*diff_all[1:3,out_p],'.')
plot(diff_all[1,out_p],diff_all[1,out_p],'.')
#%%

# data_s = data_s.mean(axis=0)
d1 = data_s.mean(axis=0)
ds = data_s.std(axis=0)
plot(d1.ravel())
plot(d1.ravel()+ds.ravel())

#%%
# data_s=data_a[0]/dfm[None]
data_norm=data_s/data_s.mean(axis=(0,1))
data_m=data_norm.mean(axis=0)
data_r=data_norm-data_m[None,:]

fig,axs=plt.subplots(2,1,figsize=(10,6))
axs[0].imshow(data_m,aspect='auto')
axs[1].plot(data_m.T)

#%%

U, S, Vh = np.linalg.svd(data_norm.reshape(data_r.shape[0],-1), full_matrices=False)
Vm=data_m.ravel()/np.sqrt(np.sum(data_m.ravel()**2))
Vh1=Vh[0]
Vh2=Vh[1]
Vh3=Vh[2]


# %%

file_cm=glob("/Users/slacour/DATA/LANTERNE/20250510/couplingmaps/*fits")[0]

couplingMap = basic.CouplingMap(file_cm)
data_2_flux = couplingMap.data_2_flux[1100:1300]
xpos = couplingMap.xpos
ypos = couplingMap.ypos

# %%

flux= np.matmul(data_2_flux,data_m.T[:,:,None])[:,:,0]


# %%


Npixel = 150
grid_x, grid_y = basic.make_image_grid(couplingMap, Npixel)
flux_maps   =   []

for w in range(len(flux)):
    # Interpolate the fluxes onto the grid
    flux_map = griddata((xpos, ypos), flux[w], (grid_x, grid_y), method='cubic')
    flux_maps += [flux_map]

flux_maps=np.array(flux_maps)
# %%


xmod=fits.getdata(files_flat[0],"MODULATION").field('xmod')
ymod=fits.getdata(files_flat[0],"MODULATION").field('ymod')
ymod[373]=ymod[371]

cm_map=(np.double(df[595:595*2])-1000)/dfm.mean(axis=(0))

cm=cm_map.reshape(595,-1)
cm = cm / np.sqrt((cm**2).sum(axis=1))[:, None]
beta = data_m.ravel()

k=(np.dot(cm,beta))/(cm**2).sum(axis=1)
fit=k[:,None]*cm
res=fit-beta[None]
chi2=(res**2).sum(axis=1)

grid_x, grid_y = basic.make_image_grid(couplingMap, Npixel)

chi2_map = griddata((xmod, ymod), chi2, (grid_x, grid_y), method='nearest')
f_map = griddata((xmod, ymod), k, (grid_x, grid_y), method='nearest')


# %%

cm=data_2_flux.transpose((1,2,0)).reshape(471,-1)
k=(np.dot(cm,beta))/(cm**2).sum(axis=1)
fit=k[:,None]*cm
res=fit-beta[None]
chi2=(res**2).sum(axis=1)
# %%
