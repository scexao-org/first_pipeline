import numpy as np
from scipy.interpolate import griddata
import runPL_library_linalg as runlib_linalg
import runPL_library_plots as runlib_plots
from tqdm import tqdm


def flux_filtering(flux):
    
    # select data only above a threshold based on flux
    flux_threshold=np.percentile(flux.mean(axis=(2)),80)/5
    flux_goodData=flux.mean(axis=(2)) > flux_threshold
    # plt.imshow(flux_goodData)
    if np.sum(flux_goodData)<57:
        #too little good data, we need to lower the bar
        flux_goodData=flux.mean(axis=(2,3)) > flux_threshold/2
        print("Not enough good data, lowering the threshold to ",flux_threshold/2)
        flux_goodData=flux.mean(axis=(2)) > flux_threshold

    return flux_goodData,flux_threshold


def svd_filtering(datacube,flux_goodData,Nsingular=19*6):

    datacube_flux_goodData = datacube[flux_goodData]
    datacube_flux_goodData = datacube_flux_goodData.reshape((datacube_flux_goodData.shape[0], -1))
    res = runlib_linalg.robust_subspace(datacube_flux_goodData, k=Nsingular, center=False, k_sigma=2.5, max_refit=1,verbose=True)
    singular_values = res["model"]["S"][:-1]
    data_svdfiltered, residuals, errors = runlib_linalg.project(datacube.reshape((datacube.shape[0]*datacube.shape[1], -1)), res["model"])
    data_svdfiltered = data_svdfiltered.reshape(datacube.shape)
    fit_goodData = errors.reshape((datacube.shape[0], -1)) < res["threshold"]
    

    return data_svdfiltered,fit_goodData,errors
