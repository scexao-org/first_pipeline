#%%
import sys
import os
import numpy as np
import matplotlib
if "VSCODE_PID" in os.environ:
    matplotlib.use('macosx')
elif os.environ.get('SPYDER_DEBUG_FILE'):
    print("Running in Spyder")
else:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.pyplot import *
from scipy.constants import speed_of_light
from matplotlib.patches import Ellipse
plt.ion()


#### Default values for the test case
poly_deg_values = (2, 3, 4, 5)

ra_dec = np.array([[ 22.86953302, -17.7121186 ],
       [ 19.28419717, -19.57605217],
       [ 21.14761847, -23.16142774],
       [ 17.5621808 , -25.02526265],
       [ 15.69834295, -21.44025519],
       [ 12.11288304, -23.30395007],
       [ 13.97610447, -26.88932688],
       [ 10.39054275, -28.7529231 ],
       [  8.52682265, -25.16775336],
       [  4.94123678, -27.03120897],
       [  6.80425966, -30.6165895 ],
       [  3.21857199, -32.47994644],
       [  1.35496967, -28.89461443],
       [ -2.23073918, -30.75783169],
       [ -4.09424236, -27.17239804],
       [ -0.50911488, -25.30923078],
       [ -2.37247805, -21.72377489],
       [  1.21275122, -19.86070629],
       [  3.07553439, -23.44621049],
       [  6.66078637, -21.58328203],
       [  4.79762317, -17.99782286],
       [  8.38297754, -16.13499116],
       [ 10.24564338, -19.72065982],
       [ 13.83101996, -17.85796811],
       [ 11.96805721, -14.27250582],
       [ 15.55353524, -12.40991383],
       [ 17.41608172, -15.99574304],
       [ 21.00158196, -14.13329104],
       [ 19.1388195 , -10.54782612],
       [ 22.72442156,  -8.68547259],
       [ 24.58685019, -12.27146419],
       [ 28.17247446, -10.40925066],
       [ 26.30991258,  -6.82378216],
       [ 29.8956386 ,  -4.9616673 ],
       [ 28.03321655,  -1.37617706],
       [ 24.44735938,  -3.23881494],
       [ 22.585036  ,   0.34677706],
       [ 18.99915662,  -1.51572082],
       [ 20.86134656,  -5.10183636],
       [ 17.27536527,  -6.96423604],
       [ 15.41315983,  -3.37848137],
       [ 11.82715444,  -5.24074045],
       [ 13.68914585,  -8.82685965],
       [ 10.10303887, -10.68901957],
       [  8.24095108,  -7.10310319],
       [  4.65482295,  -8.96512344],
       [  6.51661295, -12.55124518],
       [  2.93038293, -14.41316723],
       [  1.06841358, -10.82708835],
       [ -2.5178388 , -12.68887034],
       [ -0.65624937, -16.27499623],
       [ -4.24260348, -18.13667953],
       [ -6.10445549, -14.55043833],
       [ -9.69083178, -16.41198164],
       [ -7.82944215, -19.99810866],
       [-11.41592015, -21.85955328],
       [-13.27765386, -18.27315007],
       [-16.86415383, -20.13445476],
       [-15.0029648 , -23.72058475],
       [-18.58956649, -25.58179075],
       [-20.45118283, -21.99522523],
       [-24.03780772, -23.85629091],
       [-25.89932537, -20.26962367],
       [-22.31312924, -18.40812409],
       [-24.17450689, -14.8214347 ],
       [-20.58820904, -12.96003382],
       [-18.72725761, -16.546288  ],
       [-15.1409376 , -14.68502709],
       [-17.00211484, -11.09833469],
       [-13.41569253,  -9.23717057],
       [-11.55486057, -12.82358845],
       [ -7.96841658, -10.96256416],
       [ -9.82939294,  -7.3758689 ],
       [ -6.24284758,  -5.51494436],
       [ -4.38213293,  -9.10152336],
       [ -0.79556496,  -7.24073895],
       [ -2.65634155,  -3.65404102],
       [  0.93032817,  -1.79335511],
       [  2.79092492,  -5.38009641],
       [  6.37761629,  -3.51955033],
       [  4.51704073,   0.06715092],
       [  8.10383378,   1.9275983 ],
       [  9.96431249,  -1.65930504],
       [ 13.55112814,   0.20100222],
       [ 11.69075233,   3.78770611],
       [ 15.2776696 ,   5.64791447],
       [ 17.13803155,   2.0608489 ],
       [ 20.72497092,   3.92091729],
       [ 18.86479564,   7.50762463],
       [ 22.45183701,   9.36759536],
       [ 24.31207886,   5.78036727],
       [ 27.89914234,   7.64019803],
       [ 26.03916741,  11.22690833],
       [ 29.62633195,  13.08663847],
       [ 27.76649698,  16.67337087],
       [ 24.17960126,  14.81312233],
       [ 22.319865  ,  18.39995639],
       [ 18.73294526,  16.53984842],
       [ 20.59295091,  12.95249787],
       [ 17.00592951,  11.09248861],
       [ 15.14631111,  14.67948473],
       [ 11.55926866,  12.81961511],
       [ 13.41907227,   9.23226004],
       [  9.83192817,   7.37248913],
       [  7.97242811,  10.95964715],
       [  4.38526173,   9.10001627],
       [  6.2448655 ,   5.51265926],
       [  2.65759762,   3.65312757],
       [  0.79821481,   7.24044731],
       [ -2.78907514,   5.38105558],
       [ -0.9296718 ,   1.79369546],
       [ -4.51706354,  -0.06559802],
       [ -6.37632784,   3.52188407],
       [ -9.96374146,   1.66273048],
       [ -8.10453867,  -1.92463248],
       [-11.69205376,  -3.78368687],
       [-13.55120079,  -0.19604308],
       [-17.138739  ,  -2.05495717],
       [-15.27973544,  -5.64232318],
       [-18.86737542,  -7.50113903],
       [-20.72640439,  -3.91333274],
       [-24.31406452,  -5.77200924],
       [-22.45526354,  -9.35937858],
       [-26.04302528, -11.21795635],
       [-27.90193634,  -7.62998805],
       [-31.48972012,  -9.48842586],
       [-33.34853243,  -5.90035596],
       [-29.7608562 ,  -4.0413179 ],
       [-31.61952856,  -0.45322596],
       [-28.03175104,   1.40571231],
       [-26.17318839,  -2.18177965],
       [-22.58538883,  -0.32298134],
       [-24.44386098,   3.265113  ],
       [-20.85595975,   5.12381276],
       [-18.99751506,   1.53615861],
       [-15.40959179,   3.39471843],
       [-17.26786345,   6.98281609],
       [-13.67983859,   8.84127716],
       [-11.8215099 ,   5.25346042],
       [ -8.233463  ,   7.11178154],
       [-10.09153447,  10.69988157],
       [ -6.50338605,  12.55810376],
       [ -4.64517627,   8.97012556],
       [ -1.05700629,  10.82820795],
       [ -2.91487681,  14.41631114],
       [  0.67339507,  16.27429583],
       [  2.53148678,  12.68615455],
       [  6.11978116,  14.54399914],
       [  4.26211049,  18.13210531],
       [  7.85050583,  19.98984924],
       [  9.70848021,  16.40154791],
       [ 13.29689708,  18.25915205],
       [ 11.43942719,  21.84726089],
       [ 15.02794562,  23.70476627],
       [ 13.17061567,  27.29289711],
       [  9.58259997,  25.43519767],
       [  7.72536829,  29.02343021],
       [  4.13733058,  27.16587071],
       [  5.99506625,  23.57744579],
       [  2.40692698,  21.71998504],
       [  0.54981371,  25.30837935],
       [ -3.03834736,  23.45105847],
       [ -1.1808122 ,  19.8626308 ],
       [ -4.76907482,  18.00540867],
       [ -6.62607063,  21.59396505],
       [-10.21435629,  19.73688317],
       [ -8.35702099,  16.14845046],
       [-11.9454082 ,  14.29146734],
       [-13.80228605,  17.88018563],
       [-17.39069333,  16.02334183],
       [-15.5335599 ,  12.43490798],
       [-19.12206857,  10.57816342],
       [-20.97882861,  14.16704312],
       [-24.56735924,  12.31043848],
       [-26.42402052,  15.89941972],
       [-22.83511434,  17.75647035],
       [-24.69163553,  21.34547402],
       [-21.10262748,  23.20242693],
       [-19.24573307,  19.6138685 ],
       [-15.65670307,  21.47068147],
       [-17.51302396,  25.05968789],
       [-13.92389303,  26.91640018],
       [-12.06711598,  23.32768176],
       [ -8.4779631 ,  25.18425412],
       [-10.3340837 ,  28.77326327],
       [ -6.7448293 ,  30.62973685],
       [ -4.88816832,  27.04085593],
       [ -1.29889197,  28.89718958]])


##########################
#starting calculations of the line of interest
##########################

Ncube = 10
Nmod = 188
Noutput = 19
Nwave = 1840
wave = np.linspace(580,767, Nwave)
line_center = 656.3
line_width = 1.5

# Speed of light in km/s (precise CODATA value)
# Doppler velocity (km/s)
c = speed_of_light / 1e3
velocity = c * (wave - line_center) / line_center

# Define the wavelength regions for the line, the working area, and the fitting area
work_aera = (wave > line_center - line_width*1.5) & (wave < line_center + line_width*1.5)
wave_aera = wave[work_aera]
line_aera = (wave_aera > line_center - line_width/2) & (wave_aera < line_center + line_width/2)
line_aera_tmp = (wave > line_center - line_width/2) & (wave < line_center + line_width/2)

fit_aera = ~line_aera
line = (wave > line_center - line_width/3) & (wave < line_center + line_width/3)

ra_dec = ra_dec + np.zeros((Ncube, Nmod, 2))


jacobian_r=np.random.normal(size=(Ncube,Nmod,Noutput,2)) * 0.3

# Smooth over axis 1 with a six-value moving average.
jacobian_theo = np.apply_along_axis(
    lambda values: np.convolve(values, np.ones(6) / 6, mode="same"),
    axis=1,
    arr=jacobian_r,
)

#%%

jacobian = jacobian_theo + np.random.normal(0, np.sqrt(0.02), size=(Ncube,Nmod,Noutput,2))

jacobian = jacobian[:,:,:,None] * np.ones(Nwave)[:,None]

datacube = np.zeros((Ncube,Nmod,Noutput,Nwave))

for i in range(1,Nmod):
    datacube[:,i,:,:] = datacube[:,i-1,:,:] + np.einsum('ij,iklj->ikl', (ra_dec[:,i] - ra_dec[:,i-1]), (jacobian[:,i,:,:,:] + jacobian[:,i-1,:,:,:])/2)

flux_input2 = np.exp(-0.1*((wave-line_center)/line_width*10)**2) 

astrometry_input = np.array((0.15,0.3))
astrometry_input = (astrometry_input*jacobian).sum(axis=-1)
datacube[...,line_aera_tmp] += astrometry_input[...,line_aera_tmp]


datacube_var = np.ones_like(datacube) * 0.01
datacube+=np.random.normal(0, np.sqrt(datacube_var))

valid_basis = np.ones((Ncube,Nmod-2),dtype=bool)

# adding flat
flat_theo = np.random.normal(1, 0.01, size=(Noutput,Nwave))
flat_theo *= (1+np.random.rand(Noutput,1)*3)/4
datacube *= flat_theo

# %%
#### now is the code of data reduction I am interest in, to retreive astrometry_input
#### First define main functions

def compute_smoothed_line(data_b, wave_aera, fit_aera, poly_deg):
    """Estimate the continuum under a line with a low-order polynomial fit.

    The continuum is fitted on two side windows (each as wide as the line) on
    either side of the line and evaluated over the full span from the left to
    the right window (`fit_aera`). The wavelength axis may be last (data
    blocks) or penultimate (Jacobian blocks with a trailing RA/DEC axis).
    Inputs are already restricted to ``work_aera`` along the wavelength axis.
    """
    Nwork= wave_aera.size 
    wavelength_axis = -1 if data_b.shape[-1] == Nwork else -2
    if data_b.shape[wavelength_axis] != Nwork:
        raise ValueError("data_b has no axis matching the wavelength grid")

    data_by_wavelength = np.moveaxis(data_b, wavelength_axis, -1)
    y_cont = data_by_wavelength[..., fit_aera]
    x_cont = wave_aera[fit_aera]
    cont_shape = y_cont.shape[:-1]
    coeffs = np.polyfit(x_cont,
                        y_cont.reshape(-1, sum(fit_aera)).T, poly_deg)  # (poly_deg+1, Nseries)
    # Evaluate the polynomial continuum across the full left->right span
    V_line = np.vander(wave_aera, poly_deg + 1)                 # (n_fit, poly_deg+1)
    data_smoothed_line = (V_line @ coeffs).T.reshape(*cont_shape, -1)  # (..., n_fit)
    return np.moveaxis(data_smoothed_line, -1, wavelength_axis)


def compute_adjacent_jacobian_photon_covariance(
        var_center, var_forward, sky_step_basis_inv):
    """Compute photon cross-covariance between neighboring Jacobian blocks."""
    cross_diff_covariance = np.zeros(
        (*var_center[:, :-1].shape, 2, 2), dtype=float)
    cross_diff_covariance[..., 0, 0] = -var_forward[:, :-1]
    cross_diff_covariance[..., 0, 1] = -(
        var_center[:, :-1] + var_forward[:, :-1])
    cross_diff_covariance[..., 1, 1] = -var_center[:, :-1]
    return np.einsum(
        'cmji,cmowjk,cmkl->cmowil',
        sky_step_basis_inv[:, :-1], cross_diff_covariance,
        sky_step_basis_inv[:, 1:])


def compute_smoothed_jacobian_uncertainty(C_J, wave_aera, fit_aera,
                                          poly_deg):
    """Propagate per-wavelength Jacobian covariance through continuum fitting.

    ``C_J`` contains the two-by-two RA/DEC covariance at each wavelength;
    inter-wavelength noise correlations are assumed negligible.
    """
    fit_in_work = fit_aera
    V_cont = np.vander(wave_aera[fit_aera], poly_deg + 1)
    V_work = np.vander(wave_aera, poly_deg + 1)
    smoothing_matrix = V_work @ np.linalg.pinv(V_cont)
    return np.einsum('wc,...cij->...wij', smoothing_matrix ** 2,
                     C_J[..., fit_in_work, :, :])


def compute_smoothed_jacobian_cross_covariance(
        cross_covariance, wave_aera, fit_aera, poly_deg):
    """Propagate neighboring-block Jacobian covariance through smoothing."""
    smoothing_matrix = np.vander(wave_aera, poly_deg + 1) @ np.linalg.pinv(
        np.vander(wave_aera[fit_aera], poly_deg + 1))
    return np.einsum('wc,...cij->...wij', smoothing_matrix ** 2,
                     cross_covariance[..., fit_aera, :, :])


def estimate_jacobian_systematic_variance_2(
    jacobian_smoothed, jacobian_smoothed_covariance,
    adjacent_photon_covariance=None, valid_basis=None):
    """Estimate systematic variance from neighboring modulation blocks.

    Differences are formed along axis 1. If supplied, ``adjacent_photon_covariance``
    has shape ``(Ncube, Nblock - 1, Noutput, Nwave, 2, 2)`` and contains
    ``Cov(J[:, b], J[:, b + 1])``. The photon covariance of a difference is
    then ``C_b + C_b1 - K_b - K_b.T``. Without it, neighboring photon errors
    are treated as independent.
    """
    block_difference = np.diff(jacobian_smoothed, axis=1)
    if valid_basis is not None:
        valid_block_pair = valid_basis[:, :-1] & valid_basis[:, 1:]
        block_difference = np.where(
            valid_block_pair[:, :, None, None, None], block_difference, np.nan)
    difference_variance = np.nanvar(block_difference, axis=(1,-1))
    measured_difference_variance = np.nanmean(difference_variance, axis=(-1))

    photon_difference_covariance = (
        jacobian_smoothed_covariance[:, :-1]
        + jacobian_smoothed_covariance[:, 1:])
    valid_block_pair = None
    if valid_basis is not None:
        valid_block_pair = valid_basis[:, :-1] & valid_basis[:, 1:]
        photon_difference_covariance = np.where(
            valid_block_pair[:, :, None, None, None, None],
            photon_difference_covariance, np.nan)
    if adjacent_photon_covariance is not None:
        if adjacent_photon_covariance.shape != photon_difference_covariance.shape:
            raise ValueError(
                "adjacent_photon_covariance must match the neighboring "
                "Jacobian covariance shape")
        photon_difference_covariance -= adjacent_photon_covariance
        photon_difference_covariance -= np.swapaxes(
            adjacent_photon_covariance, -1, -2)
    photon_difference_diagonal = np.diagonal(
        photon_difference_covariance, axis1=-2, axis2=-1)
    invalid_photon_values = ~np.isfinite(photon_difference_diagonal)
    if invalid_photon_values.any():
        print(f"* Warning: ignoring {invalid_photon_values.sum()} non-finite "
              "Jacobian covariance values in photon-noise estimate")
    photon_difference_diagonal = np.where(
        invalid_photon_values, np.nan, photon_difference_diagonal)
    photon_difference_variance = np.nanmean(
        photon_difference_diagonal, axis=(1, 3, 4))
    invalid_photon_variance = ~np.isfinite(photon_difference_variance)
    if invalid_photon_variance.any():
        print(f"* Warning: no finite photon-noise estimate for "
              f"{invalid_photon_variance.sum()} cube/output entries; "
              "using zero subtraction")
        photon_difference_variance = np.nan_to_num(
            photon_difference_variance, nan=0.0, posinf=0.0, neginf=0.0)

    systematic_difference_variance = (
        measured_difference_variance - photon_difference_variance)
    systematic_difference_variance = np.maximum(
        systematic_difference_variance, 0.0) 

    return systematic_difference_variance


def compute_smoothed_cross_covariances(cov_data_J, wave_aera, fit_aera,
                                       poly_deg_sm, poly_deg_J):
    """Return ``Cov(data, Jm)`` and ``Cov(sm, Jm)`` after continuum fits.

    The first term retains covariance only where the raw data contributes to
    the Jacobian fit. The second applies the data and Jacobian polynomial
    smoothing matrices to their shared measurement covariance.
    """
    fit_in_work = fit_aera
    x_cont = wave_aera[fit_aera]
    x_work = wave_aera
    smoothing_sm = np.vander(x_work, poly_deg_sm + 1) @ np.linalg.pinv(
        np.vander(x_cont, poly_deg_sm + 1))
    smoothing_J = np.vander(x_work, poly_deg_J + 1) @ np.linalg.pinv(
        np.vander(x_cont, poly_deg_J + 1))
    cov_data_Jm = np.zeros_like(cov_data_J)
    cont_positions = np.flatnonzero(fit_in_work)
    cov_data_Jm[..., cont_positions, :] = (
        cov_data_J[..., cont_positions, :]
        * smoothing_J[cont_positions, np.arange(cont_positions.size), None])
    cov_sm_Jm = np.einsum('wc,wc,...ci->...wi', smoothing_sm, smoothing_J,
                          cov_data_J[..., fit_in_work, :])
    return cov_data_Jm, cov_sm_Jm


def solve_eiv_J(J, data, sm, C_J, cov_Jsm, var_data=None):
    """Correct the projected astrometry fit for Jacobian measurement error.

    The direct-flat model is ``J @ a = data * flat - sm``. This method keeps
    the data and continuum values fixed, and corrects only the uncertainty in
    ``J`` and its covariance with ``sm``. It uses the diagonal-in-block
    covariance approximation for the projection onto the complement of
    ``data``.

    Parameters
    ----------
    J : (Nblocks, Noutput, Nwave, 2)
        Measured response Jacobian.
    data, sm : (Nblocks, Noutput, Nwave)
        Measured data and continuum estimate.
    C_J : (Nblocks, Noutput, Nwave, 2, 2)
        Covariance of the Jacobian error.
    cov_Jsm : (Nblocks, Noutput, Nwave, 2)
        Covariance between Jacobian error and continuum error.
    var_data : (Nblocks, Noutput, Nwave), optional
        Variance of the measured data. When supplied, the returned covariance
        contains only the propagated contribution from data noise.

    Returns
    -------
    astrometry_shift : (Nwave, 2)
        Jacobian-error-corrected astrometric shift.
    flat : (Noutput, Nwave)
        Direct multiplicative gain.
    M_corrected : (Nwave, 2, 2)
        Normal matrix after subtracting the projected Jacobian covariance.
    attenuation : (Nwave, 2)
        Eigenvalues of ``M^-1 @ A``, a diagnostic for the size of the
        Jacobian-error correction.
    astrometry_covariance : (Nwave, 2, 2)
        With ``var_data`` supplied, this is the propagated data-noise
        covariance; otherwise it is ``inv(M_corrected)`` under unit residual
        variance.
    """
    D2 = np.sum(data ** 2, axis=0)
    Gd = np.sum(data[..., None] * J, axis=0)
    H = np.sum(data * sm, axis=0)

    J_proj = J - data[..., None] * (Gd / D2[..., None])[None]
    d_proj = data * (H / D2)[None] - sm
    M = np.einsum('bowi,bowj->wij', J_proj, J_proj)
    rhs = np.einsum('bowi,bow->wi', J_proj, d_proj)

    # For P = I - data data.T / D2, use diag(P) for block-diagonal
    # covariance. A is the projected Jacobian-error contribution and c is
    # the projected J-sm covariance contribution to the right-hand side.
    projected_data_diagonal = 1.0 - data ** 2 / D2[None]
    A = np.einsum('bowij,bow->wij', C_J, projected_data_diagonal)
    c = -np.einsum('bowi,bow->wi', cov_Jsm, projected_data_diagonal) 

    M_corrected = M - A
    astrometry_shift = np.linalg.solve(
        M_corrected, (rhs - c)[..., None])[..., 0]
    flat = (H + np.einsum('owi,wi->ow', Gd, astrometry_shift)) / D2

    #diagnostics:
    # r2 faible (≪ 1) et attenuation grand → dégénérescence géométrique. Votre J est bon, mais a et flat sont quasi indistinguables dans cette configuration de blocs. Aucun traitement statistique n'y remédiera ; il faut plus de diversité de blocs, ou contraindre flat par ailleurs.
    # r2 normal et attenuation grand → J réellement mal connu. Il faut améliorer la calibration.

    r2 = np.einsum('bowi,bowi->w', J_proj, J_proj) / np.einsum('bowi,bowi->w', J, J)
    attenuation = np.linalg.eigvals(np.linalg.solve(M, A)).real
    M_inverse = np.linalg.inv(M_corrected)
    if var_data is None:
        astrometry_covariance = M_inverse
    else:
        projected_data_variance = var_data * (
            1.0 - data ** 2 / D2[None])**2
        rhs_covariance = np.einsum(
            'bowi,bow,bowj->wij', J_proj,
            projected_data_variance, J_proj)
        astrometry_covariance = np.einsum(
            'wij,wjk,wlk->wil', M_inverse, rhs_covariance, M_inverse)
    return astrometry_shift, flat, M_corrected, attenuation, astrometry_covariance



# %%
####### HEre is the main code that will be used to retreive astrometry_input
#####




# Taking care if TT stepping function:

# Known sky steps from each interior modulation point to its two neighbours
sky_step_fwd = ra_dec[:,2:] - ra_dec[:,1:-1]    # p_{k+1} - p_k
sky_step_bwd = ra_dec[:,:-2] - ra_dec[:,1:-1]   # p_{k-1} - p_k

# 2x2 basis of known sky steps (columns are the two step vectors)
sky_step_basis = np.stack([sky_step_fwd, sky_step_bwd], axis=-1)
sky_step_basis_inv = np.linalg.pinv(sky_step_basis)

# Keep only well-conditioned (non-collinear) bases and good-quality data
sky_step_basis_det = np.linalg.det(sky_step_basis)

datacube_normalized = datacube[...,work_aera]
datacube_var_normalized = datacube_var[...,work_aera]

# Measured output changes for the same forward/backward steps
# data_diff_diff= np.diff(np.diff(datacube_normalized, axis=1), axis=0) 
data_diff_fwd = datacube_normalized[:,2:] - datacube_normalized[:,1:-1]    # D_{k+1} - D_k
data_diff_bwd = datacube_normalized[:,:-2] - datacube_normalized[:,1:-1]   # D_{k-1} - D_k
data_diff_basis = np.stack([data_diff_fwd,data_diff_bwd], axis=-1)
jacobian = np.einsum('cbowj,cbjk->cbowk', data_diff_basis, sky_step_basis_inv)
jacobian_smoothed = compute_smoothed_line(jacobian, wave_aera, fit_aera, 1)



# Measured variance / covariance of the same forward/backward differences (shared D_k)
var_backward = datacube_var_normalized[:,:-2]
var_center = datacube_var_normalized[:,1:-1]
var_forward = datacube_var_normalized[:,2:]
diff_covariance = np.empty((*data_diff_basis.shape, 2))
diff_covariance[..., 0, 0] = var_forward + var_center
diff_covariance[..., 0, 1] = var_center
diff_covariance[..., 1, 0] = var_center
diff_covariance[..., 1, 1] = var_backward + var_center
# Calculating the covariance matrix of the Jacobian using the inverse of the sky step basis and the covariance of the data differences
jacobian_covariance = np.einsum(
            'cmji,cmowjk,cmkl->cmowil', sky_step_basis_inv,
            diff_covariance, sky_step_basis_inv)
# Calculating the covariance of the Jacobian with respect to the data center step 
data_jacobian_covariance = np.einsum(
            'cmowj,cmji->cmowi', np.stack([-var_center, -var_center], axis=-1),
            sky_step_basis_inv)
adjacent_photon_covariance = compute_adjacent_jacobian_photon_covariance(
    var_center, var_forward, sky_step_basis_inv)

# smoothing the Jacobian over the working region (but outside the line) to gain snr on it.
# Using for the smoothing a first oder fit.
jacobian_smoothed_covariance = compute_smoothed_jacobian_uncertainty(
    jacobian_covariance, wave_aera, fit_aera, 1)
adjacent_photon_covariance = compute_smoothed_jacobian_cross_covariance(
    adjacent_photon_covariance, wave_aera, fit_aera, 1)

#estimating tip/tilt jitter -- can be ignored, just for info
# tt = estimate_position_variance(
#     jacobian_smoothed, jacobian_smoothed_covariance, sky_step_basis, valid_basis,
#     ra_dec_center=ra_dec[:, 1:-1])
# print(f"* Tip-tilt jitter: {tt['sigma']:.3f} mas "
#     f"(pas de dither {tt['step_scale']:.3f} mas, "
#     f"modele explique {tt['explained']*100:.0f}% du scatter)")

# estimating jacobiasy      n systematic variance
jacobian_systematic_variance_block = (
    estimate_jacobian_systematic_variance_2(
        jacobian_smoothed, jacobian_smoothed_covariance,
        adjacent_photon_covariance, valid_basis))


# adding the systematic variance on the diagonal of the covariance matrix
jacobian_smoothed_covariance[..., 0, 0] += (
    jacobian_systematic_variance_block[:, None, :, None])
jacobian_smoothed_covariance[..., 1, 1] += (
    jacobian_systematic_variance_block[:, None, :, None])


# print(f"* Estimated Jacobian systematic variance: {jacobian_systematic_variance:.3g}")    
# computed key data that will be used in the astrometry fit, and filtered to keep only the valid basis blocks (non-collinear triangles and good quality data)
data_b = datacube_normalized[:, 1:-1][valid_basis]
var_data_b = datacube_var_normalized[:, 1:-1][valid_basis]
J_blocks = jacobian_smoothed[valid_basis]
C_J_blocks = jacobian_smoothed_covariance[valid_basis]
cov_data_J_blocks = data_jacobian_covariance[valid_basis]

# removing outlier blocks based on the median and robust standard deviation of the Jacobian magnitude
outlier_nsigma = 5.0
J_magnitude = np.linalg.norm(J_blocks, axis=-1)
block_score = np.median(J_magnitude, axis=(1, 2))
median_score = np.median(block_score)
robust_std = 1.4826 * np.median(np.abs(block_score - median_score))
good_block = (block_score - median_score) <= outlier_nsigma * robust_std
if not good_block.all():
    print(f"* Rejecting {np.sum(~good_block)} noisy-Jacobian block(s) out of {len(good_block)}")
    data_b = data_b[good_block]
    var_data_b = var_data_b[good_block]
    J_blocks = J_blocks[good_block]
    C_J_blocks = C_J_blocks[good_block]
    cov_data_J_blocks = cov_data_J_blocks[good_block]



# Solve the variable-projection 2x2 system over the line for a list of
# polynomial continuum degrees; each degree yields one astrometry_xy track.
astrometry_xy_list = []
astrometry_covariance_list = []
attenuation_list = []
# The Jacobian fit remains linear; repeat only the continuum fit and its
# covariance propagation for every tested polynomial degree.
for poly_deg in poly_deg_values:
    # Estimate the continuum under the line (polynomial fit on the side
    # windows) instead of the notch-Hanning smoothing.
    sm_b = compute_smoothed_line(data_b, wave_aera, fit_aera, poly_deg)
    _, cov_sm_Jm = compute_smoothed_cross_covariances(
        cov_data_J_blocks, wave_aera, fit_aera, poly_deg, 1)

    J, data, sm, C_J, cov_Jsm = J_blocks, data_b, sm_b, C_J_blocks, cov_sm_Jm
    astrometry_xy, flat_eiv, M_eiv, attenuation, astrometry_covariance = solve_eiv_J(
        J, data, sm, C_J, cov_Jsm, var_data=var_data_b,
    )

    astrometry_xy_list.append(astrometry_xy)
    astrometry_covariance_list.append(astrometry_covariance)
    attenuation_list.append(attenuation)


# %%
# Make main figure showing the astrometry tracks and their covariance ellipses, colored by velocity and scaled by flux.
#####

PA= 45
object_name = "Test Object"

fig, ax = plt.subplots(1, 1, figsize=(8, 6), num="astrometry_scatter", clear=True)

velocity_line = velocity[work_aera][line_aera]
for astrometry_xy, covariance in zip(
        astrometry_xy_list[-2:-1], astrometry_covariance_list[-2:-1]):
    scatter = ax.scatter(astrometry_xy[line_aera, 0], astrometry_xy[line_aera, 1], c=velocity_line,  cmap='RdBu_r', alpha=0.6)
    ax.plot(astrometry_xy[line_aera, 0], astrometry_xy[line_aera, 1], 'k-', alpha=0.3, linewidth=1)
    for point, point_covariance in zip(
            astrometry_xy[line_aera], covariance[line_aera]):
        eigenvalues, eigenvectors = np.linalg.eigh(point_covariance)
        eigenvalues = np.maximum(eigenvalues, 0.0)
        major_axis = np.argmax(eigenvalues)
        angle = np.degrees(np.arctan2(
            eigenvectors[1, major_axis], eigenvectors[0, major_axis]))
        ellipse = Ellipse(
            point, 2 * np.sqrt(eigenvalues[major_axis]),
            2 * np.sqrt(eigenvalues[1 - major_axis]), angle=angle,
            edgecolor='black', facecolor='none', linewidth=0.6, alpha=0.45)
        ax.add_patch(ellipse)
ax.set_xlabel("RA (mas)")
ax.set_ylabel("DEC (mas)")
ax.set_title(f"{object_name} - Astrometry")
ax.plot([], [], ' ', label=f"line center = {line_center:.6g}")
ax.plot([], [], ' ', label=f"line width = {line_width:.6g}")
ax.legend()
ax.set_aspect('equal')
lim = np.max(np.abs(ax.get_xlim() + ax.get_ylim()))
ax.set_xlim(lim, -lim)
ax.set_ylim(-lim, lim)
fig.colorbar(scatter, ax=ax, label="Velocity (km/s)")
ax.set_title(
    f"{object_name} - Astrometry vs Velocity, "
    f"poly deg={list(poly_deg_values)[-2]}\n")
# fig.savefig("astrometry_scatter.png", dpi=300)
ax.grid(True, alpha=0.3)
# ax.xaxis.set_major_locator(plt.MultipleLocator(0.05))
# ax.yaxis.set_major_locator(plt.MultipleLocator(0.05))

PA_rad = PA*np.pi/180
y = np.linspace(-lim,lim,100)
x = np.tan(PA_rad)*y
ax.plot(x,y,'k--',label=f"PA={PA:.2f}°") 
ax.legend() 
# %%


# Compare separation and PA astrometry over the line for the different poly_deg
fig, axes = plt.subplots(3, 1, figsize=(10, 12), num="astromet_comparison_poly_sepPA",
                            clear=True, sharex=True)
for poly_deg, astrometry_xy in zip(poly_deg_values, astrometry_xy_list):
    separation = np.hypot(astrometry_xy[:, 0], astrometry_xy[:, 1])
    PA_deg = np.degrees(np.arctan2(astrometry_xy[:, 0], astrometry_xy[:, 1]))
    axes[0].plot(wave_aera, separation, alpha=0.8, label=f"{poly_deg}")
    axes[1].plot(wave_aera, PA_deg, alpha=0.8, label=f"{poly_deg}")
# Flux over the same wavelength span (fit_aera)
axes[2].plot(wave_aera, flux_input2[work_aera].T, 'r', alpha=0.5)
# Shade the line area
for ax in axes:
    ax.axvspan(line_center - line_width/2, line_center + line_width/2,
                color='gray', alpha=0.2)
# Reference PA and -PA given to the function
axes[1].axhline(PA, color='k', linestyle=':', alpha=0.7, label=f"PA={PA:.2f}°")
axes[1].axhline(-PA, color='k', linestyle=':', alpha=0.7, label=f"-PA={-PA:.2f}°")
axes[0].set_ylabel("Separation (mas)")
axes[1].set_ylabel("PA (deg)")
axes[2].set_ylabel("Flux (scaled)")
axes[2].set_xlabel("Wavelength")
axes[0].set_title(f"{object_name} - Separation (over the line)")
axes[1].set_title(f"{object_name} - PA (over the line)")
axes[2].set_title(f"{object_name} - Flux (over the line)")
axes[0].legend(title="polynomial degree of the continuum fit")
axes[1].legend(fontsize=8)

# %%

# Compare RA and DEC astrometry over the line for the different poly_deg
fig, axes = plt.subplots(3, 1, figsize=(10, 12), num="astromet_comparison_poly",
                            clear=True, sharex=True)
axes[1].sharey(axes[0])
for poly_deg, astrometry_xy in zip(poly_deg_values, astrometry_xy_list):
    axes[0].plot(wave_aera, astrometry_xy[:, 0], alpha=0.8, label=f"{poly_deg}")
    axes[1].plot(wave_aera, astrometry_xy[:, 1], alpha=0.8, label=f"{poly_deg}")
# Flux over the same wavelength span (fit_aera), shaded down to the
# continuum trend interpolated from the fit_aera (line-excluded) points
cont_order = np.argsort(wave_aera[fit_aera])
continuum_flux = np.interp(
    wave_aera,
    wave_aera[fit_aera][cont_order],
    flux_input2[work_aera][fit_aera][cont_order])
axes[2].fill_between(wave_aera, flux_input2[work_aera], continuum_flux,
                        color='r', alpha=0.3)
axes[2].plot(wave_aera, flux_input2[work_aera].T, 'r', alpha=0.5)
# Shade the line area
for ax in axes:
    ax.axvspan(line_center - line_width/2, line_center + line_width/2,
                color='gray', alpha=0.2)
    ax.axvline(line_center, color='black', linewidth=1)
axes[0].set_ylabel("RA astrometric signal (mas)")
axes[1].set_ylabel("DEC astrometric signal (mas)")
axes[2].set_ylabel("Flux (scaled)")
axes[2].set_xlabel("Wavelength")
axes[0].set_title(f"{object_name} - RA astrometry (over the line)")
axes[1].set_title(f"{object_name} - DEC astrometry (over the line)")
axes[2].set_title(f"{object_name} - Flux (over the line)")
axes[0].legend(title="polynomial degree of the continuum fit")
# fig.savefig("astrometry_3.pdf")
# %%

print(f"* Astrometry retrieval completed for {object_name} with {len(poly_deg_values)} polynomial degrees")

# Compare the injected astrometric signal with the largest retrieved signal
# on the spectral line.  Print the full input array for direct inspection,
# followed by compact peak values for the comparison.
line_input_peak = np.max(np.abs(astrometry_input[..., line]), axis=-1)
input_peak_index = np.unravel_index(np.argmax(line_input_peak), line_input_peak.shape)
retrieved_peak_index = np.argmax(
    np.linalg.norm(astrometry_xy_list[-1][line_aera], axis=-1))
retrieved_peak = astrometry_xy_list[-1][line_aera][retrieved_peak_index]
print(f"* Maximum injected signal on line: "
    f"{line_input_peak[input_peak_index]:.6g} "
    f"(cube, modulation, output={input_peak_index})")
print(f"* Maximum retrieved astrometric signal on line: "
    f"{np.linalg.norm(retrieved_peak):.6g} mas "
    f"(RA, DEC)=({retrieved_peak[0]:.6g}, {retrieved_peak[1]:.6g}) mas")



# %%
