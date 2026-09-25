"""
FIRST Pipeline - Spectro-astrometry core (pure numerical functions).

Everything here works on plain numpy arrays and has no dependency on the
pipeline I/O classes, so it can be exercised on a saved data set
(``np.savez`` of datacube, datacube_var, ra_dec, wave) or on simulations.

Model
-----
For a source at sky position p, the flux of lantern output o at wavelength l,
normalised by the mean spectrum of that output, is locally linear in p::

    D_o(p, l) ~= D_o(p_k, l) + J_ok(l) . (p - p_k)

``J`` is the local response Jacobian (per mas).  A spectro-astrometric signal
is a small photocentre shift ``a(l)`` (mas) shared by all outputs and poses::

    g_o(l) D_ok(l) = sm_ok(l) + J_ok(l) . a(l)

where ``sm`` is the continuum under the line (polynomial fit on the side
windows) and ``g`` a per-(output, wavelength) gain that absorbs the
pixel-to-pixel flat.  ``g`` is eliminated analytically (variable projection)
and a 2x2 system is solved per wavelength (``solve_eiv_J``), with an
errors-in-variables correction for the *photon* noise of ``J``.

Jacobian estimators
-------------------
A Jacobian is needed to fit ``a(l)``, but the choice of estimator does not
change the *expected* wavelength structure of ``a`` nor its position angle
(both are unbiased with every estimator).  It changes (i) the statistical
noise of that structure and (ii) the attenuation kappa of its amplitude,
which belongs to the second step (``astrometry_scale``) and is calibrated
there with the same estimator options.

``estimate_local_jacobian``  finite differences on a window of 2h+1 dithered
    poses (h=1: the 3-pose inversion).  Robust to slow PSF changes, but PSF
    jitter/deformation *between the poses of a block* enters the regressor
    and attenuates ``a`` by an achromatic, isotropic factor kappa.
``average_jacobian_over_cubes``  averages the local Jacobian over cubes at
    the same dither position (reduces that attenuation when the variability
    is white in time).

The wavelength structure of ``a`` (line profile, PA) is unbiased in all
cases; only the amplitude scale kappa must be calibrated by simulation
(see ``simulate_lantern.py`` / ``calibrate_scale.py``).

The reduction is organised in two steps, in two modules:

* ``astrometry_core`` (this module): estimation of ``a(l)`` and of its
  statistical covariance from the data alone.
* ``astrometry_scale``: measurement of the PSF variability on the data and
  calibration, by simulation, of the attenuation factor kappa.

Two complementary corrections of the Jacobian error are therefore applied:
``solve_eiv_J`` removes analytically the *photon* part (known covariance
C_J, ``M_corrected = M - A``), in the data and in the simulations alike, so
that the simulated kappa is independent of the photon regime (checked:
kappa varies by < 5 % over three decades of flux); kappa then only carries
the PSF-variability part, which no covariance describes.

Author: slacour (pipeline), refactoring 2026-09.
"""
import numpy as np


JAC_FIT_REGIONS = ('all', 'continuum', 'line')


def jacobian_fit_mask(line_mask, region='all'):
    """Channels of the working window on which the wavelength polynomial of
    the Jacobian is fitted: 'all' (whole window), 'continuum' (window minus
    the line) or 'line' (line only).  The polynomial is always evaluated on
    the whole window."""
    line_mask = np.asarray(line_mask, bool)
    if region == 'all':
        return np.ones_like(line_mask)
    if region == 'continuum':
        return ~line_mask
    if region == 'line':
        return line_mask.copy()
    raise ValueError(f"jac_fit_region must be one of {JAC_FIT_REGIONS}, got {region!r}")


def smoothing_matrix(x, fit_mask, poly_deg):
    """Linear operator of a polynomial fit of degree ``poly_deg`` on the
    channels ``fit_mask`` of ``x``, evaluated on every channel.

    Returns S (Nwave, Nfit) such that ``smoothed = S @ y[fit_mask]``.
    """
    fit_mask = np.asarray(fit_mask, bool)
    if fit_mask.sum() < poly_deg + 1:
        raise ValueError(f"{fit_mask.sum()} channels in the fit mask for a polynomial "
                         f"of degree {poly_deg}: widen the region or lower the degree")
    return np.vander(x, poly_deg + 1) @ np.linalg.pinv(np.vander(x[fit_mask], poly_deg + 1))


def compute_smoothed_line(data_b, wave_work, fit_mask, poly_deg):
    """Low-order polynomial fit along wavelength, evaluated on the whole window.

    The polynomial is fitted on the channels ``fit_mask`` of the working
    window (the continuum, i.e. the window minus the line, for the data; the
    region chosen by ``jac_fit_region`` for the Jacobian) and evaluated on
    every channel of ``wave_work``.  The wavelength axis may be last (data
    blocks) or penultimate (Jacobian blocks with a trailing RA/DEC axis).
    Inputs are already restricted to the working window along wavelength.
    """
    Nwork = wave_work.size
    wavelength_axis = -1 if data_b.shape[-1] == Nwork else -2
    if data_b.shape[wavelength_axis] != Nwork:
        raise ValueError("data_b has no axis matching the wavelength grid")

    data_by_wavelength = np.moveaxis(data_b, wavelength_axis, -1)
    y_fit = data_by_wavelength[..., fit_mask]
    x_fit = wave_work[fit_mask]
    fit_shape = y_fit.shape[:-1]
    coeffs = np.polyfit(x_fit,
                        y_fit.reshape(-1, sum(fit_mask)).T, poly_deg)  # (poly_deg+1, Nseries)
    # Evaluate the polynomial across the whole working window
    V_work = np.vander(wave_work, poly_deg + 1)                  # (Nwork, poly_deg+1)
    data_smoothed = (V_work @ coeffs).T.reshape(*fit_shape, -1)   # (..., Nwork)
    return np.moveaxis(data_smoothed, -1, wavelength_axis)


def compute_smoothed_jacobian_uncertainty(C_J, wave_work, fit_mask, poly_deg):
    """Propagate per-wavelength Jacobian covariance through the polynomial
    fit on the channels ``fit_mask`` (see ``compute_smoothed_line``).

    ``C_J`` contains the two-by-two RA/DEC covariance at each wavelength;
    inter-wavelength noise correlations are assumed negligible.
    """
    S = smoothing_matrix(wave_work, fit_mask, poly_deg)
    return np.einsum('wc,...cij->...wij', S ** 2,
                     C_J[..., fit_mask, :, :], optimize=True)


def compute_smoothed_cross_covariances(cov_data_J, wave_work, continuum_mask,
                                       jacobian_mask, poly_deg_sm, poly_deg_J):
    """Return ``Cov(data, Jm)`` and ``Cov(sm, Jm)`` after the polynomial fits.

    ``sm`` is the continuum of the data (fit on ``continuum_mask``, degree
    ``poly_deg_sm``) and ``Jm`` the smoothed Jacobian (fit on
    ``jacobian_mask``, degree ``poly_deg_J``).  ``cov_data_J`` is the
    per-wavelength covariance between the raw data and the raw Jacobian;
    only the channels used by both fits contribute to ``Cov(sm, Jm)``.
    """
    continuum_mask = np.asarray(continuum_mask, bool)
    jacobian_mask = np.asarray(jacobian_mask, bool)
    Nwork = wave_work.size
    S_sm = np.zeros((Nwork, Nwork))
    S_sm[:, continuum_mask] = smoothing_matrix(wave_work, continuum_mask, poly_deg_sm)
    S_J = np.zeros((Nwork, Nwork))
    S_J[:, jacobian_mask] = smoothing_matrix(wave_work, jacobian_mask, poly_deg_J)

    cov_data_Jm = np.zeros_like(cov_data_J)
    jac_positions = np.flatnonzero(jacobian_mask)
    cov_data_Jm[..., jac_positions, :] = (
        cov_data_J[..., jac_positions, :]
        * S_J[jac_positions, jac_positions, None])
    shared = continuum_mask & jacobian_mask
    cov_sm_Jm = np.einsum('wc,wc,...ci->...wi', S_sm[:, shared], S_J[:, shared],
                          cov_data_J[..., shared, :], optimize=True)
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
    M = np.einsum('bowi,bowj->wij', J_proj, J_proj, optimize=True)
    rhs = np.einsum('bowi,bow->wi', J_proj, d_proj, optimize=True)

    # For P = I - data data.T / D2, use diag(P) for block-diagonal
    # covariance. A is the projected Jacobian-error contribution and c is
    # the projected J-sm covariance contribution to the right-hand side.
    projected_data_diagonal = 1.0 - data ** 2 / D2[None]
    A = np.einsum('bowij,bow->wij', C_J, projected_data_diagonal, optimize=True)
    c = -np.einsum('bowi,bow->wi', cov_Jsm, projected_data_diagonal, optimize=True) 

    M_corrected = M - A
    astrometry_shift = np.linalg.solve(
        M_corrected, (rhs - c)[..., None])[..., 0]
    flat = (H + np.einsum('owi,wi->ow', Gd, astrometry_shift, optimize=True)) / D2

    #diagnostics:
    # r2 faible (≪ 1) et attenuation grand → dégénérescence géométrique. Votre J est bon, mais a et flat sont quasi indistinguables dans cette configuration de blocs. Aucun traitement statistique n'y remédiera ; il faut plus de diversité de blocs, ou contraindre flat par ailleurs.
    # r2 normal et attenuation grand → J réellement mal connu. Il faut améliorer la calibration.

    r2 = np.einsum('bowi,bowi->w', J_proj, J_proj, optimize=True) / np.einsum('bowi,bowi->w', J, J, optimize=True)
    attenuation = np.linalg.eigvals(np.linalg.solve(M, A)).real
    M_inverse = np.linalg.inv(M_corrected)
    if var_data is None:
        astrometry_covariance = M_inverse
    else:
        projected_data_variance = var_data * (
            1.0 - data ** 2 / D2[None])**2
        rhs_covariance = np.einsum(
            'bowi,bow,bowj->wij', J_proj,
            projected_data_variance, J_proj, optimize=True)
        astrometry_covariance = np.einsum(
            'wij,wjk,wlk->wil', M_inverse, rhs_covariance, M_inverse, optimize=True)
    return astrometry_shift, flat, M_corrected, attenuation, astrometry_covariance


def estimate_local_jacobian(datacube_n, datacube_var_n, ra_dec, half_window=1,
                            fit_order=1):
    """Local Jacobian by least squares on a window of 2*half_window+1 poses.

    For the block centred on pose k, the measured differences
    ``D_j - D_k`` (j = k-h..k+h, j != k) are fitted with
        d_j = dp_j . J_k                                      (fit_order=1)
        d_j = dp_j . J_k + 1/2 dp_j^T H_k dp_j                (fit_order=2)
    where ``dp_j = p_j - p_k``. With ``half_window=1, fit_order=1`` this is
    exactly the original 3-pose inversion (2 equations, 2 unknowns).  The
    curvature term keeps ``J_k`` a *local* gradient when the window is wide.

    Returns
    -------
    jacobian : (Ncube, Nblock, Noutput, Nwave, 2)
    jacobian_covariance : (Ncube, Nblock, Noutput, Nwave, 2, 2)
        Photon covariance of J (noise on every D_j, D_k shared).
    data_jacobian_covariance : (Ncube, Nblock, Noutput, Nwave, 2)
        Cov(D_k, J_k) induced by the shared central pose.
    condition : (Ncube, Nblock)
        Condition number of the design matrix (collinear windows -> large).
    """
    h = half_window
    Ncube, Nmod = ra_dec.shape[:2]
    Nblock = Nmod - 2 * h
    offsets = [j for j in range(-h, h + 1) if j != 0]
    centre = np.arange(h, Nmod - h)
    dp = np.stack([ra_dec[:, centre + j] - ra_dec[:, centre] for j in offsets],
                  axis=2)                                   # (Ncube, Nblock, Nj, 2)
    cols = [dp[..., 0], dp[..., 1]]
    if fit_order >= 2:
        cols += [0.5 * dp[..., 0] ** 2, dp[..., 0] * dp[..., 1],
                 0.5 * dp[..., 1] ** 2]
    if fit_order >= 3:
        raise ValueError("fit_order must be 1 or 2")
    A = np.stack(cols, axis=-1)                              # (Ncube, Nblock, Nj, Npar)
    if A.shape[-2] < A.shape[-1]:
        raise ValueError(f"half_window={h} gives {A.shape[-2]} equations for "
                         f"{A.shape[-1]} unknowns; enlarge the window")
    G = np.linalg.pinv(A)[..., :2, :]                        # (Ncube, Nblock, 2, Nj)
    condition = np.linalg.cond(A)
    d = np.stack([datacube_n[:, centre + j] - datacube_n[:, centre]
                  for j in offsets], axis=-1)                # (Ncube, Nblock, Nout, Nwave, Nj)
    jacobian = np.einsum('cbkj,cbowj->cbowk', G, d, optimize=True)
    v_off = np.stack([datacube_var_n[:, centre + j] for j in offsets], axis=-1)
    v_c = datacube_var_n[:, centre]                          # (Ncube, Nblock, Nout, Nwave)
    G1 = G.sum(axis=-1)                                      # (Ncube, Nblock, 2)
    jacobian_covariance = (
        np.einsum('cbkj,cbowj,cblj->cbowkl', G, v_off, G, optimize=True)
        + v_c[..., None, None] * (G1[:, :, None, None, :, None]
                                  * G1[:, :, None, None, None, :]))
    data_jacobian_covariance = -v_c[..., None] * G1[:, :, None, None, :]
    return jacobian, jacobian_covariance, data_jacobian_covariance, condition


def solve_eiv_J_weighted(J, data, sm, C_J, cov_Jsm, var_data, mask=None,
                         clip_nsigma=None, n_iter=3, verbose=False):
    """Weighted least-squares version of ``solve_eiv_J``.

    Every (block, output, wavelength) residual ``J.a + sm - g.data`` is
    weighted by ``1/var_data`` (all terms scaled by sqrt(w), covariances by w,
    then ``solve_eiv_J`` is called with unit variance).  ``mask`` is an
    optional boolean (Nblocks, Noutput) array of pairs to keep.  With
    ``clip_nsigma`` set, pairs with outlying mean chi2 are dropped iteratively.
    """
    w = 1.0 / var_data
    keep = np.ones(data.shape[:2], dtype=bool) if mask is None else mask.copy()
    for it in range(n_iter if clip_nsigma else 1):
        ww = w * keep[..., None]
        s = np.sqrt(ww)
        out = solve_eiv_J(J * s[..., None], data * s, sm * s,
                          C_J * ww[..., None, None], cov_Jsm * ww[..., None],
                          var_data=np.ones_like(var_data))
        if not clip_nsigma:
            break
        a, g = out[0], out[1]
        resid = np.einsum('bowi,wi->bow', J, a, optimize=True) + sm - g[None] * data
        chi2 = np.mean(resid ** 2 * w, axis=-1)
        chi2_ref = np.median(chi2[keep])
        chi2_mad = 1.4826 * np.median(np.abs(chi2[keep] - chi2_ref))
        new_keep = keep & (chi2 <= chi2_ref + clip_nsigma * chi2_mad)
        if verbose:
            print(f"    clip iter {it}: median chi2 = {chi2_ref:.2f}, "
                  f"rejected {np.sum(~new_keep)} / {keep.size} (block, output) pairs")
        if np.array_equal(new_keep, keep):
            break
        keep = new_keep
    return out + (keep,)



def average_jacobian_over_cubes(J, C_J, n_cubes, valid=None):
    """Average the Jacobian (and its covariance) over the ``n_cubes`` nearest
    cubes at the same block index (same dither position, different time).

    ``n_cubes`` must be a positive odd integer; 1 returns the inputs.  The
    covariance is that of the mean (sum / n^2).  Non-finite entries and
    entries flagged False in ``valid`` (Ncube, Nblock, Noutput) are ignored.
    """
    if n_cubes == 1:
        return J, C_J
    if n_cubes < 1 or n_cubes % 2 == 0 or n_cubes > J.shape[0]:
        raise ValueError("n_cubes must be a positive odd integer <= Ncube")
    half = n_cubes // 2
    idx = np.arange(J.shape[0])
    lo, hi = np.maximum(idx - half, 0), np.minimum(idx + half + 1, J.shape[0])

    def window_mean(x, power):
        ok = np.isfinite(x)
        if valid is not None:
            ok &= valid.reshape(valid.shape + (1,) * (x.ndim - 3))
        cs = np.concatenate([np.zeros_like(x[:1]), np.cumsum(np.where(ok, x, 0.0), axis=0)])
        cn = np.concatenate([np.zeros_like(x[:1]), np.cumsum(ok.astype(float), axis=0)])
        s, n = cs[hi] - cs[lo], cn[hi] - cn[lo]
        return s / np.maximum(n, 1.0) ** power
    return window_mean(J, 1), window_mean(C_J, 2)



def good_window_early(good, h, shape):
    """(Ncube, Nblock, Nout) mask: True where every pose of the 2h+1 window
    around the block centre is good at all wavelengths."""
    good_pose = good.all(axis=-1)
    good_window = np.ones(shape, dtype=bool)
    for j in range(-h, h + 1):
        good_window &= good_pose[:, h + j: good_pose.shape[1] - h + j]
    return good_window


def normalize_by_spectrum(datacube, datacube_var):
    """Divide each output by its mean spectrum (over cubes and poses).

    The Jacobian at a wavelength is proportional to the flux at that
    wavelength, so an emission line would otherwise look like a much larger
    response than the continuum.  After normalisation the flux is flat along
    wavelength and the astrometric shift ``a`` (mas) keeps its meaning since
    dD = J.a  <=>  dD/S = (J/S).a.
    Non-finite samples are returned as zeros with a huge variance; use
    ``finite_mask`` to drop them from the fit.
    """
    good = np.isfinite(datacube) & np.isfinite(datacube_var)
    spectrum = np.nanmean(np.where(good, datacube, np.nan), axis=(0, 1))  # (Nout, Nwave)
    data_n = np.where(good, datacube, 0.0) / spectrum
    var_n = np.where(good, datacube_var, np.inf) / spectrum ** 2
    var_n = np.where(np.isfinite(var_n), var_n, 1e12)
    return data_n, var_n, spectrum, good


def fit_astrometry(datacube, datacube_var, ra_dec, wave, line_center,
                      line_width, half_window=1, fit_order=1,
                      poly_deg_values=(2, 3, 4, 5), good=None,
                      clip_nsigma=None, jac_poly_deg=1,
                      n_cubes_average=1, jac_fit_region='all',
                      verbose=True):
    """Full reduction on a data set already restricted to the working window.

    datacube, datacube_var : (Ncube, Npose, Noutput, Nwave), flux flat in
        wavelength (see normalize_by_spectrum).
    ra_dec : (Ncube, Npose, 2) dither positions in mas.
    good : optional boolean array like datacube, False for samples to ignore.
    jac_fit_region : channels on which the wavelength polynomial (degree
        ``jac_poly_deg``) of the Jacobian is fitted: 'all' (whole working
        window), 'continuum' (window minus the line) or 'line'.
    n_cubes_average : odd number of neighbouring cubes over which the local
        Jacobian is averaged at fixed dither position (1: none).
    The Jacobian covariance given to the EIV solver is photon-only (see the
    module docstring).  The attenuation by PSF variability is NOT handled
    here: see ``astrometry_scale`` (second step of the reduction).

    Returns a dict with, for every polynomial degree, the astrometry track
    ``astrometry_xy`` (Nwave, 2), its covariance, the attenuation diagnostic,
    plus the wavelength masks and the Jacobian.
    """
    wave = np.asarray(wave, float)
    x = wave - line_center                      # centred wavelength for conditioning
    line_mask = np.abs(x) < line_width / 2
    continuum_mask = ~line_mask
    jacobian_mask = jacobian_fit_mask(line_mask, jac_fit_region)
    if good is None:   # default: drop non-finite samples and the huge-variance placeholders
        good = np.isfinite(datacube) & np.isfinite(datacube_var) & (datacube_var < 1e10)
    h = half_window
    block_slice = slice(h, ra_dec.shape[1] - h)

    J, C_J, cov_dJ, cond = estimate_local_jacobian(
        datacube, datacube_var, ra_dec, half_window=h, fit_order=fit_order)
    J_sm = compute_smoothed_line(J, x, jacobian_mask, jac_poly_deg)
    C_J_sm = compute_smoothed_jacobian_uncertainty(C_J, x, jacobian_mask, jac_poly_deg)
    if n_cubes_average > 1:
        J_sm, C_J_sm = average_jacobian_over_cubes(
            J_sm, C_J_sm, n_cubes_average, valid=good_window_early(good, h, J.shape[:3]))

    # blocks: keep every (cube, block) with a well-conditioned window, and
    # mask (block, output) pairs touching a bad sample
    valid_block = cond < 1e3
    good_window = good_window_early(good, h, J.shape[:3])
    mask = good_window[valid_block]

    data_b = datacube[:, block_slice][valid_block]
    var_b = datacube_var[:, block_slice][valid_block]
    J_b, C_b, cov_dJ_b = J_sm[valid_block], C_J_sm[valid_block], cov_dJ[valid_block]

    # outlier blocks on the Jacobian magnitude (robust 5 sigma)
    J_mag = np.linalg.norm(J_b, axis=-1)
    score = np.median(np.where(mask[..., None], J_mag, np.nan), axis=(1, 2))
    score = np.nan_to_num(score, nan=np.nanmedian(score))
    med = np.median(score); rstd = 1.4826 * np.median(np.abs(score - med))
    good_block = (score - med) <= 5.0 * rstd
    if verbose and not good_block.all():
        print(f"* Rejecting {np.sum(~good_block)} noisy-Jacobian block(s) out of {good_block.size}")
    data_b, var_b, J_b, C_b, cov_dJ_b, mask = (
        data_b[good_block], var_b[good_block], J_b[good_block], C_b[good_block],
        cov_dJ_b[good_block], mask[good_block])

    result = dict(wave=wave, line_mask=line_mask, continuum_mask=continuum_mask,
                  jacobian_mask=jacobian_mask, jac_fit_region=jac_fit_region,
                  jacobian=J_sm, jacobian_raw=J, n_blocks=len(data_b),
                  jacobian_covariance=C_J_sm, good_window=good_window,
                  data_normalized=datacube, var_normalized=datacube_var, good=good,
                  ra_dec=ra_dec, half_window=h, fit_order=fit_order,
                  n_cubes_average=n_cubes_average, jac_poly_deg=jac_poly_deg,
                  data_blocks=data_b, jacobian_blocks=J_b,
                  jacobian_blocks_covariance=C_b, block_mask=mask,
                  poly_deg_values=tuple(poly_deg_values))
    for poly_deg in poly_deg_values:
        sm_b = compute_smoothed_line(data_b, x, continuum_mask, poly_deg)
        _, cov_sm_Jm = compute_smoothed_cross_covariances(
            cov_dJ_b, x, continuum_mask, jacobian_mask, poly_deg, jac_poly_deg)
        a, g, M, att, cov_a, keep = solve_eiv_J_weighted(
            J_b, data_b, sm_b, C_b, cov_sm_Jm, var_b, mask=mask,
            clip_nsigma=clip_nsigma, verbose=verbose)
        result[poly_deg] = dict(astrometry_xy=a, covariance=cov_a, flat=g,
                                attenuation=att, kept=keep)
        if verbose:
            on = a[line_mask]; sig = np.sqrt(np.diagonal(cov_a, axis1=-2, axis2=-1))
            print(f"* poly {poly_deg}: on line mean (RA,DEC)=({on[:,0].mean():+.3f},{on[:,1].mean():+.3f}) mas, "
                  f"per-lambda sigma ~{sig[line_mask].mean():.3f}, continuum rms {a[continuum_mask].std():.3f}, "
                  f"|attenuation| {np.abs(att).mean():.2f}")
    return result
