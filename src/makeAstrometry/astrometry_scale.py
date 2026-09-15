"""
FIRST Pipeline - Spectro-astrometry, step 2: amplitude scale (kappa).

The local Jacobian measured from neighbouring dithered poses is corrupted
by the PSF jitter and deformation *between* those poses.  This error is not
photon noise (it has no known covariance, is neither Gaussian nor
independent of J) and it attenuates the fitted astrometric shift by a
factor kappa (a_measured = kappa * a_true) that is achromatic and isotropic:
the wavelength structure and the position angle from ``astrometry_core``
are unbiased, only the amplitude needs this second step.

This module

1. quantifies the variability on the data (``estimate_jacobian_variability``:
   excess block-to-block variance of J; ``estimate_psf_variability``:
   pointing jitter in mas and flux deformation per output and pose);
2. calibrates kappa by simulating the same dither pattern, line profile and
   fit options with that jitter and deformation (``calibrate_kappa``, using
   ``simulate_lantern``);
3. ``calibrate_attenuation`` chains 1 and 2 on the result of
   ``astrometry_core.fit_astrometry`` and adds ``kappa``, ``kappa_err``,
   ``jitter``, ``deformation`` to it.

usage (offline): python astrometry_scale.py data.npz [jitter_mas deformation]
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from makeAstrometry import astrometry_core as core, simulate_lantern as sl


# ---------------------------------------------------------------------------
# 1. PSF variability measured on the data
# ---------------------------------------------------------------------------

def estimate_jacobian_variability(J_sm, C_J_sm, fit_aera, x, jac_poly_deg=1,
                                  lag=3, good_block=None):
    """Excess block-to-block variance of the smoothed Jacobian (per cube,
    output and RA/DEC component) beyond photon noise.

    Compares ``Var(J[k+lag] - J[k])`` of the continuum-averaged Jacobian with
    its photon prediction.  With ``lag >= 3`` (3-pose blocks) the two blocks
    share no pose, so photon errors are independent and
        Var(J[k+lag] - J[k]) = 2 sigma_ph^2 + 2 sigma_var^2 .
    The excess is an error on the regressor J that is independent of the
    astrometric signal (PSF motion / deformation between the poses of a
    block), and attenuates the fitted astrometry unless it is added to the
    Jacobian covariance given to the EIV solver.

    Returns sigma_var^2 with shape (Ncube, Noutput, 2).
    """
    V_cont = np.vander(x[fit_aera], jac_poly_deg + 1)
    V_work = np.vander(x, jac_poly_deg + 1)
    S = V_work @ np.linalg.pinv(V_cont)                      # (Nwave, Ncont)
    r = S[fit_aera].mean(axis=0)                             # continuum-mean of the fit
    Jc = J_sm[..., fit_aera, :].mean(axis=-2)                # (Ncube, Nblock, Nout, 2)
    var_ph = np.einsum('c,...cii->...i', r ** 2,
                       C_J_sm[..., fit_aera, :, :])          # (Ncube, Nblock, Nout, 2)
    ok = np.ones(J_sm.shape[:2], dtype=bool) if good_block is None else good_block
    pair = (ok[:, lag:] & ok[:, :-lag])[..., None, None]
    dJ2 = np.where(pair, (Jc[:, lag:] - Jc[:, :-lag]) ** 2, np.nan)
    ph = np.where(pair, var_ph[:, lag:] + var_ph[:, :-lag], np.nan)
    excess = 0.5 * (np.nanmean(dJ2, axis=1) - np.nanmean(ph, axis=1))
    return np.maximum(np.nan_to_num(excess), 0.0)


def estimate_psf_variability(data_n, var_n, ra_dec, fit_aera, good=None, deg=6,
                             scale=30.0):
    """Measure the pose-to-pose PSF variability that corrupts the local
    Jacobian: pointing jitter and flux deformation.

    Per cube, the continuum flux of every output is fitted by a smooth
    polynomial of the dither position.  The residuals are then decomposed,
    pose by pose, into a 2-D shift common to all outputs (pointing jitter,
    using the model gradients) and what remains (PSF deformation, i.e. a
    redistribution of the flux between outputs).

    Returns
    -------
    jitter : float
        rms of the per-pose common shift, in mas.
    deformation : float
        rms of the remaining relative flux residual (fraction of the flux).
    detail : dict with per-cube values and the fraction of residual variance
        explained by the shift.
    """
    Ncube, Npose = ra_dec.shape[:2]
    Dc = data_n[..., fit_aera].mean(-1)                       # (Ncube, Npose, Nout)
    Vc = var_n[..., fit_aera].mean(-1) / fit_aera.sum()
    ok = np.ones(Dc.shape[:2], bool) if good is None else good.all(axis=(-1, -2))
    powers = [(i, j) for i in range(deg + 1) for j in range(deg + 1 - i)]
    jitter, deform, explained = [], [], []
    for c in range(Ncube):
        m = ok[c]
        px, py = ra_dec[c, m, 0] / scale, ra_dec[c, m, 1] / scale
        X = np.stack([px ** i * py ** j for i, j in powers], -1)
        dX = np.stack([(i * px ** (i - 1) if i else 0 * px) * py ** j for i, j in powers], -1) / scale
        dY = np.stack([px ** i * (j * py ** (j - 1) if j else 0 * py) for i, j in powers], -1) / scale
        coef, *_ = np.linalg.lstsq(X, Dc[c, m], rcond=None)
        R = Dc[c, m] - X @ coef                                # (Ngood, Nout)
        Jx, Jy = dX @ coef, dY @ coef
        w = 1.0 / Vc[c, m]
        delta = np.zeros((m.sum(), 2)); res = np.empty_like(R)
        for k in range(m.sum()):
            A = np.stack([Jx[k], Jy[k]], 1) * np.sqrt(w[k])[:, None]
            b = R[k] * np.sqrt(w[k])
            delta[k], *_ = np.linalg.lstsq(A, b, rcond=None)
            res[k] = R[k] - (A @ delta[k]) / np.sqrt(w[k])
        jitter.append(np.sqrt(np.mean(delta ** 2)))
        deform.append(np.median(res.std(0) / np.abs(Dc[c, m]).mean(0)))
        explained.append(1 - np.sum(res ** 2 * w[:, None]) / np.sum(R ** 2 * w[:, None]))
    detail = dict(jitter_per_cube=np.array(jitter), deformation_per_cube=np.array(deform),
                  shift_explained_fraction=np.array(explained))
    return float(np.mean(jitter)), float(np.mean(deform)), detail


def report_jacobian_variability(result, verbose=True):
    """Fraction of <J^2> that is pose-to-pose variability rather than photon
    noise (diagnostic, local Jacobian only).  Stored as ``result['variability_fraction']``."""
    if result['jacobian_method'] != 'local':
        return None
    x = result['wave'] - float(np.mean(result['wave'][result['line_aera']])) \
        if 'line_center' not in result else result['wave'] - result['line_center']
    sigma_var2 = estimate_jacobian_variability(
        result['jacobian'], result['jacobian_covariance'], result['fit_aera'], x,
        result['jac_poly_deg'], lag=3, good_block=result['good_window'].all(axis=-1))
    J2 = np.mean(result['jacobian'][..., result['fit_aera'], :] ** 2, axis=(0, 1, 3))
    fraction = float(np.mean(sigma_var2.mean(0) / J2))
    result['variability_fraction'] = fraction
    if verbose:
        msg = (f"* Jacobian variability: {100 * fraction:.0f}% of <J^2> is pose-to-pose "
               "variability, not photon noise")
        print(msg + (" -> strong attenuation expected, calibrate kappa" if fraction > 0.3 else ""))
    return fraction


# ---------------------------------------------------------------------------
# 2. Calibration of kappa by simulation
# ---------------------------------------------------------------------------

def calibrate_kappa(ra_dec, wave, line_center, line_width, profile, jitter,
                    deformation, fit_kwargs=None, seeds=(11, 12, 13),
                    variation=0.3, a_true=(0.2, -0.12), verbose=True):
    """Return ``(kappa, kappa_err_seed, kappa_err_model, table)``.

    ``kappa`` is the mean recovery factor over ``seeds`` at the measured
    (jitter, deformation), relative to the recovery of the same chain without
    variability (so that photon noise and continuum-fit effects cancel out); ``kappa_err_model`` is the half-spread of kappa
    when both parameters are varied by +-``variation`` (the dominant
    uncertainty).  ``fit_kwargs`` are passed to ``fit_astrometry`` so that
    the simulation uses exactly the options of the real reduction.
    """
    fit_kwargs = dict(fit_kwargs or {})
    fit_kwargs.setdefault('poly_deg_values', (3,))
    x = wave - line_center
    core_mask = np.abs(x) < 0.5 * line_width / 1.8 * 1.0
    dil = 1 - 1 / profile
    a_true = np.asarray(a_true, float)

    def one(jit, defo, seed):
        cube, var, _, _ = sl.simulate(ra_dec, wave, line_center, line_width, a_true,
                                      jitter=jit, deform=defo, seed=seed, profile=profile)
        dn, vn, _, _ = core.normalize_by_spectrum(cube, var)
        r = core.fit_astrometry(dn, vn, ra_dec, wave, line_center, line_width,
                                verbose=False, **fit_kwargs)
        pd = fit_kwargs['poly_deg_values'][0]
        a = r[pd]['astrometry_xy']; w = 1 / np.diagonal(r[pd]['covariance'], axis1=-2, axis2=-1)
        k = [(a[core_mask, i] * w[core_mask, i] * dil[core_mask]).sum()
             / (w[core_mask, i] * dil[core_mask] ** 2).sum() / a_true[i] for i in (0, 1)]
        return float(np.mean(k))

    # Reference: same simulation and fit without jitter nor deformation.  It
    # is close to 1 but not exactly (line wings in the continuum windows,
    # channel weighting); dividing by it makes kappa a pure attenuation.
    reference = one(0.0, 0.0, seeds[0])
    centre = [one(jitter, deformation, s) / reference for s in seeds]
    table = {}
    for fj in (1 - variation, 1 + variation):
        for fd in (1 - variation, 1 + variation):
            table[(fj, fd)] = one(jitter * fj, deformation * fd, seeds[0]) / reference
    kappa = float(np.mean(centre))
    err_seed = float(np.std(centre, ddof=1)) if len(centre) > 1 else 0.0
    err_model = 0.5 * (max(table.values()) - min(table.values()))
    if verbose:
        print(f"* kappa calibration: ideal-case recovery {reference:.3f} (used as reference)")
        print(f"* kappa calibration: jitter {jitter:.2f} mas, deformation {deformation:.0%} -> "
              f"kappa = {kappa:.3f} +- {err_seed:.3f} (seeds) +- {err_model:.3f} (model, +-{variation:.0%})")
    return kappa, err_seed, err_model, table




def calibrate_attenuation(result, line_center, line_width, verbose=True, seeds=(11, 12, 13)):
    """Second step of the reduction: measure the PSF variability on the data,
    then calibrate by simulation the attenuation factor kappa with the same
    dither pattern, line profile and fit options as ``result`` (output of
    ``astrometry_core.fit_astrometry``).  Adds ``kappa``, ``kappa_err``
    (seed and model errors in quadrature), ``jitter``, ``deformation``."""
    jitter, deformation, _ = estimate_psf_variability(
        result['data_normalized'], result['var_normalized'], result['ra_dec'],
        result['fit_aera'], result['good'])
    if verbose:
        print(f"* PSF variability measured on the data: pointing jitter {jitter:.2f} mas rms, "
              f"flux deformation {deformation:.0%} rms per output and pose")
    spectrum = result['spectrum'] if 'spectrum' in result else np.ones((1, result['wave'].size))
    spec_tot = np.nansum(spectrum, axis=0)
    outer = np.abs(result['wave'] - line_center) > 1.1 * line_width
    profile = spec_tot / spec_tot[outer].mean()
    fit_kwargs = dict(half_window=result['half_window'], fit_order=result['fit_order'],
                      jacobian_method=result['jacobian_method'], model_deg=result['model_deg'],
                      n_cubes_average=result['n_cubes_average'],
                      poly_deg_values=(result['poly_deg_values'][0],))
    kappa, err_seed, err_model, _ = calibrate_kappa(
        result['ra_dec'], result['wave'], line_center, line_width, profile,
        jitter, deformation, fit_kwargs=fit_kwargs, seeds=seeds, verbose=verbose)
    result.update(kappa=kappa, kappa_err=float(np.hypot(err_seed, err_model)),
                  jitter=jitter, deformation=deformation)
    return result


if __name__ == "__main__":
    f = np.load(sys.argv[1] if len(sys.argv) > 1 else 'toto.npz')
    d, v, rd, wave = f['datacube'], f['datacube_var'], f['ra_dec'][:f['datacube'].shape[0]], f['wave']
    line_center, line_width = 656.5, 1.8
    dn, vn, spectrum, good = core.normalize_by_spectrum(d, v)
    # step 1: astrometry
    result = core.fit_astrometry(dn, vn, rd, wave, line_center, line_width, poly_deg_values=(3,))
    result['spectrum'] = spectrum
    report_jacobian_variability(result)
    # step 2: scale
    if len(sys.argv) > 3:
        jitter, deformation = float(sys.argv[2]), float(sys.argv[3])
        x = wave - line_center; spec_tot = np.nansum(spectrum, 0)
        profile = spec_tot / spec_tot[np.abs(x) > 1.1 * line_width].mean()
        calibrate_kappa(rd, wave, line_center, line_width, profile, jitter, deformation)
    else:
        calibrate_attenuation(result, line_center, line_width)
