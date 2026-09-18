"""Offline spectro-astrometry on a saved data set (see --save_npz of main.py).

usage: python reduce_toto.py [toto.npz]
"""
import sys, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from makeAstrometry import astrometry_core as la

path = sys.argv[1] if len(sys.argv) > 1 else 'toto.npz'
line_center, line_width = 656.5, 1.8
half_window, fit_order, poly_deg = 1, 1, 3          # 3-pose local Jacobian
c_kms = 299792.458

f = np.load(path)
datacube, datacube_var, wave = f['datacube'], f['datacube_var'], f['wave']
ra_dec = f['ra_dec'][:datacube.shape[0]]           # the 40 patterns differ by < 2 mas
velocity = c_kms * (wave - line_center) / line_center

# 1) flatten the stellar spectrum: the Jacobian at a wavelength scales with the flux
data_n, var_n, spectrum, good = la.normalize_by_spectrum(datacube, datacube_var)

# 2) full reduction, all cubes
print("=== all cubes")
res = la.fit_astrometry(data_n, var_n, ra_dec, wave, line_center, line_width,
                           half_window=half_window, fit_order=fit_order,
                           poly_deg_values=(2, 3, 4))
line_aera, fit_aera = res['line_aera'], res['fit_aera']

# 3) reproducibility: even vs odd cubes
halves = []
for name, sel in (("even cubes", slice(0, None, 2)), ("odd cubes", slice(1, None, 2))):
    print(f"=== {name}")
    halves.append(la.fit_astrometry(data_n[sel], var_n[sel], ra_dec[sel], wave,
                                       line_center, line_width, half_window=half_window,
                                       fit_order=fit_order, poly_deg_values=(poly_deg,)))
a1, a2 = halves[0][poly_deg]['astrometry_xy'], halves[1][poly_deg]['astrometry_xy']
diff = (a1 - a2) / np.sqrt(2)
print(f"* half-split: rms(diff)/sqrt2 on continuum = {diff[fit_aera].std():.3f} mas, "
      f"on line = {diff[line_aera].std():.3f} mas")

# error bars: predicted covariance, rescaled so that the continuum (a = 0) has chi2 = 1
a = res[poly_deg]['astrometry_xy']; C = res[poly_deg]['covariance']
sig = np.sqrt(np.diagonal(C, axis1=-2, axis2=-1))
scale = np.sqrt(np.mean((a[fit_aera] / sig[fit_aera]) ** 2))
print(f"* error rescaling from continuum: x{scale:.2f}")
sig_s, C_s = sig * scale, C * scale ** 2
on = a[line_aera]; w_on = 1 / sig_s[line_aera] ** 2
mean_on = (on * w_on).sum(0) / w_on.sum(0)
print(f"* weighted mean on the line: RA={mean_on[0]:+.4f} DEC={mean_on[1]:+.4f} mas "
      f"(naive sigma {1/np.sqrt(w_on.sum(0))[0]:.4f}, {1/np.sqrt(w_on.sum(0))[1]:.4f}; "
      f"channels are correlated, treat as lower bound)")

# ---- figure 1: RA/DEC vs wavelength ------------------------------------
spec_tot = np.nansum(spectrum, axis=0)
fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
for k, lab in enumerate(("RA", "DEC")):
    for pd, col in zip((2, 3, 4), ("C1", "C0", "C2")):
        ak = res[pd]['astrometry_xy'][:, k]
        axes[k].plot(wave, ak, color=col, alpha=0.8, lw=1, label=f"poly deg {pd}")
    axes[k].errorbar(wave, a[:, k], sig_s[:, k], fmt='o', color="C0", ms=3, capsize=2)
    axes[k].axhline(0, color='k', lw=0.8)
    axes[k].set_ylabel(f"{lab} astrometric signal (mas)")
axes[2].plot(wave, spec_tot / spec_tot[fit_aera].mean(), 'r')
axes[2].set_ylabel("flux / continuum"); axes[2].set_xlabel("wavelength (nm)")
for ax in axes:
    ax.axvspan(line_center - line_width/2, line_center + line_width/2, color='gray', alpha=0.2)
    ax.axvline(line_center, color='k', lw=0.8)
axes[0].legend(title=f"{2*half_window+1}-pose Jacobian; errors x{scale:.1f} from continuum")
axes[0].set_title("H-alpha spectro-astrometry, toto.npz")
fig.tight_layout(); fig.savefig("toto_astrometry_vs_wavelength.png", dpi=150)

# ---- figure 2: RA-DEC track coloured by velocity -------------------------
fig, ax = plt.subplots(figsize=(7, 6))
sc = ax.scatter(on[:, 0], on[:, 1], c=velocity[line_aera], cmap='RdBu_r', zorder=3)
ax.plot(on[:, 0], on[:, 1], 'k-', alpha=0.3, lw=1)
for pt, cov in zip(on, C_s[line_aera]):
    ev, evec = np.linalg.eigh(cov); ev = np.maximum(ev, 0)
    ang = np.degrees(np.arctan2(evec[1, 1], evec[0, 1]))
    ax.add_patch(Ellipse(pt, 2*np.sqrt(ev[1]), 2*np.sqrt(ev[0]), angle=ang,
                         edgecolor='k', facecolor='none', lw=0.6, alpha=0.4))
ax.plot(a[fit_aera, 0], a[fit_aera, 1], '.', color='gray', ms=4, label='continuum channels')
ax.set_xlabel("RA (mas)"); ax.set_ylabel("DEC (mas)"); ax.set_aspect('equal')
lim = 1.1 * np.abs(a).max(); ax.set_xlim(lim, -lim); ax.set_ylim(-lim, lim)
ax.grid(alpha=0.3); ax.legend()
fig.colorbar(sc, ax=ax, label="velocity (km/s)")
ax.set_title("Astrometry on the line (colour = velocity), 1-sigma ellipses")
fig.tight_layout(); fig.savefig("toto_astrometry_scatter.png", dpi=150)
# ---- step 2 (optional, a few minutes): amplitude scale kappa -----------
if '--scale' in sys.argv:
    from makeAstrometry import astrometry_scale as scale
    res['spectrum'] = spectrum
    scale.report_jacobian_variability(res)
    scale.calibrate_attenuation(res, line_center, line_width)
    from makeAstrometry.astrometry_plots import plot_kappa_diagnostics
    fig, _ = plot_kappa_diagnostics(res)
    fig.savefig("toto_kappa_diagnostics.png", dpi=150)
    amp = np.hypot(*mean_on)
    print(f"* corrected line-mean amplitude: {amp:.4f} / {res['kappa']:.3f} = "
          f"{amp / res['kappa']:.3f} +- {amp * res['kappa_err'] / res['kappa'] ** 2:.3f} mas (scale error only)")

np.savez("toto_astrometry_result.npz", wave=wave, astrometry_xy=a, covariance=C_s,
         line_aera=line_aera, spectrum=spectrum)
