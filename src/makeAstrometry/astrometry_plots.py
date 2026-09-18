"""Plotting helpers for the FIRST astrometry pipeline (makeAstrometry)."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


def _shade_line(axes, line_center, line_width):
    for ax in np.atleast_1d(axes):
        ax.axvspan(line_center - line_width / 2, line_center + line_width / 2,
                   color='gray', alpha=0.2)
        ax.axvline(line_center, color='black', linewidth=1)


def _covariance_ellipse(point, covariance, **kw):
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    major = np.argmax(eigenvalues)
    angle = np.degrees(np.arctan2(eigenvectors[1, major], eigenvectors[0, major]))
    return Ellipse(point, 2 * np.sqrt(eigenvalues[major]),
                   2 * np.sqrt(eigenvalues[1 - major]), angle=angle, **kw)


def plot_correlation_lag_histogram(data_corr_lag, good_data_flux,
                                   threshold_corr=0.5):
    correlation_values = data_corr_lag[np.isfinite(data_corr_lag)]
    rejected_flux_mask = ~(good_data_flux[:, 1:] & good_data_flux[:, :-1])
    rejected_correlation_values = data_corr_lag[
        rejected_flux_mask & np.isfinite(data_corr_lag)]

    below_threshold_percent = 100 * np.mean(correlation_values < threshold_corr)
    percentile_levels = np.array([5, 16, 50, 84, 95])
    correlation_percentiles = np.percentile(correlation_values, percentile_levels)
    fig, ax = plt.subplots(1, 1, figsize=(8, 6),
                           num="correlation_lag_histogram", clear=True)
    bin_edges = np.linspace(0, 1, 21)
    ax.hist(correlation_values, bins=bin_edges, color="steelblue",
            edgecolor="white", alpha=0.7, label="All data")
    if rejected_correlation_values.size:
        ax.hist(rejected_correlation_values, bins=bin_edges, color="tomato",
                edgecolor="white", alpha=0.7,
                label="Rejected by flux filter")
    ax.axvline(threshold_corr, color="goldenrod", linewidth=2,
               label=f"Threshold: {threshold_corr:.2f} ({below_threshold_percent:.1f}% below)")
    for percentile, value in zip(percentile_levels, correlation_percentiles):
        ax.axvline(value, color="black", linestyle="--", linewidth=1,
                   label=f"P{percentile:g}: {value:.3f}")
    ax.set_xlabel("Correlation between adjacent modulation steps")
    ax.set_ylabel("Count")
    ax.set_title("Adjacent-step correlation across all cubes")
    ax.set_xlim(0, 1)
    ax.legend()
    return fig, ax


def plot_astrometry_comparison(wave_aera, astrometry_xy_list, poly_deg_values,
                               mean_flux, work_aera, fit_aera, object_name,
                               line_center, line_width):
    fig, axes = plt.subplots(3, 1, figsize=(10, 12),
                             num="astromet_comparison_poly", clear=True,
                             sharex=True)
    axes[1].sharey(axes[0])
    for poly_deg, astrometry_xy in zip(poly_deg_values, astrometry_xy_list):
        axes[0].plot(wave_aera, astrometry_xy[:, 0], alpha=0.8, label=f"{poly_deg}")
        axes[1].plot(wave_aera, astrometry_xy[:, 1], alpha=0.8, label=f"{poly_deg}")
    cont_order = np.argsort(wave_aera[fit_aera])
    continuum_flux = np.interp(wave_aera, wave_aera[fit_aera][cont_order],
                               mean_flux[work_aera][fit_aera][cont_order])
    axes[2].fill_between(wave_aera, mean_flux[work_aera], continuum_flux,
                         color='r', alpha=0.3)
    axes[2].plot(wave_aera, mean_flux[work_aera].T, 'r', alpha=0.5)
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
    return fig, axes


def plot_separation_pa(wave_aera, astrometry_xy_list, poly_deg_values,
                       mean_flux, work_aera, line_center, line_width, PA):
    fig, axes = plt.subplots(3, 1, figsize=(10, 12),
                             num="astromet_comparison_poly_sepPA", clear=True,
                             sharex=True)
    for poly_deg, astrometry_xy in zip(poly_deg_values, astrometry_xy_list):
        separation = np.hypot(astrometry_xy[:, 0], astrometry_xy[:, 1])
        PA_deg = np.degrees(np.arctan2(astrometry_xy[:, 0], astrometry_xy[:, 1]))
        axes[0].plot(wave_aera, separation, alpha=0.8, label=f"{poly_deg}")
        axes[1].plot(wave_aera, PA_deg, alpha=0.8, label=f"{poly_deg}")
    axes[2].plot(wave_aera, mean_flux[work_aera].T, 'r', alpha=0.5)
    for ax in axes:
        ax.axvspan(line_center - line_width/2, line_center + line_width/2,
                   color='gray', alpha=0.2)
    axes[1].axhline(PA, color='k', linestyle=':', alpha=0.7, label=f"PA={PA:.2f}\u00b0")
    axes[1].axhline(-PA, color='k', linestyle=':', alpha=0.7, label=f"-PA={-PA:.2f}\u00b0")
    axes[0].set_ylabel("Separation (mas)")
    axes[1].set_ylabel("PA (deg)")
    axes[2].set_ylabel("Flux (scaled)")
    axes[2].set_xlabel("Wavelength")
    axes[0].legend(title="polynomial degree of the continuum fit")
    axes[1].legend(fontsize=8)
    return fig, axes


def plot_astrometry_scatter(astrometry_xy, covariance, line_aera, velocity_line,
                            flux_scaled_filtered, object_name, line_center,
                            line_width, poly_deg, PA, subtitle="", kappa=None,
                            kappa_err=None):
    """RA/DEC track over the line, coloured by velocity, with 1-sigma
    covariance ellipses; the continuum channels are shown in grey.

    With ``kappa`` (amplitude attenuation, ``astrometry_scale``), the top and
    right axes give the corrected astrometry a / kappa.  kappa is achromatic
    and isotropic, so the track, the PA and the ellipses keep their shape and
    only the scale changes; ``kappa_err`` is quoted as a scale uncertainty."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 6), num="astrometry_scatter", clear=True)
    on = astrometry_xy[line_aera]
    scatter = ax.scatter(on[:, 0], on[:, 1], c=velocity_line,
                         s=flux_scaled_filtered * 1000 + 10, cmap='RdBu_r',
                         alpha=0.6, zorder=3)
    ax.plot(on[:, 0], on[:, 1], 'k-', alpha=0.3, linewidth=1)
    for point, point_covariance in zip(on, covariance[line_aera]):
        ax.add_patch(_covariance_ellipse(point, point_covariance, edgecolor='black',
                                         facecolor='none', linewidth=0.6, alpha=0.45))
    ax.plot(astrometry_xy[~line_aera, 0], astrometry_xy[~line_aera, 1], '.',
            color='gray', ms=4, label="continuum channels")
    ax.set_xlabel("RA measured (mas)" if kappa else "RA (mas)")
    ax.set_ylabel("DEC measured (mas)" if kappa else "DEC (mas)")
    ax.plot([], [], ' ', label=f"line center = {line_center:.6g}")
    ax.plot([], [], ' ', label=f"line width = {line_width:.6g}")
    ax.set_aspect('equal')
    lim = 1.1 * np.nanmax(np.abs(astrometry_xy))
    ax.set_xlim(lim, -lim)
    ax.set_ylim(-lim, lim)
    if kappa:
        to_true = (lambda v: v / kappa, lambda v: v * kappa)
        ax.secondary_xaxis('top', functions=to_true).set_xlabel(
            r"RA corrected $= a/\kappa$ (mas)", color='darkred')
        ax.secondary_yaxis('right', functions=to_true).set_ylabel(
            r"DEC corrected $= a/\kappa$ (mas)", color='darkred')
        err = f" $\\pm$ {kappa_err:.3f} ({100 * kappa_err / kappa:.0f}% scale error)" \
            if kappa_err is not None else ""
        ax.plot([], [], ' ', label=f"$\\kappa$ = {kappa:.3f}{err}")
        w = 1 / np.diagonal(covariance[line_aera], axis1=-2, axis2=-1)
        mean = (on * w).sum(0) / w.sum(0)                  # weighted, as in print_summary
        ax.plot([], [], ' ', label=f"line mean: {np.hypot(*mean):.3f} mas measured, "
                                    f"{np.hypot(*mean) / kappa:.2f} mas corrected")
    fig.colorbar(scatter, ax=ax, label="Velocity (km/s)", pad=0.12 if kappa else 0.05)
    ax.grid(True, alpha=0.3)
    y = np.linspace(-lim, lim, 100)
    ax.plot(np.tan(np.radians(PA)) * y, y, 'k--', label=f"PA={PA:.2f}\u00b0")
    ax.legend(fontsize=8)
    ax.set_title(f"{object_name} - Astrometry vs velocity, poly deg={poly_deg}\n{subtitle}",
                 fontsize=9, pad=28 if kappa else 6)
    if kappa:
        fig.tight_layout()                                   # room for the top axis + title
    return fig, ax


def plot_astrometry_with_errors(wave_aera, astrometry_xy, covariance,
                                flux_scaled, fit_aera, object_name,
                                line_center, line_width, poly_deg):
    """RA and DEC versus wavelength with error bars, and the line profile.

    The error bars are rescaled so that the continuum channels (where the
    signal must be zero) have a reduced chi2 of one.
    """
    sigma = np.sqrt(np.diagonal(covariance, axis1=-2, axis2=-1))
    scale = np.sqrt(np.mean((astrometry_xy[fit_aera] / sigma[fit_aera]) ** 2))
    sigma = sigma * max(scale, 1.0)
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), num="astrometry_errors",
                             clear=True, sharex=True)
    for k, label in enumerate(("RA", "DEC")):
        axes[k].errorbar(wave_aera, astrometry_xy[:, k], sigma[:, k], fmt='o-',
                         ms=3, capsize=2, color='C0', alpha=0.8)
        axes[k].axhline(0, color='k', linewidth=0.8)
        axes[k].set_ylabel(f"{label} astrometric signal (mas)")
    axes[2].plot(wave_aera, flux_scaled.T, 'r', alpha=0.5)
    axes[2].set_ylabel("Flux (scaled)")
    axes[2].set_xlabel("Wavelength")
    _shade_line(axes, line_center, line_width)
    axes[0].set_title(f"{object_name} - astrometry with 1-sigma errors, poly deg={poly_deg} "
                      f"(errors x{max(scale, 1.0):.1f} from the continuum scatter)")
    return fig, axes


def plot_kappa_diagnostics(result, title=None):
    """Attenuation factor kappa and the PSF variability it comes from.

    Uses only what ``astrometry_scale.calibrate_attenuation`` already computed
    (no extra simulation): ``psf_variability`` (per-pose measurements on the
    data), ``kappa_table`` (the simulations of the calibration), ``kappa``,
    ``kappa_err`` and, if present, ``variability_fraction``.

    Panels: (a) pointing jitter per pose; (b) flux deformation per pose and
    output; (c) simulated kappa around the measured (jitter, deformation),
    with the local power law and the per-cube measurements; (d) kappa per
    cube predicted by that power law, with the calibrated value.
    """
    from matplotlib.colors import LogNorm, Normalize
    detail, table = result['psf_variability'], result['kappa_table']
    kappa, kappa_err = result['kappa'], result['kappa_err']
    jit, defo = table['jitter'], 100 * table['deformation']
    pl = table['power_law']
    law = lambda j, d: pl['kappa0'] * (j / jit) ** pl['alpha'] * (d / defo) ** pl['beta']
    jc, dc = detail['jitter_per_cube'], 100 * detail['deformation_per_cube']
    kc = law(jc, dc)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9.5), num="kappa_diagnostics", clear=True)
    (ax_j, ax_d), (ax_m, ax_k) = axes

    # (a) pointing jitter ---------------------------------------------------
    shift = detail['pose_shift']
    ax_j.plot(shift[:, 0], shift[:, 1], '.', ms=3, color='steelblue', alpha=0.5,
              label=f"per pose ({len(shift)})")
    t = np.linspace(0, 2 * np.pi, 200)
    for n, ls in ((1, '-'), (2, '--')):
        ax_j.plot(n * np.sqrt(2) * jit * np.cos(t), n * np.sqrt(2) * jit * np.sin(t), 'k', ls=ls,
                  lw=1, label=f"{n}x rms radius ({n * np.sqrt(2) * jit:.1f} mas)")
    lim = 1.05 * np.abs(shift).max()
    ax_j.set_xlim(lim, -lim); ax_j.set_ylim(-lim, lim); ax_j.set_aspect('equal')
    ax_j.set_xlabel("pose shift RA (mas)"); ax_j.set_ylabel("pose shift DEC (mas)")
    ax_j.grid(alpha=0.3); ax_j.legend(fontsize=8, loc='upper right')
    ax_j.set_title(f"(a) pointing jitter: {jit:.2f} mas rms per axis\n"
                   f"(common shift explains {100 * detail['shift_explained_fraction'].mean():.0f}% "
                   "of the flux residuals)", fontsize=10)

    # (b) deformation -------------------------------------------------------
    rr = 100 * detail['relative_residual']
    rr = rr[np.isfinite(rr)]
    edge = np.percentile(np.abs(rr), 99.5)
    ax_d.hist(rr, bins=np.linspace(-edge, edge, 61), color='indianred', alpha=0.7,
              edgecolor='white', density=True, label="per pose and output")
    x = np.linspace(-edge, edge, 300)
    sd = rr.std()
    ax_d.plot(x, np.exp(-0.5 * (x / sd) ** 2) / (sd * np.sqrt(2 * np.pi)), 'k--', lw=1,
              label=f"Gaussian, same rms ({sd:.0f}%)")
    ax_d.axvspan(-defo, defo, color='gray', alpha=0.15, lw=0,
                 label=f"$\\pm$ deformation used ({defo:.0f}%, median over outputs)")
    ax_d.set_xlabel("flux residual after removing the pose shift (% of the flux)")
    ax_d.set_ylabel("density"); ax_d.legend(fontsize=8)
    ax_d.set_title("(b) PSF deformation (flux redistribution between outputs)", fontsize=10)

    # (c) kappa versus jitter and deformation (simulations of the calibration)
    corners = table['corners']
    fjs = sorted({c[0] for c in corners}); fds = sorted({c[1] for c in corners})
    pad = 0.1
    J = np.linspace((fjs[0] - pad) * jit, (fjs[-1] + pad) * jit, 60)
    D = np.linspace((fds[0] - pad) * defo, (fds[-1] + pad) * defo, 60)
    with np.errstate(divide='ignore', invalid='ignore'):
        K = law(J[:, None], D[None, :])
    values = np.concatenate([list(corners.values()), table['seeds'], kc, K.ravel()])
    pos = values[np.isfinite(values) & (values > 0)]
    n_bad = int(np.sum(~(np.asarray(list(corners.values()) + list(table['seeds'])) > 0)))
    if pos.size:
        # kappa can be <= 0 (or nan) when the simulated signal is lost in the
        # noise: a log colour scale is then built on the positive values only
        vmin, vmax = 0.8 * pos.min(), 1.25 * pos.max()
        norm, levels = LogNorm(vmin, vmax), np.geomspace(vmin, vmax, 25)
        line_levels = np.geomspace(vmin, vmax, 7)[1:-1]
    else:
        finite = values[np.isfinite(values)]
        vmin, vmax = (finite.min(), finite.max()) if finite.size else (0.0, 1.0)
        if vmax <= vmin:
            vmin, vmax = vmin - 0.5, vmax + 0.5
        norm, levels = Normalize(vmin, vmax), np.linspace(vmin, vmax, 25)
        line_levels = levels[4:-4:4]
    clip = lambda v: np.clip(np.nan_to_num(np.asarray(v, float), nan=vmin), vmin, vmax)
    cf = ax_m.contourf(J, D, clip(K).T, levels=levels, cmap='viridis', norm=norm)
    cs = ax_m.contour(J, D, clip(K).T, levels=line_levels, colors='white', linewidths=0.7)
    ax_m.clabel(cs, fmt='%.3f', fontsize=7)
    for (fj, fd), k in corners.items():
        ax_m.scatter(fj * jit, fd * defo, c=clip([k]), cmap='viridis', norm=norm, marker='s', s=120,
                     edgecolors='w', linewidths=1.5, zorder=3)
        ax_m.annotate(f"{k:.3f}", (fj * jit, fd * defo), xytext=(0, 11), textcoords='offset points',
                      ha='center', fontsize=8, color='w', weight='bold')
    ax_m.scatter([jit], [defo], c=clip([kappa]), cmap='viridis', norm=norm, marker='*', s=300,
                 edgecolors='w', linewidths=1.5, zorder=4, label=f"measured: {kappa:.3f} (simulated seeds)")
    ax_m.scatter(jc, dc, c=clip(kc), cmap='viridis', norm=norm, marker='o', s=30, edgecolors='w',
                 zorder=3, label="per cube (data)")
    ax_m.scatter([], [], marker='s', c='gray', edgecolors='k', label="simulated +-30% corners")
    cbar = fig.colorbar(cf, ax=ax_m, label=r"$\kappa$")
    ticks = [v for v in (0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1) if vmin <= v <= vmax]
    if n_bad:
        ax_m.text(0.02, 0.02, f"warning: {n_bad} simulated kappa <= 0\n(calibration unreliable)",
                  transform=ax_m.transAxes, color='tomato', fontsize=8, weight='bold')
    if ticks and isinstance(norm, LogNorm):
        cbar.set_ticks(ticks); cbar.set_ticklabels([f"{v:g}" for v in ticks])
    ax_m.set_xlabel("pointing jitter rms (mas)"); ax_m.set_ylabel("flux deformation rms (%)")
    ax_m.legend(fontsize=7, loc='upper center', framealpha=0.8)
    ax_m.set_title(f"(c) $\\kappa \\approx {pl['kappa0']:.3f}\\,"
                   f"(\\sigma_j/{jit:.2f})^{{{pl['alpha']:+.2f}}}\\,"
                   f"(\\sigma_d/{defo:.0f}\\%)^{{{pl['beta']:+.2f}}}$ (local fit)", fontsize=10)

    # (d) kappa distribution ------------------------------------------------
    cubes = np.arange(len(kc))
    ax_k.axhspan(kappa - kappa_err, kappa + kappa_err, color='goldenrod', alpha=0.2, lw=0)
    ax_k.axhline(kappa, color='goldenrod', lw=2,
                 label=f"calibrated $\\kappa$ = {kappa:.3f} $\\pm$ {kappa_err:.3f}")
    ax_k.plot(cubes, kc, 'o-', color='C0', label=f"per cube (power law): "
              f"{kc.mean():.3f}, spread {kc.std():.3f}")
    xs = np.full(len(table['seeds']), -1.0)
    ax_k.plot(xs, table['seeds'], 'k_', ms=14, mew=2,
              label=f"simulation seeds (spread {result.get('kappa_err_seed', 0):.3f})")
    ax_k.plot([-1] * len(corners), list(corners.values()), 's', mfc='none', mec='gray',
              label=f"$\\pm$30% corners (model error {result.get('kappa_err_model', 0):.3f})")
    if result.get('variability_fraction') is not None:
        f = result['variability_fraction']
        ax_k.axhline(1 - f, color='k', ls='--', lw=1,
                     label=f"regression dilution 1 - f = {1 - f:.2f} (data)")
    ax_k.set_xticks([-1] + list(cubes)); ax_k.set_xticklabels(['sim.'] + [str(c) for c in cubes])
    ax_k.set_xlabel("cube"); ax_k.set_ylabel(r"$\kappa$")
    ax_k.axhline(0, color='k', lw=0.6); ax_k.grid(alpha=0.3); ax_k.legend(fontsize=8)
    ax_k.set_title("(d) amplitude attenuation $\\kappa$", fontsize=10)

    fig.suptitle(title or "Amplitude attenuation of the local-Jacobian astrometry", fontsize=12)
    fig.tight_layout()
    return fig, axes
