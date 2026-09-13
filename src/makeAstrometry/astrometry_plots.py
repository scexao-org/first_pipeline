"""Plotting helpers for the FIRST astrometry pipeline."""

import numpy as np
import matplotlib.pyplot as plt


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


def compare_jacobian_covariance(J, C_J):
    J_norm = np.linalg.norm(J, axis=-1, keepdims=True)
    J_normalized = J / (J_norm + 1e-30)
    empirical_variance = np.nanvar(J_normalized, axis=0).sum(axis=-1)
    normalized_covariance = C_J / (J_norm[..., None]**2 + 1e-30)
    covariance_trace = np.nanmean(
        np.trace(normalized_covariance, axis1=-2, axis2=-1), axis=0)
    variance_ratio = empirical_variance / (covariance_trace + 1e-30)
    return empirical_variance, covariance_trace, variance_ratio


def plot_jacobian_diagnostics(wave_aera, J, data, C_J,
                              smoothing_length, line_center=None,
                              line_width=None):
    fig, axes = plt.subplots(4, 1, figsize=(10, 14),
                             num="jacobian_diagnostics", clear=True,
                             sharex=True)
    snr_J, r2, lam = diagnose_J(J, data, C_J)
    _, _, variance_ratio = compare_jacobian_covariance(J, C_J)
    ratio = np.nanmedian(variance_ratio, axis=0)
    label = f"Ncube={smoothing_length}"
    axes[0].plot(wave_aera, snr_J, color='steelblue', label=label)
    axes[1].plot(wave_aera, r2, color='steelblue', label=label)
    axes[2].plot(wave_aera, np.max(lam, axis=-1) if lam.ndim > 1 else lam,
                 color='steelblue', label=label)
    axes[3].plot(wave_aera, ratio, color='steelblue', label=label)
    axes[0].axhline(1, color='k', linestyle=':', alpha=0.7, label="SNR = 1")
    axes[0].set_ylabel("Jacobian SNR")
    axes[0].set_yscale('log')
    axes[1].set_ylabel("R\u00b2 (J retained after data projection)")
    axes[1].set_ylim(0, 1.05)
    axes[2].set_ylabel("Max attenuation eigenvalue")
    axes[3].axhline(1/3, color='k', linestyle=':', alpha=0.7)
    axes[3].axhline(3, color='k', linestyle=':', alpha=0.7)
    axes[3].set_ylabel("empirical / C_J variance")
    axes[3].set_xlabel("Wavelength")
    if line_center is not None and line_width is not None:
        for ax in axes:
            ax.axvspan(line_center - line_width/2, line_center + line_width/2,
                       color='gray', alpha=0.2)
            ax.axvline(line_center, color='black', linewidth=1)
    axes[0].legend(fontsize=8)
    axes[0].set_title("Jacobian diagnostics for cube smoothing lengths")
    return fig, axes


def plot_eiv_effects(wave_aera, astrometry_uncorrected_list,
                     astrometry_cj_list, astrometry_full_list,
                     attenuation_list, poly_deg_values, line_center,
                     line_width):
    fig, axes = plt.subplots(3, 1, figsize=(10, 11),
                             num="eiv_effects", clear=True, sharex=True)
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(poly_deg_values)))
    for color, poly_deg, uncorrected, cj_only, full, attenuation in zip(
            colors, poly_deg_values, astrometry_uncorrected_list,
            astrometry_cj_list, astrometry_full_list, attenuation_list):
        cj_effect = cj_only - uncorrected
        cov_effect = full - cj_only
        axes[0].plot(wave_aera, cj_effect[:, 0], color=color,
                     linestyle='-', label=f"C_J, deg={poly_deg}")
        axes[0].plot(wave_aera, cj_effect[:, 1], color=color, linestyle='--')
        axes[1].plot(wave_aera, cov_effect[:, 0], color=color,
                     linestyle='-', label=f"cov_Jsm, deg={poly_deg}")
        axes[1].plot(wave_aera, cov_effect[:, 1], color=color, linestyle='--')
        axes[2].plot(wave_aera, np.max(attenuation, axis=1), color=color,
                     label=f"deg={poly_deg}")
    axes[0].axhline(0, color='black', linewidth=0.8)
    axes[1].axhline(0, color='black', linewidth=0.8)
    axes[0].set_ylabel("C_J effect on astrometry")
    axes[1].set_ylabel("cov_Jsm astrometric offset")
    axes[2].set_ylabel("Max attenuation eigenvalue")
    axes[2].set_xlabel("Wavelength")
    axes[0].set_title("EIV corrections: C_J attenuation and cov_Jsm offset")
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    axes[2].legend(fontsize=8)
    for ax in axes:
        ax.axvspan(line_center - line_width / 2,
                   line_center + line_width / 2, color='gray', alpha=0.2)
        ax.axvline(line_center, color='black', linewidth=1)
    return fig, axes


def diagnose_J(J, data, C_J):
    D2 = np.sum(data**2, axis=0)
    Gd = np.sum(data[..., None] * J, axis=0)
    J_proj = J - data[..., None] * (Gd / D2[..., None])[None]
    Pd = 1.0 - data**2 / D2[None]
    M = np.einsum('bowi,bowj->wij', J_proj, J_proj)
    A = np.einsum('bowij,bow->wij', C_J, Pd)
    snr_J = np.einsum('bowi,bowi->w', J, J) / np.einsum('bowii->w', C_J)
    r2 = np.einsum('bowi,bowi->w', J_proj, J_proj) / np.einsum('bowi,bowi->w', J, J)
    lam = np.linalg.eigvalsh(np.linalg.solve(M, A)).max(axis=1)
    return snr_J, r2, lam


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


def plot_astrometry_scatter(astrometry_xy_list, line_aera, velocity_line,
                            flux_scaled_filtered, object_name,
                            line_center, line_width, poly_deg, PA):
    fig, ax = plt.subplots(1, 1, figsize=(8, 6),
                           num="astrometry_scatter_2", clear=True)
    for astrometry_xy in astrometry_xy_list[-2:-1]:
        scatter = ax.scatter(astrometry_xy[line_aera, 0], astrometry_xy[line_aera, 1],
                             c=velocity_line, s=flux_scaled_filtered*1000,
                             cmap='RdBu_r', alpha=0.6)
        ax.plot(astrometry_xy[:, 0], astrometry_xy[:, 1], 'k-', alpha=0.3, linewidth=1)
    ax.set_xlabel("RA (mas)")
    ax.set_ylabel("DEC (mas)")
    ax.set_title(f"{object_name} - Astrometry vs Velocity, poly deg={poly_deg}")
    ax.plot([], [], ' ', label=f"line center = {line_center:.6g}")
    ax.plot([], [], ' ', label=f"line width = {line_width:.6g}")
    ax.legend()
    ax.set_aspect('equal')
    lim = np.max(np.abs(ax.get_xlim() + ax.get_ylim()))
    ax.set_xlim(lim, -lim)
    ax.set_ylim(-lim, lim)
    fig.colorbar(scatter, ax=ax, label="Velocity (km/s)")
    ax.grid(True, alpha=0.3)
    PA_rad = PA * np.pi / 180
    y = np.linspace(-lim, lim, 100)
    ax.plot(np.tan(PA_rad) * y, y, 'k--', label=f"PA={PA_rad * 180 / np.pi:.2f}\u00b0")
    ax.legend()
    return fig, ax
