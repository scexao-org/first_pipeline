#%%
import numpy as np
from scipy import linalg


def center(X):
    """Center columns of X."""
    return X - X.mean(axis=0, keepdims=True)


def inv_sqrtm_psd(A, eps=1e-12):
    """
    Inverse square root of a symmetric PSD matrix.
    Uses eigen-decomposition with numerical stabilization.
    """
    w, V = linalg.eigh(A)
    w = np.maximum(w, eps)
    return V @ np.diag(1.0 / np.sqrt(w)) @ V.T


def cca(X, Y, n_components=None, reg=1e-6):
    """
    Canonical Correlation Analysis (CCA)

    Parameters
    ----------
    X : (n, p) array
    Y : (n, q) array
    n_components : int or None
        Number of canonical components
    reg : float
        Ridge regularization strength

    Returns
    -------
    A : (p, k) canonical directions for X
    B : (q, k) canonical directions for Y
    r : (k,) canonical correlations
    U : (n, k) canonical variables for X
    V : (n, k) canonical variables for Y
    """

    # Center data
    Xc = center(X)
    Yc = center(Y)

    n = Xc.shape[0]
    scale = 1.0 / (n - 1)

    # Covariance matrices
    Sxx = scale * Xc.T @ Xc + reg * np.eye(Xc.shape[1])
    Syy = scale * Yc.T @ Yc + reg * np.eye(Yc.shape[1])
    Sxy = scale * Xc.T @ Yc

    # Whitening transforms
    Wx = inv_sqrtm_psd(Sxx)
    Wy = inv_sqrtm_psd(Syy)

    # Whitened cross-covariance
    T = Wx @ Sxy @ Wy

    # SVD
    U_svd, s, Vt_svd = linalg.svd(T, full_matrices=False)

    if n_components is not None:
        U_svd = U_svd[:, :n_components]
        Vt_svd = Vt_svd[:n_components, :]
        s = s[:n_components]

    # Canonical directions
    A = Wx @ U_svd
    B = Wy @ Vt_svd.T

    # Canonical variables
    U = Xc @ A
    V = Yc @ B

    return A, B, s, U, V

import numpy as np

if __name__ == "__main__":
    rng = np.random.default_rng(0)

    n = 3000

    z = rng.normal(size=(n, 2))  # shared latent factors

    # Choose feature sizes for each view
    px = 1
    qy = 5

    # Mixing matrices map latent (2) -> observed (px or qy)
    Wx = rng.normal(size=(2, px))
    Wy = rng.normal(size=(2, qy))

    # Build correlated views + noise
    X = z @ Wx + 0.5 * rng.normal(size=(n, px))   # (n, 5)
    Y = z @ Wy + 0.5 * rng.normal(size=(n, qy))   # (n, 4)

    A, B, r, U, V = cca(X, Y, n_components=2)

    print("Canonical correlations:", r)
    print("Empirical corr of first pair:",
          np.corrcoef(U[:, 0], V[:, 0])[0, 1])

# %%

import numpy as np
import matplotlib
matplotlib.use("Qt5Agg")  # for headless environments
import matplotlib.pyplot as plt
plt.ion()

# ------------------------------------------------------------
# OLS vs OCA (PCA/Total Least Squares) plots for Y.shape=(N,2)
# Optionally uses XY for a "rotation-around-mean" fairness test.
# ------------------------------------------------------------

def fit_ols_y_on_x(Y: np.ndarray):
    """
    Ordinary least squares: y2 = a*y1 + b
    Returns: a, b
    """
    x = Y[:, 0]
    y = Y[:, 1]
    A = np.vstack([x, np.ones_like(x)]).T
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]
    return a, b


def fit_oca_pca(Y: np.ndarray):
    """
    Orthogonal Component Analysis for a line in 2D via PCA:
    direction = first principal component of centered Y.
    Returns: mean (2,), direction (2,) unit vector
    """
    mu = Y.mean(axis=0)
    Yc = Y - mu
    _, _, Vt = np.linalg.svd(Yc, full_matrices=False)
    direction = Vt[0]  # unit vector
    return mu, direction


def line_from_point_dir(mu: np.ndarray, v: np.ndarray, eps: float = 1e-12):
    """
    Convert (point mu, direction v) to slope/intercept if not vertical.
    Returns:
      - (a, b) if non-vertical
      - (None, x0) if vertical line x = x0
    """
    if abs(v[0]) > eps:
        a = v[1] / v[0]
        b = mu[1] - a * mu[0]
        return a, b
    else:
        return None, mu[0]


def orthogonal_rss_to_pca_line(Y: np.ndarray, mu: np.ndarray, v: np.ndarray):
    """
    Sum of squared orthogonal distances to PCA line through mu with direction v.
    """
    Yc = Y - mu
    proj = (Yc @ v)[:, None] * v[None, :]
    orth = Yc - proj
    return float(np.sum(np.sum(orth**2, axis=1)))


def orthogonal_rss_to_ols_line(Y: np.ndarray, a: float, b: float):
    """
    Sum of squared orthogonal distances to OLS line y = a x + b.
    Distance to line ax + by + c = 0 form:
      (-a)x + (1)y + (-b) = 0
    """
    x = Y[:, 0]
    y = Y[:, 1]
    A, B, C = -a, 1.0, -b
    dist2 = (A * x + B * y + C) ** 2 / (A**2 + B**2)
    return float(np.sum(dist2))


def vertical_rss_ols(Y: np.ndarray, a: float, b: float):
    """
    OLS objective: sum of squared vertical residuals in y (second column).
    """
    x = Y[:, 0]
    y = Y[:, 1]
    return float(np.sum((y - (a * x + b)) ** 2))


def plot_ols_vs_oca(Y: np.ndarray, title="OLS vs OCA (PCA) on Y"):
    """
    Scatter of Y plus OLS line and OCA/PCA line.
    """
    if Y.ndim != 2 or Y.shape[1] != 2:
        raise ValueError("Y must have shape (N, 2).")

    # Fit
    a_ols, b_ols = fit_ols_y_on_x(Y)
    print("angle direction of OLS line:", np.arctan(a_ols)*180/np.pi)
    mu, v = fit_oca_pca(Y)
    print("vector direction of OCA/PCA line:", v)
    a_oca, b_or_x0 = line_from_point_dir(mu, v)

    # Plot range for x-axis
    x = Y[:, 0]
    xx = np.linspace(np.percentile(x, 0.5), np.percentile(x, 99.5), 200)
    yy_ols = a_ols * xx + b_ols

    plt.figure("plot OLS vs OCA",figsize=(7, 6),clear=True)
    plt.scatter(Y[:, 0], Y[:, 1], s=4, alpha=0.25)

    plt.plot(xx, yy_ols, linewidth=2, label="OLS regression (min vertical error)")

    if a_oca is not None:
        yy_oca = a_oca * xx + b_or_x0
        plt.plot(xx, yy_oca, linewidth=2, label="OCA / PCA line (min orthogonal error)")
    else:
        # vertical line
        plt.axvline(b_or_x0, linewidth=2, label="OCA / PCA line (vertical)")

    plt.title(title)
    plt.xlabel("Y[:, 0]")
    plt.ylabel("Y[:, 1]")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Print objective values
    print("Objective values on this Y:")
    print(f"  OLS vertical RSS:         {vertical_rss_ols(Y, a_ols, b_ols):,.2f}")
    print(f"  OLS orthogonal RSS:       {orthogonal_rss_to_ols_line(Y, a_ols, b_ols):,.2f}")
    print(f"  OCA (PCA) orthogonal RSS: {orthogonal_rss_to_pca_line(Y, mu, v):,.2f}")


def plot_rotation_test(Y: np.ndarray, degrees=60.0, title="Rotation test"):
    """
    Rotate Y around its mean and show OLS vs OCA again.
    Demonstrates that OCA is rotation-invariant while OLS depends on axis choice.
    """
    mu = Y.mean(axis=0)
    theta = np.deg2rad(degrees)
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])
    Y_rot = (Y - mu) @ R.T + mu

    plot_ols_vs_oca(Y_rot, title=f"{title}: rotated by {degrees:.1f}°")

    return Y_rot

# --------------------------
# EXAMPLE (synthetic dataset)
# --------------------------
if __name__ == "__main__":
    rng = np.random
    n = 100000

    # latent line direction
    true_angle = np.deg2rad(80)
    v_true = np.array([np.cos(true_angle), np.sin(true_angle)])

    t = rng.normal(0, 2.0, size=n)          # coordinate along the line
    noise = rng.normal(0, 1, size=(n, 2)) # noise in BOTH axes
    Y = t[:, None] * v_true[None, :] + noise*10 + np.array([1.0, -0.5])


    mu = Y.mean(axis=0)
    Yc = Y - mu
    _, _, Vt = np.linalg.svd(Yc, full_matrices=False)
    direction = Vt[0]  # unit vector
    v = direction
    print("angle direction of OCA/PCA line:", np.arctan(v[1]/v[0])*180/np.pi)


    # Y[:,1] += noise[:,1]*1  # more noise in y-axis
    # Plot comparisons
    plot_ols_vs_oca(Y, title="OLS vs OCA (PCA) on Y ∈ R^{10000×2}")
    
    a_ols, b_ols = fit_ols_y_on_x(Y)
    print("angle direction of OLS line:", np.arctan(a_ols)*180/np.pi)
    mu, v = fit_oca_pca(Y)
    print("angle direction of OCA/PCA line:", np.arctan(v[1]/v[0])*180/np.pi)
    print(np.tan(true_angle),"ratios of OLS vrs PCA ", a_ols, "vs", v[1]/v[0])
    print(np.rad2deg(true_angle), "vrs", np.arctan(a_ols)*180/np.pi, "vrs", np.arctan(v[1]/v[0])*180/np.pi)

                                      
    plt.ylim([-20,20])
    plt.xlim([-20,20])

    A, B, r, U, V = cca(Y[:,:1], Y[:,1:])
    # X, Y = Y[:,:1], Y[:,1:]
    # title="Rotation test (OCA invariant; OLS axis-dependent)"
    # degrees = 54.5
    # mu = Y.mean(axis=0)
    # theta = np.deg2rad(degrees)
    # R = np.array([[np.cos(theta), -np.sin(theta)],
    #               [np.sin(theta),  np.cos(theta)]])
    # Y_rot = (Y - mu) @ R.T + mu

    # plot_ols_vs_oca(Y_rot, title=f"{title}: rotated by {degrees:.1f}°")

# %%

import numpy as np

rng = np.random.default_rng(13)
n = 100000

# latent line direction
true_angle = np.deg2rad(80)
v_true = np.array([np.cos(true_angle), np.sin(true_angle)])

t = rng.normal(0, 2.0, size=n)            # coordinate along the line
noise = rng.normal(0, 1, size=(n, 2))     # isotropic noise
Y = t[:, None] * v_true[None, :] + noise * 10 + np.array([1.0, -0.5])

# center
Yc = Y - Y.mean(axis=0)

# --- empirical covariance matrix ---
C = np.cov(Yc, rowvar=False, bias=False)  # 2x2
sxx, sxy, syy = C[0, 0], C[0, 1], C[1, 1]

print("Empirical covariance C =\n", C)

# --- PCA slope beta from closed-form formula ---
# beta = vy/vx for the top eigenvector, assuming sxy != 0
disc = (syy - sxx)**2 + 4 * (sxy**2)
beta_pca = (syy - sxx + np.sqrt(disc)) / (2 * sxy)

angle_from_beta = np.degrees(np.arctan(beta_pca))  # angle of line with slope beta
print("\nbeta_PCA (closed-form):", beta_pca)
print("angle from beta_PCA (deg):", angle_from_beta)

# --- PCA direction from SVD ---
_, _, Vt = np.linalg.svd(Yc, full_matrices=False)
v = Vt[0]  # first principal direction (unit vector), sign arbitrary

angle_svd = np.degrees(np.arctan2(v[1], v[0]))
print("\nPCA direction from SVD v:", v)
print("angle from SVD direction (deg):", angle_svd)

# --- reconcile sign ambiguity (v and -v are same PC) ---
# Compare absolute angle modulo 180
def angle_mod_180(a):
    a = (a + 180) % 360 - 180
    return a

print("\nangle(beta) mod 180:", angle_mod_180(angle_from_beta))
print("angle(SVD)  mod 180:", angle_mod_180(angle_svd))
print("true angle  mod 180:", angle_mod_180(np.degrees(true_angle)))
# %%
