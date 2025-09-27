# file: ell2d_odr.py
import numpy as np
from numpy.linalg import svd, lstsq, pinv
from dataclasses import dataclass
from scipy import odr
from scipy.optimize import least_squares

@dataclass
class Ell2DModel:
    """Stores coefficients of the six quadratic functions ℓ_i(x,y)."""
    V: np.ndarray        # (n,6) base matrix [A B C D E F]
    coefs: np.ndarray    # (6,6) rows = [a,b,c,d,e,f] for each ℓ_i

    def ell_params(self):
        """Return all coefficients as a (6,6) array."""
        return self.coefs.copy()

    def ell_i(self, i):
        """Return ℓ_i(x,y) as a Python function."""
        a,b,c,d,e,f = self.coefs[i]
        return lambda x, y: a*x*x + b*y*y + c*x*y + d*x + e*y + f

    def lambda_hat(self, x, y):
        """Evaluate all six ℓ_i(x,y) at (x,y)."""
        Phi = np.array([x*x, y*y, x*y, x, y, 1.0])
        return self.coefs @ Phi

    def Y_hat(self, x, y):
        """Reconstruct Y(x,y) = V @ λ(x,y)."""
        return self.V @ self.lambda_hat(x, y)

def fit_ell2d(
    V,
    x_tilde, y_tilde,
    Y_meas,
    sigma_x=0.1,
    sigma_y=0.1,
    init_with_ridge=1e-8, ODR = False
):
    """
    Fit six quadratic functions ℓ_i(x, y) from noisy (x, y, Y) samples, where each ℓ_i(x, y) is parameterized as:
        ℓ_i(x, y) = a_i x^2 + b_i y^2 + c_i x y + d_i x + e_i y + f_i
    The goal is to model the relationship between measured vectors Y and their (x, y) positions using a known base matrix V.

    Parameters
    ----------
    V : ndarray, shape (n, 6)
        Base matrix whose columns define the subspace for the quadratic models. Each row corresponds to a vector in the measurement space.
    x_tilde, y_tilde : array-like, shape (N,)
        Noisy observed positions for each sample.
    Y_meas : ndarray, shape (N, n)
        Measured data vectors at each (x, y) position.
    sigma_x, sigma_y : float or array-like, optional
        Standard deviations of noise in x and y. Can be scalars or arrays of length N. Used only if ODR=True.
    init_with_ridge : float, optional
        Small ridge regularization parameter for stable initialization of the least-squares fit.
    ODR : bool, optional
        If True, use Orthogonal Distance Regression (ODR) to account for errors in x and y. If False, use ordinary least squares.

    Returns
    -------
    Ell2DModel
        An object containing the fitted coefficients for the six quadratic functions and the base matrix V.
    """
    N, n = Y_meas.shape
    if V.shape[0] != n:
        raise ValueError("Dimension mismatch between V and Y_meas")

    # Project Y -> Lambda
    V_pinv = np.linalg.pinv(V)
    Lambda = (V_pinv @ Y_meas.T).T  # (N,6)
    
    # --- Initialisation par OLS (ridge léger) : Lambda ≈ Phi @ (coefs0.T) ---
    # Matrice de features quadratiques
    Phi = np.column_stack([
        x_tilde**2,            # x^2
        y_tilde**2,            # y^2
        x_tilde * y_tilde,     # x*y
        x_tilde,               # x
        y_tilde,               # y
        np.ones_like(x_tilde)  # 1
    ])  # shape (N,6)

    # Ridge très léger pour stabiliser si Phi^T Phi est près-singulière
    G = Phi.T @ Phi + init_with_ridge * np.eye(6)

    # Solution fermée vectorisée pour les 6 fonctions :
    # Pour chaque i: beta_i = (Phi^T Phi + αI)^(-1) Phi^T Lambda[:, i]
    # On les empile d'un coup : coefs0.T = solve(G, Phi^T @ Lambda)
    coefs0 = np.linalg.solve(G, Phi.T @ Lambda).T   # shape (6,6)
    # NB: chaque LIGNE i de coefs0 est [a_i,b_i,c_i,d_i,e_i,f_i]


    # -- Build 2×N input-error array for ODR --
    s_x = np.full(N, sigma_x) if np.isscalar(sigma_x) else np.asarray(sigma_x, dtype=float).reshape(-1)
    s_y = np.full(N, sigma_y) if np.isscalar(sigma_y) else np.asarray(sigma_y, dtype=float).reshape(-1)
    if s_x.shape != (N,) or s_y.shape != (N,):
        raise ValueError("sigma_x/sigma_y must be scalar or length-N arrays.")
    sx_inputs = np.vstack([s_x, s_y]).astype(float)   # shape (2, N)

    # -- Prepare input matrix (2, N) --
    xy_inputs = np.vstack([x_tilde, y_tilde]).astype(float)

    if not np.all(np.isfinite(xy_inputs)):
        raise ValueError("x_tilde/y_tilde contain NaN/Inf.")
    if not np.all(np.isfinite(Lambda)):
        raise ValueError("Projected Lambda contains NaN/Inf (check V rank and Y_meas).")
    if np.any(sx_inputs <= 0) or not np.all(np.isfinite(sx_inputs)):
        raise ValueError("Input std-devs must be positive and finite.")

    coefs = np.zeros((6, 6))
    if ODR:
        # Quadratic model for ODR
        def quad2d_beta(beta, data):
            x, y = data
            return (beta[0]*x**2 + beta[1]*y**2 + beta[2]*x*y +
                    beta[3]*x + beta[4]*y + beta[5])
        
        # -- Run ODR for each function i --
        for i in range(6):
            z = np.asarray(Lambda[:, i], dtype=float).reshape(-1)   # output (N,)
            data = odr.RealData(xy_inputs, z, sx=sx_inputs, sy=None)  # sy=None = no output error
            model = odr.Model(quad2d_beta)
            odr_run = odr.ODR(data, model, beta0=coefs0[i]).run()
            coefs[i] = odr_run.beta
    else:
        for i in range(6):
            coefs[i] = coefs0[i]


    return Ell2DModel(V=V, coefs=coefs)


import numpy as np
from scipy.optimize import least_squares

def invert_lambda_batch(Lambda_tgt, C, 
                        n_starts=1, jitter=0.1, random_state=0):
    """
    Estimate (x,y) for each Lambda_tgt[k] such that C @ phi(x,y) ≈ Lambda_tgt[k].

    Parameters
    ----------
    Lambda_tgt : (N,6) array
        Target λ vectors.
    C : (6,6) array
        Coefficient matrix (rows = [a,b,c,d,e,f]).
    n_starts : int
        Number of random restarts per point.
    jitter : float
        Std of Gaussian jitter added to initialization.
    random_state : int
        RNG seed.

    Returns
    -------
    X, Y : arrays of shape (N,)
    residual_norm : array of shape (N,)
    success : array of bools
    """
    rng = np.random.default_rng(random_state)
    Lambda_tgt = np.asarray(Lambda_tgt, dtype=float)
    N, d = Lambda_tgt.shape
    if d != 6:
        raise ValueError("Lambda_tgt must have shape (N,6).")
    C = np.asarray(C, dtype=float)
    if C.shape != (6,6):
        raise ValueError("C must be (6,6).")

    def phi(k,x,y):
        return np.array([x*x, y*y, x*y, x, y, k], dtype=float)
    def dphi_dx(k,x,y):
        return np.array([2*x, 0, y, 1, 0, 0], dtype=float)
    def dphi_dy(k,x,y):
        return np.array([0, 2*y, x, 0, 1, 0], dtype=float)
    def dphi_dk(k,x,y):
        return np.array([0, 0, 0, 0, 0, 1], dtype=float)

    X_hat = np.zeros(N)
    Y_hat = np.zeros(N)
    res_norms = np.zeros(N)
    success = np.zeros(N, dtype=bool)

    def jac_fun(p):
        k,x,y = p
        return np.column_stack((C @ dphi_dk(k,x,y),
                                C @ dphi_dx(k,x,y),
                                C @ dphi_dy(k,x,y))) 
    for k in range(N):

        def res_fun(p):
            k,x,y = p
            return C @ phi(k,x,y) - Lambda_tgt[k]

        best = None
        for _ in range(max(1,n_starts)):
            p0 = (1, jitter * rng.normal(size=2))
            res = least_squares(res_fun, p0, jac=jac_fun,
                                method="lm")
            if (best is None) or (res.cost < best.cost):
                best = res

        X_hat[k] = best.x[1]/best.x[0]
        Y_hat[k] = best.x[0]/best.x[0]
        res_norms[k] = np.linalg.norm(best.fun)
        success[k] = best.success

    return X_hat, Y_hat, res_norms, success

def svd_subspace(X, k, center=True):
    if X.ndim != 2:
        raise ValueError("X must be 2D (m x n)")
    m, n = X.shape
    if k < 1 or k > min(m, n):
        raise ValueError(f"k must be in [1, {min(m, n)}]")
    mu = X.mean(axis=0) if center else np.zeros(n, dtype=X.dtype)
    Xc = X - mu if center else X
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    V = Vt[:k].T
    var = S**2
    explained = var / var.sum() if var.sum() > 0 else np.zeros_like(var)
    model = {"mu": mu, "V": V, "S": S, "explained_var_ratio": explained, "k": int(k), "center": bool(center)}
    return model, Xc

def project(X, model):
    if X.ndim != 2:
        raise ValueError("X must be 2D (m x n)")
    V, mu, center = model["V"], model["mu"], model["center"]
    Xc = X - mu if center else X
    Xc_proj = (Xc @ V) @ V.T
    X_proj = Xc_proj + (mu if center else 0)
    residuals = X - X_proj
    errors = np.linalg.norm(residuals, axis=1)
    return X_proj, residuals, errors

def mad_threshold(errors, k_sigma=3.5):
    if errors.ndim != 1:
        raise ValueError("errors must be 1D")
    med = np.median(errors)
    mad = np.median(np.abs(errors - med))
    c = 1.4826
    robust_sigma = c * mad if mad > 0 else (np.std(errors) if errors.size > 1 else 0.0)
    thr = med + k_sigma * robust_sigma
    mask = errors <= thr
    return float(thr), mask

def robust_subspace(X, k=6, center=True, k_sigma=3.5, max_refit=1, verbose=False):
    if verbose:
        print(f"Robust subspace estimation with Nsingular={k}, center={center}, k_sigma={k_sigma}, max_refit={max_refit}")
    model, _ = svd_subspace(X, k=k, center=center)
    if verbose:
        print(f"Subspace computed.")
    X_proj, residuals, errors = project(X, model)
    thr, mask_in = mad_threshold(errors, k_sigma=k_sigma)
    if verbose:
        print(f"Initial threshold: {thr:.4f}. Inliers: {np.sum(mask_in)} / {X.shape[0]}")
    inliers_idx = np.where(mask_in)[0]
    outliers_idx = np.where(~mask_in)[0]

    refits = 0
    while refits < max_refit and inliers_idx.size >= k:
        model, _ = svd_subspace(X[inliers_idx], k=k, center=center)
        X_proj, residuals, errors = project(X, model)
        thr, mask_in = mad_threshold(errors, k_sigma=k_sigma)
        inliers_idx = np.where(mask_in)[0]
        outliers_idx = np.where(~mask_in)[0]
        refits += 1
        if verbose:
            print(f"Refit {refits}: threshold: {thr:.4f}, {inliers_idx.size} / {X.shape[0]} inliers")

    return {
        "model": model,
        "X_proj": X_proj,
        "residuals": residuals,
        "errors": errors,
        "threshold": thr,
        "inliers_mask": mask_in,
        "inliers_idx": inliers_idx,
        "outliers_idx": outliers_idx,
    }


