import numpy as np
from scipy.optimize import least_squares
from scipy.linalg import solve_triangular


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
    model = {"mu": mu, "U": U ,"V": V, "S": S, "explained_var_ratio": explained, "k": int(k), "center": bool(center)}
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
        if verbose:
            print(f"Refit {refits}: {inliers_idx.size} inliers, recomputing subspace...")
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



def fit_QR_6(QTdata, R, init=None, bounds=None, lsq_kwargs=None):
    """
    Solve: minimize over (x,y) the residual r(x,y) = k*(R @ phi(x,y)) - b,
    where k is eliminated in closed form: k = (b^T R phi) / ||R phi||^2.

    Parameters
    ----------
    data : (6,) array-like
        Your data vector.
    QT   : (6,6) array-like
        Orthogonal Q^T (so b = Q^T data).
    R    : (6,6) array-like, upper-triangular (non-singular on relevant subspace)
    init : (x0, y0) iterable or None
        Optional single initial guess. If None, robust inits are auto-generated.
    bounds : 2-tuple of array-like or None
        Bounds for (x,y), e.g. ([-10,-10], [10,10]). Passed to least_squares if given.
    lsq_kwargs : dict or None
        Extra kwargs forwarded to least_squares (e.g., ftol, xtol, verbose).

    Returns
    -------
    x_hat, y_hat, k_hat, cost
    """
    b = QTdata  # shape (6,)
    Nqr = len(b)  # should be 
    if Nqr !=6 :
        raise ValueError("data and QT must have length 6 (pyramids)")
    
    # --- basis and helpers ---
    def phi_vec(x, y):
        return np.array([1.0, x, y, x*y, x*x, y*y], dtype=float)

    def optimal_k(R, b, x, y):
        v = phi_vec(x, y)
        Rv = R @ v
        denom = float(Rv @ Rv)
        if denom < 1e-14:
            return 0.0
        return float((b @ Rv) / denom)

    def analytic_cost(x, y):
        # ||k R phi - b||^2 = ||b||^2 - (b^T R phi)^2 / ||R phi||^2
        v = phi_vec(x, y)
        Rv = R @ v
        denom = float(Rv @ Rv)
        if denom < 1e-14:
            return float(b @ b)  # k = 0 => cost = ||b||^2
        num = float(b @ Rv)
        return float((b @ b) - (num * num) / denom)

    def resid(z):
        x, y = float(z[0]), float(z[1])
        k = optimal_k(R, b, x, y)
        return k * (R @ phi_vec(x, y)) - b  # shape (6,)


    def jac(z):
        # analytic jacobian of residual wrt x,y
        x, y = float(z[0]), float(z[1])
        v = phi_vec(x, y)
        dv_dx = np.array([0.0, 1.0, 0.0, y, 2.0*x, 0.0])
        dv_dy = np.array([0.0, 0.0, 1.0, x, 0.0, 2.0*y])

        Rv = R @ v
        denom = float(Rv @ Rv)
        if denom < 1e-14:
            return np.zeros((Nqr, 2), dtype=float)

        num = float(b @ Rv)
        dRv_dx = R @ dv_dx
        dRv_dy = R @ dv_dy
        # derivatives of k wrt x,y
        dk_dx = (float(b @ dRv_dx) * denom - num * (2.0 * float(Rv @ dRv_dx))) / (denom**2)
        dk_dy = (float(b @ dRv_dy) * denom - num * (2.0 * float(Rv @ dRv_dy))) / (denom**2)

        k = num / denom
        dr_dx = dk_dx * Rv + k * dRv_dx
        dr_dy = dk_dy * Rv + k * dRv_dy
        return np.column_stack([dr_dx, dr_dy])

    # --- better initializations ---
    def build_inits():
        cands = [(0.0, 0.0)]  # always include safe fallback
        # v* ∝ R^{-1} b
        try:
            cs = solve_triangular(R, b, lower=False)  # shape (6,)
        except np.linalg.LinAlgError:
            return cands

        # Normalize so t[0] ~ 1 to match phi[0]=1
        if abs(cs[0]) > 1e-14:
            t = cs / cs[0]
        else:
            t = cs.copy()
            t[0] = 1.0

        def safe_sqrt(u):
            return np.sqrt(u) if u > 0 else np.nan

        # Linear, quadratic, and cross-based guesses
        xL, yL = float(t[1]), float(t[2])
        xQ = float(np.sign(xL)) * safe_sqrt(float(t[4]))
        yQ = float(np.sign(yL)) * safe_sqrt(float(t[5]))
        xC = float(t[3] / yL) if abs(yL) > 1e-12 else np.nan
        yC = float(t[3] / xL) if abs(xL) > 1e-12 else np.nan

        cands.append((xL, yL))
        if not np.isnan(xQ) and not np.isnan(yQ):
            cands.append((xQ, yQ))
        if not np.isnan(xQ):
            cands.append((xQ, yL))
        if not np.isnan(yQ):
            cands.append((xL, yQ))
        if not np.isnan(xC) and not np.isnan(yC):
            cands.append((xC, yC))

        # Robust median combo
        vx = [v for v in [xL, xQ, xC] if not np.isnan(v)]
        vy = [v for v in [yL, yQ, yC] if not np.isnan(v)]
        if vx and vy:
            cands.append((float(np.median(vx)), float(np.median(vy))))

        # Tiny projection refinement on phi(x,y) ≈ t
        def refine_xy(x, y, iters=2):
            for _ in range(iters):
                vx = np.array([0, 1, 0, y, 2*x, 0], float)
                vy = np.array([0, 0, 1, x, 0, 2*y], float)
                r  = np.array([1, x, y, x*y, x*x, y*y], float) - t
                J  = np.column_stack([vx, vy])
                H  = J.T @ J + 1e-8 * np.eye(2)
                g  = J.T @ r
                try:
                    dx = -np.linalg.solve(H, g)
                except np.linalg.LinAlgError:
                    break
                x += dx[0]; y += dx[1]
            return float(x), float(y)

        refined = [refine_xy(xi, yi, iters=2) for (xi, yi) in cands]

        # Uniquify
        def uniq(seq, tol=1e-10):
            out = []
            for x in seq:
                if not any(np.hypot(x[0]-y[0], x[1]-y[1]) < tol for y in out):
                    out.append(x)
            return out

        return uniq(cands + refined)

    # assemble initial guesses
    if init is not None:
        inits = [tuple(map(float, init))]
    else:
        inits = build_inits()

    # Pre-screen by analytic cost; try best few first
    scored = sorted(((analytic_cost(x, y), (x, y)) for (x, y) in inits), key=lambda t: t[0])
    # Limit to a reasonable number to keep it fast
    scored = scored[:12]
    inits_ordered = [xy for _, xy in scored]

    # Prepare lsq kwargs
    if lsq_kwargs is None:
        lsq_kwargs = {}
    if bounds is not None:
        lsq_kwargs = dict(lsq_kwargs)  # copy
        lsq_kwargs.setdefault("bounds", bounds)

    # Solve from multiple starts; keep best
    best_res = None
    for x0 in inits_ordered:
        try:
            if bounds is None:
                res = least_squares(resid, x0=np.array(x0, dtype=float),
                                    jac=jac, method="lm", **lsq_kwargs)
            else:
                res = least_squares(resid, x0=np.array(x0, dtype=float),
                                jac=jac, method="trf", **lsq_kwargs)
        except Exception:
            continue
        if (best_res is None) or (res.cost < best_res.cost):
            best_res = res

    if best_res is None:
        # Fallback: return (0,0,k=0) with cost ||b||^2
        return 0.0, 0.0, 0.0, float(b @ b)

    x_hat, y_hat = best_res.x
    k_hat = optimal_k(R, b, x_hat, y_hat)
    return float(x_hat), float(y_hat), float(k_hat), float(best_res.cost), resid(best_res.x)


def solve_QR_3(QTdata, R):
    """

    x_hat, y_hat, k_hat, cost
    """
    
    b = QTdata  # shape (6,)
    Nqr = len(b)  # should be 
    if Nqr !=3 :
        raise ValueError("data and QT must have length 3 (triangles)")
    
    # Check for NaN values in QTdata
    if np.any(np.isnan(b)):
        return np.nan, np.nan, np.nan, np.nan, np.full_like(b, np.nan)
    
    cs = solve_triangular(R, b, lower=False)  # shape (6,)
    t = cs / cs[0]
    x_hat, y_hat, k_hat = float(t[1]), float(t[2]), float(cs[0])
    resid = k_hat * (R @ np.array([1.0, x_hat, y_hat]) - b)
    cost = float(np.sum(resid**2))
    return x_hat, y_hat, k_hat, cost, resid


def solve_QR_2(QTdata, R):
    """

    x_hat, y_hat
    """
    b = QTdata  # shape (6,)
    Nqr = len(b)  # should be 
    if Nqr !=3 :
        raise ValueError("data and QT must have length 3")
    
    cs = solve_triangular(R, b, lower=False)  # shape (6,)
    x_hat, y_hat, = float(cs[0]), float(cs[1])
    resid = R @ np.array([x_hat, y_hat]) - b
    cost = float(np.sum(resid**2))
    return x_hat, y_hat, cost, resid


def flux_filtering(flux):
    
    # select data only above a threshold based on flux
    flux_threshold=np.nanpercentile(flux.mean(axis=(2)),80)/5
    flux_goodData=np.nanmean(flux,axis=(2)) > flux_threshold
    # plt.imshow(flux_goodData)
    if np.sum(flux_goodData)<57:
        #too little good data, we need to lower the bar
        flux_threshold /= 2
        print("Not enough good data, lowering the threshold to ",flux_threshold)
        flux_goodData=np.nanmean(flux,axis=(2)) > flux_threshold

    return flux_goodData,flux_threshold


def svd_filtering(datacube,flux_goodData,Nsingular=19*6,verbose=True):

    datacube_flux_goodData = datacube[flux_goodData]
    datacube_flux_goodData = datacube_flux_goodData.reshape((datacube_flux_goodData.shape[0], -1))
    if Nsingular >= len(datacube_flux_goodData):
        Nsingular = len(datacube_flux_goodData) - 1
    res = robust_subspace(datacube_flux_goodData, k=Nsingular, center=True, k_sigma=2.5, max_refit=1,verbose=verbose)
    singular_values = res["model"]["S"][:-1]
    data_svdfiltered, residuals, errors = project(datacube.reshape((datacube.shape[0]*datacube.shape[1], -1)), res["model"])
    data_svdfiltered = data_svdfiltered.reshape(datacube.shape)
    fit_goodData = errors.reshape((datacube.shape[0], -1)) < res["threshold"]
    

    return data_svdfiltered,fit_goodData,errors