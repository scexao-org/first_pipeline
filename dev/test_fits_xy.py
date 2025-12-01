#%%

import numpy as np
from scipy.optimize import minimize

# ---------- Vandermonde & bases ----------
def _vandermonde_xy(x_nodes, y_nodes):
    x = np.asarray(x_nodes, dtype=float); y = np.asarray(y_nodes, dtype=float)
    return np.column_stack([x*x, x*y, y*y, x, y, np.ones_like(x)])

def _build_M(x_nodes, y_nodes, rcond=1e12):
    Phi = _vandermonde_xy(x_nodes, y_nodes)
    try:
        cond = np.linalg.cond(Phi)
    except np.linalg.LinAlgError:
        cond = np.inf
    return np.linalg.inv(Phi.T) if (np.isfinite(cond) and cond < rcond) else np.linalg.pinv(Phi.T)

def _phi_xy(x, y):
    return np.array([x*x, x*y, y*y, x, y, 1.0], dtype=float)

def _dphi_xy(x, y):
    dphix = np.array([2*x,  y , 0.0, 1.0, 0.0, 0.0], dtype=float)
    dphiy = np.array([0.0,  x , 2*y, 0.0, 1.0, 0.0], dtype=float)
    return dphix, dphiy

def _ddphi_xy():
    # Hessienne des monômes (constante)
    ddxx = np.array([2.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
    ddyy = np.array([0.0, 0.0, 2.0, 0.0, 0.0, 0.0], dtype=float)
    ddxy = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
    return ddxx, ddyy, ddxy

# ---------- J, ∇J, (option) ∇²J ----------
def make_objectives(x_nodes, y_nodes, G, k, s):
    M = _build_M(x_nodes, y_nodes)
    ddxx, ddyy, ddxy = _ddphi_xy()
    G = np.asarray(G, float); k = np.asarray(k, float).ravel(); s = float(s)

    def fun(z):
        x, y = float(z[0]), float(z[1])
        phi = _phi_xy(x, y)
        ell = M @ phi
        return s - 2.0*(ell @ k) + ell @ (G @ ell)

    def jac(z):
        x, y = float(z[0]), float(z[1])
        dphix, dphiy = _dphi_xy(x, y)
        ex = M @ dphix
        ey = M @ dphiy
        # On évite de recalculer ell via fun pour rester auto-contenu
        phi = _phi_xy(x, y); ell = M @ phi
        dJx = -2.0*(ex @ k) + 2.0*(ex @ (G @ ell))
        dJy = -2.0*(ey @ k) + 2.0*(ey @ (G @ ell))
        return np.array([dJx, dJy])

    def hess(z):
        # Hessienne exacte (utile pour 'trust-constr'/'trust-ncg')
        x, y = float(z[0]), float(z[1])
        phi = _phi_xy(x, y); ell = M @ phi
        dphix, dphiy = _dphi_xy(x, y)
        ex = M @ dphix; ey = M @ dphiy
        ddxx_l = M @ ddxx   # d²ℓ/dx²
        ddyy_l = M @ ddyy   # d²ℓ/dy²
        ddxy_l = M @ ddxy   # d²ℓ/dxdy

        # Formules:
        # ∂xx J = -2(ddxx_l^T k) + 2[(ddxx_l^T G ell) + (ex^T G ex)]
        # ∂yy J = -2(ddyy_l^T k) + 2[(ddyy_l^T G ell) + (ey^T G ey)]
        # ∂xy J = -2(ddxy_l^T k) + 2[(ddxy_l^T G ell) + (ex^T G ey)]
        J_xx = -2*(ddxx_l @ k) + 2*((ddxx_l @ (G @ ell)) + (ex @ (G @ ex)))
        J_yy = -2*(ddyy_l @ k) + 2*((ddyy_l @ (G @ ell)) + (ey @ (G @ ey)))
        J_xy = -2*(ddxy_l @ k) + 2*((ddxy_l @ (G @ ell)) + (ex @ (G @ ey)))
        return np.array([[J_xx, J_xy],
                         [J_xy, J_yy]], dtype=float)

    return fun, jac, hess

# ---------- Solveur SciPy (multi-start, rapide) ----------
def argmin_xy_quadratic_interp_noisy_scipy(
    x_nodes, y_nodes, G, k, s,
    method="L-BFGS-B",    # "L-BFGS-B" (rapide), ou "trust-constr"/"trust-ncg" (avec Hessienne)
    box_pad=0.10,         # boîte de recherche = bbox des nœuds ± pad
    starts_grid=3,        # multi-start sur une grille
    max_iter=200,
    tol=1e-10,
    verbose=False
):
    x_nodes = np.asarray(x_nodes, float).ravel()
    y_nodes = np.asarray(y_nodes, float).ravel()
    assert x_nodes.shape == (6,) and y_nodes.shape == (6,), "Besoin de 6 nœuds pour le quadratique 2D."
    fun, jac, hess = make_objectives(x_nodes, y_nodes, G, k, s)

    # Boîte de recherche raisonnable (accélère L-BFGS-B)
    xmin, xmax = x_nodes.min(), x_nodes.max()
    ymin, ymax = y_nodes.min(), y_nodes.max()
    dx, dy = xmax - xmin, ymax - ymin
    xmin -= box_pad * dx; xmax += box_pad * dx
    ymin -= box_pad * dy; ymax += box_pad * dy
    bounds = [(xmin, xmax), (ymin, ymax)]

    # Grille de départs
    xs = np.linspace(xmin, xmax, starts_grid)
    ys = np.linspace(ymin, ymax, starts_grid)
    starts = [(xi, yi) for xi in xs for yi in ys]

    best = (np.inf, None, None)
    for (x0, y0) in starts:
        if method.lower().startswith("trust"):
            res = minimize(fun, x0=[x0, y0], method=method,
                           jac=jac, hess=hess, bounds=bounds if method=="trust-constr" else None,
                           options={"maxiter": max_iter, "gtol": tol, "verbose": 3 if verbose else 0})
        else:
            # L-BFGS-B (rapide, mémoire économe), avec bornes
            res = minimize(fun, x0=[x0, y0], method="L-BFGS-B",
                           jac=jac, bounds=bounds,
                           options={"maxiter": max_iter, "ftol": tol, "gtol": tol, "iprint": 1 if verbose else -1})
        if res.fun < best[0]:
            best = (res.fun, res.x[0], res.x[1])

    Jmin, xstar, ystar = best
    return xstar, ystar, Jmin

# ---------- Exemple rapide ----------
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    p = 4
    # f(x,y) = A x^2 + B xy + C y^2 + D x + E y + F
    A = rng.normal(size=p); B = rng.normal(size=p); C = rng.normal(size=p)
    D = rng.normal(size=p); E = rng.normal(size=p); F = rng.normal(size=p)
    def f_true(xx, yy): return A*xx*xx + B*xx*yy + C*yy*yy + D*xx + E*yy + F

    x_nodes = np.array([-1.0, 1.0, 0.0,  2.0, -1.5, 0.5])
    y_nodes = np.array([ 0.5,-0.5,1.5, -1.0,  1.0,-1.5])
    Y = np.stack([f_true(x_nodes[i], y_nodes[i]) for i in range(6)], axis=0)  # (6,p)

    G = Y @ Y.T
    x_true, y_true = 0.2, -0.3
    y_tgt = f_true(x_true, y_true) + 0.05*rng.normal(size=p)
    k = Y @ y_tgt
    s = float(y_tgt @ y_tgt)

    xhat, yhat, Jmin = argmin_xy_quadratic_interp_noisy_scipy(
        x_nodes, y_nodes, G, k, s,
        method="L-BFGS-B", starts_grid=4, box_pad=0.15, verbose=True
    )
    print("Estimation (x*, y*) =", (xhat, yhat))
    print("J* =", Jmin)
# %%
