"""Physical simulation of the lantern data set with PSF jitter and deformation.

Each output o has a smooth coupling map F_o(p) (sum of Gaussian lobes, width
~ PSF).  Pose k is observed at p_k + delta_k (pointing jitter) with a
per-output, per-pose multiplicative deformation (1 + eps_ko).  On the line
channels the emission comes from a source displaced by ``a`` from the star,
so the exact line flux is F_o(p_k + delta_k + a) scaled by the line profile.
"""
import numpy as np

def make_maps(Nout, rng, fwhm=22.0, n_lobes=2, extent=25.0):
    lobes = []
    for o in range(Nout):
        centres = rng.uniform(-extent, extent, size=(n_lobes, 2))
        amps = rng.uniform(0.3, 1.0, size=n_lobes)
        widths = fwhm / 2.355 * rng.uniform(0.8, 1.4, size=n_lobes)
        lobes.append((centres, amps, widths))
    def F(p, o):                       # p (..., 2) -> flux (...)
        centres, amps, widths = lobes[o]
        r2 = ((p[..., None, :] - centres) ** 2).sum(-1)
        return 0.05 + (amps * np.exp(-0.5 * r2 / widths ** 2)).sum(-1)
    return F

def simulate(ra_dec, wave, line_center, line_width, a_true, Nout=19,
             jitter=3.5, deform=0.4, photons=3000.0, seed=0, line_ratio=4.0,
             profile=None):
    rng = np.random.default_rng(seed)
    Ncube, Npose = ra_dec.shape[:2]
    Nwave = wave.size
    F = make_maps(Nout, rng)
    if profile is None:   # line profile relative to the continuum (use the measured one when possible)
        profile = 1 + (line_ratio - 1) * np.exp(-0.5 * ((wave - line_center) / (line_width / 2.355 * 0.55)) ** 2)
    delta = rng.normal(0, jitter, size=(Ncube, Npose, 2))
    q = ra_dec + delta
    eps = np.exp(deform * rng.normal(size=(Ncube, Npose, Nout)))   # log-normal, stays positive
    cube = np.zeros((Ncube, Npose, Nout, Nwave))
    for o in range(Nout):
        cont = F(q, o) * eps[..., o]                          # (Ncube, Npose)
        line = F(q + a_true, o) * eps[..., o]                 # line photons come from p + a
        cube[:, :, o, :] = cont[..., None] + (profile - 1)[None, None, :] * line[..., None]
    var = cube / photons + (0.02) ** 2                        # photon + read noise
    cube = cube + rng.normal(size=cube.shape) * np.sqrt(var)
    return cube, var, delta, F
