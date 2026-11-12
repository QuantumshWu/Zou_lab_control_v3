"""The tweezer's focal spot from vector diffraction (Richards & Wolf, Proc. R. Soc. A 253, 358 (1959)): an aplanatic lens
of numerical aperture NA, a linearly polarized Gaussian input of fill factor f0 (its waist over the aperture radius), the
intensity |E|² in the focal plane along the polarization and across it -- how much a linearly polarized focus is
elongated along its polarization, at the fill that gives a scene's waist.

    python focus.py [--na 0.6] [--nm 852] --waist <µm> [--waist <µm> ...]

The focal-plane field (polarization along X, azimuth φ from X; Richards & Wolf; Novotny & Hecht, Principles of
Nano-Optics, §3.6): E_X ∝ I0 + I2 cos 2φ, E_Y ∝ I2 sin 2φ, E_Z ∝ −2i I1 cos φ, with
I0 = ∫ A sinθ (1 + cosθ) J0(k r sinθ) dθ, I1 = ∫ A sin²θ J1(k r sinθ) dθ, I2 = ∫ A sinθ (1 − cosθ) J2(k r sinθ) dθ over
0..arcsin NA, A(θ) = √cosθ exp(−sin²θ / (f0² NA²)). Printed per waist: the fill f0 whose spot's geometric-mean 1/e² radius
is that waist, the 1/e² radii along and across the polarization, and the curvature radii w = √(−4 I / I'') at the centre
(what sets the radial trap frequencies, ω ∝ 1/w)."""
import argparse, math
import numpy as np
from scipy.special import jv
from scipy.optimize import brentq

def spot(na, lam_um, f0, r):
    """|E|² along (φ = 0) and across (φ = 90°) the polarization at the radii r (µm), normalized to the centre."""
    a = math.asin(na); th = np.linspace(0.0, a, 401)[:, None]; k = 2 * math.pi / lam_um
    A = np.sqrt(np.cos(th)) * np.exp(-np.sin(th) ** 2 / (f0 * na) ** 2)
    u = k * np.asarray(r)[None, :] * np.sin(th)
    w = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz
    I0 = w(A * np.sin(th) * (1 + np.cos(th)) * jv(0, u), th[:, 0], axis=0)
    I1 = w(A * np.sin(th) ** 2 * jv(1, u), th[:, 0], axis=0)
    I2 = w(A * np.sin(th) * (1 - np.cos(th)) * jv(2, u), th[:, 0], axis=0)
    along = (I0 + I2) ** 2 + 4 * I1 ** 2; across = (I0 - I2) ** 2
    return along / along[0], across / across[0]

def radii(na, lam_um, f0):
    r = np.concatenate([[0.0, 1e-3], np.linspace(0.005, 6.0, 400)])
    al, ac = spot(na, lam_um, f0, r)
    def e2(I):
        n = int(np.argmax(I < math.exp(-2))); return float(np.interp(-math.exp(-2), -I[:n + 1], r[:n + 1]))
    c = lambda I: math.sqrt(-4.0 / ((I[1] - 1.0) * 2 / r[1] ** 2))   # I ≈ 1 + I'' r²/2 at r = 1 nm
    return e2(al), e2(ac), c(al), c(ac)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--na', type=float, default=0.6); ap.add_argument('--nm', type=float, default=852.0)
    ap.add_argument('--waist', type=float, action='append', required=True); a = ap.parse_args()
    lam = a.nm * 1e-3
    for w0 in a.waist:
        def mean(f):
            ra = radii(a.na, lam, f); return math.sqrt(ra[0] * ra[1]) - w0
        f0 = brentq(mean, 0.2, 20.0, xtol=1e-4)
        wa, wc, ca, cc = radii(a.na, lam, f0)
        print(f"NA {a.na}, {a.nm:.0f} nm, waist {w0} µm: fill f0 {f0:.3f}; 1/e² radius along the polarization {wa:.4f} µm, across {wc:.4f} µm "
              f"(ratio {wa / wc:.4f}); curvature radius along {ca:.4f}, across {cc:.4f} (ratio {ca / cc:.4f}: radial frequencies differ by {cc / ca - 1:+.2%})")

if __name__ == '__main__':
    main()
