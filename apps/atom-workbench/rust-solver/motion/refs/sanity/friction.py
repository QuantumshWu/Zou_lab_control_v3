"""The light force on an atom at constant velocity: the textbook friction test (Dalibard & Cohen-Tannoudji, JOSA B 6,
2023 (1989); Ungar, Weiss, Riis, Chu, JOSA B 6, 2058 (1989)), with coherent.py's Hamiltonian (F=2 + F'=3, model.py's
tables) and the master equation instead of quantum jumps -- no noise: the density matrix along the path r(t) = r0 + v t û,
û the probe's axis, its force −Tr(ρ ∇H) averaged over whole periods of the light's pattern after the transient.

    python friction.py --request <S3 or S4 request JSON> [--case free|shifts|field|trap] [--field G] [--out <json>]

Cases (the request's beams, s and offset):
- free: no tweezer (its shifts zero), B = 0 (or --field);
- shifts: the tweezer's focus shifts as uniform constants -- ground and m'-dependent excited --, no gradient;
- field: as `shifts`, with the request's field (or --field);
- trap: the atom passing through the focus along û with the tweezer's shifts at its place; the force less the ground
  trap's own (U0 ∇g), averaged over the pass from −2 w0 to +2 w0.
Each speed is run both ways (±v): the force's odd part F_odd = (F(v) − F(−v))/2 is the friction, its even part
F_even = (F(v) + F(−v))/2 the velocity-independent push (the two beams scattering unequally: a π and a σ beam at different
detunings). Printed per speed: F_odd and F_even in ħkΓ, and α/(ħk²) = −F_odd/(ħ k² v) (dimensionless, positive where it
cools; the capture velocity is where −F_odd peaks); and the two-level Doppler and the lin⊥lin / σ+σ− Sisyphus (low
intensity, J=1/2→3/2 and J=1→2) values of α/(ħk²) at the same s and δ for comparison.
"""
import argparse, json, math, sys, os
import numpy as np
import numba as nb
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import coherent as C
M = C.M

@nb.njit(inline='always')
def deriv(H, rho, Cg, gamma, out):
    """dρ/dt = −i(H ρ − ρ H†) + Σ_q L_q ρ L_q† (H the non-Hermitian generator, L_q = √Γ Σ_m <2 m;1 q|3 m+q> |g_m><e_{m+q}|)."""
    for i in range(12):
        for j in range(12):
            acc = 0j
            for l in range(12):
                acc += H[i, l] * rho[l, j] - rho[i, l] * H[j, l].conjugate()
            out[i, j] = -1j * acc
    for qi in range(3):
        for a in range(5):
            ea = a - 2 + qi - 1
            if ea < -3 or ea > 3: continue
            for b in range(5):
                eb = b - 2 + qi - 1
                if eb < -3 or eb > 3: continue
                out[a, b] += gamma * Cg[qi, a] * Cg[qi, b] * rho[5 + ea + 3, 5 + eb + 3]

@nb.njit(parallel=True)
def forces(vs, P, H0, diagL, Cg, k, eps, Om, u, mode, gfix, dt, t_trans, t_avg, out_F, out_rate):
    """For each velocity: out_F[i] the force along û per ħ (rad/µs per µm), averaged over t_avg after t_trans;
    out_rate[i] the scattering rate Γ ρ_ee. mode 0: g fixed at gfix, no gradient; mode 1: the pass through the focus."""
    gamma, w0, zR, U0 = P[0], P[1], P[2], P[3]
    nb_ = k.shape[0]
    for iv in nb.prange(len(vs)):
        v = vs[iv]
        H = np.zeros((12, 12), np.complex128); Hb = np.zeros((nb_, 12, 12), np.complex128)
        rho = np.zeros((12, 12), np.complex128)
        for a in range(5): rho[a, a] = 0.2
        k1 = np.zeros((12, 12), np.complex128); k2 = np.zeros((12, 12), np.complex128); k3 = np.zeros((12, 12), np.complex128)
        k4 = np.zeros((12, 12), np.complex128); tmp = np.zeros((12, 12), np.complex128)
        n = int((t_trans[iv] + t_avg[iv]) / dt); n0 = int(t_trans[iv] / dt)
        start = -2.0 * w0 * math.copysign(1.0, v) if mode == 1 else 0.0
        if mode == 1:
            n0 = 0; n = int(4.0 * w0 / abs(v) / dt)
        Fsum = 0.0; Rsum = 0.0; cnt = 0
        for s in range(n):
            t = s * dt
            for half in range(3):
                tt = t + 0.5 * dt * half
                d = start + v * tt
                x = d * u[0]; y = d * u[1]; z = d * u[2]
                g, gx, gy, gz = C.hamiltonian(x, y, z, P, H0, diagL, Cg, k, eps, Om, H, Hb)
                if mode == 0:
                    for i in range(12): H[i, i] += diagL[i] * (gfix - g)
                if half == 0:
                    deriv(H, rho, Cg, gamma, k1)
                    for i in range(12):
                        for j in range(12): tmp[i, j] = rho[i, j] + 0.5 * dt * k1[i, j]
                elif half == 1:
                    deriv(H, tmp, Cg, gamma, k2)
                    for i in range(12):
                        for j in range(12): tmp[i, j] = rho[i, j] + 0.5 * dt * k2[i, j]
                    deriv(H, tmp, Cg, gamma, k3)
                    for i in range(12):
                        for j in range(12): tmp[i, j] = rho[i, j] + dt * k3[i, j]
                else:
                    deriv(H, tmp, Cg, gamma, k4)
            for i in range(12):
                for j in range(12): rho[i, j] += dt / 6.0 * (k1[i, j] + 2 * k2[i, j] + 2 * k3[i, j] + k4[i, j])
            if s >= n0:
                # the force along û at the step's end: the couplings' 2 k_b Im Tr(H_b⁺ ρ), and the tweezer's −Tr(ρ diagL)∇g less
                # the ground trap's U0 ∇g (mode 1); its place's H
                d = start + v * (t + dt)
                x = d * u[0]; y = d * u[1]; z = d * u[2]
                g, gx, gy, gz = C.hamiltonian(x, y, z, P, H0, diagL, Cg, k, eps, Om, H, Hb)
                tr = 0.0
                for i in range(12): tr += rho[i, i].real
                F = 0.0
                for b in range(nb_):
                    A = 0j
                    for i in range(5, 12):
                        for j in range(5):
                            if Hb[b, i, j] != 0.0: A += Hb[b, i, j] * rho[j, i]
                    F += 2.0 * (k[b, 0] * u[0] + k[b, 1] * u[1] + k[b, 2] * u[2]) * A.imag / tr
                if mode == 1:
                    lev = 0.0
                    for i in range(12): lev += rho[i, i].real * diagL[i]
                    lev /= tr
                    gu = gx * u[0] + gy * u[1] + gz * u[2]
                    F += -lev * gu - U0 * gu
                ee = 0.0
                for i in range(5, 12): ee += rho[i, i].real
                Fsum += F; Rsum += gamma * ee / tr; cnt += 1
        out_F[iv] = Fsum / max(cnt, 1); out_rate[iv] = Rsum / max(cnt, 1)
    return 0

def two_level_doppler(s, delta, gamma, kv):
    """F/(ħk) of two counter-propagating two-level beams, each s, the total s in the saturation (rad/µs)."""
    R = lambda d: 0.5 * gamma * s / (1 + 2 * s + 4 * d * d / gamma ** 2)
    return R(delta - kv) - R(delta + kv)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--request', required=True); ap.add_argument('--case', default='free'); ap.add_argument('--field', type=float, default=None)
    ap.add_argument('--dt', type=float, default=0.001); ap.add_argument('--out', default=None)
    ap.add_argument('--vs', default='0.005,0.01,0.02,0.05,0.1,0.2,0.5,1,2,4')
    a = ap.parse_args()
    req = json.load(open(a.request, encoding='utf-8'))
    field = a.field if a.field is not None else (0.0 if a.case in ('free', 'shifts') else None)
    S = C.setup(req, field_G=field)
    p = S['p']; G = p['gamma']
    diagL = S['diagL'].copy()
    if a.case == 'free': diagL[:] = 0.0
    P = C.trap_params(S)
    Om = G * np.sqrt(S['s'] / 2)
    kk = float(np.linalg.norm(S['k'][0])); u = S['k'][0] / kk
    vs = np.array([float(x) for x in a.vs.split(',')]); nv = len(vs); both = np.concatenate([vs, -vs])
    lam2 = math.pi / kk
    t_trans = np.array([min(max(10.0, lam2 / abs(v)), 200.0) for v in both]); t_avg = np.array([min(max(20.0, 2 * lam2 / abs(v)), 400.0) for v in both])
    mode = 1 if a.case == 'trap' else 0
    out_F = np.zeros(2 * nv); out_R = np.zeros(2 * nv)
    forces(both, P, S['H0'], diagL, S['C'], S['k'], S['eps'].astype(np.complex128), Om, u, mode, 1.0, a.dt, t_trans, t_avg, out_F, out_R)
    F_odd = 0.5 * (out_F[:nv] - out_F[nv:]); F_even = 0.5 * (out_F[:nv] + out_F[nv:]); R_mean = 0.5 * (out_R[:nv] + out_R[nv:])
    # the probe's detuning from the cycling line at the case's centre (free: the offset with no Zeeman; else the focus's)
    d0 = S['H0'].diagonal().real; dl = diagL
    delta = -((d0[11] + dl[11]) - (d0[4] + dl[4]))
    s = S['s'][0]
    rows = []
    print(f"case {a.case}: field {np.round(S['Bvec'], 3).tolist()} G, s per beam {s:.2f}, δ(|2,2>→|3',3'>) {delta / G:+.2f}Γ, beams {S['names']}, k {kk:.3f}/µm, Γ/k {G / kk:.2f} µm/µs")
    sis_lin = -3 * delta / G; sis_pm = -(120 / 17) * delta * G / (5 * G * G + 4 * delta * delta)
    for v, F, Fe, R in zip(vs, F_odd, F_even, R_mean):
        alpha = -F / (kk * kk * v)
        # α/(ħk²) = −F_odd/(ħ k² v), F/ħ = k ΔR: −ΔR/(k v); positive where it cools
        alpha_D = -two_level_doppler(s * 7 / 15, delta, G, kk * v) / (kk * v)
        rows.append(dict(v=float(v), Fodd_over_hbark_gamma=float(F / kk / G), Feven_over_hbark_gamma=float(Fe / kk / G), alpha=float(alpha), rate=float(R), alphaDoppler=float(alpha_D)))
        print(f"  v {v:7.3f} µm/µs (kv/Γ {kk * v / G:.4f}): F_odd/(ħkΓ) {F / kk / G:+.5f}, F_even {Fe / kk / G:+.5f}, α/(ħk²) {alpha:+9.3f}, Doppler (2-level, s·7/15) {alpha_D:+8.3f}; scattering {R:.3f}/µs")
    print(f"  low-intensity Sisyphus α/(ħk²): lin⊥lin J=1/2→3/2 {sis_lin:+.2f}; σ+σ− J=1→2 {sis_pm:+.2f} (Dalibard & Cohen-Tannoudji 1989)")
    if a.out: json.dump(dict(case=a.case, field=S['Bvec'].tolist(), s=float(s), deltaGamma=float(delta / G), rows=rows, sisyphusLin=sis_lin, sisyphusPM=sis_pm), open(a.out, 'w'))

if __name__ == '__main__':
    main()
