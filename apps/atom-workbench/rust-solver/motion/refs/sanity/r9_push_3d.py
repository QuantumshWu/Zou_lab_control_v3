"""R9_PUSH by a three-dimensional Monte Carlo (an independent check of refs/expect.mjs `r9`, LOSSCAP 2026-11-21).

    python r9_push_3d.py [--atoms 20000] [--end 100] [--seed 1] [--expect ../expect/R9_PUSH.json] [--out r9_push_3d.json]

The expectation's own derivation (expect.mjs `r9`) moves each atom along the tweezer's axis and carries the radial
motion as its second moments only: the loss is the radial energy X = λ₁χ²₂ + λ₂χ²₂ in the harmonic approximation of the
well added to the axial energy, the path's chance of having passed 0 taken at its largest so far. Here every atom has
its own x, y, z and momenta, in the exact well of the ground state V = −U₀ g e^{−2ρ²g/w₀²}, g = 1/(1 + z²/z_R²), and is
lost the first time its own loss energy E = (ħ/m)κ²/2 + V passes 0, the far value (SPEC §1.7) -- no moment closure, no
harmonic well, no largest-so-far. The same physics otherwise, as the expectation states it (the scene's numbers from
its `info.model`): the two-level atom's steady rate R = Γ ρ_ee(δ), ρ_ee = (s/2)/(1 + s + (2δ/Γ)²), at the local and
Doppler-shifted detuning δ = δ_free − δ_LS g e^{−s} − k v_z; the push ħkR along z; the ground well's force and the
excited state's extra one, ρ_ee δ_LS ∇(g e^{−s}) against it; the momentum diffusion (ħk)²(1 + Q + ξ)R along z and
(ħk)²(1 − ξ)R/2 on each radial axis; the prepared ensemble's Gaussian start (z, κ_z from `sz`, `sk`; x, v_x from B₀, A₀).
Steps of the expectation's dt, the kick then the drift (symplectic Euler), the loss tested at every step's end.
Writes L(t) at the expectation's outputs with its binomial standard error, and the times of L = 0.05, 0.5, 0.95.
"""
import argparse, json, math, os
import numpy as np


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    a = argparse.ArgumentParser()
    a.add_argument('--atoms', type=int, default=20000)
    a.add_argument('--end', type=float, default=100.0)
    a.add_argument('--seed', type=int, default=1)
    a.add_argument('--expect', default=os.path.join(here, '..', 'expect', 'R9_PUSH.json'))
    a.add_argument('--out', default=None)
    o = a.parse_args()
    e = json.load(open(o.expect, encoding='utf-8'))
    info, m = e['info'], e['info']['model']
    G, U0, dLS, dfree, Om, k, r = m['gamma'], m['U0'], m['dLS'], m['deltaFree'], m['omega'], m['k'], m['hbarOverMass']
    w2, zR, xi, Q, dt = m['w0'] ** 2, info['zR'], m['xiSigma'], info['mandelQ'], m['dt']
    s = 2 * Om * Om / (G * G)
    rng = np.random.default_rng(o.seed)
    N = o.atoms
    # The start: the expectation's prepared Gaussians.
    pos = np.zeros((N, 3))
    kap = np.zeros((N, 3))
    pos[:, 2] = info['sz'] * rng.standard_normal(N)
    kap[:, 2] = info['sk'] * rng.standard_normal(N)
    for x in (0, 1):
        pos[:, x] = math.sqrt(m['B0']) * rng.standard_normal(N)
        kap[:, x] = math.sqrt(m['A0']) / r * rng.standard_normal(N)
    times = np.array([e['durationUs'] * i / (e['samples'] - 1) for i in range(e['samples'])])
    times = times[times <= o.end + 1e-9]
    lost = np.full(N, np.inf)
    alive = np.ones(N, bool)

    def fields(p):
        g = 1.0 / (1.0 + p[:, 2] ** 2 / zR ** 2)
        rho2 = p[:, 0] ** 2 + p[:, 1] ** 2
        sg = 2.0 * rho2 * g / w2
        a2 = g * np.exp(-sg)  # |a|², the intensity over the focus'
        # ∇|a|²: along z −(2z/z_R²) g a2 (1 − sg); across −(4 g/w₀²) x a2.
        dz = -(2.0 * p[:, 2] / zR ** 2) * g * a2 * (1.0 - sg)
        dx = -(4.0 * g / w2) * a2
        return a2, dz, dx

    t, step, out = 0.0, 0, [0.0]
    ends = np.rint(times / dt).astype(int)
    for k_out in range(1, len(times)):
        while step < ends[k_out]:
            idx = np.nonzero(alive)[0]
            p, q = pos[idx], kap[idx]
            a2, dz, dx = fields(p)
            delta = dfree - dLS * a2 - k * r * q[:, 2]
            ree = (s / 2) / (1 + s + (2 * delta / G) ** 2)
            R = G * ree
            # Forces (rates per µm, κ per µs): −∇V = U₀∇|a|², the excited share's −ρ_ee δ_LS ∇|a|², the push kR.
            fz = U0 * dz - ree * dLS * dz + k * R
            fx = (U0 - ree * dLS) * dx
            q[:, 2] += fz * dt + np.sqrt(k * k * (1 + Q + xi) * R * dt) * rng.standard_normal(len(idx))
            for x in (0, 1):
                q[:, x] += fx * p[:, x] * dt + np.sqrt(k * k * (1 - xi) / 2 * R * dt) * rng.standard_normal(len(idx))
            p += r * q * dt
            pos[idx], kap[idx] = p, q
            step += 1
            # The loss test at the step's end: the atom's own loss energy above 0.
            a2, _, _ = fields(p)
            E = 0.5 * r * (q ** 2).sum(axis=1) - U0 * a2
            gone = idx[E > 0]
            lost[gone] = step * dt
            alive[gone] = False
        out.append(float((lost <= times[k_out] + 1e-9).mean()))
    L = np.array(out)
    se = np.sqrt(L * (1 - L) / N)

    def crossing(q):
        i = int(np.argmax(L >= q))
        if L[i] < q or i < 1:
            return float('nan')
        return float(times[i - 1] + (times[i] - times[i - 1]) * (q - L[i - 1]) / (L[i] - L[i - 1]))

    ref = e['series'][0]
    n = len(times)
    d = np.array(ref['values'][:n]) - L
    bound = np.hypot(np.array(ref['u'][:n]), se)
    res = {'atoms': N, 'seed': o.seed, 'times': times.tolist(), 'L': L.tolist(), 'se': se.tolist(), 'crossings': [crossing(q) for q in (0.05, 0.5, 0.95)],
           'expectation': ref['values'][:n], 'u': ref['u'][:n], 'largest |d|/sqrt(u²+se²)': float(np.max(np.abs(d) / np.where(bound > 0, bound, np.inf))), 'largest |d|': float(np.max(np.abs(d)))}
    print(json.dumps({k: v for k, v in res.items() if k not in ('times', 'L', 'se', 'expectation', 'u')}, indent=1))
    for i in range(0, n, 5):
        print(f"t {times[i]:6.1f}  3D {L[i]:.4f} ± {se[i]:.4f}   expectation {ref['values'][i]:.4f} ± {ref['u'][i]:.4f}")
    if o.out:
        json.dump(res, open(o.out, 'w', encoding='utf-8'), indent=1)


if __name__ == '__main__':
    main()
