"""R9_HEAT by a three-dimensional Monte Carlo (an independent check of refs/expect.mjs `r9heat`, LOSSCAP 2026-11-21).

    python r9_heat_3d.py [--atoms 4000] [--dt 0.1] [--seed 1] [--expect ../expect/R9_HEAT.json] [--out r9_heat_3d.json]

The expectation reduces the motion to an energy Fokker–Planck equation, its spread between two closures (all the
kinetic energy shared alike among the axes, or all of it along the beams), absorbing at the far value. Here every atom
moves in three dimensions in the exact well of the ground state, V = −U₀ g e^{−2ρ²g/w₀²}, g = 1/(1 + z²/z_R²), and is lost
the first time its own loss energy E = (ħ/m)κ²/2 + V passes 0 (SPEC §1.7) -- no energy equation, no closure. The same
light as the expectation states it (the scene's numbers from its `info.model`): each beam b's steady rate
R_b = Γ ρ_ee(δ_b), ρ_ee = (s/2)/(1 + s + (2δ/Γ)²), at δ_b + κU₀(1 − |a|²) − k_b v_z (the line's shift with the
potential, the Doppler shift); the beams' pressures ħk_b R_b along z, which cancel at the bottom; the excited share's
weakening of the well, (1 − κΣρ_ee); the momentum diffusion (ħk)²(1 + Q_b + ξ)R_b along z and (ħk)²(1 − ξ)R_b/2 on each
radial axis. The start: the lowered isothermal f ∝ (e^{−E/k_BT} − 1)₊ in that well, drawn by a Metropolis chain per atom.
Steps of dt, the kick then the drift (symplectic Euler), the loss tested at every step's end. Writes L(t) at the
expectation's outputs with its binomial standard error, beside the expectation and its two closures.
"""
import argparse, json, math, os
import numpy as np

KT_PER_UK = 1.380649e-23 / (6.62607015e-34 / (2 * math.pi)) * 1e-12  # k_B/ħ in rad/µs per µK


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    a = argparse.ArgumentParser()
    a.add_argument('--atoms', type=int, default=4000)
    a.add_argument('--dt', type=float, default=0.1)
    a.add_argument('--seed', type=int, default=1)
    a.add_argument('--expect', default=os.path.join(here, '..', 'expect', 'R9_HEAT.json'))
    a.add_argument('--out', default=None)
    o = a.parse_args()
    e = json.load(open(o.expect, encoding='utf-8'))
    info, m = e['info'], e['info']['model']
    G, U0, r, w2, zR, kap, xi = m['gamma'], m['U0'], m['hbarOverMass'], m['w0'] ** 2, info['zR'], info['kappa'], m['xiSigma']
    beams, Q = m['beams'], info['mandelQ']
    kT = m['temperatureUK'] * KT_PER_UK
    rng = np.random.default_rng(o.seed)
    N, dt = o.atoms, o.dt

    def intensity(p):
        g = 1.0 / (1.0 + p[:, 2] ** 2 / zR ** 2)
        sg = 2.0 * (p[:, 0] ** 2 + p[:, 1] ** 2) * g / w2
        a2 = g * np.exp(-sg)
        return a2, -(2.0 * p[:, 2] / zR ** 2) * g * a2 * (1.0 - sg), -(4.0 * g / w2) * a2

    # The lowered isothermal by a Metropolis chain per atom, from the bottom at rest.
    def weight(p, q):
        a2, _, _ = intensity(p)
        E = 0.5 * r * (q ** 2).sum(axis=1) - U0 * a2
        return np.maximum(np.exp(-E / kT) - 1.0, 0.0)

    pos, mom = np.zeros((N, 3)), np.zeros((N, 3))
    step_r = 0.5 * np.array([math.sqrt(w2 * kT / (4 * U0))] * 2 + [zR * math.sqrt(kT / (2 * U0))])
    step_k = 0.5 * math.sqrt(kT / r)
    f = weight(pos, mom)
    accepted = 0
    for _ in range(4000):
        p2, q2 = pos + step_r * rng.standard_normal((N, 3)), mom + step_k * rng.standard_normal((N, 3))
        f2 = weight(p2, q2)
        take = rng.random(N) * f < f2
        pos[take], mom[take], f[take] = p2[take], q2[take], f2[take]
        accepted += take.sum()
    a2, _, _ = intensity(pos)
    E0 = 0.5 * r * (mom ** 2).sum(axis=1) - U0 * a2
    kinetic0 = r * (mom ** 2).mean(axis=0) / KT_PER_UK
    times = np.array([e['durationUs'] * i / (e['samples'] - 1) for i in range(e['samples'])])
    ends = np.rint(times / dt).astype(int)
    lost = np.full(N, np.inf)
    alive = np.ones(N, bool)
    out, step = [0.0], 0
    for k_out in range(1, len(times)):
        while step < ends[k_out]:
            idx = np.nonzero(alive)[0]
            p, q = pos[idx], mom[idx]
            a2, dz, dx = intensity(p)
            fz = np.zeros(len(idx))
            ree = np.zeros(len(idx))
            dzz, dxx = np.zeros(len(idx)), np.zeros(len(idx))
            for b, Qb in zip(beams, Q):
                delta = b['delta'] + kap * U0 * (1.0 - a2) - b['along'] * b['k'] * r * q[:, 2]
                s = 2 * b['omega'] ** 2 / G ** 2
                rb = (s / 2) / (1 + s + (2 * delta / G) ** 2)
                R = G * rb
                ree += rb
                fz += b['along'] * b['k'] * R
                dzz += b['k'] ** 2 * (1 + Qb + xi) * R
                dxx += b['k'] ** 2 * (1 - xi) / 2 * R
            well = U0 * (1.0 - kap * ree)
            q[:, 2] += (well * dz + fz) * dt + np.sqrt(dzz * dt) * rng.standard_normal(len(idx))
            for x in (0, 1):
                q[:, x] += well * dx * p[:, x] * dt + np.sqrt(dxx * dt) * rng.standard_normal(len(idx))
            p += r * q * dt
            pos[idx], mom[idx] = p, q
            step += 1
            a2, _, _ = intensity(p)
            E = 0.5 * r * (q ** 2).sum(axis=1) - U0 * a2
            gone = idx[E > 0]
            lost[gone] = step * dt
            alive[gone] = False
        out.append(float((lost <= times[k_out] + 1e-9).mean()))
    L = np.array(out)
    se = np.sqrt(L * (1 - L) / N)
    ref, cl = e['series'][0], info['closures']

    def crossing(curve, q):
        curve = np.asarray(curve)
        i = int(np.argmax(curve >= q))
        if curve[i] < q or i < 1:
            return float('nan')
        return float(times[i - 1] + (times[i] - times[i - 1]) * (q - curve[i - 1]) / (curve[i] - curve[i - 1]))

    d = np.array(ref['values']) - L
    between = np.mean((L >= np.minimum(cl['ergodic'], cl['axial']) - 2 * se) & (L <= np.maximum(cl['ergodic'], cl['axial']) + 2 * se))
    res = {'atoms': N, 'dt': dt, 'seed': o.seed, 'metropolis acceptance': float(accepted / (4000 * N)), 'initial E/U0 largest': float(E0.max() / U0),
           'initial kinetic T (uK, per axis)': kinetic0.tolist(),
           'crossings 3D': [crossing(L, q) for q in (0.05, 0.5)], 'crossings expectation': [crossing(ref['values'], q) for q in (0.05, 0.5)],
           'largest |d|': float(np.max(np.abs(d))), 'largest |d|/sqrt(u²+se²)': float(np.max(np.abs(d) / np.maximum(np.hypot(ref['u'], se), 1e-12))),
           'share of outputs between the closures (2 se)': float(between),
           'times': times.tolist(), 'L': L.tolist(), 'se': se.tolist()}
    print(json.dumps({k: v for k, v in res.items() if k not in ('times', 'L', 'se')}, indent=1))
    for i in range(0, len(times), 10):
        print(f"t {times[i]:7.1f}  3D {L[i]:.4f} ± {se[i]:.4f}   expectation {ref['values'][i]:.4f} ± {ref['u'][i]:.4f}   ergodic {cl['ergodic'][i]:.4f} axial {cl['axial'][i]:.4f}")
    if o.out:
        json.dump(res, open(o.out, 'w', encoding='utf-8'), indent=1)


if __name__ == '__main__':
    main()
