"""A classical toy for the transverse part: an atom in the tweezer's radial plane (y, z), the Gaussian's central potential
of R4_TW2 (ω, waist 1 µm), a cooling force and a white momentum noise along z alone, nothing along y but the emitted
photons' recoil. Does releasing y change T_z? Units of the references (µm, µs, κ = p/ħ)."""
import sys, numpy as np
HM, OMEGA, KT, W = 0.0007307375228700508, 0.6317695417004418, 0.1309203, 1.0
rng = np.random.default_rng(7)


def run(n, ty, tz, free_y, span=1000.0, dt=0.05, power=0.0077, vc=0.01, twoD=8.6, recoil=1.73, harmonic=False, soft=0.0, jitter=0.0, memory=10.0, omega=OMEGA, waist=W, every=10.0):
    """Cooling force along z of constant power at |v| ≫ vc: dκ/dt = −power v/(v² + vc²) (so that v dκ/dt = −power), noise 2D along z,
    2D_y = recoil along y (if free). Velocity-Verlet with the kicks at the step's ends. The internal state's potential
    along z as a change of z's spring alone: ω_z² = ω²(1 + soft + jitter s(t)), s an Ornstein–Uhlenbeck process of unit
    variance (held within ±2) and correlation time `memory` per atom, on a Gaussian profile of z alone (a potential
    of z: nothing of it acts along y, and it does no work around a loop)."""
    OMEGA, W = omega, waist            # this run's trap
    kz = np.sqrt(KT * tz / HM) * rng.standard_normal(n)
    z = np.sqrt(KT * tz * HM) / OMEGA * rng.standard_normal(n)
    ky = np.sqrt(KT * ty / HM) * rng.standard_normal(n) if free_y else np.zeros(n)
    y = np.sqrt(KT * ty * HM) / OMEGA * rng.standard_normal(n) if free_y else np.zeros(n)
    def force(y, z):
        g = OMEGA ** 2 / HM * (np.ones_like(z) if harmonic else np.exp(-2 * (y * y + z * z) / W ** 2))
        fz = -g * z - OMEGA ** 2 / HM * (soft + jitter * np.clip(state, -2.0, 2.0)) * z * np.exp(-2 * z * z / W ** 2)
        return -g * y, fz
    state = rng.standard_normal(n)
    fy, fz = force(y, z)
    out = []
    steps = int(round(span / dt))
    per = int(round(every / dt))
    acc = np.zeros(2)
    for s in range(steps):
        ky += 0.5 * dt * fy
        kz += 0.5 * dt * fz
        y += HM * ky * dt * free_y
        z += HM * kz * dt
        state += -state * dt / memory + np.sqrt(2 * dt / memory) * rng.standard_normal(n)
        fy, fz = force(y, z)
        ky += 0.5 * dt * fy
        kz += 0.5 * dt * fz
        v = HM * kz
        kz += dt * (-power * v / (v * v + vc * vc)) + np.sqrt(twoD * dt) * rng.standard_normal(n)
        if free_y:
            ky += np.sqrt(recoil * dt) * rng.standard_normal(n)
        else:
            ky[:] = 0.0
        acc += [np.mean(ky * ky), np.mean(kz * kz)]
        if (s + 1) % per == 0:
            out.append([(s + 1) * dt, HM * acc[0] / per / KT, HM * acc[1] / per / KT])
            acc[:] = 0.0
    return np.array(out)


if __name__ == '__main__':
    n = 8192
    variants = {'soft -0.09, jitter 0.2': dict(soft=-0.09, jitter=0.2), 'jitter 0.2': dict(jitter=0.2), 'soft -0.09, jitter 0.4': dict(soft=-0.09, jitter=0.4), 'jitter 0.4, memory 30': dict(jitter=0.4, memory=30.0)}
    print('T_z (T_y) at 300, 500, 1000 µs: 1D | y free 30 µK | y free cold')
    for name, kw in variants.items():
        res = [run(n, ty, 30, free, **kw) for ty, free in ((0, False), (30, True), (1e-9, True))]
        print('%-24s' % name + ' || '.join(' '.join('%6.2f (%5.2f)' % (r[i, 2], r[i, 1]) for i in (29, 49, 99)) for r in res), flush=True)
