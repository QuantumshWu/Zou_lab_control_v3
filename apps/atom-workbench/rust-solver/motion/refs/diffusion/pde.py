"""The semiclassical phase-space equation itself, in 1D along z (diffusion.status.md §4.3): the Wigner function W(z, κ),
a matrix on the internal states (block diagonal over the model's clusters), to the second order in ħk/Δp:

  ∂_t W = −(ħ/m) κ ∂_z W + 𝓛(z)[W] − ∂_κ 𝓕(z)[W] + ∂_κ² 𝓢(z)[W],

𝓛 the internal master equation, 𝓕 the force superoperator (ref.py; its Hermitian part's gradient holds the trap's force),
and 𝓢 the untraced second order,
  𝓢[X] = (i/8)[∇²H_h, X] + (1/16){∇²N, X} − (1/8)Σ(∇²J X J† + J X ∇²J†) + (1/4)Σ ∇J X ∇J† + ½ Σ k_φ²⟨n_z²⟩ J X J†,
whose trace is half the local second moment's rate B (ref.py). The model's operators are taken at κ = 0 (their
first-order Doppler part moves the friction by 1 % at the thermal velocity, §4.1).

Strang splitting: the flight (a shift in z, exact in z's Fourier space) for dt/2, then everything local in z for dt --
in κ's Fourier space, W̃(z, q): ∂_t W̃ = (𝓛 − i q 𝓕 − q² 𝓢) W̃, a matrix exponential per (z, q), exact, the internal dynamics
and the state-dependent force together --, then the flight for dt/2.

python pde.py <tag> <chunk µs> [T0 µK] [nz] [nk] [dt] [exact]: continues the run `tag` by a chunk; prints its series.
`exact`: the master equation in the position representation, no expansion (the quantum problem of the same model).
"""
import sys, os, time, json
import numpy as np
import scipy.linalg as sl
from common import *
import ref

EPS = 1e-4


def operators(z, kind='closed', waist=0):
    """Per point of z the matrices of 𝓛, 𝓕, 𝓢 on vec(ρ), the trace row, and the model."""
    m0 = dump(kind, waist, z, 0.0, 'pde0')
    mm = dump(kind, waist, z - EPS, 0.0, 'pdem')
    mp = dump(kind, waist, z + EPS, 0.0, 'pdep')
    L, F, B = m0.supers()
    I = np.eye(m0.dim)
    S = []
    for i in range(m0.n):
        J, dJ = m0.J[i], m0.dJ[i]
        d2J = (mp.J[i] + mm.J[i] - 2 * J) / EPS ** 2
        d2H = (mp.G[i] - mm.G[i]) / (2 * EPS)
        d2N = np.einsum('qba,qbc->ac', d2J.conj(), J) + 2 * np.einsum('qba,qbc->ac', dJ.conj(), dJ) + np.einsum('qba,qbc->ac', J.conj(), d2J)
        c = np.sqrt(m0.kphi ** 2 * m0.nz2)[:, None, None]
        S.append(0.125j * (m0._kron(d2H, I) - m0._kron(I, d2H)) + (m0._kron(d2N, I) + m0._kron(I, d2N)) / 16
                 - 0.125 * (m0._sandwich(d2J, J) + m0._sandwich(J, d2J)) + 0.25 * m0._sandwich(dJ, dJ) + 0.5 * m0._sandwich(c * J, c * J))
    S = np.array(S)
    worst = max(np.abs(m0.trace @ S[i] - 0.5 * B[i]).max() for i in range(m0.n))
    assert worst < 1e-6 * np.abs(B).max(), f'the second order\'s trace is not B/2: {worst}'
    return L, F, S, m0


def pattern(s, axis):
    """⟨e^{i s n_z}⟩ over the directions of a photon emitted by a dipole along a lab axis: 3(sin s − s cos s)/s³ for z,
    (3/2)(sin s/s + cos s/s² − sin s/s³) for x and y (1 − s²/10 and 1 − s²/5 at small s)."""
    s = np.where(np.abs(s) < 1e-3, 1e-3, s)
    return 3 * (np.sin(s) - s * np.cos(s)) / s ** 3 if axis == 2 else 1.5 * (np.sin(s) / s + np.cos(s) / s ** 2 - np.sin(s) / s ** 3)


def exact_generators(z, dz, jmax, kind='closed', waist=0):
    """The master equation itself in the position representation, ρ(x, x') at x = z − q/2, x' = z + q/2 with q = 2 j dz,
    no expansion in q: M(z, q)[X] = −i(H(x) X − X H(x')†) + Σ ⟨e^{i k_φ n_z q}⟩ J(x) X J(x')† -- the quantum problem of
    the model (its operators at κ = 0), of which `operators` is the second order in q. Returns M[z, j] for
    j = −jmax..jmax (the q of numpy's FFT order is the caller's) and the model on the extended grid."""
    ext = np.concatenate([z[0] + dz * np.arange(-jmax, 0), z, z[-1] + dz * np.arange(1, jmax + 1)])
    m = dump(kind, waist, ext, 0.0, 'pdex')
    I = np.eye(m.dim)
    nz, nv = len(z), m.nv
    M = np.zeros((nz, 2 * jmax + 1, nv, nv), complex)
    axes = np.array([a for (_, _, a) in m.keys])
    for j in range(-jmax, jmax + 1):
        q = 2 * j * dz
        phi = np.where(axes == 2, pattern(m.kphi * q, 2), pattern(m.kphi * q, 0))[:, None, None]
        for i in range(nz):
            a, b = i + jmax - j, i + jmax + j
            M[i, j + jmax] = -1j * (m._kron(m.H[a], I) - m._kron(I, m.H[b].conj().T)) + m._sandwich(phi * m.J[a], m.J[b])
    return M, m


class Run:
    def __init__(self, tag, T0=30.0, nz=128, nk=128, dt=0.1, zmax=0.48, kmax=380.0, state='ground-2-2', exact=False):
        """`exact`: the master equation in the position representation (`exact_generators`) instead of its second order
        in q; then q = 2 j dz on the z grid, which fixes κ's range at π/(2 dz), and the start is the trap's quantum thermal
        state (its kinetic energy ħω(n̄ + ½), 30.06 µK at T = 30 µK)."""
        self.tag = tag
        self.file = f'pde_{tag}.npz'
        if os.path.exists(self.file):
            d = np.load(self.file, allow_pickle=True)
            self.W, self.t, self.series = d['W'], float(d['t']), list(d['series'])
            (T0, nz, nk, dt, zmax, kmax, exact) = d['setup']
            nz, nk, exact = int(nz), int(nk), bool(exact)
        self.z = (np.arange(nz) - nz // 2) * (2 * zmax / nz)
        self.dz = self.z[1] - self.z[0]
        if exact:
            kmax = np.pi / (2 * self.dz)
        self.setup = (T0, nz, nk, dt, zmax, kmax, float(exact))
        self.dt = dt
        self.kappa = (np.arange(nk) - nk // 2) * (2 * kmax / nk)
        self.dk = self.kappa[1] - self.kappa[0]
        q = 2 * np.pi * np.fft.fftfreq(nk, self.dk)
        cut = float(os.environ.get('PDE_CUT', '1.6' if exact else '1.0')) / K
        self.share, self.scalar = None, None
        if exact:
            jmax = int(round(cut / (2 * self.dz)))
            Mj, m = exact_generators(self.z, self.dz, jmax)
            order = np.round(q / (2 * self.dz)).astype(int)
            assert np.allclose(order * 2 * self.dz, q), 'q is not on the grid of 2 dz'
            M = np.zeros((nz, nk) + Mj.shape[2:], complex)
            keep = np.abs(order) <= jmax
            M[:, keep] = Mj[:, order[keep] + jmax]
        else:
            L, F, S, m = operators(self.z)
            # The switch on the first order's traceless part (`PDE_SHARE` = λ; unset: the equation as it is). With c(z) the
            # force every state shares (the trap's: −tr G/n) and 𝓕' = 𝓕 − c, the force term is taken as
            #   𝓕_λ[W] = c W + λ 𝓕'[W] + (1 − λ) f̄'(z, κ) W,   f̄' = Tr 𝓕'[W] / Tr W,
            # the same in its trace at every (z, κ): at λ = 1 the equation itself, at λ = 0 every atom of a phase-space
            # cell under the cell's mean force -- the first order's traceless part (relative to the cell's own state)
            # removed and nothing else changed. The mean-force part is nonlinear: it is stepped apart (`advect`).
            # `PDE_SHARE=trap`: λ = 1 with c itself stepped by `advect` (the equation unchanged: the check of that step).
            share = os.environ.get('PDE_SHARE')
            self.share = None if share is None else 1.0 if share == 'trap' else float(share)
            if self.share is not None:
                c = -np.real(np.einsum('zaa->z', m.G)) / m.dim
                Fp = F - c[:, None, None] * np.eye(m.nv)[None]
                self.row = np.einsum('v,zvw->zw', m.trace, Fp)
                # No state feels more than the force operator's largest eigenvalue: the cell's mean is held within it
                # (where the density is its own rounding the quotient is not a force).
                self.most = max(np.abs(np.linalg.eigvalsh(0.5 * (g + g.conj().T) + ci * np.eye(m.dim))).max() for g, ci in zip(m.G, c))
                self.scalar = c if share == 'trap' else None
                F = (0.0 if share == 'trap' else c[:, None, None]) * np.eye(m.nv)[None] + self.share * Fp
            M = L[:, None] - 1j * q[None, :, None, None] * F[:, None] - (q ** 2)[None, :, None, None] * S[:, None]
        self.m = m
        self.growth = max(np.linalg.eigvals(M[i]).real.max() for i in range(0, nz, 8))
        self.E = sl.expm(M * dt)
        # The second order is an expansion in q·∇ (q the separation the coherence ρ(z − q/2, z + q/2) is taken at): beyond
        # |q| ≈ 1/k it does not hold (its modes grow from |q| ≈ 0.15 µm on, growth.py). The separations beyond the cut
        # are left out (a thermal atom's own coherence there is e^{−q²σ_κ²/2} = 10⁻¹⁶ at 30 µK; what the dynamics put
        # there is read by moving the cut).
        self.E[:, np.abs(q) > cut + 1e-12] = 0.0
        self.iq = (1j * q * (np.abs(q) <= cut + 1e-12))[None, :, None]
        p = 2 * np.pi * np.fft.fftfreq(nz, self.dz)
        self.flight = np.exp(-1j * p[:, None] * HM * self.kappa[None, :] * dt / 2)[:, :, None]
        self.f1 = np.array([n.startswith('ground-1') for n in m.names])
        if not os.path.exists(self.file):
            kt = ref.KT_PER_UK * T0
            quanta = 0.5 / np.tanh(OMEGA / (2 * kt)) if exact else kt / OMEGA      # n̄ + ½, or its classical limit
            sz, sk = np.sqrt(HM / OMEGA * quanta), np.sqrt(OMEGA / HM * quanta)
            g = np.exp(-self.z[:, None] ** 2 / (2 * sz ** 2) - self.kappa[None, :] ** 2 / (2 * sk ** 2))
            g /= g.sum() * self.dz * self.dk
            rho = np.zeros((m.dim, m.dim), complex)
            s = m.names.index(state)
            rho[s, s] = 1.0
            self.W = g[:, :, None] * rho.ravel()[m.idx][None, None, :]
            self.t, self.series = 0.0, []
            self.observe()

    def observe(self):
        m, w = self.m, self.dz * self.dk
        P = (self.W @ m.trace).real * w                      # the phase-space density's weights
        pops = np.einsum('zkv->v', self.W) * w
        diag = np.real(np.diag(m.mat(pops)))
        norm = P.sum()
        row = [self.t, norm, (P * self.z[:, None]).sum() / norm, (P * self.kappa[None, :]).sum() / norm, (P * self.z[:, None] ** 2).sum() / norm, (P * self.kappa[None, :] ** 2).sum() / norm,
               diag[self.f1].sum() / norm, np.abs(P[[0, -1], :]).sum() / norm, np.abs(P[:, [0, -1]]).sum() / norm, P.min() / P.max()]
        self.series.append(row)
        return row

    def advect(self, W, dt):
        """∂_tW = −∂_κ(f(z, κ) W) over dt, f the part of the force stepped apart (`__init__`): fourth-order Runge–Kutta
        on the spectral derivative (|f| dt q ≤ 1: within its stability)."""
        tr, weight = self.m.trace, 1.0 - self.share
        def rate(W):
            if self.scalar is not None:
                f = np.broadcast_to(self.scalar[:, None], W.shape[:2])
            else:
                density = (W @ tr).real
                floor = 1e-6 * density.max()
                f = weight * np.where(density > floor, np.clip(np.einsum('zkv,zv->zk', W, self.row).real / np.maximum(density, floor), -self.most, self.most), 0.0)
            return -np.fft.ifft(self.iq * np.fft.fft(f[:, :, None] * W, axis=1), axis=1)
        k1 = rate(W)
        k2 = rate(W + 0.5 * dt * k1)
        k3 = rate(W + 0.5 * dt * k2)
        k4 = rate(W + dt * k3)
        return W + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    def step(self):
        apart = getattr(self, 'share', None) is not None and (self.share != 1.0 or self.scalar is not None)
        W = np.fft.ifft(np.fft.fft(self.W, axis=0) * self.flight, axis=0)
        if apart:
            W = self.advect(W, 0.5 * self.dt)
        Wq = np.fft.fft(W, axis=1)
        Wq = np.einsum('zqab,zqb->zqa', self.E, Wq)
        W = np.fft.ifft(Wq, axis=1)
        if apart:
            W = self.advect(W, 0.5 * self.dt)
        self.W = np.fft.ifft(np.fft.fft(W, axis=0) * self.flight, axis=0)
        self.t += self.dt

    def advance(self, span, every=1.0, limit=240.0):
        clock, end = time.time(), self.t + span
        per = int(round(every / self.dt))
        while self.t < end - 1e-9 and time.time() - clock < limit:
            for _ in range(per):
                self.step()
            self.t = round(self.t / every) * every
            self.observe()
        np.savez(self.file, W=self.W, t=self.t, series=np.array(self.series), setup=np.array(self.setup))


def temperatures(series):
    """t, T from the kinetic energy and from the position's spread at the trap's frequency (µK), F = 1 share."""
    s = np.array(series)
    return s[:, 0], HM * (s[:, 5] - s[:, 3] ** 2) / ref.KT_PER_UK, OMEGA ** 2 * (s[:, 4] - s[:, 2] ** 2) / HM / ref.KT_PER_UK, s[:, 6]


if __name__ == '__main__':
    tag, span = sys.argv[1], float(sys.argv[2])
    kw = {}
    for name, cast, i in (('T0', float, 3), ('nz', int, 4), ('nk', int, 5), ('dt', float, 6), ('exact', lambda x: x == 'exact', 7)):
        if len(sys.argv) > i:
            kw[name] = cast(sys.argv[i])
    clock = time.time()
    run = Run(tag, **kw)
    print(f'setup {run.setup}: operators and exponentials in {time.time() - clock:.0f} s; largest growth rate of a (z, q) mode {run.growth:.3e} /µs', flush=True)
    clock = time.time()
    run.advance(span)
    print(f'to t = {run.t} µs in {time.time() - clock:.0f} s')
    t, tk, tz, f1 = temperatures(run.series)
    s = np.array(run.series)
    for i in range(0, len(t), max(1, len(t) // 20)):
        w = slice(max(0, i - 5), i + 6)
        print('%7.1f  T_kin %7.3f (±5 µs mean %7.3f)  T_pos %7.3f (%7.3f)  F=1 %.4f  norm %.6f  edges %.1e %.1e  min/max %.1e' % (t[i], tk[i], tk[w].mean(), tz[i], tz[w].mean(), f1[i], s[i, 1], s[i, 7], s[i, 8], s[i, 9]))
