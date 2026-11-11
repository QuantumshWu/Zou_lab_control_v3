"""Exact semiclassical references from a master equation dumped by `diffusion dump` (diffusion.status.md).

Units: time µs, length µm, momentum κ = p/ħ (1/µm), energies rad/µs. Force dκ/dt = Tr 𝓕[ρ].
A model is a set of clusters; ρ is block diagonal over them. The global space is the concatenation of the clusters'
states; vec(ρ) holds the blocks' entries only (row-major).

Internal master equation at a point of the path: 𝓛[X] = −i(H X − X H†) + Σ J X J†.
Force superoperator (the first order of the Wigner expansion in ħk/Δp, untraced):
  𝓕[X] = −½{G, X} + (i/4)[∇N, X] − (i/2) Σ (∇J X J† − J X ∇J†),  G = ∂_z H_h, N = Σ J†J,
its trace −⟨G⟩ + Σ Im⟨u|∇u⟩; the local second moment's rate (the second order's trace)
  B[X] = Σ Re Tr[∇J X ∇J†] + k_φ² ⟨n_z²⟩ Tr[J X J†]   (⟨n_z²⟩ = 2/5 for the emitted dipole axes x, y; 1/5 for z).
With every excited state explicit ∇J = 0: 𝓕 is the anticommutator with the force operator and B the emitted recoil.

On a prescribed path (quantum regression theorem as moment equations, exact for the master equation):
  ρ̇ = 𝓛ρ,   d⟨κ⟩/dt = F(t) = Tr 𝓕[ρ],
  μ̇ = 𝓛μ + 𝓕[ρ] − F ρ          (μ = ∫(κ − ⟨κ⟩) W dκ, traceless),
  d Var(κ)/dt = 2 Tr 𝓕[μ] + B[ρ].
With a weight w(t) on every increment (X = ∫ w dκ): 𝓕 → w𝓕, B → w²B.
"""
import json
import numpy as np
import scipy.linalg as sl

KT_PER_UK = 1.380649e-23 / 1.054571817e-34 * 1e-12


class Model:
    def __init__(self, path):
        d = json.load(open(path))
        self.hm = d['hbar_over_mass']
        self.clusters = d['clusters']
        sizes = [len(c['states']) for c in self.clusters]
        self.offsets = np.concatenate([[0], np.cumsum(sizes)])
        self.dim = N = int(self.offsets[-1])
        self.names = sum([c['names'] for c in self.clusters], [])
        pts = d['points']
        self.z = np.array([p['z'] for p in pts])
        self.kappa = np.array([p['kappa'] for p in pts])
        self.n = len(pts)
        eps = d['epsilon']
        cm = lambda flat, r, c: (np.array(flat[0::2]) + 1j * np.array(flat[1::2])).reshape(r, c)
        keys = sorted({(ci, j['class'], a) for p in pts for ci, c in enumerate(p['clusters']) for j in c['jumps'] for a in range(3)})
        index = {key: q for q, key in enumerate(keys)}
        self.keys = keys
        self.H = np.zeros((self.n, N, N), complex)
        self.G = np.zeros((self.n, N, N), complex)
        self.J = np.zeros((self.n, len(keys), N, N), complex)
        self.dJ = np.zeros((self.n, len(keys), N, N), complex)
        self.kphi = np.zeros(len(keys))
        self.nz2 = np.array([0.4 if a < 2 else 0.2 for (_, _, a) in keys])
        for i, p in enumerate(pts):
            for ci, c in enumerate(p['clusters']):
                o, n = self.offsets[ci], sizes[ci]
                self.H[i, o:o + n, o:o + n] = cm(c['h'], n, n)
                self.G[i, o:o + n, o:o + n] = cm(c['g'], n, n)
                for which, sign in (('jumps', 0.0), ('before', -1.0), ('after', 1.0)):
                    for j in c[which]:
                        ol, m = self.offsets[j['landing']], sizes[j['landing']]
                        for a in range(3):
                            q = index[(ci, j['class'], a)]
                            block = cm(j['j'][a], m, n)
                            if sign == 0.0:
                                self.J[i, q, ol:ol + m, o:o + n] = block
                                self.kphi[q] = j['k']
                            else:
                                self.dJ[i, q, ol:ol + m, o:o + n] += sign * block / (2 * eps)
        mask = np.zeros((N, N), bool)
        for ci in range(len(sizes)):
            o, n = self.offsets[ci], sizes[ci]
            mask[o:o + n, o:o + n] = True
        self.idx = np.flatnonzero(mask.ravel())
        self.nv = len(self.idx)
        self.trace = np.eye(N).ravel()[self.idx]

    def mat(self, v):
        X = np.zeros(self.dim * self.dim, complex)
        X[self.idx] = v
        return X.reshape(self.dim, self.dim)

    def _kron(self, A, B):
        """The matrix of X ↦ A X B on the block-diagonal entries (row-major vec: kron(A, Bᵀ))."""
        return np.kron(A, B.T)[np.ix_(self.idx, self.idx)]

    def _sandwich(self, A, B):
        """The matrix of X ↦ Σ_q A_q X B_q† on the block-diagonal entries: entry (ab, cd) = Σ_q A_q[a,c] conj(B_q[b,d])."""
        N = self.dim
        full = np.tensordot(A.reshape(len(A), -1).T, B.conj().reshape(len(B), -1), axes=1).reshape(N, N, N, N).transpose(0, 2, 1, 3).reshape(N * N, N * N)
        return full[np.ix_(self.idx, self.idx)]

    def supers(self):
        """Per point the matrices of 𝓛 and 𝓕 and the row of B on vec(ρ)."""
        if not hasattr(self, '_supers'):
            I = np.eye(self.dim)
            L, F, B = [], [], []
            for i in range(self.n):
                H, G, J, dJ = self.H[i], self.G[i], self.J[i], self.dJ[i]
                L.append(-1j * (self._kron(H, I) - self._kron(I, H.conj().T)) + self._sandwich(J, J))
                dN = np.einsum('qba,qbc->ac', dJ.conj(), J) + np.einsum('qba,qbc->ac', J.conj(), dJ)
                F.append(-0.5 * (self._kron(G, I) + self._kron(I, G)) + 0.25j * (self._kron(dN, I) - self._kron(I, dN)) - 0.5j * (self._sandwich(dJ, J) - self._sandwich(J, dJ)))
                M = np.einsum('qba,qbc->ac', dJ.conj(), dJ) + np.einsum('q,qba,qbc->ac', self.kphi ** 2 * self.nz2, J.conj(), J)
                B.append(M.T.ravel()[self.idx])
            self._supers = (np.array(L), np.array(F), np.array(B))
        return self._supers

    def checks(self):
        worst = 0.0
        for i in range(self.n):
            H, J = self.H[i], self.J[i]
            worst = max(worst, np.abs(1j * (H - H.conj().T) - np.einsum('qba,qbc->ac', J.conj(), J)).max())
        return {'decay_identity': worst, 'G_hermitian': np.abs(self.G - self.G.conj().transpose(0, 2, 1)).max(), 'G_size': np.abs(self.G).max(), 'dJ_size': np.abs(self.dJ).max(), 'J_size': np.abs(self.J).max()}


def steady(L, trace):
    """The trace-one null vector of a Liouvillian."""
    A = L.copy()
    A[0, :] = trace
    b = np.zeros(L.shape[0], complex)
    b[0] = 1.0
    return np.linalg.solve(A, b)


def at_rest(model, gammas=(0.0,)):
    """Per point of the dump, the atom held there: the mean force, B, the equal-time variance C₀ = Tr 𝓕(𝓕 − F)ρ and
    2D(γ) = 2 Re Tr 𝓕 (γ − 𝓛)⁻¹ (𝓕 − F)ρ + B (γ = 0: the momentum's diffusion; γ > 0: a Laplace cutoff)."""
    L, F, B = model.supers()
    tr = model.trace
    rows = []
    for i in range(model.n):
        rho = steady(L[i], tr)
        f = (tr @ (F[i] @ rho)).real
        src = F[i] @ rho - f * rho
        local = (B[i] @ rho).real
        twoD = []
        for g in gammas:
            A = g * np.eye(model.nv) - L[i]
            s = src.copy()
            if g == 0.0:
                A[0, :] = tr
                s[0] = 0.0
            twoD.append(2 * (tr @ (F[i] @ np.linalg.solve(A, s))).real + local)
        rows.append({'z': model.z[i], 'force': f, 'local': local, 'twoD': twoD, 'c0': (tr @ (F[i] @ src)).real, 'rho': rho})
    return rows


class BlockTridiagonal:
    """A block-tridiagonal system (lower[i], diag[i], upper[i]) factored once (block Thomas), solved for many right sides."""
    def __init__(self, lower, diag, upper):
        self.lower, self.n = lower, len(diag)
        self.lu, self.cp = [], []
        for i in range(self.n):
            d = diag[i] if i == 0 else diag[i] - lower[i] @ self.cp[i - 1]
            lu = sl.lu_factor(d)
            self.lu.append(lu)
            self.cp.append(sl.lu_solve(lu, upper[i]) if i < self.n - 1 else None)

    def solve(self, b):
        y = []
        for i in range(self.n):
            y.append(sl.lu_solve(self.lu[i], b[i] if i == 0 else b[i] - self.lower[i] @ y[i - 1]))
        x = [None] * self.n
        x[-1] = y[-1]
        for i in range(self.n - 2, -1, -1):
            x[i] = y[i] - self.cp[i] @ x[i + 1]
        return x


class Periodic:
    """The periodic steady state on a path whose samples (the model's points, in order, equally spaced in time) make one
    period 2π/Ω: ρ(t) = Σ_n ρ_n e^{inΩt} by harmonic balance, the path's own harmonics of 𝓛 kept up to `band` (those
    above `tol` of the largest), ρ's up to `harmonics`."""
    def __init__(self, model, Omega, harmonics, tol=1e-10):
        self.model, self.Omega = model, Omega
        L, F, B = model.supers()
        self.nt = nt = model.n
        nv, tr = model.nv, model.trace
        self.Lm = Lm = np.fft.fft(L, axis=0) / nt                # 𝓛(t) = Σ_m Lm[m] e^{imΩt}
        size = np.abs(Lm).max(axis=(1, 2))
        order = np.abs(np.fft.fftfreq(nt, 1.0 / nt)).astype(int)
        self.band = g = max(1, int(order[size > tol * size.max()].max()))
        assert g < nt // 2 - 1, 'the path is not resolved by its samples'
        blocks = 2 * int(np.ceil(harmonics / g)) + 1
        lo = -(blocks // 2) * g
        self.ns = ns = np.arange(lo, lo + blocks * g)
        self.pos = {int(n): q for q, n in enumerate(ns)}
        sl_ = lambda q: slice(q * nv, (q + 1) * nv)
        self.sl_, self.g, self.blocks = sl_, g, blocks
        def block(bi, bj):
            out = np.zeros((g * nv, g * nv), complex)
            for a in range(g):
                for b in range(g):
                    n, n2 = ns[bi * g + a], ns[bj * g + b]
                    if abs(n - n2) <= g:
                        out[sl_(a), sl_(b)] -= Lm[n - n2]
                    if n == n2:
                        out[sl_(a), sl_(b)] += 1j * n * Omega * np.eye(nv)
            return out
        diag = [block(i, i) for i in range(blocks)]
        lower = [None] + [block(i, i - 1) for i in range(1, blocks)]
        upper = [block(i, i + 1) for i in range(blocks - 1)] + [None]
        self.mid, a0 = blocks // 2, self.pos[0] - (blocks // 2) * g
        self.row = a0 * nv
        diag[self.mid][self.row, :] = 0.0
        diag[self.mid][self.row, sl_(a0)] = tr
        lower[self.mid][self.row, :] = 0.0
        upper[self.mid][self.row, :] = 0.0
        self.system = BlockTridiagonal(lower, diag, upper)
        self.rho = self.solve([np.zeros(nv, complex)] * len(ns), 1.0)
        self.tail = max(np.abs(self.rho[0]).max(), np.abs(self.rho[-1]).max()) / np.abs(self.rho[self.pos[0]]).max()

    def solve(self, vecs, first):
        g, sl_ = self.g, self.sl_
        b = [np.concatenate([vecs[bi * g + a] for a in range(g)]) for bi in range(self.blocks)]
        b[self.mid][self.row] = first
        x = self.system.solve(b)
        return [x[bi][sl_(a)] for bi in range(self.blocks) for a in range(g)]

    def _conv(self, Xm, y, width):
        """(X y)_n = Σ_m X_m y_{n−m} over the harmonics kept."""
        return [sum(Xm[n - n2] @ y[self.pos[n2]] for n2 in range(n - width, n + width + 1) if n2 in self.pos) for n in self.ns]

    def moments(self, weight=None):
        """With the weight w(t) (samples; None: 1) on every increment of the momentum: the time averages of the drift
        ⟨w F⟩ and of the variance's rate, 2 Tr w𝓕[μ] (`slow`) + w² B[ρ] (`local`)."""
        m, nt, tr = self.model, self.nt, self.model.trace
        _, F, B = m.supers()
        w = np.ones(nt) if weight is None else np.asarray(weight)
        Fm = np.fft.fft(F * w[:, None, None], axis=0) / nt
        Bm = np.fft.fft(B * (w * w)[:, None], axis=0) / nt
        size = np.abs(Fm).max(axis=(1, 2))
        order = np.abs(np.fft.fftfreq(nt, 1.0 / nt)).astype(int)
        width = int(order[size > 1e-10 * size.max()].max())
        Frho = self._conv(Fm, self.rho, width)
        f_n = {int(n): tr @ Frho[self.pos[n]] for n in self.ns}
        Fbar = [sum(f_n[j] * self.rho[self.pos[n - j]] for j in f_n if (n - j) in self.pos) for n in self.ns]
        mu = self.solve([a - b for a, b in zip(Frho, Fbar)], 0.0)
        Fmu = self._conv(Fm, mu, width)
        local = sum((Bm[k] @ self.rho[self.pos[-k]]) for k in range(-width, width + 1) if -k in self.pos).real
        slow = 2 * (tr @ Fmu[self.pos[0]]).real
        return {'drift': f_n[0].real, 'slow': slow, 'local': local, 'rate': slow + local}

    def population(self, select):
        """The time-averaged population of the states `select` (a boolean per global state)."""
        return float(np.real(np.diag(self.model.mat(self.rho[self.pos[0]]))[select].sum()))


def write_points(path, z, kappa):
    json.dump([[float(a), float(b)] for a, b in zip(z, np.broadcast_to(kappa, np.shape(z)))], open(path, 'w'))


if __name__ == '__main__':
    import sys
    np.set_printoptions(linewidth=200, precision=4, suppress=True)
    m = Model(sys.argv[1])
    print('dim', m.dim, 'vec', m.nv, 'points', m.n, 'jumps', len(m.keys))
    print(m.checks())
