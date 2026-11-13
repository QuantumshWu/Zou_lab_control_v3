"""The exact mean force on an Rb87 atom held at rest, from the request's full 32-state master equation alone (no engine
code, no closure, no fold; motion3d_status/rewrite/restforce.status.md).

The request (contract: request.rs header) gives, at the position r (µm from the focus) in the states' frames (rad/µs),
  H(r, t) = H0 + Σ_ω Σ_i (x_{ω,i}(r) − x_{ω,i}(0)) O_{ω,i} + Σ_l [a_l(r) X_l e^{−iΔ_l t} + a_l(r)* R_l e^{+i(Δ_l + 2ν)t} + h.c.]
and the emission Σ_a D[C_a]. Here every term is built from the request's numbers and a textbook Gaussian beam:
  - the beams at Δ_l = 0 (the probe and its reflection): exact, static;
  - the repump (Δ = 2π·6588 MHz from the probe): exact, Floquet harmonics n = −N..N of Δ (ρ(t) = Σ ρ_n e^{−inΔt});
  - the tweezer (Δ = −2π·32.4 THz) and every counter-rotating term (2ν ≈ 2π·768 THz): their second-order effective
    Hamiltonian with each pair's own energy denominator (D1 sits 7 THz below the D2 frame: not negligible against 32 THz);
    the error is of the order (Ω_t/Δ_t)² ≈ 4e-6 of the trap's shift, the scattering they add ≈ 3e-5 /µs.
Force along ê: F = −Σ_n Tr(ρ_n ∂_ê H_{−n}) (the time average of −Tr ρ(t) ∂H(t)), ∂ by central differences (ε = 1e-4 µm).
  python restforce_fullref.py <request.json> <s,s,...> [N=2] [axis x,y,z] [out.json]
"""
import sys, json
import numpy as np

N = 32


def load(path):
    d = json.load(open(path))['model']
    mo = d['motion']
    mat = lambda o: (np.array(o['real'], float) + 1j * np.array(o['imag'], float)).reshape(N, N)
    freqs = np.array(d['frequencyCoordinates']['frequenciesMHz'], float)
    nu = 2 * np.pi * np.array(mo['frameFrequenciesMHz'], float)
    beams = []
    for b in mo['beams']:
        n = np.array(b['direction'], float)
        n = n / np.linalg.norm(n)
        beams.append(dict(id=b['id'], n=n, k=b['waveNumberPerUm'], w0=b.get('waistUm'), delta=2 * np.pi * float(np.dot(b['frequencyIndex'], freqs)),
                          idx=tuple(b['frequencyIndex']), X=mat(b['coupling']), R=mat(b['counterCoupling']),
                          E=np.array([c['re'] + 1j * c['im'] for c in b['electricField']])))
    forms = [dict(idx=tuple(f['frequencyIndex']), O=[mat(m) for m in f['matrices']]) for f in mo['outsideForms']]
    C = [mat(m) for m in mo['emission']]
    ground = np.array([s.startswith('ground') for s in d['states']])
    return dict(states=d['states'], H0=mat(mo['hamiltonian']), beams=beams, forms=forms, C=C, nu=nu[~ground][0], ground=ground,
                hm=1.054571817e-34 / mo['massKg'] * 1e6, trap=mo['trapBeamId'])


def amplitude(b, r):
    """A paraxial Gaussian beam focused at the origin, relative to its focus value (textbook form):
    (w0/w) exp(−ρ²/w²) exp(i(kζ + kρ²/(2R) − atan(ζ/z_R)))."""
    k, n, w0 = b['k'], b['n'], b['w0']
    zeta = float(np.dot(r, n))
    if not w0:
        return np.exp(1j * k * zeta)
    rho2 = float(np.dot(r, r)) - zeta * zeta
    zr = k * w0 * w0 / 2
    w2 = w0 * w0 * (1 + (zeta / zr) ** 2)
    curv = zeta / (zeta * zeta + zr * zr)          # 1/R
    return np.sqrt(w0 * w0 / w2) * np.exp(-rho2 / w2) * np.exp(1j * (k * zeta + 0.5 * k * rho2 * curv - np.arctan(zeta / zr)))


def coherence(E):
    """x(E): |E_q|² for q = −1, 0, +1, then Re, Im of E_p* E_q for (−1, 0), (−1, +1), (0, +1)."""
    x = [abs(E[0]) ** 2, abs(E[1]) ** 2, abs(E[2]) ** 2]
    for p, q in ((0, 1), (0, 2), (1, 2)):
        z = np.conj(E[p]) * E[q]
        x += [z.real, z.imag]
    return np.array(x)


def second_order(A, omega, E):
    """The effective Hamiltonian of A e^{−iωt} + h.c. (|ω| ≫ the static part's spread), symmetrized, with each pair's
    own energy denominator (E: the static diagonal): Σ_m A_im A†_mj ½[1/(E_i − E_m − ω) + 1/(E_j − E_m − ω)]
    + Σ_m A†_im A_mj ½[1/(E_i − E_m + ω) + 1/(E_j − E_m + ω)]."""
    Ad = A.conj().T
    d1 = 1.0 / (E[:, None] - E[None, :] - omega)     # [i, m]
    d2 = 1.0 / (E[:, None] - E[None, :] + omega)
    P = (A * d1) @ Ad + A @ (Ad * d1.T)               # Σ_m A_im A†_mj (1/(E_i−E_m−ω) + 1/(E_j−E_m−ω))
    Q = (Ad * d2) @ A + Ad @ (A * d2.T)
    return 0.5 * (P + Q)


def hamiltonian(m, r, slow):
    """(H_0, V) at r: the static part (exact near light, the outside shifts' change, the far light's second order) and V,
    the coefficient of e^{−iΔt} of the beams at the frequency Δ = `slow` (the repump: exact by Floquet)."""
    E0 = np.real(np.diag(m['H0']))
    H = m['H0'].copy()
    V = np.zeros((N, N), complex)
    fields = {}
    for b in m['beams']:
        a, a0 = amplitude(b, r), amplitude(b, np.zeros(3))
        fields.setdefault(b['idx'], [np.zeros(3, complex), np.zeros(3, complex)])
        fields[b['idx']][0] += a * b['E']
        fields[b['idx']][1] += a0 * b['E']
        A = a * b['X']
        if b['delta'] == 0.0:
            H += A + A.conj().T
        elif abs(b['delta'] - slow) < 1e-9 * abs(slow):
            V += A
        else:
            H += second_order(A, b['delta'], E0)
        # The counter-rotating term a* R e^{+i(Δ + 2ν)t} = B e^{−iω t}, ω = −(Δ + 2ν).
        H += second_order(np.conj(a) * b['R'], -(b['delta'] + 2 * m['nu']), E0)
    for f in m['forms']:
        if f['idx'] in fields:
            x = coherence(fields[f['idx']][0]) - coherence(fields[f['idx']][1])
            H += sum(xi * O for xi, O in zip(x, f['O']))
    return H, V


def kron_l(A):   # X ↦ A X (row-major vec)
    return np.kron(A, np.eye(N))


def kron_r(B):   # X ↦ X B
    return np.kron(np.eye(N), B.T)


def liouvillian(H, C):
    L = -1j * (kron_l(H) - kron_r(H))
    for c in C:
        cd = c.conj().T @ c
        L += np.kron(c, c.conj()) - 0.5 * (kron_l(cd) + kron_r(cd))
    return L


def floquet_steady(L0, Lp, Lm, delta, harmonics):
    """ρ(t) = Σ_n ρ_n e^{−inΔt}: (L0 + inΔ) ρ_n + Lp ρ_{n−1} + Lm ρ_{n+1} = 0, Tr ρ_0 = 1."""
    nv = L0.shape[0]
    ns = list(range(-harmonics, harmonics + 1))
    M = np.zeros((len(ns) * nv, len(ns) * nv), complex)
    for q, n in enumerate(ns):
        sl = slice(q * nv, (q + 1) * nv)
        M[sl, sl] = L0 + 1j * n * delta * np.eye(nv)
        if q > 0:
            M[sl, (q - 1) * nv:q * nv] = Lp
        if q < len(ns) - 1:
            M[sl, (q + 1) * nv:(q + 2) * nv] = Lm
    b = np.zeros(len(ns) * nv, complex)
    q0 = ns.index(0)
    row = q0 * nv
    M[row, :] = 0.0
    M[row, q0 * nv:(q0 + 1) * nv] = np.eye(N).ravel()
    b[row] = 1.0
    x = np.linalg.solve(M, b)
    return {n: x[q * nv:(q + 1) * nv].reshape(N, N) for q, n in enumerate(ns)}


def at(m, r, e, harmonics, slow, eps=1e-4):
    H, V = hamiltonian(m, r, slow)
    L0 = liouvillian(H, m['C'])
    Lp = -1j * (kron_l(V) - kron_r(V))
    Vd = V.conj().T
    Lm = -1j * (kron_l(Vd) - kron_r(Vd))
    rho = floquet_steady(L0, Lp, Lm, slow, harmonics)
    Hp, Vp = hamiltonian(m, r + eps * e, slow)
    Hm, Vm = hamiltonian(m, r - eps * e, slow)
    dH, dV = (Hp - Hm) / (2 * eps), (Vp - Vm) / (2 * eps)
    force = -np.trace(rho[0] @ dH).real
    if harmonics >= 1:
        force -= (np.trace(rho[-1] @ dV) + np.trace(rho[1] @ dV.conj().T)).real
    # The parts: −Tr ρ ∂(beam terms) per beam group is read below by the caller from `parts`.
    return dict(rho=rho, force=force, H=H, dH=dH, dV=dV, V=V)


if __name__ == '__main__':
    path = sys.argv[1]
    ss = [float(x) for x in sys.argv[2].split(',')]
    harmonics = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    e = np.array([float(x) for x in sys.argv[4].split(',')]) if len(sys.argv) > 4 else np.array([0.0697564737441253, 0.9975640502598242, 0.0])
    e = e / np.linalg.norm(e)
    m = load(path)
    slow = [b['delta'] for b in m['beams'] if b['delta'] != 0.0 and abs(b['delta']) < 1e6]
    slow = slow[0]
    # The decay identity and the frames.
    G = sum(c.conj().T @ c for c in m['C'])
    print('beams', [(b['id'], round(b['delta'], 3), b['w0']) for b in m['beams']], 'nu', m['nu'], 'slow Δ', slow, 'harmonics', harmonics)
    print('decay rates (diag of Σ C†C) D1/D2:', np.round(np.real(np.diag(G))[8:16].mean(), 4), np.round(np.real(np.diag(G))[16:].mean(), 4))
    out = []
    for s in ss:
        r = s * e
        x = at(m, r, e, harmonics, slow)
        p = np.real(np.diag(x['rho'][0]))
        names = m['states']
        F1, F2 = p[:3].sum(), p[3:8].sum()
        exc = p[8:].sum()
        tr = np.trace(x['rho'][0]).real
        print(f's {s:.4f}: F = {x["force"]:+.3f}  (Tr ρ0 {tr:.6f}; F=1 {F1:.4f}, F=2 {F2:.4f}, excited {exc:.4f})')
        print('   ground pops', np.round(p[:8], 4))
        print('   excited pops >1e-4', {names[i]: round(p[i], 4) for i in range(8, 32) if p[i] > 1e-4})
        out.append(dict(s=s, force=x['force'], pop=p.tolist(), names=names))
    if len(sys.argv) > 5:
        json.dump(out, open(sys.argv[5], 'w'), indent=1)
