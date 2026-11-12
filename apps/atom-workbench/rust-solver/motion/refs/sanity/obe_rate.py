"""The photon scattering rate of a ⁸⁷Rb atom at rest on the D2 line, written fresh from Steck's data (D. A. Steck,
"Rubidium 87 D Line Data", rev. 2.3.3: Γ = 2π · 6.0666 MHz; the cycling line |2,±2⟩ → |3′,±3⟩ I_sat = 1.669 mW/cm²; the
isotropic I_sat = 3.576 mW/cm² and the π-pumped effective one 2.503 mW/cm² as references; the hyperfine constants
A(5S1/2) = 3417.341305 MHz, A(5P3/2) = 84.7185 MHz, B(5P3/2) = 12.4965 MHz) -- none of coherent.py, model.py or the
app's tables in the fresh models -- against coherent.py's Hamiltonian in the same steady state and the two-level
formulas labs quote.

    python obe_rate.py                                   the free atom (B 0, no tweezer): configurations × s × δ
    python obe_rate.py --request <S3/S4 request JSON> [--s-scale x] [--repump-scales 0.1,1,10] [--out <json>]
                                                         the request's beams at the tweezer's focus
                       [--bins 25,8] [--skip-pattern]   ... and the rate from |2,2⟩ at r = 0 in bins: an engine run on
                                                         the same request with the atom held there (a heavy mass)

# The two fresh models
- closed (12 levels): |2,m⟩ and |3′,m′⟩, the coupling of |2,m⟩ → |3′,m+q⟩ (Ω_b/2) ε_{b,q} e^{i k_b·r} ⟨2 m; 1 q|3 m+q⟩
  with Ω_b = Γ √(I_b/(2 I_sat,cyc)) (the cycling line's Clebsch–Gordan coefficient 1: Steck's I_sat,cyc) and
  ε_q = ê_q*·ε (ê_±1 = ∓(x̂ ± iŷ)/√2, ê_0 = ẑ); the decay Γ Σ_q D[L_q], L_q = Σ_m ⟨2 m; 1 q|3 m+q⟩ |2,m⟩⟨3′,m+q|.
- D2 (24 levels): every |F=1,2; m⟩ and |F′=0..3; m′⟩, built in the uncoupled basis |m_J, m_I⟩ (8 ground, 16 excited):
  the hyperfine Hamiltonians A I·J (+ B's quadrupole on 5P3/2), the dipole acting on J alone,
  ⟨3/2 m_J′|d_q|1/2 m_J⟩ = ⟨1/2 m_J; 1 q|3/2 m_J′⟩ (the same normalization: |2,2⟩ → |3′,3′⟩ is 1), the decay
  L_q = d_q† (Σ_q L_q†L_q = 1 on 5P3/2: closed on the D2 manifold); the probe on F=2 (every F′), a repump on F=1 (every F′),
  each in its own rotating frame (F=1 turns at the difference of the two frequencies: H(F=1) = (E_F′=2 − E_F′=3) + δ_r −
  δ_p; each laser's coupling of the other ground level, 6.8 GHz off, left out: its light shift ≲ Ω²/(4·6.8 GHz) ~ 0.03
  MHz); the tweezer's focus: the ground's scalar shift U_g on every ground level, 5P3/2's shift δE(m_J) (linear
  polarization along z: the scalar and tensor parts, by |m_J|) on the uncoupled excited states -- in the hyperfine basis
  m′-dependent and mixing F′ (coherent.py keeps its F′=3 diagonal part).
R = Γ Σ ρ_e′e′ of the steady state of dρ/dt = −i[H, ρ] + Γ Σ_q D[L_q]ρ, solved exactly (the Liouvillian with the trace as
one row); a standing wave's rate averaged over its pattern along the beams (the mirror's distance is not fixed in a lab).
Each beam's intensity and polarization at the atom, the probe's and repump's offsets from their free lines and the
tweezer's shifts are the only inputs; with --request they are read from the request (I_b = intensityMWcm2, ε_b =
electricField, the shifts from coherent.setup: model.py's sum over states at the tweezer's power and waist).

# What it checks
- the fresh closed model's σ+ traveling wave is the two-level cycling formula (it is closed: exactly);
- coherent.py's Hamiltonian (its tables' Γ, I_sat and Clebsch–Gordan coefficients) in each configuration at the same
  intensity: the ratio to the fresh model (a shared-input error in Γ, I_sat or the couplings would show here);
- the request's beams at the focus: free → uniform shift → m′-dependent shifts → the hyperfine levels and the repump."""
import argparse, json, math, os, sys
import numpy as np

GAMMA = 2 * math.pi * 6.0666          # rad/µs (Steck Table 3)
ISAT_CYC, ISAT_ISO, ISAT_PI = 1.669, 3.576, 2.503   # mW/cm² (Steck Table 3, §5.1)
A_G, A_E, B_E = 3417.341305452145, 84.7185, 12.4965  # MHz (Steck Tables 4 and 6)
K_UM = 2 * math.pi / 0.780241209686   # 1/µm (Steck: λ = 780.241209686 nm)

def cg(j1, m1, j2, m2, J, M):
    """⟨j1 m1; j2 m2|J M⟩ by Racah's formula (half-integers allowed)."""
    if abs(m1 + m2 - M) > 1e-9 or abs(m1) > j1 + 1e-9 or abs(m2) > j2 + 1e-9 or abs(M) > J + 1e-9: return 0.0
    if J < abs(j1 - j2) - 1e-9 or J > j1 + j2 + 1e-9: return 0.0
    f = lambda x: math.factorial(int(round(x)))
    pre = math.sqrt((2 * J + 1) * f(J + j1 - j2) * f(J - j1 + j2) * f(j1 + j2 - J) / f(j1 + j2 + J + 1))
    pre *= math.sqrt(f(J + M) * f(J - M) * f(j1 - m1) * f(j1 + m1) * f(j2 - m2) * f(j2 + m2))
    s = 0.0
    for k in range(0, int(round(j1 + j2 - J)) + 1):
        den = [k, j1 + j2 - J - k, j1 - m1 - k, j2 + m2 - k, J - j2 + m1 + k, J - j1 - m2 + k]
        if min(den) < -1e-9: continue
        s += (-1) ** k / np.prod([f(d) for d in den])
    return pre * s

def sph(v):
    """ε_q = ê_q*·v, q = −1, 0, +1."""
    v = np.asarray(v, complex)
    return {-1: (v[0] + 1j * v[1]) / math.sqrt(2), 0: v[2], 1: -(v[0] - 1j * v[1]) / math.sqrt(2)}

def liouvillian(H, jumps, gamma=GAMMA):
    """dρ/dt = −i[H, ρ] + Γ Σ D[L]ρ (H Hermitian) on ρ as a row-major vector."""
    n = H.shape[0]; I = np.eye(n)
    Lv = -1j * (np.kron(H, I) - np.kron(I, H.T))
    for L in jumps:
        LdL = L.conj().T @ L
        Lv += gamma * (np.kron(L, L.conj()) - 0.5 * np.kron(LdL, I) - 0.5 * np.kron(I, LdL.T))
    return Lv

def steady(H, jumps, gamma=GAMMA):
    """The steady state (the trace as one row of the Liouvillian)."""
    n = H.shape[0]; Lv = liouvillian(H, jumps, gamma)
    b = np.zeros(n * n, complex); Lv[0, :] = 0; Lv[0, [i * n + i for i in range(n)]] = 1; b[0] = 1
    return np.linalg.solve(Lv, b).reshape(n, n)

def binned(H, jumps, rho0, excited, width, count, gamma=GAMMA):
    """The photon rate Γ Tr_e ρ averaged over consecutive bins of `width` µs from ρ0, exactly: one step's propagator
    e^{Lw} and its integral ∫₀ʷ e^{Lt} dt from the exponential of [[L, 1], [0, 0]] w."""
    from scipy.linalg import expm
    n = H.shape[0]; Lv = liouvillian(H, jumps, gamma); m = n * n
    A = np.zeros((2 * m, 2 * m), complex); A[:m, :m] = Lv; A[:m, m:] = np.eye(m)
    X = expm(A * width); E, F = X[:m, :m], X[:m, m:]
    r = rho0.reshape(-1).astype(complex); out = []
    for _ in range(count):
        mean = (F @ r).reshape(n, n) / width
        out.append(gamma * float(np.real(np.trace(mean[excited, excited])))); r = E @ r
    return out

def two_level(s_total, delta, gamma=GAMMA):
    return 0.5 * gamma * s_total / (1 + s_total + 4 * delta ** 2 / gamma ** 2)

# ---------------------------------------------------------------------------------------------------------------------
# The closed F=2 → F'=3 model (12 levels)
N12 = 12
ig = lambda m: m + 2; ie = lambda m: 5 + m + 3
CG23 = {(m, q): cg(2, m, 1, q, 3, m + q) for m in range(-2, 3) for q in (-1, 0, 1)}
JUMPS12 = []
for q in (-1, 0, 1):
    L = np.zeros((N12, N12), complex)
    for m in range(-2, 3):
        if abs(m + q) <= 3: L[ig(m), ie(m + q)] = CG23[(m, q)]
    JUMPS12.append(L)
assert np.allclose(sum(L.conj().T @ L for L in JUMPS12)[5:, 5:], np.eye(7)), 'closed F=3 decay'

def h12(beams, r, delta, excited_shift=None, ground_shift=0.0):
    """H (rad/µs, rotating at the probe): −δ on F'=3 (+ `excited_shift[m']` − `ground_shift`: the tweezer); couplings."""
    H = np.zeros((N12, N12), complex)
    for m in range(-3, 4): H[ie(m), ie(m)] = -delta + (0.0 if excited_shift is None else excited_shift[m + 3] - ground_shift)
    for b in beams:
        a = b['Omega'] / 2 * np.exp(1j * np.dot(b['k'], r)); e = sph(b['eps'])
        for m in range(-2, 3):
            for q in (-1, 0, 1):
                if abs(m + q) <= 3 and e[q] != 0: H[ie(m + q), ig(m)] += a * e[q] * CG23[(m, q)]
    return H + np.tril(H, -1).conj().T

def rate12(H):
    rho = steady(H, JUMPS12)
    return GAMMA * float(np.real(np.trace(rho[5:, 5:]))), rho

def pattern(beams, points):
    """Points along the first beam over its standing wave's period π/k (one point for a single beam)."""
    if len(beams) == 1: return [np.zeros(3)]
    k = np.asarray(beams[0]['k'], float); u = k / np.linalg.norm(k); period = math.pi / np.linalg.norm(k)
    return [u * period * (i + 0.5) / points for i in range(points)]

def averaged12(beams, delta, points=64, **kw):
    return float(np.mean([rate12(h12(beams, r, delta, **kw))[0] for r in pattern(beams, points)]))

def configurations(s):
    """The free-atom configurations at s = I/I_sat,cyc per beam."""
    Om = GAMMA * math.sqrt(s / 2); ky = np.array([0, K_UM, 0]); kz = np.array([0, 0, K_UM])
    z, x = [0, 0, 1], [1, 0, 0]
    sp = [-1 / math.sqrt(2), -1j / math.sqrt(2), 0]; sm = [1 / math.sqrt(2), -1j / math.sqrt(2), 0]   # ê_+1, ê_−1 about z
    return {
        'σ+ traveling': [dict(Omega=Om, k=kz, eps=sp)],
        'π traveling': [dict(Omega=Om, k=ky, eps=z)],
        'linear ⊥ traveling': [dict(Omega=Om, k=ky, eps=x)],
        'σ+σ−': [dict(Omega=Om, k=kz, eps=sp), dict(Omega=Om, k=-kz, eps=sm)],
        'lin⊥lin': [dict(Omega=Om, k=ky, eps=z), dict(Omega=Om, k=-ky, eps=x)],
        'lin∥lin': [dict(Omega=Om, k=ky, eps=z), dict(Omega=Om, k=-ky, eps=z)],
    }

# ---------------------------------------------------------------------------------------------------------------------
# The D2 model (24 levels), in the uncoupled basis |m_J, m_I⟩
JG, JE, II = 0.5, 1.5, 1.5
MJG, MJE, MI = [-0.5, 0.5], [-1.5, -0.5, 0.5, 1.5], [-1.5, -0.5, 0.5, 1.5]
NG, NE = 8, 16; N24 = NG + NE
gidx = lambda mj, mi: MJG.index(mj) * 4 + MI.index(mi)
eidx = lambda mj, mi: NG + MJE.index(mj) * 4 + MI.index(mi)

def angular(j):
    """J_z, J_+ on |j, m⟩, m ascending."""
    ms = np.arange(-j, j + 1); n = len(ms); Jp = np.zeros((n, n))
    for i in range(n - 1): Jp[i + 1, i] = math.sqrt(j * (j + 1) - ms[i] * (ms[i] + 1))
    return np.diag(ms), Jp

def i_dot_j(j):
    Jz, Jp = angular(j); Iz, Ip = angular(II)
    return np.kron(Jz, Iz) + 0.5 * (np.kron(Jp, Ip.T) + np.kron(Jp.T, Ip))

H_HFS_G = A_G * i_dot_j(JG)
_IJ = i_dot_j(JE)
H_HFS_E = A_E * _IJ + B_E * (3 * _IJ @ _IJ + 1.5 * _IJ - II * (II + 1) * JE * (JE + 1) * np.eye(16)) / (2 * II * (2 * II - 1) * JE * (2 * JE - 1))

def projectors(H, groups):
    """Eigenprojectors of a Hermitian H onto its levels, grouped by energy (ascending)."""
    w, v = np.linalg.eigh(H); out = []; i = 0
    for size in groups:
        out.append((float(np.mean(w[i:i + size])), v[:, i:i + size] @ v[:, i:i + size].conj().T)); i += size
    return out

(E_G1, P_G1), (E_G2, P_G2) = projectors(H_HFS_G, [3, 5])
(E_E0, P_E0), (E_E1, P_E1), (E_E2, P_E2), (E_E3, P_E3) = projectors(H_HFS_E, [1, 3, 5, 7])
assert abs(E_G2 - E_G1 - 6834.682611) < 1e-3 and abs(E_E3 - E_E2 - 266.650) < 0.01 and abs(E_E2 - E_E1 - 156.947) < 0.01 and abs(E_E1 - E_E0 - 72.218) < 0.01, "Steck splittings (A, B alone: within 7 kHz)"

D24 = {}
for q in (-1, 0, 1):
    D = np.zeros((N24, N24))
    for mj in MJG:
        for mi in MI:
            if abs(mj + q) <= JE: D[eidx(mj + q, mi), gidx(mj, mi)] = cg(JG, mj, 1, q, JE, mj + q)
    D24[q] = D
JUMPS24 = [D24[q].T.astype(complex) for q in (-1, 0, 1)]
assert np.allclose(sum(L.conj().T @ L for L in JUMPS24)[NG:, NG:], np.eye(NE)), 'closed 5P3/2 decay'
PG = [np.zeros((N24, N24), complex) for _ in range(2)]
PG[0][:NG, :NG] = P_G1; PG[1][:NG, :NG] = P_G2
_stretched = np.zeros(N24, complex); _stretched[:NG] = P_G2[:, gidx(0.5, 1.5)]
assert abs(abs(_stretched[gidx(0.5, 1.5)]) - 1) < 1e-12, '|2,2⟩ is |1/2, 3/2⟩'

def h24(probes, repump, r, delta_p, delta_r=0.0, shifts=None):
    """H (rad/µs): F=2 at 0, F=1 at (E_F'=2 − E_F'=3) + δ_r − δ_p, 5P3/2 at its hyperfine Hamiltonian less E_F'=3, less δ_p;
    `shifts` = (U_g, δE(|m_J| = 1/2), δE(3/2)) in MHz: the tweezer's focus (5P3/2 less the ground's); the probes on F=2,
    the repump on F=1 (dict(Omega, eps) or None)."""
    tw = 2 * math.pi; H = np.zeros((N24, N24), complex)
    H[:NG, :NG] = tw * ((E_E2 - E_E3) + (delta_r - delta_p) / tw) * P_G1
    H[NG:, NG:] = tw * (H_HFS_E - E_E3 * np.eye(NE)) - delta_p * np.eye(NE)
    if shifts is not None:
        Ug, half, three = shifts
        for mj in MJE:
            for mi in MI:
                i = eidx(mj, mi); H[i, i] += tw * ((three if abs(mj) > 1 else half) - Ug)
    V = np.zeros((N24, N24), complex)
    for b in probes:
        e = sph(b['eps']); a = b['Omega'] / 2 * np.exp(1j * np.dot(b['k'], r))
        V += a * sum(e[q] * D24[q] for q in (-1, 0, 1)) @ PG[1]
    if repump is not None:
        e = sph(repump['eps'])
        V += repump['Omega'] / 2 * sum(e[q] * D24[q] for q in (-1, 0, 1)) @ PG[0]
    return H + V + V.conj().T

def rate24(H):
    rho = steady(H, JUMPS24)
    return GAMMA * float(np.real(np.trace(rho[NG:, NG:]))), rho

def averaged24(probes, repump, delta_p, delta_r=0.0, shifts=None, points=32):
    out = [rate24(h24(probes, repump, r, delta_p, delta_r, shifts)) for r in pattern(probes, points)]
    return float(np.mean([o[0] for o in out])), sum(o[1] for o in out) / len(out)

EXC_LOW = "F'<3"

def f_m_populations(rho):
    """ρ's ground populations by |F, m⟩ and excited by |F'=3, m'⟩ (the hyperfine basis from the projectors)."""
    out = {}
    for F, P in ((1, P_G1), (2, P_G2)):
        Jz = np.kron(np.diag(MJG), np.eye(4)) + np.kron(np.eye(2), np.diag(MI)); r = rho[:NG, :NG]
        w, v = np.linalg.eigh(P @ Jz @ P + 10 * (np.eye(NG) - P))
        for i in range(len(w)):
            if w[i] < 5: out[f'{F},{int(round(w[i]))}'] = float(np.real(v[:, i].conj() @ r @ v[:, i]))
    re = rho[NG:, NG:]; Jz = np.kron(np.diag(MJE), np.eye(4)) + np.kron(np.eye(4), np.diag(MI))
    out["F'=3"] = float(np.real(np.trace(P_E3 @ re))); out["F'<3"] = float(np.real(np.trace(re))) - out["F'=3"]
    return out

# ---------------------------------------------------------------------------------------------------------------------
def free_tables(deltas, ss, points):
    print(f"fresh closed OBE (Steck): Γ = 2π·6.0666 MHz; s = I/I_sat,cyc per beam (1.669 mW/cm²); R photons/µs; B 0, no tweezer,")
    print(f"the atom at rest; standing waves averaged over their pattern. δ = {deltas} Γ")
    print(f"two-level references (S = Σ_b s_b): 'cyc' Γ/2·S/(1+S+4δ²/Γ²); 'π-eff' the same with I_sat 2.503 (S × 0.6668)")
    out = {}
    for s in ss:
        for name, beams in configurations(s).items():
            row = [averaged12(beams, d * GAMMA, points) for d in deltas]; out[f'{name} s{s}'] = row
            S = s * len(beams)
            print(f"  s {s:>2} {name:<19} {' '.join(f'{x:7.3f}' for x in row)}  | cyc {' '.join(f'{two_level(S, d * GAMMA):6.3f}' for d in deltas)}"
                  f"  π-eff {' '.join(f'{two_level(S * ISAT_CYC / ISAT_PI, d * GAMMA):6.3f}' for d in deltas)}")
    return out

def coherent_tables(deltas, ss, points, fresh):
    """coherent.py's Hamiltonian (its tweezer off, B 0) in each configuration at the same intensity I = s · 1.669 mW/cm²:
    its own Γ, I_sat (model.py from the app's tables) and Clebsch–Gordan coefficients (model.wigner3j)."""
    here = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, here)
    import coherent as C, model as M
    tables = M.app_tables(); sh = M.light_shifts(tables, wavelength_nm=852.0, power_mw=1.0, waist_um=1.2); p = M.parameters(tables, sh)
    Cm = np.array([[C.cg(2, m, 1, q, 3, m + q) if abs(m + q) <= 3 else 0.0 for m in range(-2, 3)] for q in (-1, 0, 1)])
    P = np.array([p['gamma'], 1.2, 5.3, 0.0, p['hbar_m'], p['uK'], 0.0, 0.0, 0.0])
    cj = []
    for q in (-1, 0, 1):
        L = np.zeros((12, 12), complex)
        for mi in range(5):
            if abs(mi - 2 + q) <= 3: L[mi, 5 + mi - 2 + q + 3] = Cm[q + 1, mi]
        cj.append(L)
    print(f"\ncoherent.py's Hamiltonian: Γ {p['gamma']:.5f} rad/µs (Steck {GAMMA:.5f}), I_sat {p['Isat']:.5f} mW/cm² (Steck {ISAT_CYC}); "
          f"max |CG − fresh| {max(abs(Cm[q + 1, m + 2] - CG23[(m, q)]) for m in range(-2, 3) for q in (-1, 0, 1)):.1e}")
    Hc = np.zeros((12, 12), complex); worst = 0.0; out = {}
    for s in ss:
        I = s * ISAT_CYC
        for name, beams in configurations(s).items():
            k = np.array([b['k'] for b in beams], float); eps = np.array([[sph(b['eps'])[q] for q in (-1, 0, 1)] for b in beams], complex)
            Om = p['gamma'] * np.sqrt(np.full(len(beams), I / p['Isat']) / 2); Hb = np.zeros((len(beams), 12, 12), complex)
            row = []
            for d in deltas:
                H0 = np.zeros((12, 12), complex)
                for i in range(5, 12): H0[i, i] = -d * GAMMA      # the same δ in rad/µs (Γ's definition apart)
                rs = []
                for r in pattern(beams, points):
                    C.hamiltonian(r[0], r[1], r[2], P, H0, np.zeros(12), Cm, k, eps, Om, Hc, Hb)
                    rs.append(p['gamma'] * float(np.real(np.trace(steady(0.5 * (Hc + Hc.conj().T), cj, p['gamma'])[5:, 5:]))))
                row.append(float(np.mean(rs)))
            ratio = [a / b for a, b in zip(row, fresh[f'{name} s{s}'])]; worst = max(worst, max(abs(x - 1) for x in ratio))
            out[f'{name} s{s}'] = row
            print(f"  s {s:>2} {name:<19} {' '.join(f'{x:7.3f}' for x in row)}  | / fresh {' '.join(f'{x:.4f}' for x in ratio)}")
    print(f"  largest |coherent/fresh − 1| {worst:.2e}")
    return out

def in_trap(request_path, s_scale, repump_scales, points, out, bins=None, point_only=False):
    here = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, here)
    import coherent as C
    req = json.load(open(request_path, encoding='utf-8'))
    S = C.setup(req, field_G=0.0, s_scale=s_scale); p = S['p']; sh = S['shifts']; d = req['model']['motion']
    shifts = (float(sh['groundShiftMHz']), float(sh['excitedHalfShiftMHz']), float(sh['excitedShiftMHz']))
    off = S['offset']; dp = 2 * math.pi * off
    I = [float(x) * p['Isat'] for x in S['s']]                      # the probe beams' intensities (mW/cm², s_scale in)
    probes = [dict(Omega=GAMMA * math.sqrt(Ib / (2 * ISAT_CYC)), k=S['k'][i], eps=C.cartesian(S['eps'][i])) for i, Ib in enumerate(I)]
    # the repump: the beam neither the tweezer nor on the probes' frequency; its offset from the free F=1 → F'=2 line (Steck:
    # F=1 4271.676631 MHz below the 5S1/2 centre, F'=2 72.9113 MHz below 5P3/2's)
    trap = d['trapBeamId']; others = [b for b in d['beams'] if b['id'] != trap]
    rep = [b for b in others if b['name'] not in S['names']]
    line12 = C.M.app_tables()['lines']['D2']['nu'] / 1e6 - 72.9113 + 4271.676631
    coords = req['model']['frequencyCoordinates']['frequenciesMHz']
    repump = None; dr = 0.0
    if rep:
        b = rep[0]; E = np.array([complex(c['re'], c['im']) for c in b['electricField']]); E /= np.linalg.norm(E)
        dr = 2 * math.pi * (max(d['frameFrequenciesMHz']) + sum(n * f for n, f in zip(b['frequencyIndex'], coords)) - line12)
        repump = dict(Omega=GAMMA * math.sqrt(b['intensityMWcm2'] / (2 * ISAT_CYC)), eps=C.cartesian(E), I=b['intensityMWcm2'])
    U = S['diagL'] / (2 * math.pi)
    print(f"\n{os.path.basename(os.path.dirname(request_path))}: probe offset {off:.3f} MHz, beams {S['names']}, I {np.round(I, 3).tolist()} mW/cm² (s_cyc {np.round(np.array(I) / ISAT_CYC, 2).tolist()}), "
          f"ε {[np.round(C.cartesian(e), 3).tolist() for e in S['eps']]}")
    if repump: print(f"  repump {rep[0]['name']}: {repump['I']:.3f} mW/cm², ε {np.round(repump['eps'], 3).tolist()}, offset {dr / 2 / math.pi:+.3f} MHz from free F=1 → F'=2")
    print(f"  tweezer {p['tweezerNm']:.1f} nm {p['tweezerMW']:.3f} mW w0 {p['w0']} µm: U_g {shifts[0]:.3f} MHz ({-shifts[0] * 1e6 * 6.62607015e-34 / 1.380649e-23 * 1e3:.3f} mK), "
          f"5P3/2 δE(|m_J| 1/2) {shifts[1]:+.3f}, (3/2) {shifts[2]:+.3f} MHz; F'=3 by m' (coherent.py) {np.round(U[5:], 2).tolist()} MHz")
    dc = dp - 2 * math.pi * (shifts[2] - shifts[0]); dm = dp - 2 * math.pi * (U[8] - shifts[0])
    print(f"  at the focus: |2,±2⟩ → |3′,±3⟩ {dc / GAMMA:+.2f}Γ, |2,0⟩ → |3′,0⟩ {dm / GAMMA:+.2f}Γ (free {dp / GAMMA:+.2f}Γ); repump F=1 → F'=2 by m_J ~ "
          f"{(dr - 2 * math.pi * (shifts[2] - shifts[0])) / GAMMA:+.2f}…{(dr - 2 * math.pi * (shifts[1] - shifts[0])) / GAMMA:+.2f}Γ")
    res = dict(request=request_path, sScale=s_scale, offsetMHz=off, I=I, shifts=shifts, excitedF3=U[5:].tolist())
    # closed 12-level: free, the cycling line's shift on every level (uniform), the m'-dependent shifts; at the focus point
    # (r = 0, the request's phases) and over the pattern
    zero = np.zeros(3); Ug2 = 2 * math.pi * shifts[0]
    uni = [2 * math.pi * shifts[2]] * 7; mdep = list(2 * math.pi * U[5:])
    rows = {}
    for label, kw in (('free', {}), ('uniform (cycling line\'s shift)', dict(excited_shift=uni, ground_shift=Ug2)), ("m'-dependent", dict(excited_shift=mdep, ground_shift=Ug2))):
        r0, rho0 = rate12(h12(probes, zero, dp, **kw)); ra = averaged12(probes, dp, points, **kw)
        rows[label] = (r0, ra)
        pops = np.real(np.diag(rho0))[:5] / np.real(np.trace(rho0[:5, :5]))
        print(f"  closed  {label:<30} r=0 {r0:.4f}/µs  pattern {ra:.4f}/µs   |2,m⟩ (m −2..2, r=0) {np.round(pops, 3).tolist()}, ρ_ee {np.real(np.trace(rho0[5:, 5:])):.4f}")
    res['closed'] = rows
    # coherent.py's own focus Hamiltonian: the same steady state (its I_sat, Γ, CG, diagL), r = 0 and the pattern
    P = C.trap_params(S); Hc = np.zeros((12, 12), complex); Hb = np.zeros((len(S['k']), 12, 12), complex)
    Om = p['gamma'] * np.sqrt(S['s'] / 2)
    cj = []
    for q in (-1, 0, 1):
        L = np.zeros((12, 12), complex)
        for mi in range(5):
            if abs(mi - 2 + q) <= 3: L[mi, 5 + mi - 2 + q + 3] = S['C'][q + 1, mi]
        cj.append(L)
    cr = []
    for r in pattern(probes, points) + [zero]:
        g = C.hamiltonian(r[0], r[1], r[2], P, S['H0'], S['diagL'], S['C'], S['k'], S['eps'].astype(complex), Om, Hc, Hb)[0]
        for j in range(12): Hc[j, j] += S['diagL'][j] * (1 - g)     # the focus's shifts at every point of the pattern
        cr.append(p['gamma'] * float(np.real(np.trace(steady(0.5 * (Hc + Hc.conj().T), cj, p['gamma'])[5:, 5:]))))
    print(f"  coherent.py's focus Hamiltonian (closed)      r=0 {cr[-1]:.4f}/µs  pattern {np.mean(cr[:-1]):.4f}/µs")
    res['coherent'] = (cr[-1], float(np.mean(cr[:-1])))
    S_tot = sum(I) / ISAT_CYC
    print(f"  two-level, S = Σ s_cyc = {S_tot:.2f}: at the cycling line's focus detuning {two_level(S_tot, dc):.4f}/µs, at |2,0⟩→|3′,0⟩'s {two_level(S_tot, dm):.4f}/µs, "
          f"π-eff (I_sat 2.503) at |2,0⟩→|3′,0⟩'s {two_level(S_tot * ISAT_CYC / ISAT_PI, dm):.4f}/µs; free (no tweezer) {two_level(S_tot, dp):.4f}/µs")
    # D2 (24 levels) with the repump: free, at the focus; the repump scaled
    if repump and point_only:
        r0, rho0 = rate24(h24(probes, repump, zero, dp, dr, shifts)); fm = f_m_populations(rho0)
        print(f"  D2 24-level focus, the request's repump: r=0 {r0:.4f}/µs, F=1 share {sum(v for k_, v in fm.items() if k_.startswith('1,')):.4f}")
        res['d2r0'] = r0
    elif repump:
        d2 = {}
        for label, sh_ in (('free', None), ('focus', shifts)):
            for sc in repump_scales:
                rp = dict(repump, Omega=repump['Omega'] * math.sqrt(sc))
                r0, rho0 = rate24(h24(probes, rp, zero, dp, dr, sh_)); ra, rhoa = averaged24(probes, rp, dp, dr, sh_, points)
                fm = f_m_populations(rhoa); F1 = sum(v for k_, v in fm.items() if k_.startswith('1,'))
                d2[f'{label} ×{sc}'] = dict(r0=r0, pattern=ra, F1=F1, pops=fm)
                print(f"  D2 24-level {label:<5} repump ×{sc:<5} r=0 {r0:.4f}/µs  pattern {ra:.4f}/µs  F=1 share {F1:.4f}; F'<3 excited {fm[EXC_LOW]:.2e}")
        res['d2'] = d2
    if bins:
        # from |2,2⟩ at r = 0 (an engine run with the atom held there): the rate in bins, closed and D2 with the repump
        width, count = bins
        rho12 = np.zeros((N12, N12), complex); rho12[ig(2), ig(2)] = 1
        rho24 = np.zeros((N24, N24), complex); rho24[gidx(0.5, 1.5), gidx(0.5, 1.5)] = 1
        c12 = binned(h12(probes, zero, dp, excited_shift=mdep, ground_shift=Ug2), JUMPS12, rho12, slice(5, None), width, count)
        c24 = binned(h24(probes, repump, zero, dp, dr, shifts), JUMPS24, rho24, slice(NG, None), width, count) if repump else None
        print(f"  from |2,2⟩ at r = 0, {width:g} µs bins: closed {' '.join(f'{x:.4f}' for x in c12)}" + (f"; D2 + repump {' '.join(f'{x:.4f}' for x in c24)}" if c24 else ''))
        res['bins'] = dict(width=width, closed=c12, d2=c24)
    out.append(res)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--request', action='append', default=[]); ap.add_argument('--s-scale', type=float, default=1.0)
    ap.add_argument('--repump-scales', default='0.1,1,10'); ap.add_argument('--points', type=int, default=32)
    ap.add_argument('--bins', default=None, help='width,count (µs): the rate from |2,2⟩ at r = 0 in bins (an engine run held there)')
    ap.add_argument('--skip-pattern', action='store_true', help='the D2 model at r = 0 only (no pattern average, no repump scan)')
    ap.add_argument('--free', action='store_true', help='the free-atom tables (default when no --request)'); ap.add_argument('--out', default=None)
    a = ap.parse_args()
    deltas, ss = [-2, -4, -6, -10], [1, 8, 38]
    result = {}
    # self-check: σ+ traveling is the closed two-level cycle
    beams = configurations(2)['σ+ traveling']; d = -3 * GAMMA
    r, _ = rate12(h12(beams, np.zeros(3), d)); r24, _ = rate24(h24(beams, dict(Omega=GAMMA, eps=[0, 0, 1]), np.zeros(3), d))
    print(f"self-check σ+ traveling, s 2, δ −3Γ: closed {r:.6f}, D2 {r24:.6f}, two-level {two_level(2, d):.6f}/µs")
    if a.free or not a.request:
        fresh = free_tables(deltas, ss, a.points); result['fresh'] = fresh
        result['coherent'] = coherent_tables(deltas, ss, a.points, fresh)
        # the D2 model's free atom (repump strong, resonant, π): the F'≠3 lines' share at each δ
        print('\nD2 24-level, free, a resonant π repump at s 10 (F=1 emptied): R and its ratio to the closed model')
        for s in ss:
            for name, beams in configurations(s).items():
                rp = dict(Omega=GAMMA * math.sqrt(10 / 2), eps=[0, 0, 1])
                row = [averaged24(beams, rp, d_ * GAMMA, 0.0, None, a.points)[0] for d_ in deltas]
                print(f"  s {s:>2} {name:<19} {' '.join(f'{x:7.3f}' for x in row)}  | / closed {' '.join(f'{x / y:.4f}' for x, y in zip(row, fresh[f'{name} s{s}']))}")
    traps = []
    for rq in a.request:
        in_trap(rq, a.s_scale, [float(x) for x in a.repump_scales.split(',')], a.points, traps,
                (float(a.bins.split(',')[0]), int(a.bins.split(',')[1])) if a.bins else None, a.skip_pattern)
    result['trap'] = traps
    if a.out: json.dump(result, open(a.out, 'w'), default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))

if __name__ == '__main__':
    main()
