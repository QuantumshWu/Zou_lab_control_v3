"""S3, the user's experiment in the lab's geometry (refs/sanity.mjs), by an independent coherent model: quantum jumps of
the full internal state of F=2 and F'=3 along classical trajectories in the tweezer. It shares no code with the motion
engine; it takes model.py's tables (light shifts by its own sum over states, Clebsch–Gordan, Zeeman, I_sat).

    python coherent.py --request <S3 request JSON> [--atoms 400] [--end 300] [--every 5] [--dt 0.001] [--noise 1]
                       [--field G] [--offset MHz] [--s-scale x] [--depth-scale x] [--phase deg] [--still] [--out <json>]
                       [--selftest]

# Why a coherent model: the rate model's premise fails here
model.py's multilevel rate model (S2Q) leaves the ground coherences out: there the probe is 3.18 mW/cm² (s ≈ 1.9), its
ground light shift ~0.4 MHz against the 3 G field's Larmor 2.1 MHz per Δm. S3's probe is 1 mW in 1 mm, 63.7 mW/cm² a
beam: s = I/I_sat ≈ 38 on the cycling line, Ω = Γ√(s/2) ≈ 4.4Γ ≈ 27 MHz, and the ground light shift Ω²|c|²/4Δ is
~7|c|² MHz at Δ ≈ −4Γ (the probe at the free line, the trap's centre) and ~4|c|² MHz at −7Γ (20 MHz red of it) --
above the Larmor splitting either way. And the two beams drive the same Δm = ±1 Raman pairs (π from one,
σ± from the other: lin⊥lin about a field along the π beam's polarization), with a Raman Rabi frequency
Ω_π Ω_σ |c c'|/2Δ of several MHz: the ground coherences are driven, not dephased, so the populations do not obey rate
equations and the force does not follow from them. This model keeps them.

# The model
Internal state ψ on |F=2, m⟩ (5) and |F'=3, m'⟩ (7), in the frame rotating at the probe's frequency (one frequency: the
repump acts on F=1, which this closed F=2 → F'=3 cycle does not reach -- off-resonant F'=2 excitation by the probe,
267 MHz = 44Γ below, is left out: its light shift Ω²|c|²/(4·267 MHz) ≲ 0.7 MHz and its depumping to F=1 ≲ 0.1/µs).
H/ħ = Σ_m (Z_g m + U_g g(r)) |g_m⟩⟨g_m| + Σ_m' (Z_e m' + U_e(m') g(r) − ν) |e_m'⟩⟨e_m'| − iΓ/2 Σ|e⟩⟨e|
      + ½ Σ_b Σ_q Ω_b ε_b,q e^{i k_b·r} Σ_m ⟨2 m; 1 q|3 m+q⟩ |e_{m+q}⟩⟨g_m| + h.c.
- the beams b: the probe and its reflection, each its wave vector, its intensity's s_b = I_b/I_sat (Ω_b = Γ√(s_b/2)) and
  its polarization's spherical components ε_b,q = ê_q*·ε_b with their phases at the origin -- read from the request
  (the app's own scene: `electricField`, ordered q = −1, 0, +1, |E|² ∝ I);
- the tweezer: g(r) its intensity over the focus's (model.py `beam`), U_g = the ground shift (model.py's sum over
  states), U_e(m') = Σ |⟨m_J m_I|3 m'⟩|² δE(m_J) (model.py `multilevel_tables`);
- Z: the 3 G (or `--field`) Zeeman shifts, ν the probe's offset from the free F=2 → F'=3 line: the request's rotating
  frame on the excited states less the line (or `--offset`; −20 MHz in S3).
Between jumps ψ follows the non-Hermitian H (RK4, dt = 1 ns ≪ 1/|H|); a jump where ‖ψ‖² reaches a uniform threshold
(Mølmer–Castin–Dalibard), through L_q = √Γ Σ_m ⟨2 m; 1 q|3 m+q⟩ |g_m⟩⟨e_{m+q}| drawn by ‖L_q ψ‖², the emitted photon in
the q pattern (π: sin²θ, σ: (1 + cos²θ)/2 about z), its recoil −ħk n. The atom's centre: classical, under the mean force
of its internal state, F = −⟨∇H⟩/‖ψ‖²: the tweezer's gradient on every level, and of each beam's coupling the radiation
pressure and the polarization gradient's dipole force together, 2 k_b Im⟨ψ|H_b⁺|ψ⟩ (H_b⁺ the beam's |e⟩⟨g| part) --
for one running wave ħkΓρ_ee (`--selftest`). The absorbed photon's momentum fluctuation (which beam was absorbed) is
not in a mean force; `--noise 1` adds it at each jump, ħ(k_b − Σ p k) with b drawn by p_b ∝ max(2 Im⟨ψ|H_b⁺|ψ⟩, 0) --
the variance of the absorption without a change of the mean.
Initial atoms: thermal (model.py's sampling) at the request's temperature in |2, 2⟩. Lost: the app's definition, outside
1/20 of the focus intensity with E = mv²/2 + U_g g > 0. T: m Var(v)/k_B over the held atoms, per lab axis.
"""
import argparse, json, math, os, sys, time
import numpy as np
import numba as nb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model as M

def cg(j1, m1, j2, m2, J, Mm):
    return (-1) ** int(round(j1 - j2 + Mm)) * math.sqrt(2 * J + 1) * M.wigner3j(j1, j2, J, m1, m2, -Mm)

def spin(F):
    """F_x, F_y, F_z on |F, m>, m = -F..F (hbar = 1)."""
    ms = np.arange(-F, F + 1); n = len(ms)
    Fp = np.zeros((n, n), complex)
    for i, m in enumerate(ms[:-1]):
        Fp[i + 1, i] = math.sqrt(F * (F + 1) - m * (m + 1))
    return (Fp + Fp.conj().T) / 2, (Fp - Fp.conj().T) / 2j, np.diag(ms).astype(complex)

def spherical(v):
    """A field vector's components ε_q = ê_q*·v, q = −1, 0, +1 (the request's order and convention)."""
    return np.array([(v[0] + 1j * v[1]) / math.sqrt(2), v[2], -(v[0] - 1j * v[1]) / math.sqrt(2)])

def cartesian(e):
    """The field vector of spherical components ε_q (q = −1, 0, +1; `spherical`'s inverse): v = Σ_q ε_q ê_q."""
    return np.array([(e[0] - e[2]) / math.sqrt(2), -1j * (e[0] + e[2]) / math.sqrt(2), e[1]])

def tilted(beams, axis, deg):
    """The beams turned by `deg` toward `axis` (the tweezer's) about k̂₁ × axis -- k and the polarization together, so each
    keeps its polarization about its own k (σ stays σ, π stays ⊥ k)."""
    k1 = beams[0]['k'] / np.linalg.norm(beams[0]['k']); n = np.cross(k1, axis); n /= np.linalg.norm(n)
    t = math.radians(deg); K = np.array([[0, -n[2], n[1]], [n[2], 0, -n[0]], [-n[1], n[0], 0]])
    R = np.eye(3) * math.cos(t) + math.sin(t) * K + (1 - math.cos(t)) * np.outer(n, n)
    return [dict(b, k=R @ b['k'], eps=spherical(R @ cartesian(b['eps']))) for b in beams]

def lab_beams(k_um, tilt_deg=4.0, power_mW=1.0, waist_um=1000.0, plate_deg=45.0, mirror_deg=180.0):
    """The probe and its reflection from the lab's description alone, none of the app's conversion: the probe along +y
    tilted toward +x, its polarization along the field's axis z (π), I = 2P/(πw²) at the focus; the reflection back along
    −k through a λ/4 plate at `plate_deg` to the polarization, passed twice -- a λ/2 plate at that angle: in the probe's
    transverse basis (a = z, b = z × k̂), J = [[cos 2θ, sin 2θ], [sin 2θ, −cos 2θ]] --, the mirror reversing the
    tangential field, and the round trip's phase `mirror_deg` (the mirror's distance: no fixed value in the lab; 180° is
    the app's convention, found by compare.py)."""
    t = math.radians(tilt_deg); k1 = np.array([math.sin(t), math.cos(t), 0.0]); a = np.array([0.0, 0.0, 1.0]); b = np.cross(a, k1)
    th = math.radians(plate_deg); J = np.array([[math.cos(2 * th), math.sin(2 * th)], [math.sin(2 * th), -math.cos(2 * th)]])
    o = J @ np.array([1.0, 0.0])
    e2 = -(o[0] * a + o[1] * b) * np.exp(1j * math.radians(mirror_deg))
    I = 2 * power_mW / (math.pi * (waist_um * 1e-4) ** 2)
    return [dict(name='probe (lab)', k=k_um * k1, eps=spherical(a.astype(complex)), I=I), dict(name='reflection (lab)', k=-k_um * k1, eps=spherical(e2), I=I)]

def setup(request, field_G=None, offset_MHz=None, s_scale=1.0, depth_scale=1.0, phase_deg=0.0, field_dir=None, reflect=1.0, lab=False, magic=False, tilt_deg=0.0,
          ellipticity=0.0, ell_deg=0.0):
    """`tilt_deg`: the probe beams turned that much further toward the tweezer's axis (`tilted`).
    `ellipticity`, `ell_deg`: the tweezer's focus elliptical (`trap`), its long axis `ell_deg` from the probe's transverse
    direction in the radial plane (x̂ × the probe's radial direction) toward the probe's radial direction.
    `magic`: the dipole-force-fluctuation control -- every excited level's tweezer shift made the ground's, g(r) U_g, plus
    its own focus value less the ground's as a constant: the excited states feel the ground's trap force (no force
    difference between the internal states, no DFF), and every transition keeps its detuning at the focus everywhere (so the
    probe also no longer comes nearer its lines at the trap's edge: that and the DFF are one thing, ħ∇δ)."""
    tables = M.app_tables()
    # the tweezer as the request has it: its wavelength, power and waist (model.py's sum over states at that wavelength)
    tw_ = next(b for b in request['model']['motion']['beams'] if b['id'] == request['model']['motion']['trapBeamId'])
    lam_nm = 2 * math.pi / tw_['waveNumberPerUm'] * 1e3; w0 = tw_['waistUm']
    power_mW = tw_['intensityMWcm2'] * math.pi * (w0 * 1e-4) ** 2 / 2
    shifts = M.light_shifts(tables, wavelength_nm=lam_nm, power_mw=power_mW, waist_um=w0); p = M.parameters(tables, shifts)
    p = dict(p); p['w0'] = w0; p['zR'] = math.pi * w0 ** 2 / (lam_nm * 1e-3); p['tweezerNm'] = lam_nm; p['tweezerMW'] = power_mW
    mo = request['model']['motion']
    Bvec = np.array(mo['fieldVectorG'], float)
    B = float(np.linalg.norm(Bvec)) if field_G is None else field_G
    # the field's axis: the request's (z where it is zero), or `field_dir`
    axis = (Bvec / np.linalg.norm(Bvec) if np.linalg.norm(Bvec) > 0 else np.array([0.0, 0.0, 1.0])) if field_dir is None else np.array(field_dir, float) / np.linalg.norm(field_dir)
    Bvec = B * axis
    mt = M.multilevel_tables(shifts, field_G=B)
    # g_F μ_B (MHz/G) of F=2 and F'=3: the tables' Zeeman shift of m = 1 at 1 G (model.py, with g_I)
    m1 = M.multilevel_tables(shifts, field_G=1.0)
    gg, ge = m1['Zg'][M.GROUND.index((2, 1))], m1['Ze'][M.EXCITED.index((3, 1))]
    # the probe beams: of the beams but the tweezer, those on the frequency most of them share (S3: the probe and its
    # reflection; the repump has its own)
    trap = mo['trapBeamId']
    others = [b for b in mo['beams'] if b['id'] != trap]
    index = max((tuple(b['frequencyIndex']) for b in others), key=lambda f: sum(tuple(b['frequencyIndex']) == f for b in others))
    probes = [b for b in others if tuple(b['frequencyIndex']) == index]
    # The probe's offset from the free F=2 -> F'=3 line: the probe's frequency is the request's rotating frame on the
    # excited states plus its frequency index on the coordinates (expect.mjs `frameDetuning`; the frame is not always the
    # probe's -- with the repump's index at 0 it is the repump's); the line is the D2 centre (the app's table) with the
    # hyperfine shifts of F'=3 (+193.7407 MHz) and F=2 (+2563.005979 MHz; Steck, Rb 87 D line data, Table 7).
    if offset_MHz is None:
        line = tables['lines']['D2']['nu'] / 1e6 + 193.7407 - 2563.005979
        coords = request['model']['frequencyCoordinates']['frequenciesMHz']
        offset_MHz = max(mo['frameFrequenciesMHz']) + sum(n * f for n, f in zip(index, coords)) - line
    gi = [M.GROUND.index((2, m)) for m in range(-2, 3)]
    ei = [M.EXCITED.index((3, m)) for m in range(-3, 4)]
    tw = 2 * math.pi
    # H0: the Zeeman energy g_F μ_B B·F on each manifold (any direction of B; the basis quantized along z, the tweezer's
    # polarization) and the probe's offset on the excited states
    H0 = np.zeros((12, 12), complex)
    for (lo, F, gf) in ((0, 2, gg), (5, 3, ge)):
        Fx, Fy, Fz = spin(F); n = 2 * F + 1
        H0[lo:lo + n, lo:lo + n] = tw * gf * (Bvec[0] * Fx + Bvec[1] * Fy + Bvec[2] * Fz)
    for i in range(5, 12):
        H0[i, i] -= tw * offset_MHz
    diagL = depth_scale * np.array([tw * mt['LSg'][i] for i in gi] + [tw * mt['LSe'][i] for i in ei])
    p['U0'] *= depth_scale
    if magic:
        for i in range(5, 12):
            H0[i, i] += diagL[i] - diagL[0]
            diagL[i] = diagL[0]
    # Clebsch–Gordan C[q+1, m+2] = <2 m; 1 q | 3 m+q>
    C = np.array([[cg(2, m, 1, q, 3, m + q) if abs(m + q) <= 3 else 0.0 for m in range(-2, 3)] for q in (-1, 0, 1)])
    # the probe beams: of the beams but the tweezer, those on the frequency most of them share (S3: the probe and its
    # reflection; the repump has its own)
    trap = mo['trapBeamId']
    others = [b for b in mo['beams'] if b['id'] != trap]
    index = max((tuple(b['frequencyIndex']) for b in others), key=lambda f: sum(tuple(b['frequencyIndex']) == f for b in others))
    probes = [b for b in others if tuple(b['frequencyIndex']) == index]
    # `phase_deg`: the second probe beam's phase at the focus against the first's moved by this much (the mirror's
    # distance, which the lab does not fix to λ/2): its field times e^{iφ}
    # `reflect`: the reflection's power over the probe's (the mirror's reflectivity R): its s times R
    if lab:
        beams = lab_beams(probes[0]['waveNumberPerUm'], mirror_deg=180.0 + phase_deg)
    else:
        beams = []
        for i, b in enumerate(sorted(probes, key=lambda b: '反射' in b['name'])):
            E = np.array([complex(c['re'], c['im']) for c in b['electricField']])
            d = np.array(b['direction'], float); d /= np.linalg.norm(d)
            if i: E = E * np.exp(1j * math.radians(phase_deg))
            beams.append(dict(name=b['name'], k=d * b['waveNumberPerUm'], eps=E / np.linalg.norm(E), I=b['intensityMWcm2']))
    if tilt_deg:
        beams = tilted(beams, np.array(tw_['direction'], float) / np.linalg.norm(tw_['direction']), tilt_deg)
    kvec = [b['k'] for b in beams]; eps = [b['eps'] for b in beams]
    kr = np.array(kvec[0], float); kr[0] = 0.0; kr /= np.linalg.norm(kr); tr = np.cross([1.0, 0.0, 0.0], kr)
    ell_axis = math.cos(math.radians(ell_deg)) * tr + math.sin(math.radians(ell_deg)) * kr
    s = [s_scale * b['I'] / p['Isat'] * (reflect if i else 1.0) for i, b in enumerate(beams)]
    return dict(p=p, mt=mt, magic=magic, shifts=shifts, B=B, Bvec=Bvec, reflect=reflect, lab=lab, tilt=tilt_deg, ell=ellipticity, ellAxis=ell_axis, ellDeg=ell_deg, phase=phase_deg, offset=offset_MHz, sScale=s_scale, depthScale=depth_scale, H0=H0, diagL=diagL, C=C,
                k=np.array(kvec), eps=np.array(eps), s=np.array(s), names=[b['name'] for b in beams], T=mo['temperatureUK'])

@nb.njit(inline='always')
def trap(x, y, z, P):
    """g = I/I0 of the tweezer (along x) and its gradient: model.py's paraxial Gaussian, or with P[6] = e > 0 an elliptical
    one -- its 1/e² radii w0·√(1+e) along â = (0, P[7], P[8]) and w0/√(1+e) along x̂ × â (their product w0², so the
    depth at a given power is kept), each with its own Rayleigh range ∝ w²."""
    w0, zR, e = P[1], P[2], P[6]
    if e == 0.0:
        return M.beam(x, y, z, w0, zR)
    ay, az = P[7], P[8]
    wa2 = w0 * w0 * (1.0 + e); wb2 = w0 * w0 / (1.0 + e); zA = zR * (1.0 + e); zB = zR / (1.0 + e)
    a = ay * y + az * z; b = -az * y + ay * z
    Aa = 1.0 / (1.0 + (x / zA) ** 2); Ab = 1.0 / (1.0 + (x / zB) ** 2)
    g = math.sqrt(Aa * Ab) * math.exp(-2.0 * a * a * Aa / wa2 - 2.0 * b * b * Ab / wb2)
    gx = g * (-x * (Aa / zA ** 2 + Ab / zB ** 2) + 4.0 * x * (a * a * Aa * Aa / (wa2 * zA ** 2) + b * b * Ab * Ab / (wb2 * zB ** 2)))
    ga = -4.0 * g * a * Aa / wa2; gb = -4.0 * g * b * Ab / wb2
    return g, gx, ay * ga - az * gb, az * ga + ay * gb

def trap_params(S):
    """The P array of `hamiltonian`, `run` and friction.py: Γ, w0, z_R, U0, ħ/m, the µK unit, and the trap's ellipticity
    and long axis."""
    p = S['p']
    return np.array([p['gamma'], p['w0'], p['zR'], p['U0'], p['hbar_m'], p['uK'], S['ell'], S['ellAxis'][1], S['ellAxis'][2]])

@nb.njit(inline='always')
def hamiltonian(x, y, z, P, H0, diagL, C, k, eps, Om, H, Hb):
    """H (12×12, non-Hermitian) at r and each beam's |e><g| part Hb[b] (for the force); g and its gradient."""
    gamma = P[0]
    g, gx, gy, gz = trap(x, y, z, P)
    for i in range(12):
        for j in range(12):
            H[i, j] = H0[i, j]
    # less the first ground level's energy on every level -- a real multiple of the identity: no change to the state's
    # evolution but its overall phase, to its norm (the jumps) or to the force (Hb, diagL) -- so that the ground
    # amplitudes, most of the norm, turn slowly and RK4 does not damp them (|R(iy)|² = 1 − y⁶/72 a step: at the
    # tweezer's ground shift of ~200 rad/µs, 1e-6 a 1 ns step, 6e-5 a 2 ns step otherwise)
    c0 = H0[0, 0].real + diagL[0] * g
    for i in range(12):
        H[i, i] += diagL[i] * g - c0
    for i in range(5, 12):
        H[i, i] -= 0.5j * gamma
    nb_ = k.shape[0]
    for b in range(nb_):
        ph = k[b, 0] * x + k[b, 1] * y + k[b, 2] * z
        e = complex(math.cos(ph), math.sin(ph)) * 0.5 * Om[b]
        for i in range(12):
            for j in range(12):
                Hb[b, i, j] = 0.0
        for qi in range(3):
            q = qi - 1
            a = e * eps[b, qi]
            for mi in range(5):
                mp = mi - 2 + q
                if mp < -3 or mp > 3: continue
                c = C[qi, mi]
                if c == 0.0: continue
                ee = 5 + mp + 3
                Hb[b, ee, mi] = a * c
                H[ee, mi] += a * c
                H[mi, ee] += (a * c).conjugate()
    return g, gx, gy, gz

@nb.njit(inline='always')
def matvec(H, v, out):
    for i in range(12):
        acc = 0j
        for j in range(12):
            acc += H[i, j] * v[j]
        out[i] = -1j * acc

@nb.njit(inline='always')
def force(psi, H, Hb, k, diagL, gx, gy, gz, f):
    nrm = 0.0
    for i in range(12): nrm += psi[i].real ** 2 + psi[i].imag ** 2
    lev = 0.0
    for i in range(12): lev += (psi[i].real ** 2 + psi[i].imag ** 2) * diagL[i]
    lev /= nrm
    f[0] = -lev * gx; f[1] = -lev * gy; f[2] = -lev * gz
    nb_ = k.shape[0]
    for b in range(nb_):
        A = 0j
        for i in range(5, 12):
            for j in range(5):
                if Hb[b, i, j] != 0.0:
                    A += psi[i].conjugate() * Hb[b, i, j] * psi[j]
        im = 2.0 * A.imag / nrm
        f[0] += k[b, 0] * im; f[1] += k[b, 1] * im; f[2] += k[b, 2] * im

@nb.njit(parallel=True)
def run(n_atoms, seed, T_uK, dt, n_steps, every, P, H0, diagL, C, k, eps, Om, noise, still,
        out_v, out_r, out_alive, out_pop, out_jumps, out_fy, out_loss, out_flow, resume, state, rng):
    """`resume`: each atom starts from `state` (x, y, z, v, ψ's 12 real and 12 imaginary parts, θ, jumps, alive) and its
    random stream from `rng` -- a segment continuing the one before; at the end both hold the atoms' state (`--state`)."""
    gamma, w0, zR, U0, hbar_m, uK = P[0], P[1], P[2], P[3], P[4], P[5]
    kT = T_uK * uK; vth = math.sqrt(kT * hbar_m)
    wR = math.sqrt(4 * U0 * hbar_m / w0 ** 2); wA = math.sqrt(2 * U0 * hbar_m / zR ** 2)
    sig = np.array([1.5 * vth / wA, 1.25 * vth / wR, 1.25 * vth / wR])
    n_out = out_v.shape[0]; nb_ = k.shape[0]
    for a in nb.prange(n_atoms):
        st = np.zeros(1, np.uint64)
        z0 = np.uint64(seed) * np.uint64(0x9E3779B97F4A7C15) + np.uint64(a + 1) * np.uint64(0xBF58476D1CE4E5B9)
        z0 = (z0 ^ (z0 >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9); z0 = (z0 ^ (z0 >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        z0 = z0 ^ (z0 >> np.uint64(31)); st[0] = z0 if z0 != np.uint64(0) else np.uint64(1)
        for _ in range(4): M.rnd(st)
        if resume: st[0] = rng[a]
        H = np.zeros((12, 12), np.complex128); Hb = np.zeros((nb_, 12, 12), np.complex128)
        psi = np.zeros(12, np.complex128); k1 = np.zeros(12, np.complex128); k2 = np.zeros(12, np.complex128)
        k3 = np.zeros(12, np.complex128); k4 = np.zeros(12, np.complex128); tmp = np.zeros(12, np.complex128)
        f = np.zeros(3)
        if resume:
            x, y, z, vx, vy, vz = state[a, 0], state[a, 1], state[a, 2], state[a, 3], state[a, 4], state[a, 5]
            for i in range(12): psi[i] = complex(state[a, 6 + i], state[a, 18 + i])
            theta = state[a, 30]; jumps = int(state[a, 31]); alive = state[a, 32] > 0.5; k_out = 0
        else:
            while True:
                x = sig[0] * M.gauss(st); y = sig[1] * M.gauss(st); z = sig[2] * M.gauss(st)
                g, _, _, _ = trap(x, y, z, P)
                if g < 0.05: continue
                logw = -U0 * (1 - g) / kT + 0.5 * ((x / sig[0]) ** 2 + (y / sig[1]) ** 2 + (z / sig[2]) ** 2)
                if math.log(max(M.rnd(st), 1e-300)) < logw: break
            vx = vth * M.gauss(st); vy = vth * M.gauss(st); vz = vth * M.gauss(st)
            if still: vx = 0.0; vy = 0.0; vz = 0.0; x = 0.0; y = 0.0; z = 0.0
            psi[4] = 1.0
            theta = M.rnd(st)
            while theta <= 0.0: theta = M.rnd(st)
            alive = True; k_out = 0; jumps = 0
        g, gx, gy, gz = hamiltonian(x, y, z, P, H0, diagL, C, k, eps, Om, H, Hb)
        force(psi, H, Hb, k, diagL, gx, gy, gz, f)
        fy_acc = 0.0; fy_n = 0
        for step in range(n_steps + 1):
            if step % every == 0 and k_out < n_out:
                out_v[k_out, a, 0] = vx; out_v[k_out, a, 1] = vy; out_v[k_out, a, 2] = vz
                out_r[k_out, a, 0] = x; out_r[k_out, a, 1] = y; out_r[k_out, a, 2] = z
                out_alive[k_out, a] = alive; out_jumps[k_out, a] = jumps
                out_fy[k_out, a] = fy_acc / max(fy_n, 1); fy_acc = 0.0; fy_n = 0
                nrm = 0.0
                for i in range(12): nrm += psi[i].real ** 2 + psi[i].imag ** 2
                for i in range(5): out_pop[k_out, a, i] = (psi[i].real ** 2 + psi[i].imag ** 2) / nrm
                pe = 0.0
                for i in range(5, 12): pe += (psi[i].real ** 2 + psi[i].imag ** 2) / nrm
                out_pop[k_out, a, 5] = pe
                k_out += 1
            if step == n_steps or not alive: continue
            # velocity Verlet: half kick, drift, the internal state over dt at the midpoint, half kick
            if not still:
                vx += 0.5 * dt * hbar_m * f[0]; vy += 0.5 * dt * hbar_m * f[1]; vz += 0.5 * dt * hbar_m * f[2]
                x += 0.5 * dt * vx; y += 0.5 * dt * vy; z += 0.5 * dt * vz
            hamiltonian(x, y, z, P, H0, diagL, C, k, eps, Om, H, Hb)
            matvec(H, psi, k1)
            for i in range(12): tmp[i] = psi[i] + 0.5 * dt * k1[i]
            matvec(H, tmp, k2)
            for i in range(12): tmp[i] = psi[i] + 0.5 * dt * k2[i]
            matvec(H, tmp, k3)
            for i in range(12): tmp[i] = psi[i] + dt * k3[i]
            matvec(H, tmp, k4)
            for i in range(12): psi[i] += dt / 6.0 * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i])
            if not still:
                x += 0.5 * dt * vx; y += 0.5 * dt * vy; z += 0.5 * dt * vz
            g, gx, gy, gz = hamiltonian(x, y, z, P, H0, diagL, C, k, eps, Om, H, Hb)
            nrm = 0.0
            for i in range(12): nrm += psi[i].real ** 2 + psi[i].imag ** 2
            v0x, v0y, v0z = vx, vy, vz
            if nrm < theta:
                # the jump: its channel q by ||L_q psi||^2, the absorbed beam's noise, the emitted photon's recoil
                w = np.zeros(3)
                for qi in range(3):
                    for mi in range(5):
                        mp = mi - 2 + qi - 1
                        if mp < -3 or mp > 3: continue
                        amp = C[qi, mi] * psi[5 + mp + 3]
                        w[qi] += amp.real ** 2 + amp.imag ** 2
                if noise:
                    pb = np.zeros(nb_); tot = 0.0
                    for b in range(nb_):
                        A = 0j
                        for i in range(5, 12):
                            for j in range(5):
                                if Hb[b, i, j] != 0.0: A += psi[i].conjugate() * Hb[b, i, j] * psi[j]
                        pb[b] = max(2.0 * A.imag, 0.0); tot += pb[b]
                    if tot > 0.0:
                        u = M.rnd(st) * tot; pick = nb_ - 1; acc = 0.0
                        for b in range(nb_):
                            acc += pb[b]
                            if u < acc:
                                pick = b; break
                        mx = 0.0; my = 0.0; mz = 0.0
                        for b in range(nb_):
                            mx += pb[b] / tot * k[b, 0]; my += pb[b] / tot * k[b, 1]; mz += pb[b] / tot * k[b, 2]
                        if not still:
                            vx += hbar_m * (k[pick, 0] - mx); vy += hbar_m * (k[pick, 1] - my); vz += hbar_m * (k[pick, 2] - mz)
                W = w[0] + w[1] + w[2]
                u = M.rnd(st) * W; qi = 2
                if u < w[0]: qi = 0
                elif u < w[0] + w[1]: qi = 1
                for i in range(12): tmp[i] = 0.0
                for mi in range(5):
                    mp = mi - 2 + qi - 1
                    if mp < -3 or mp > 3: continue
                    tmp[mi] = C[qi, mi] * psi[5 + mp + 3]
                nrm = 0.0
                for i in range(12): nrm += tmp[i].real ** 2 + tmp[i].imag ** 2
                for i in range(12): psi[i] = tmp[i] / math.sqrt(nrm)
                if qi == 1: ex, ey, ez = M.emission_pi(st)
                else: ex, ey, ez = M.emission(st)
                kk = math.sqrt(k[0, 0] ** 2 + k[0, 1] ** 2 + k[0, 2] ** 2) * hbar_m
                if not still:
                    vx += kk * ex; vy += kk * ey; vz += kk * ez
                theta = M.rnd(st)
                while theta <= 0.0: theta = M.rnd(st)
                jumps += 1
                if alive:
                    out_flow[a, 3] += 0.5 * (vx * vx - v0x * v0x) / hbar_m; out_flow[a, 4] += 0.5 * (vy * vy - v0y * v0y) / hbar_m
                    out_flow[a, 5] += 0.5 * (vz * vz - v0z * v0z) / hbar_m; out_flow[a, 11] += 1.0
            force(psi, H, Hb, k, diagL, gx, gy, gz, f)
            fy_acc += f[1]; fy_n += 1
            if alive and not still:
                # the light force: all of it but the ground trap's, U0 ∇g; the trap's coupling: its force less the cuts'
                lx, ly, lz = f[0] - U0 * gx, f[1] - U0 * gy, f[2] - U0 * gz
                _, cx, _, _ = trap(x, 0.0, 0.0, P); _, _, cy, _ = trap(0.0, y, 0.0, P); _, _, _, cz = trap(0.0, 0.0, z, P)
                out_flow[a, 0] += dt * vx * lx; out_flow[a, 1] += dt * vy * ly; out_flow[a, 2] += dt * vz * lz
                out_flow[a, 6] += dt * vx * U0 * (gx - cx); out_flow[a, 7] += dt * vy * U0 * (gy - cy); out_flow[a, 8] += dt * vz * U0 * (gz - cz)
                out_flow[a, 9] += dt; out_flow[a, 10] += dt * ly
            if not still:
                vx += 0.5 * dt * hbar_m * f[0]; vy += 0.5 * dt * hbar_m * f[1]; vz += 0.5 * dt * hbar_m * f[2]
                E = 0.5 * (vx * vx + vy * vy + vz * vz) / hbar_m - U0 * g
                if g < 0.05 and E > 0:
                    alive = False
                    out_loss[a, 0] = (step + 1) * dt; out_loss[a, 1] = x; out_loss[a, 2] = y; out_loss[a, 3] = z
                    out_loss[a, 4] = vx; out_loss[a, 5] = vy; out_loss[a, 6] = vz
        state[a, 0] = x; state[a, 1] = y; state[a, 2] = z; state[a, 3] = vx; state[a, 4] = vy; state[a, 5] = vz
        for i in range(12):
            state[a, 6 + i] = psi[i].real; state[a, 18 + i] = psi[i].imag
        state[a, 30] = theta; state[a, 31] = jumps; state[a, 32] = 1.0 if alive else 0.0; rng[a] = st[0]
    return 0

def flow_summary(F, p):
    """The held atoms' energy flow per lab axis (µK/µs, over every atom's held time together): the light force's work,
    the kicks' heat, the trap's coupling from the other axes; their sum; the photons a µs; the mean light force along y
    (µm/µs²) and over the trap's largest radial restoring acceleration."""
    held = F[:, 9].sum()
    if held <= 0: return None
    rate = lambda c: float(F[:, c].sum() / held / p['uK'])
    a_max = p['U0'] * p['hbar_m'] * 2 / p['w0'] * math.exp(-0.5)
    light = [rate(c) for c in (0, 1, 2)]; kick = [rate(c) for c in (3, 4, 5)]; coupling = [rate(c) for c in (6, 7, 8)]
    push = float(F[:, 10].sum() / held * p['hbar_m'])
    return dict(heldUs=float(held), light=light, kick=kick, coupling=coupling, net=[light[i] + kick[i] + coupling[i] for i in range(3)],
                photonsPerUs=float(F[:, 11].sum() / held), pushY=push, pushOverAmax=push / a_max, aMax=a_max)

def simulate(S, atoms, end_us, every_us, dt, noise, seed=12345, still=False, state_path=None):
    """`state_path`: an .npz of the atoms' state -- read if it exists (this run continues it from its time to `end_us`, the
    held atoms' flows and the losses carried on), written at the end -- so that a long run is a chain of segments, each
    its own process."""
    p = S['p']
    resume = state_path is not None and os.path.exists(state_path)
    if resume:
        saved = np.load(state_path)
        state, rng, out_loss, out_flow, t0 = saved['state'].copy(), saved['rng'].copy(), saved['loss'].copy(), saved['flow'].copy(), float(saved['t'])
        atoms = state.shape[0]
    else:
        state, rng, out_loss, out_flow, t0 = np.zeros((atoms, 33)), np.zeros(atoms, np.uint64), np.full((atoms, 7), np.nan), np.zeros((atoms, 12)), 0.0
    n_steps = int(round((end_us - t0) / dt)); every = int(round(every_us / dt)); n_out = n_steps // every + 1
    out_v = np.zeros((n_out, atoms, 3)); out_r = np.zeros((n_out, atoms, 3)); out_alive = np.zeros((n_out, atoms), np.bool_)
    out_pop = np.zeros((n_out, atoms, 6)); out_jumps = np.zeros((n_out, atoms), np.int64); out_fy = np.zeros((n_out, atoms))
    P = trap_params(S)
    Om = p['gamma'] * np.sqrt(S['s'] / 2)
    t_start = time.time()
    run(atoms, seed, S['T'], dt, n_steps, every, P, S['H0'], S['diagL'], S['C'], S['k'], S['eps'].astype(np.complex128), Om, noise, still,
        out_v, out_r, out_alive, out_pop, out_jumps, out_fy, out_loss, out_flow, resume, state, rng)
    wall = time.time() - t_start
    if state_path is not None:
        np.savez(state_path, state=state, rng=rng, loss=out_loss, flow=out_flow, t=t0 + n_steps * dt)
    times = t0 + np.arange(n_out) * every * dt
    r = M.summarize('S3C', 'coherent F=2 -> F\'=3 quantum jumps', p, atoms, dt, wall, times, out_v, out_alive)
    pops = np.zeros((n_out, 6)); rate = np.zeros(n_out); fy = np.zeros(n_out); ry = np.zeros(n_out)
    for kk in range(n_out):
        h = out_alive[kk]
        if h.sum():
            pops[kk] = out_pop[kk][h].mean(0); fy[kk] = out_fy[kk][h].mean() * p['hbar_m']; ry[kk] = out_r[kk][h][:, 1].mean()
        if kk:
            hp = out_alive[kk - 1] & out_alive[kk]
            if hp.sum(): rate[kk] = (out_jumps[kk][hp] - out_jumps[kk - 1][hp]).mean() / (every * dt)
    r.update(groundPopulationsF2=pops[:, :5].round(4).tolist(), excited=pops[:, 5].round(4).tolist(), jumpRatePerUs=rate.round(3).tolist(),
             meanForceY_um_per_us2=fy.tolist(), meanY_um=ry.tolist(), noise=noise, fieldG=S['B'], beams=S['names'], s=S['s'].tolist(),
             lostAt=[[round(float(v), 4) for v in row] for row in out_loss if np.isfinite(row[0])],
             flow=flow_summary(out_flow, p),
             offsetMHz=S['offset'], depthScale=S['depthScale'], phaseDeg=S['phase'], still=still, fieldVectorG=S['Bvec'].tolist(), reflect=S['reflect'], lab=S['lab'],
             magic=S['magic'], tiltDeg=S['tilt'], ellipticity=S['ell'], ellAngleDeg=S['ellDeg'], ellAxis=S['ellAxis'].tolist(), tweezerNm=p['tweezerNm'], tweezerMW=p['tweezerMW'], depthUK=p['U0'] / p['uK'], excitedShiftsMHz=(S['diagL'][5:] / (2 * math.pi)).tolist())
    return r

def selftest(S):
    """One running wave along y driving σ+ on |2,2> -> |3',3'> (the cycling line: closed), no tweezer, the atom held at
    the origin at rest, s = 2: the mean force (its y part, ħk along the beam) and the jump rate against the two-level
    steady state at the line's Zeeman detuning, ħkΓρ_ee and Γρ_ee, ρ_ee = ½s/(1 + s + (2δ/Γ)²)."""
    p = S['p']
    T = dict(S); T['diagL'] = np.zeros(12); kp = float(np.linalg.norm(S['k'][0]))
    T['k'] = np.array([[0.0, kp, 0.0]]); T['eps'] = np.array([[0, 0, 1]], complex); T['s'] = np.array([2.0])
    det = -(S['H0'][11, 11].real - S['H0'][4, 4].real)      # δ = ω_L − ω_eg of |2,2> -> |3',3'> (B along z)
    r = simulate(T, 64, 60.0, 5.0, 0.001, 0, still=True)
    G = p['gamma']; s = 2.0; ree = 0.5 * s / (1 + s + 4 * det * det / G ** 2)
    return dict(detuningGamma=det / G, jumpRate=float(np.mean(r['jumpRatePerUs'][2:])), jumpRateFormula=G * ree,
                force=float(np.mean(r['meanForceY_um_per_us2'][2:])), forceFormula=p['hbar_m'] * kp * G * ree)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--request', required=True); ap.add_argument('--atoms', type=int, default=400)
    ap.add_argument('--end', type=float, default=300.0); ap.add_argument('--every', type=float, default=5.0)
    ap.add_argument('--dt', type=float, default=0.001); ap.add_argument('--noise', type=int, default=1)
    ap.add_argument('--field', type=float, default=None); ap.add_argument('--seed', type=int, default=12345)
    ap.add_argument('--out', default=None); ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--offset', type=float, default=None, help="the probe's offset from the free F=2 -> F'=3 line (MHz); default: the request's")
    ap.add_argument('--s-scale', type=float, default=1.0, help="the probe beams' intensity times this")
    ap.add_argument('--depth-scale', type=float, default=1.0, help="the tweezer's intensity (its shifts and depth) times this")
    ap.add_argument('--phase', type=float, default=0.0, help="the second probe beam's phase at the focus moved by this (degrees)")
    ap.add_argument('--still', action='store_true', help='every atom held at rest at the focus: the light force there')
    ap.add_argument('--field-dir', default=None, help="the field along this axis (x, y or z) instead of the request's")
    ap.add_argument('--reflect', type=float, default=1.0, help="the reflection's power over the probe's (R)")
    ap.add_argument('--lab', action='store_true', help="the probe beams from the lab's description (lab_beams), not the request's")
    ap.add_argument('--magic', action='store_true', help='the excited levels feel the ground trap (no dipole-force fluctuation), their focus detunings kept')
    ap.add_argument('--state', default=None, help="an .npz of the atoms' state: continued from if it exists, written at the end (a segment of a long run)")
    ap.add_argument('--tilt', type=float, default=0.0, help="the probe beams turned this much further toward the tweezer's axis (degrees)")
    ap.add_argument('--ellipticity', type=float, default=0.0, help="the tweezer's focus: its long 1/e² radius over the short one, less 1 (focus.py)")
    ap.add_argument('--ell-angle', type=float, default=0.0, help="the long axis's angle from the probe's transverse radial direction (degrees)")
    a = ap.parse_args()
    req = json.load(open(a.request, encoding='utf-8'))
    axes = dict(x=(1, 0, 0), y=(0, 1, 0), z=(0, 0, 1))
    S = setup(req, a.field, a.offset, a.s_scale, a.depth_scale, a.phase, axes[a.field_dir] if a.field_dir else None, a.reflect, a.lab, a.magic, a.tilt, a.ellipticity, a.ell_angle)
    p = S['p']
    print(f"probe offset {S['offset']:.3f} MHz from the free F=2 -> F'=3 line; field {np.round(S['Bvec'], 4).tolist()} G; R {S['reflect']}; beams {S['names']}: s {np.round(S['s'], 3)}, k {np.round(S['k'], 4).tolist()}; eps {np.round(S['eps'], 4).tolist()}")
    print(f"U0 {p['U0']/p['uK']:.1f} uK, Gamma {p['gamma']:.3f}/us, Isat {p['Isat']:.4f} mW/cm2; detunings at the focus (MHz) |2,m> -> |3',m+q>:")
    for q in (-1, 0, 1):
        d = S['H0'].diagonal().real
        print('  q', q, [round(float((-(d[5 + m + q + 3] + S['diagL'][5 + m + q + 3]) + d[m + 2] + S['diagL'][m + 2]) / (2 * math.pi)), 2) if abs(m + q) <= 3 else None for m in range(-2, 3)])
    if a.selftest:
        print('selftest', selftest(S)); return
    r = simulate(S, a.atoms, a.end, a.every, a.dt, a.noise, a.seed, still=a.still, state_path=a.state)
    n = len(r['timesUs'])
    print(f"wall {r['wallS']:.1f} s")
    for kk in range(0, n, max(1, n // 12)):
        print(f"t {r['timesUs'][kk]:7.1f} us L {r['loss'][kk]:.3f} T {r['Ttotal'][kk]} ({'/'.join(str(r['T'][i][kk]) for i in range(3))}) "
              f"rate {r['jumpRatePerUs'][kk]:.2f}/us Fy {r['meanForceY_um_per_us2'][kk]:.4f} <y> {r['meanY_um'][kk]:.3f} pops {r['groundPopulationsF2'][kk]} ee {r['excited'][kk]}")
    if a.out: json.dump(r, open(a.out, 'w'))

if __name__ == '__main__':
    main()
