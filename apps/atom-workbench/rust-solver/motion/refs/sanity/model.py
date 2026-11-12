"""Independent physics model of the two sanity scenes (refs/sanity.mjs): S1, the app's default tweezer + one σ+ probe;
S2, the same + the probe's retro-reflection (the app's plain mirror) + a repump. It shares no code and no approximation
with the motion engine: the light shifts come from a sum over states on the app's own atomic tables (read from
src/physics.ts and src/spectroscopy-data.ts), the dynamics from a classical-atom Monte Carlo written here.

    python model.py [--atoms N] [--end µs] [--every µs] [--models S1K,S1F,S2K,S2F] [--out model.json]

(the science env's python with numba; a few seconds to a minute on all cores). Writes the expectations the check reads
(sanity/model.json); refs/sanity.mjs never runs python.

# The scenes (as the app builds them; refs/sanity.mjs prints and checks this)
Tweezer: 852 nm, 4 mW, waist w0 = 1.2 µm, along +x, linear along z = π on the bias field (3 G along +z).
Probe (the app's ＋ laser, newLaser): 50 µW in a 1 mm waist, I = 2P/(πw²) = 3.1831 mW/cm², σ+ along the field
(the app's σ+ preset), at the free-space F=2 → F'=3 frequency. Initial |F=2, m=2⟩, 30 µK.
S2: + the reflection (reflection.ts default mirror: R = 1, φ = 0, no λ/4): k = −z, and on the quantization axis it is
again σ+ (a plain mirror keeps the lab field; it drives Δm = +1 about z): the field at the atom is
E = E1 ê₊ (e^{ikz} + e^{−ikz}) = 2E1 ê₊ cos kz -- a pure-σ+ standing wave (a "σ+σ+" pair on the field's axis, no
polarization gradient), antinode at the trap centre (φ = 0). + repump: ＋ laser on F=1 → F'=2, along +x, π.
|2,2⟩ ↔ |3',3'⟩ is closed for pure σ+ light (F'=3 cannot decay to F=1; σ+ from |2,2⟩ reaches only m' = 3, which only
F'=3 has; the π tweezer conserves m, and |2,2⟩, |3',3'⟩ are stretched states, unmixed by hyperfine structure), so both
scenes are two-level problems and the repump never acts. Its only effect is off-resonant scattering from |2,2⟩
(6.6–6.8 GHz away): ≈ 2/s, < 1 % of atoms in 10 ms, and they are pumped straight back. Neglected.

# Light shifts (part A)
Sum over states, rotating and counter-rotating terms, for |J m_J⟩ in light of polarization q (π: q = 0):
  α_{m,q}(ω) = Σ_n Σ_{m_n} (J 1 J_n; −m q m_n)² |⟨J‖D‖J_n⟩|² · 2ΔE_n/(ΔE_n² − (ħω)²)  (+ the Rb⁺ core, scalar),
  ΔE = −α I/(2ε0 c), I0 = 2P/(πw0²).
5S–5P1/2 and 5S–5P3/2 reduced elements from the app's Γ and line frequencies (physics.ts, Steck):
|⟨J‖D‖J'⟩|² = 3πε0ħc³(2J'+1)Γ/ω³; every other line (5S → 6–9P; 5P3/2 → 6–10S, 4–9D) and the core from
spectroscopy-data.ts (Arora & Sahoo, PRA 86, 033416 (2012)). |2,2⟩ = |m_J=1/2⟩|m_I=3/2⟩, |3',3'⟩ = |3/2⟩|3/2⟩.
For |3/2, 3/2⟩ in π light the 5S term vanishes ((3/2 1 1/2; −3/2 0 m) = 0): the excited state's shift is the upper
states' (4D mostly) -- small and positive (anti-trapping): κ = δE_e/|δE_g| ≈ 0.11.
Approximations: hyperfine structure in the 852 nm detunings (relative 2e-4); the tables' own uncertainty (~1 %).
The app's level chart (all-order Floquet, spectroscopy.ts) gives −17.2483 MHz and +1.8274 MHz at the focus; this sum
reproduces both (printed). The local detuning of the probe from |2,2⟩ → |3',3'⟩:
  δ(r) = δ_Z − δ_LS g(r),  δ_Z/2π = −4.1987 MHz (3 G Zeeman: (g'·3 − g·2)μ_B B), δ_LS = (δE_e − δE_g)/ħ,
  g(r) = I_tw(r)/I0 = exp(−2ρ²/w(x)²)/(1 + x²/z_R²) (paraxial Gaussian, ρ² = y² + z², z_R = πw0²/λ).
The light shifts are linear in I to (δ_LS/Δ_852)² ~ 1e-12.

# Dynamics (part B): a classical atom in the tweezer, internal state at its local steady state
The internal relaxation time 1/Γ = 26 ns is ≪ the trap periods (13 µs radial, 84 µs axial) and kv ≪ Γ (kv/Γ = 0.01 at
30 µK, 0.04 at 400 µK): the internal state follows the motion; the semiclassical (Gordon–Ashkin) description applies
(recoil E_r = h·3.8 kHz ≪ ħΓ). Two independent implementations:

K (kicks; the brief's model): scattering events at the two-level steady-state rate at the local light-shifted and
Doppler-shifted detuning, R = (Γ/2)s/(1 + s + (2δ_eff/Γ)²), δ_eff = δ(r) − k·v (exact for the optical Bloch equations'
steady state; no expansion in s or 1/δ). Each event: absorption kick ħk along the beam, emission kick ħk in the σ+
dipole pattern (1 + cos²θ about z: ⟨k̂_z²⟩ = 2/5, ⟨k̂_x²⟩ = ⟨k̂_y²⟩ = 3/10). Events Poissonian (the two-level atom's
sub-Poissonian absorption -- Mandel Q < 0 -- is left out: an overestimate of the absorption noise by |Q|, which F
includes). Mean force: −∇U_g − ρ_ee ∇(U_e − U_g): the atom spends ρ_ee = R/Γ of its time in |3',3'⟩, whose potential
U_e = +κ U0 g is a hill (the trap is weakened by ρ_ee(1 + κ) -- up to ~30 % near resonance); radiation pressure is
the mean of the kicks; Doppler cooling is R's dependence on v.
S2 in K: the two beams as independent running waves (R_± at δ ∓ kv_z, saturated by s_tot = 2s, kicks ±ħk): the
textbook Doppler-molasses model, which drops the standing wave's interference (its dipole lattice, the localization of
friction and absorption noise at the nodes and of scattering at the antinodes).
F (Fokker–Planck, Gordon & Ashkin, PRA 21, 1606 (1980)): the two-level optical Bloch equations (RWA; H = −ħδ(r)|e⟩⟨e|
+ ħ/2 (Ω(r)|e⟩⟨g| + h.c.), Ω = Ω1 e^{ikz} for S1, 2Ω1 cos kz for S2), solved numerically at every step: the force
operator f = −∇H; the mean force ⟨f⟩ at ρ0 + ρ1, ρ0 the local steady state and ρ1 = L⁻¹(v·∇ρ0) its first order in v
(Doppler/non-adiabatic friction, the standing wave's sin²kz friction); the momentum diffusion of the induced dipole and
of the excited fraction's light-shift force by the quantum regression theorem, D_ij = Re∫⟨δf_i(τ)δf_j(0)⟩dτ
(symmetrized; includes Mandel Q and the fluctuating excited-state potential); spontaneous emission as Poisson kicks in
the dipole pattern at Γρ_ee. Valid for kv ≪ Γ (first order in v). F is the model for S2 (it keeps the interference);
for S1 it cross-checks K (they differ by Q, the light-shift force noise and K's exact Doppler nonlinearity).

Initial state: Boltzmann positions in U_g at 30 µK (exact rejection), Maxwell velocities (the Wigner factor
coth(ħω/2k_BT)·ħω/2k_BT − 1 < 2e-3 is left out). Loss: the app's definition (ui-design.md §6.20, engine SPEC §1.7):
the atom is first outside the region where the tweezer's intensity exceeds 1/20 of its focus value, with energy above
the value far away (E = mv²/2 + U_g > 0). T_a = m Var(v_a)/k_B over the atoms still held (the app's T); total = mean of
the three axes. Velocity Verlet (dt below), stochastic kicks after each step. No gravity (2e-4 of the trap's force).

# What this gives (printed by --summary; the numbers are in model.json)
Centre: U0/k_B = 828 µK, ω/2π = 74.5 kHz radial, 11.9 kHz axial; δ(0) = −23.27 MHz = −3.84Γ at the focus,
−4.20 MHz = −0.69Γ at the trap's edge: the probe is red-detuned everywhere, so its z motion is Doppler-COOLED (one
red beam on a trapped atom cools along it, as for a trapped ion) to T_D(δ) = (7/5)(ħΓ/16)(1+s+(2δ/Γ)²)/(|δ|/Γ)
≈ 410 µK at the focus (half the depth), less at the edge; x and y have no cooling beam: spontaneous emission heats
each by (3/10)E_r per photon. S1 has the radiation pressure ħkR(r) along +z, which grows as the atom leaves the
light shift and lowers the barrier on +z. S2 has no net push, twice the photons (four times at the antinode).
"""
import argparse, json, math, os, re, sys, time
import numpy as np
import numba as nb

HERE = os.path.dirname(os.path.abspath(__file__))
APP_SRC = os.path.normpath(os.path.join(HERE, '..', '..', '..', '..', 'src'))

# CODATA 2018 exact/recommended constants.
H = 6.62607015e-34; HBAR = H / (2 * math.pi); C = 299792458.0; EPS0 = 8.8541878128e-12; KB = 1.380649e-23
E_CH = 1.602176634e-19; A0 = 5.29177210903e-11; EH = 4.3597447222071e-18; AU_POL = 4 * math.pi * EPS0 * A0 ** 3
EV = E_CH  # J per eV

# ---------------------------------------------------------------------------------------------------------------------
# Part A: the app's atomic tables and the light shifts.

def app_tables():
    """RB87 (physics.ts) and the virtual manifolds and core polarizabilities (spectroscopy-data.ts), read from the app."""
    phys = open(os.path.join(APP_SRC, 'physics.ts'), encoding='utf8').read()
    data = open(os.path.join(APP_SRC, 'spectroscopy-data.ts'), encoding='utf8').read()
    mass = re.search(r'massKg:\s*([\d.eE+-]+)\*([\d.eE+-]+)', phys)
    lines = {}
    for name in ('D1', 'D2'):
        m = re.search(name + r':\s*\{\s*J:\s*([\d.]+).*?frequencyHz:\s*([\d.eE+-]+),\s*gammaHz:\s*([\d.eE+-]+)', phys)
        lines[name] = dict(J=float(m.group(1)), nu=float(m.group(2)), gamma=float(m.group(3)))
    virtual = {}
    for name in ('ground', 'D1', 'D2'):
        block = re.search(name + r':\[(.*?)\]\.map', data, re.S).group(1)
        virtual[name] = [dict(name=r[0], J=float(r[1]), E=float(r[3]), D=float(r[4]))
                         for r in re.findall(r"\['([^']+)',([.\d]+),(\d),([.\d]+),([.\d]+)\]", block)]
    core = re.search(r'CORE_POLARIZABILITY_AU[^=]*=\{ground:([\d.+-]+),D1:([\d.]+),D2:([\d.]+)\}', data)
    return dict(mass=float(mass.group(1)) * float(mass.group(2)), lines=lines, virtual=virtual,
                core=dict(ground=eval(core.group(1)), D1=float(core.group(2)), D2=float(core.group(3))))

def wigner3j(j1, j2, j3, m1, m2, m3):
    """Racah's formula (half-integers allowed)."""
    if abs(m1 + m2 + m3) > 1e-9 or abs(m1) > j1 or abs(m2) > j2 or abs(m3) > j3: return 0.0
    if j3 < abs(j1 - j2) - 1e-9 or j3 > j1 + j2 + 1e-9: return 0.0
    f = lambda x: math.factorial(int(round(x)))
    t = math.sqrt(f(j1 + j2 - j3) * f(j1 - j2 + j3) * f(-j1 + j2 + j3) / f(j1 + j2 + j3 + 1))
    t *= math.sqrt(f(j1 + m1) * f(j1 - m1) * f(j2 + m2) * f(j2 - m2) * f(j3 + m3) * f(j3 - m3))
    s = 0.0
    for k in range(0, 100):
        a = [k, j1 + j2 - j3 - k, j1 - m1 - k, j2 + m2 - k, j3 - j2 + m1 + k, j3 - j1 - m2 + k]
        if min(a) < -1e-9: continue
        if k > j1 + j2 + j3 + 1: break
        s += (-1) ** k / np.prod([f(x) for x in a])
    return (-1) ** int(round(j1 - j2 - m3)) * t * s

def light_shifts(tables, wavelength_nm=852.0, power_mw=4.0, waist_um=1.2):
    """α of |5S1/2, 1/2⟩ and |5P3/2, 3/2⟩ in π light at the tweezer's wavelength (a.u.), the focus shifts (MHz)."""
    hw = H * C / (wavelength_nm * 1e-9) / EH   # photon energy, Hartree
    L = tables['lines']
    def reduced(line, Jp):  # |<J||D||J'>|^2 in (e a0)^2 from Γ and ω
        w = 2 * math.pi * L[line]['nu']
        d2 = 3 * math.pi * EPS0 * HBAR * C ** 3 * (2 * Jp + 1) * L[line]['gamma'] / w ** 3
        return d2 / (E_CH * A0) ** 2
    E_D1, E_D2 = H * L['D1']['nu'] / EH, H * L['D2']['nu'] / EH
    def alpha(J, m, partners, core):
        a = core; parts = {}
        for (name, Jn, dE, D2) in partners:
            ang = sum(wigner3j(J, 1, Jn, -m, 0, mn) ** 2 for mn in np.arange(-Jn, Jn + 1))
            term = ang * D2 * 2 * dE / (dE ** 2 - hw ** 2)
            parts[name] = term; a += term
        return a, parts
    g_partners = [('5P1/2', .5, E_D1, reduced('D1', .5)), ('5P3/2', 1.5, E_D2, reduced('D2', 1.5))]
    g_partners += [(v['name'], v['J'], v['E'] * EV / EH, v['D'] ** 2) for v in tables['virtual']['ground']]
    e_partners = [('5S1/2', .5, -E_D2, reduced('D2', 1.5) * 4 / 2)]  # |<3/2||D||1/2>|² = |<1/2||D||3/2>|²
    e_partners += [(v['name'], v['J'], v['E'] * EV / EH - E_D2, v['D'] ** 2) for v in tables['virtual']['D2']]
    ag, gparts = alpha(.5, .5, g_partners, tables['core']['ground'])
    ae, eparts = alpha(1.5, 1.5, e_partners, tables['core']['D2'])
    # scalar and tensor of 5P3/2 (α(m=J, π) = α0 + α2)
    a0e = np.mean([sum(alpha(1.5, m, e_partners, 0)[0] for m in (1.5, .5, -.5, -1.5)) / 4])
    I0 = 2 * power_mw * 1e-3 / (math.pi * (waist_um * 1e-6) ** 2)
    shift = lambda a: -a * AU_POL * I0 / (2 * EPS0 * C) / H / 1e6   # MHz
    return dict(I0Wm2=I0, alphaGroundAU=ag, alphaExcitedAU=ae, alphaExcitedScalarAU=a0e + tables['core']['D2'],
                alphaExcitedTensorAU=ae - (a0e + tables['core']['D2']),
                groundShiftMHz=shift(ag), excitedShiftMHz=shift(ae),
                excitedParts={k: v for k, v in eparts.items() if abs(v) > 1e-3}, groundParts=gparts,
                reducedD1=math.sqrt(reduced('D1', .5)), reducedD2=math.sqrt(reduced('D2', 1.5)))

# The app's level chart at the focus (spectroscopy.ts, all-order Floquet; refs/sanity.mjs re-reads it): for comparison.
CHART = dict(groundShiftMHz=-17.2483, excitedShiftMHz=1.8274, zeemanDetuningMHz=-4.1987)

def parameters(tables, shifts):
    """The model's numbers, in µm, µs, rad/µs (energies as E/ħ)."""
    m = tables['mass']; L = tables['lines']
    gamma = L['D2']['gamma'] * 1e-6                     # 1/µs
    lam_p = C / L['D2']['nu']                            # probe wavelength (m), F=2→F'=3 differs by 1e-9
    kp = 2 * math.pi / lam_p * 1e-6                      # rad/µm
    lam_t, w0 = 852e-9, 1.2
    zR = math.pi * w0 ** 2 / (lam_t * 1e6)               # µm
    U0 = -2 * math.pi * shifts['groundShiftMHz']         # rad/µs
    dLS = 2 * math.pi * (shifts['excitedShiftMHz'] - shifts['groundShiftMHz'])
    # The Zeeman detuning of the probe from |2,2> -> |3',3'> at 3 G: (g_F'·m' − g_F·m) μ_B B with g_F = g_J/2 (F=2, I=3/2)
    # and g_F' = g_J'·(3·4+3/4−15/4)/(2·12) ... = g_J'/2·... -- here from the app's own levels (sanity.mjs checks it):
    dZ = 2 * math.pi * CHART['zeemanDetuningMHz']
    hbar_m = HBAR / m * 1e12 / 1e6                        # µm²/µs
    # saturation intensity of the cycling transition: I_sat = cε0Γ²ħ²/(4|d|²), |d|² = |<1/2||D||3/2>|²/4 (stretched)
    d = shifts['reducedD2'] * E_CH * A0 / 2
    Isat = C * EPS0 * (gamma * 1e6) ** 2 * HBAR ** 2 / (4 * d ** 2) * 0.1   # mW/cm²
    Iprobe = 2 * 0.05 / (math.pi * 0.1 ** 2)              # mW/cm² (50 µW, 1 mm)
    s1 = Iprobe / Isat
    return dict(mass=m, gamma=gamma, kp=kp, w0=w0, zR=zR, U0=U0, dLS=dLS, dZ=dZ, hbar_m=hbar_m, Isat=Isat, Iprobe=Iprobe,
                s1=s1, omegaR=math.sqrt(4 * U0 * hbar_m / w0 ** 2), omegaAx=math.sqrt(2 * U0 * hbar_m / zR ** 2),
                uK=KB / HBAR * 1e-6 * 1e-6)               # rad/µs per µK

# ---------------------------------------------------------------------------------------------------------------------
# Part B: the Monte Carlo (numba). State per atom: r (µm), v (µm/µs).

@nb.njit(inline='always')
def rnd(state):
    x = state[0]
    x ^= x >> np.uint64(12); x ^= x << np.uint64(25); x ^= x >> np.uint64(27)
    state[0] = x
    return float((x * np.uint64(2685821657736338717)) >> np.uint64(11)) * (1.0 / 9007199254740992.0)

@nb.njit(inline='always')
def gauss(state):
    u1 = rnd(state)
    while u1 <= 0.0: u1 = rnd(state)
    return math.sqrt(-2.0 * math.log(u1)) * math.cos(2 * math.pi * rnd(state))

@nb.njit(inline='always')
def poisson(state, lam):
    k = 0; p = math.exp(-lam); F = p; u = rnd(state)
    while u > F and k < 200:
        k += 1; p *= lam / k; F += p
    return k

@nb.njit(inline='always')
def emission(state):
    """A unit vector in the σ dipole pattern (1 + cos²θ about z)."""
    while True:
        c = 2 * rnd(state) - 1
        if 2 * rnd(state) < 1 + c * c: break
    s = math.sqrt(max(0.0, 1 - c * c)); ph = 2 * math.pi * rnd(state)
    return s * math.cos(ph), s * math.sin(ph), c

@nb.njit(inline='always')
def beam(x, y, z, w0, zR):
    """g = I/I0 of the tweezer (paraxial Gaussian along x) and its gradient."""
    X2 = (x / zR) ** 2; A = 1.0 / (1.0 + X2); r2 = y * y + z * z
    E = math.exp(-2.0 * r2 * A / (w0 * w0)); g = A * E
    gx = -(2.0 * x * A * A / (zR * zR)) * E * (1.0 - 2.0 * r2 * A / (w0 * w0))
    gy = g * (-4.0 * y * A / (w0 * w0)); gz = g * (-4.0 * z * A / (w0 * w0))
    return g, gx, gy, gz

@nb.njit(inline='always')
def rate(gamma, s, d):
    return 0.5 * gamma * s / (1.0 + s + 4.0 * d * d / (gamma * gamma))

# ---- the optical Bloch equations, ρ as [gg, ge, eg, ee] ----
@nb.njit(inline='always')
def liouvillian(Om, d, gamma, L):
    """dρ/dt = L ρ for H/ħ = −δ|e><e| + (Ω|e><g| + Ω*|g><e|)/2 and decay Γ (gamma = 0: the coherent part alone)."""
    i = 1j; Oc = Om.conjugate()
    for a in range(4):
        for b in range(4): L[a, b] = 0
    # gg
    L[0, 3] = gamma; L[0, 2] = -i * Oc / 2; L[0, 1] = i * Om / 2
    # ee
    L[3, 3] = -gamma; L[3, 1] = -i * Om / 2; L[3, 2] = i * Oc / 2
    # eg
    L[2, 2] = -gamma / 2 + i * d; L[2, 0] = -i * Om / 2; L[2, 3] = i * Om / 2
    # ge
    L[1, 1] = -gamma / 2 - i * d; L[1, 0] = i * Oc / 2; L[1, 3] = -i * Oc / 2

@nb.njit(inline='always')
def lu(A, piv):
    """In-place LU with partial pivoting of a 4×4 complex matrix."""
    n = 4
    for k in range(n):
        p = k; best = abs(A[k, k])
        for r in range(k + 1, n):
            if abs(A[r, k]) > best: best = abs(A[r, k]); p = r
        piv[k] = p
        if p != k:
            for c in range(n):
                t = A[k, c]; A[k, c] = A[p, c]; A[p, c] = t
        for r in range(k + 1, n):
            A[r, k] /= A[k, k]
            for c in range(k + 1, n): A[r, c] -= A[r, k] * A[k, c]

@nb.njit(inline='always')
def lusolve(A, piv, b):
    n = 4
    for k in range(n):
        p = piv[k]
        if p != k:
            t = b[k]; b[k] = b[p]; b[p] = t
    for r in range(1, n):
        for c in range(r): b[r] -= A[r, c] * b[c]
    for r in range(n - 1, -1, -1):
        for c in range(r + 1, n): b[r] -= A[r, c] * b[c]
        b[r] /= A[r, r]

@nb.njit(inline='always')
def opmul(o, rho, out):
    """out = O ρ for 2×2 matrices stored [gg, ge, eg, ee]."""
    out[0] = o[0] * rho[0] + o[1] * rho[2]; out[1] = o[0] * rho[1] + o[1] * rho[3]
    out[2] = o[2] * rho[0] + o[3] * rho[2]; out[3] = o[2] * rho[1] + o[3] * rho[3]

@nb.njit(inline='always')
def optrace(o, rho):
    """Tr(O ρ)."""
    return o[0] * rho[0] + o[1] * rho[2] + o[2] * rho[1] + o[3] * rho[3]

@nb.njit(inline='always')
def fp_coefficients(Om, dOm_z, d, grad_d, gamma, vx, vy, vz, out_f, out_D, L, Ld, piv, wk):
    """Gordon–Ashkin: the mean force (per ħ, besides −∇U_g) at first order in v, the induced momentum diffusion
    (per ħ², 3×3) and ρ_ee of ρ0 + ρ1. grad_d = ∇δ, dOm_z = ∂Ω/∂z (Ω depends on z only). Workspace: L, Ld 4×4, piv,
    wk 8×4 complex. Linear systems L y = b with b traceless are solved with the gg row replaced by the trace (y
    traceless): exact, as Tr(Lρ) = 0 for every ρ makes that row follow from the others."""
    liouvillian(Om, d, gamma, L)
    L[0, 0] = 1; L[0, 1] = 0; L[0, 2] = 0; L[0, 3] = 1
    lu(L, piv)
    rho0 = wk[0]; t = wk[1]; rho1 = wk[2]; fe = wk[3]; fc = wk[4]; Ye = wk[5]; Yc = wk[6]
    for a in range(4): rho0[a] = 0
    rho0[0] = 1
    lusolve(L, piv, rho0)                            # steady state ρ0 (trace 1)
    # ∂_t ρ0 along the path = −L⁻¹ (L̇ ρ0), L̇ the coherent part at (Ω̇, δ̇) = (v_z ∂_zΩ, v·∇δ)
    liouvillian(dOm_z * vz, grad_d[0] * vx + grad_d[1] * vy + grad_d[2] * vz, 0.0, Ld)
    for a in range(4):
        acc = 0j
        for b in range(4): acc += Ld[a, b] * rho0[b]
        t[a] = -acc
    t[0] = 0
    lusolve(L, piv, t)                               # t = ∂_t ρ0 (traceless)
    for a in range(4): rho1[a] = t[a]
    rho1[0] = 0
    lusolve(L, piv, rho1)                            # ρ1 = L⁻¹ ∂_t ρ0 (traceless)
    # force operators per ħ: f_i = ∂_iδ σ_ee − (∂_iΩ σ_eg + ∂_iΩ* σ_ge)/2; operators stored [gg, ge, eg, ee]
    for a in range(4): fe[a] = 0; fc[a] = 0
    fe[3] = 1
    fc[2] = -dOm_z / 2; fc[1] = -dOm_z.conjugate() / 2
    ee = (optrace(fe, rho0) + optrace(fe, rho1)).real; cz = (optrace(fc, rho0) + optrace(fc, rho1)).real
    out_f[0] = grad_d[0] * ee; out_f[1] = grad_d[1] * ee; out_f[2] = grad_d[2] * ee + cz
    # diffusion by the quantum regression theorem: Y_O = −L⁻¹ (O ρ0 − <O> ρ0); C_AB = Tr(A Y_B)
    ee0 = optrace(fe, rho0); c0 = optrace(fc, rho0)
    opmul(fe, rho0, Ye); opmul(fc, rho0, Yc)
    for a in range(4):
        Ye[a] = -(Ye[a] - ee0 * rho0[a]); Yc[a] = -(Yc[a] - c0 * rho0[a])
    Ye[0] = 0; Yc[0] = 0
    lusolve(L, piv, Ye); lusolve(L, piv, Yc)
    See = optrace(fe, Ye).real; Scc = optrace(fc, Yc).real
    Sec = 0.5 * (optrace(fe, Yc).real + optrace(fc, Ye).real)
    g = grad_d
    for i in range(3):
        for j in range(3):
            out_D[i, j] = g[i] * g[j] * See
    for i in range(3):
        out_D[i, 2] += g[i] * Sec; out_D[2, i] += g[i] * Sec
    out_D[2, 2] += Scc
    return ee

@nb.njit
def fp_probe(Om, dOm_z, d, gz, gamma, vz):
    """fp_coefficients at one point (tests): (f_z, D_zz, ρ_ee)."""
    L = np.zeros((4, 4), np.complex128); Ld = np.zeros((4, 4), np.complex128); piv = np.zeros(4, np.int64)
    wk = np.zeros((8, 4), np.complex128); f = np.zeros(3); D = np.zeros((3, 3)); gd = np.array([0.0, 0.0, gz])
    ee = fp_coefficients(Om, dOm_z, d, gd, gamma, 0.0, 0.0, vz, f, D, L, Ld, piv, wk)
    return f[2], D[2, 2], ee

@nb.njit(parallel=True)
def run(model, n_atoms, seed, T_uK, dt, n_steps, every, P, out_v, out_alive, out_unbound):
    """model: 0 = S1 kicks, 1 = S1 Fokker–Planck, 2 = S2 kicks (independent beams), 3 = S2 Fokker–Planck (standing wave).
    P = [gamma, kp, w0, zR, U0, dLS, dZ, hbar_m, s1, uK]. Writes, per output time and atom, the velocity and whether
    the atom is still held (and whether it is unbound, E > 0)."""
    gamma, kp, w0, zR, U0, dLS, dZ, hbar_m, s1, uK = P[0], P[1], P[2], P[3], P[4], P[5], P[6], P[7], P[8], P[9]
    vr = hbar_m * kp
    kT = T_uK * uK                       # rad/µs
    vth = math.sqrt(kT * hbar_m)         # µm/µs
    wR = math.sqrt(4 * U0 * hbar_m / w0 ** 2); wA = math.sqrt(2 * U0 * hbar_m / zR ** 2)
    # proposal widths wider than the harmonic trap's (the Gaussian trap is softer): the acceptance ratio stays ≤ 1 out
    # to 1.6 z_R axially and the region's edge radially, beyond e^-14 of the proposal's weight
    sig = np.array([1.5 * vth / wA, 1.25 * vth / wR, 1.25 * vth / wR])
    Om1 = gamma * math.sqrt(s1 / 2.0)    # s = 2Ω²/Γ²
    n_out = out_v.shape[0]
    for a in nb.prange(n_atoms):
        # each atom's own stream (results do not depend on the thread count): splitmix64 of (seed, atom) seeds xorshift64*
        st = np.zeros(1, np.uint64)
        z = np.uint64(seed) * np.uint64(0x9E3779B97F4A7C15) + np.uint64(a + 1) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9); z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        z = z ^ (z >> np.uint64(31)); st[0] = z if z != np.uint64(0) else np.uint64(1)
        for _ in range(4): rnd(st)
        Lw = np.zeros((4, 4), np.complex128); Ldw = np.zeros((4, 4), np.complex128); piv = np.zeros(4, np.int64)
        wk = np.zeros((8, 4), np.complex128)
        f = np.zeros(3); D = np.zeros((3, 3)); gd = np.zeros(3)
        # initial position: rejection from a Gaussian proposal against exp(−(U_g + U0)/kT) inside the region
        while True:
            x = sig[0] * gauss(st); y = sig[1] * gauss(st); z = sig[2] * gauss(st)
            g, _, _, _ = beam(x, y, z, w0, zR)
            if g < 0.05: continue
            logw = -U0 * (1 - g) / kT + 0.5 * ((x / sig[0]) ** 2 + (y / sig[1]) ** 2 + (z / sig[2]) ** 2)
            if math.log(max(rnd(st), 1e-300)) < logw: break   # logw ≤ 0 inside the region for these widths (checked)
        vx = vth * gauss(st); vy = vth * gauss(st); vz = vth * gauss(st)
        alive = True; unbound = False
        k_out = 0
        ax = 0.0; ay = 0.0; az = 0.0; ree = 0.0; Rp = 0.0; Rm = 0.0
        for step in range(n_steps + 1):
            if step % every == 0 and k_out < n_out:
                out_v[k_out, a, 0] = vx; out_v[k_out, a, 1] = vy; out_v[k_out, a, 2] = vz
                out_alive[k_out, a] = alive; out_unbound[k_out, a] = unbound
                k_out += 1
            if step == n_steps or not alive: continue
            for half in range(2):
                g, gx, gy, gz = beam(x, y, z, w0, zR)
                d = dZ - dLS * g
                if model == 0:
                    Rp = rate(gamma, s1, d - kp * vz); Rm = 0.0; ree = Rp / gamma
                    fx = 0.0; fy = 0.0; fz = 0.0
                elif model == 2:
                    Rp = 0.5 * gamma * s1 / (1 + 2 * s1 + 4 * (d - kp * vz) ** 2 / gamma ** 2)
                    Rm = 0.5 * gamma * s1 / (1 + 2 * s1 + 4 * (d + kp * vz) ** 2 / gamma ** 2)
                    ree = (Rp + Rm) / gamma
                    fx = 0.0; fy = 0.0; fz = 0.0
                else:
                    gd[0] = -dLS * gx; gd[1] = -dLS * gy; gd[2] = -dLS * gz
                    if model == 1:
                        ph = kp * z; Om = Om1 * complex(math.cos(ph), math.sin(ph)); dOm = 1j * kp * Om
                    else:
                        Om = complex(2 * Om1 * math.cos(kp * z), 0.0); dOm = complex(-2 * Om1 * kp * math.sin(kp * z), 0.0)
                    ree = fp_coefficients(Om, dOm, d, gd, gamma, vx, vy, vz, f, D, Lw, Ldw, piv, wk)
                    # the σ_ee part of f is already −ρ_ee ∇ΔU: the mean force below adds only −∇U_g
                    fx = f[0]; fy = f[1]; fz = f[2]
                if model == 0 or model == 2:
                    # mean force besides −∇U_g: −ρ_ee ∇ΔU = ρ_ee ∇δ·ħ... (∇δ = −dLS ∇g)
                    fx = -dLS * gx * ree; fy = -dLS * gy * ree; fz = -dLS * gz * ree
                ax = hbar_m * (U0 * gx + fx); ay = hbar_m * (U0 * gy + fy); az = hbar_m * (U0 * gz + fz)
                if half == 0:
                    vx += 0.5 * dt * ax; vy += 0.5 * dt * ay; vz += 0.5 * dt * az
                    x += dt * vx; y += dt * vy; z += dt * vz
                else:
                    vx += 0.5 * dt * ax; vy += 0.5 * dt * ay; vz += 0.5 * dt * az
            # stochastic part at the end of the step (the coefficients of the last evaluation)
            if model == 0 or model == 2:
                n_p = poisson(st, Rp * dt); n_m = poisson(st, Rm * dt) if model == 2 else 0
                vz += vr * (n_p - n_m)
                for _ in range(n_p + n_m):
                    ex, ey, ez = emission(st); vx += vr * ex; vy += vr * ey; vz += vr * ez
            else:
                # Gaussian momentum noise with covariance 2 D dt (ħ² units → velocity: hbar_m² factor)
                # Cholesky of the symmetric 3×3 (clamped at zero for roundoff)
                l00 = math.sqrt(max(D[0, 0], 0.0))
                l10 = D[1, 0] / l00 if l00 > 0 else 0.0
                l11 = math.sqrt(max(D[1, 1] - l10 * l10, 0.0))
                l20 = D[2, 0] / l00 if l00 > 0 else 0.0
                l21 = (D[2, 1] - l20 * l10) / l11 if l11 > 0 else 0.0
                l22 = math.sqrt(max(D[2, 2] - l20 * l20 - l21 * l21, 0.0))
                sc = hbar_m * math.sqrt(2 * dt)
                n0 = gauss(st); n1 = gauss(st); n2 = gauss(st)
                vx += sc * l00 * n0; vy += sc * (l10 * n0 + l11 * n1); vz += sc * (l20 * n0 + l21 * n1 + l22 * n2)
                n_e = poisson(st, gamma * max(ree, 0.0) * dt)
                for _ in range(n_e):
                    ex, ey, ez = emission(st); vx += vr * ex; vy += vr * ey; vz += vr * ez
            g, _, _, _ = beam(x, y, z, w0, zR)
            E = 0.5 * (vx * vx + vy * vy + vz * vz) / hbar_m - U0 * g
            if E > 0: unbound = True
            if g < 0.05 and E > 0: alive = False
    return 0

MODELS = {'S1K': 0, 'S1F': 1, 'S2K': 2, 'S2F': 3}
DESCRIPTION = {'S1K': 'S1, kicks at the two-level rate (Poisson)', 'S1F': 'S1, Gordon–Ashkin Fokker–Planck (cross-check)',
               'S2K': 'S2, independent running waves (no interference), kicks', 'S2F': 'S2, standing wave, Gordon–Ashkin Fokker–Planck'}

def simulate(name, p, atoms, end_us, every_us, dt, seed=12345):
    n_steps = int(round(end_us / dt)); every = int(round(every_us / dt)); n_out = n_steps // every + 1
    out_v = np.zeros((n_out, atoms, 3)); out_alive = np.zeros((n_out, atoms), np.bool_); out_unbound = np.zeros((n_out, atoms), np.bool_)
    P = np.array([p['gamma'], p['kp'], p['w0'], p['zR'], p['U0'], p['dLS'], p['dZ'], p['hbar_m'], p['s1'], p['uK']])
    t0 = time.time(); run(MODELS[name], atoms, seed, 30.0, dt, n_steps, every, P, out_v, out_alive, out_unbound)
    wall = time.time() - t0
    times = np.arange(n_out) * every * dt
    held = out_alive.sum(1); loss = 1 - held / atoms
    T = np.full((n_out, 3), np.nan); se = np.full((n_out, 3), np.nan)
    for k in range(n_out):
        v = out_v[k][out_alive[k]]
        if len(v) > 10:
            var = v.var(0, ddof=1); T[k] = var / p['hbar_m'] / p['uK']
            # the standard error of a variance: sqrt((m4 − σ⁴(n−3)/(n−1))/n), with the sample's 4th central moment
            m4 = ((v - v.mean(0)) ** 4).mean(0); se[k] = np.sqrt(np.maximum(m4 - var ** 2 * (len(v) - 3) / (len(v) - 1), 0) / len(v)) / p['hbar_m'] / p['uK']
    Ttot = T.mean(1); setot = np.sqrt((se ** 2).sum(1)) / 3
    return dict(model=name, description=DESCRIPTION[name], atoms=atoms, dtUs=dt, wallS=wall, timesUs=times.tolist(),
                loss=loss.tolist(), lossSE=np.sqrt(np.maximum(loss * (1 - loss), 1 / atoms) / atoms).tolist(),
                unbound=out_unbound.mean(1).tolist(),
                T=[[None if not np.isfinite(x) else round(float(x), 3) for x in T[:, a]] for a in range(3)],
                TSE=[[None if not np.isfinite(x) else round(float(x), 3) for x in se[:, a]] for a in range(3)],
                Ttotal=[None if not np.isfinite(x) else round(float(x), 3) for x in Ttot],
                TtotalSE=[None if not np.isfinite(x) else round(float(x), 3) for x in setot])

def analytic(p):
    """The semi-analytic numbers at the focus (two-level formulas): rates, Doppler limit, initial heating per axis."""
    G = p['gamma']; s = p['s1']; out = {}
    for where, g in (('focus', 1.0), ('edge', 0.0)):
        d = p['dZ'] - p['dLS'] * g; R = rate(G, s, d)
        # one running wave: friction β/m = ħk² R·8|δ|/Γ²/(1+s+(2δ/Γ)²)/m; D_z = (ħk)²R(1 + 2/5)/2 → k_BT = D/β
        TD = 1.4 * (1 + s + 4 * d * d / G ** 2) * G ** 2 / (16 * abs(d)) / p['uK']
        gam = p['hbar_m'] * p['kp'] ** 2 * R * 8 * abs(d) / G ** 2 / (1 + s + 4 * d * d / G ** 2)
        Er = 0.5 * p['hbar_m'] * p['kp'] ** 2 / p['uK']     # recoil energy in µK
        out[where] = dict(detuningMHz=d / (2 * math.pi), detuningGamma=d / G, photonRatePerUs=R, excited=R / G,
                          dopplerLimitUK=TD, coolingRatePerMs=gam * 1e3, heatXYuKPerMs=0.3 * Er * R * 1e3,
                          radiationForceOverTrapMax=(p['kp'] * R) / (p['U0'] * 2 / p['w0'] * math.exp(-0.5)))
    # standing wave at the antinode (S2 at the focus): s = 4 s1
    d = p['dZ'] - p['dLS']; R = rate(G, 4 * s, d); out['antinode'] = dict(photonRatePerUs=R, heatXYuKPerMs=0.3 * 0.5 * p['hbar_m'] * p['kp'] ** 2 / p['uK'] * R * 1e3)
    return out

def selftest(p):
    """The Fokker–Planck coefficients against closed forms: a running wave's radiation pressure ħkΓρ_ee and friction
    −ħk²R'(δ); its absorption diffusion (ħk)²R(1+Q)/2 with Mandel's Q from the Bloch equations integrated in time
    (scipy's expm, independent of the LU code); a weak standing wave's friction averaged over z = twice the running
    wave's (the interference terms average out at low saturation)."""
    from scipy.linalg import expm
    G, k = p['gamma'], p['kp']; out = []
    for s, d in ((p['s1'], -3.84 * G), (p['s1'], -0.69 * G), (0.01, -1.0 * G)):
        Om1 = G * math.sqrt(s / 2)
        R = lambda dd: rate(G, s, dd)
        f0, D0, ee = fp_probe(complex(Om1, 0), 1j * k * Om1, d, 0.0, G, 0.0)
        f1, _, _ = fp_probe(complex(Om1, 0), 1j * k * Om1, d, 0.0, G, 1e-3)
        dR = (R(d + 1e-4) - R(d - 1e-4)) / 2e-4
        # Q: ∫(ρ_ee(τ|g) − ρ_ee) dτ by quadrature of expm
        L = np.zeros((4, 4), complex); liouvillian.py_func(complex(Om1, 0), d, G, L)
        r0 = np.array([1, 0, 0, 0], complex); ss = R(d) / G
        taus = np.linspace(0, 40 / G, 4001); dtau = taus[1] - taus[0]; Eh = expm(L * dtau); r = r0.copy(); acc = []
        for _ in taus: acc.append(r[3].real - ss); r = Eh @ r
        Q = 2 * G * np.trapezoid(acc, taus)   # 2R/ρ_ee ∫(ρ_ee(τ|g) − ρ_ee) = 2Γ ∫(...)
        out.append(dict(s=s, deltaGamma=d / G, pressure=f0 / (k * G * ee), excitedVsFormula=ee / (R(d) / G),
                        friction=((f1 - f0) / 1e-3) / (-k * k * dR), diffusion=D0 / (k * k * R(d) * (1 + Q) / 2), Q=Q))
    # weak standing wave: friction averaged over a period vs twice the running wave's
    s, d = 0.01, -1.0 * G; Om1 = G * math.sqrt(s / 2); fr = []
    for z in np.linspace(0, math.pi / k, 64, endpoint=False):
        Om = complex(2 * Om1 * math.cos(k * z), 0); dOm = complex(-2 * Om1 * k * math.sin(k * z), 0)
        fa, _, _ = fp_probe(Om, dOm, d, 0.0, G, 0.0); fb, _, _ = fp_probe(Om, dOm, d, 0.0, G, 1e-3); fr.append((fb - fa) / 1e-3)
    dR = (rate(G, s, d + 1e-4) - rate(G, s, d - 1e-4)) / 2e-4
    out.append(dict(standingFrictionOverTwoRunning=np.mean(fr) / (-2 * k * k * dR)))
    for o in out: print('  selftest', {kk: round(float(v), 5) for kk, v in o.items()})
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--atoms', type=int, default=2000); ap.add_argument('--end', type=float, default=10000.0)
    ap.add_argument('--every', type=float, default=50.0); ap.add_argument('--models', default='S1K,S1F,S2K,S2F')
    ap.add_argument('--dt', type=float, default=0.05); ap.add_argument('--out', default=os.path.join(HERE, 'model.json'))
    ap.add_argument('--summary', action='store_true')
    ap.add_argument('--seed', type=int, default=12345)
    ap.add_argument('--s-scale', type=float, default=1.0, help='tests only: the probe intensity times this (0: the conservative trap)')
    a = ap.parse_args()
    tables = app_tables(); shifts = light_shifts(tables); p = parameters(tables, shifts)
    print(f"light shifts at the focus: ground {shifts['groundShiftMHz']:.4f} MHz (chart {CHART['groundShiftMHz']}), "
          f"|3',3'> {shifts['excitedShiftMHz']:.4f} MHz (chart {CHART['excitedShiftMHz']}); α(5S) {shifts['alphaGroundAU']:.1f} au, "
          f"α(5P3/2, m=3/2, π) {shifts['alphaExcitedAU']:.1f} au (scalar {shifts['alphaExcitedScalarAU']:.1f}, tensor {shifts['alphaExcitedTensorAU']:.1f})")
    print('  5P3/2 parts (au):', {k: round(v * 1, 1) for k, v in shifts['excitedParts'].items()})
    print(f"U0 {p['U0']/p['uK']:.1f} µK, ω/2π radial {p['omegaR']/2/math.pi*1e3:.2f} kHz, axial {p['omegaAx']/2/math.pi*1e3:.2f} kHz, "
          f"δ_LS/2π {p['dLS']/2/math.pi:.4f} MHz, s {p['s1']:.4f} (I_sat {p['Isat']:.5f} mW/cm²)")
    an = analytic(p)
    for k, v in an.items(): print(' ', k, {kk: round(vv, 5) for kk, vv in v.items()})
    tests = selftest(p)
    if a.summary or a.selftest: return
    p['s1'] *= a.s_scale
    models = [m for m in a.models.split(',') if m]
    result = dict(generator='motion/refs/sanity/model.py', lightShifts={k: v for k, v in shifts.items() if not isinstance(v, dict)},
                  chart=CHART, parameters={k: v for k, v in p.items()}, analytic=an,
                  selftest=[{k: float(v) for k, v in t.items()} for t in tests], runs={})
    for m in models:
        r = simulate(m, p, a.atoms, a.end, a.every, a.dt, a.seed)
        print(f"{m}: {r['wallS']:.1f} s; " + '; '.join(f"t {r['timesUs'][k]/1e3:.1f} ms T {r['Ttotal'][k]} ({'/'.join(str(r['T'][i][k]) for i in range(3))}) L {r['loss'][k]:.3f} unb {r['unbound'][k]:.3f}"
                                               for k in range(0, len(r['timesUs']), max(1, len(r['timesUs']) // 10))))
        result['runs'][m] = r
    json.dump(result, open(a.out, 'w'))
    print('wrote', a.out)

if __name__ == '__main__':
    main()
