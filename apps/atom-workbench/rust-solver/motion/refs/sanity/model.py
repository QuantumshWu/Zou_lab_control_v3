"""Independent physics model of the two sanity scenes (refs/sanity.mjs): S1, the app's default tweezer + one σ+ probe;
S2, the same + the probe's retro-reflection (the app's plain mirror) + a repump. It shares no code and no approximation
with the motion engine: the light shifts come from a sum over states on the app's own atomic tables (read from
src/physics.ts and src/spectroscopy-data.ts), the dynamics from a classical-atom Monte Carlo written here.

    python model.py [--atoms 4000] [--end 5000] [--every 50] [--models S1K,S1F,S2K,S2F,S2R,S2QR,S2QZ,S2QV] [--out model.json]

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

# S2Q: the mirror with its λ/4 plate (σ+ and σ− about the field) -- part C, the multilevel rate model
With σ− present the atom leaves |2,2⟩: quantum jumps over the 8 ground sublevels; every beam excites every allowed
|F',m'⟩ (F' = 0..3) at the two-level rate at its own detuning -- hyperfine intervals, linear Zeeman shifts (3 G), the
tweezer's shift of each sublevel (Σ|⟨m_J m_I|F'm'⟩|² δE(m_J), δE(±1/2) from the same sum over states), Doppler --,
decays by the dipole branching (F'=2 → F=1: 1/2, F'=1: 5/6, checked), each photon's kick in its pattern. Ground
coherences are left out: the 3 G field's Larmor precession (2.1 MHz per Δm; the Δm = 2 Raman pair σ+σ− drives is
4.2 MHz apart) is ≫ the probe's ground light shift (~0.4 MHz) and the pumping rates (≲ 1/µs), so they are dephased.
That same fact disables σ+σ− sub-Doppler cooling near v = 0: the motion-induced orientation responds to kv + ω_L, so
its resonance sits at v = ω_L/k ≈ 1.6 m/s, far above the atoms' 0.05–0.3 m/s; near v = 0 what remains is a friction of
either sign, ≲ (Δ'/ω_L)² ≈ 3 % of the zero-field σ+σ− friction (Δ' the ground light shift) -- comparable to the Doppler
friction at most. What the rate model cannot say is that remainder along z; it is bracketed by two variants: z frozen
(v_z damped at 1/µs: the best ANY cooling acting on z alone can do, sub-Doppler included) and z without Doppler shifts
(no velocity-dependent force along z). x and y have no cooling beam in any variant.
Result: the atom is pumped mostly into |2,−2⟩ (56 %; |2,2⟩ 22 %, the rest spread): the 3 G field puts the σ− cycle
|2,−2⟩ → |3',−3'⟩ 8.4 MHz nearer the probe than σ+'s: −2.45Γ at the centre, +0.69Γ -- blue, heating and pushing -- at
the trap's edge. Half the atoms lost at 0.52 ms (z frictionless 0.45 ms; z frozen 2.99 ms): no faster cooling along
z can make S2Q "much slower" than S1 (1.82 ms). What would: cooling on all three axes (the x, y spontaneous-emission
heating is what loses the atom in every variant), with the light red of the light-shifted line throughout the trap,
and at a field whose Larmor frequency is below the ground light shift if sub-Doppler cooling is wanted.
S2 by the same model (S2R, independent beams) agrees with S2K within 0.04 in L, and gives S2's physical leaks out of
the cycle (all from the repump, 6.5–6.8 GHz off its F=2 lines): 2.1/s to |2,1⟩, 1.2/s to |1,1⟩ at the centre.

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
    e_partners = [('5S1/2', .5, -E_D2, reduced('D2', 1.5))]  # |<3/2||D||1/2>|² = |<1/2||D||3/2>|² (Edmonds)
    e_partners += [(v['name'], v['J'], v['E'] * EV / EH - E_D2, v['D'] ** 2) for v in tables['virtual']['D2']]
    ag, gparts = alpha(.5, .5, g_partners, tables['core']['ground'])
    ae, eparts = alpha(1.5, 1.5, e_partners, tables['core']['D2'])
    ae_half, _ = alpha(1.5, .5, e_partners, tables['core']['D2'])    # |3/2, ±1/2⟩ in π (for S2Q's other sublevels)
    # scalar and tensor of 5P3/2 (α(m=J, π) = α0 + α2)
    a0e = np.mean([sum(alpha(1.5, m, e_partners, 0)[0] for m in (1.5, .5, -.5, -1.5)) / 4])
    I0 = 2 * power_mw * 1e-3 / (math.pi * (waist_um * 1e-6) ** 2)
    shift = lambda a: -a * AU_POL * I0 / (2 * EPS0 * C) / H / 1e6   # MHz
    return dict(I0Wm2=I0, alphaGroundAU=ag, alphaExcitedAU=ae, alphaExcitedScalarAU=a0e + tables['core']['D2'],
                alphaExcitedTensorAU=ae - (a0e + tables['core']['D2']),
                groundShiftMHz=shift(ag), excitedShiftMHz=shift(ae), excitedHalfShiftMHz=shift(ae_half),
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

# ---------------------------------------------------------------------------------------------------------------------
# Part C: the multilevel rate model (S2Q, where σ+σ− makes the atom leave |2,2⟩; and S2 again as its check).
# Rate equations over the 8 ground sublevels, excitation of each |F',m'⟩ (F' = 0..3) by each beam at the two-level rate
# at its own local light-shifted, Zeeman-shifted, Doppler-shifted detuning, decay by the dipole branching, kicks of
# both photons (π emission ∝ sin²θ, σ ∝ 1 + cos²θ): a quantum-jump Monte Carlo of the internal state along each
# classical trajectory (Gillespie within the step). Ground-state coherences are left out: they are dephased by the
# 3 G field (Larmor 2.1 MHz per Δm, 4.2 MHz for the Δm = 2 Raman pair σ+σ− couples) against a ground light shift of
# ≈ 0.4 MHz and pumping rates ≲ 1/µs -- this is what removes σ+σ− sub-Doppler cooling (see SUB_DOPPLER below) and
# what makes rate equations valid. Excited sublevels: linear Zeeman (g_F' with g_I; ≤ 0.3 MHz = 0.05Γ nonlinear
# corrections for F' ≠ 3 left out), tweezer shift Σ|⟨m_J m_I|F'm'⟩|² δE(m_J) (the tensor's F'-mixing, ~(6 MHz)²/267 MHz,
# left out). Hyperfine intervals: Steck, Rb 87 D line data, Table 7 (5P3/2: 266.650, 156.947, 72.218 MHz; 5S1/2
# 6834.682611 MHz) -- the values physics.ts holds as A, B.

def sixj(a, b, c, d, e, f):
    """Racah's formula for {a b c; d e f}."""
    def tri(x, y, z):
        if z < abs(x - y) - 1e-9 or z > x + y + 1e-9 or abs((x + y + z) - round(x + y + z)) > 1e-9: return None
        F = lambda t: math.factorial(int(round(t)))
        return F(x + y - z) * F(x - y + z) * F(-x + y + z) / F(x + y + z + 1)
    ts = [tri(a, b, c), tri(a, e, f), tri(d, b, f), tri(d, e, c)]
    if any(t is None for t in ts): return 0.0
    pre = math.sqrt(np.prod(ts)); F = lambda t: math.factorial(int(round(t)))
    s = 0.0
    for k in range(0, 60):
        args = [k - a - b - c, k - a - e - f, k - d - b - f, k - d - e - c, a + b + d + e - k, a + c + d + f - k, b + c + e + f - k]
        if min(args) < -1e-9: continue
        s += (-1) ** k * F(k + 1) / np.prod([F(x) for x in args])
    return pre * s

GROUND = [(2, m) for m in (-2, -1, 0, 1, 2)] + [(1, m) for m in (-1, 0, 1)]
EXCITED = [(Fp, m) for Fp in (0, 1, 2, 3) for m in range(-Fp, Fp + 1)]
HFS_GROUND = {2: 0.0, 1: -6834.682611}
HFS_EXCITED = {3: 0.0, 2: -266.650, 1: -266.650 - 156.947, 0: -266.650 - 156.947 - 72.218}

def multilevel_tables(shifts, field_G=3.0, gJ=2.00233107, gJp=1.3341, gI=-0.0009951414, muB=1.39962449361):
    """Strengths S[g, e] (cycling = 1), branching B[g, e] (e → g), and per state its zero-field energy, Zeeman shift and
    tweezer shift at the focus (MHz), relative to |2⟩ → |3'⟩."""
    I, J, Jp = 1.5, 0.5, 1.5
    def strength(F, m, Fp, mp):
        q = mp - m
        if abs(q) > 1: return 0.0
        # |⟨F'm'|d_q|F m⟩|² = |⟨J'‖d‖J⟩|² (2F+1)(2F'+1) {J J' 1; F' F I}² (F' 1 F; −m' q m)² (Edmonds)
        return (2 * F + 1) * (2 * Fp + 1) * sixj(J, Jp, 1, Fp, F, I) ** 2 * wigner3j(Fp, 1, F, -mp, q, m) ** 2
    S = np.array([[strength(F, m, Fp, mp) for (Fp, mp) in EXCITED] for (F, m) in GROUND])
    S /= strength(2, 2, 3, 3)
    B = S / S.sum(0)
    gF = lambda F, j, gj: gj * (F * (F + 1) - I * (I + 1) + j * (j + 1)) / (2 * F * (F + 1)) + gI * (F * (F + 1) + I * (I + 1) - j * (j + 1)) / (2 * F * (F + 1)) if F > 0 else 0.0
    Zg = np.array([muB * field_G * gF(F, J, gJ) * m for (F, m) in GROUND])
    Ze = np.array([muB * field_G * gF(Fp, Jp, gJp) * m for (Fp, m) in EXCITED])
    Eg = np.array([HFS_GROUND[F] for (F, m) in GROUND]); Ee = np.array([HFS_EXCITED[Fp] for (Fp, m) in EXCITED])
    def cg(j1, m1, j2, m2, Jt, M): return (-1) ** int(round(j1 - j2 + M)) * math.sqrt(2 * Jt + 1) * wigner3j(j1, j2, Jt, m1, m2, -M)
    LSmJ = {1.5: shifts['excitedShiftMHz'], 0.5: shifts['excitedHalfShiftMHz']}
    LSe = np.array([sum(cg(Jp, mJ, I, mp - mJ, Fp, mp) ** 2 * LSmJ[abs(mJ)] for mJ in (-1.5, -.5, .5, 1.5) if abs(mp - mJ) <= I) for (Fp, mp) in EXCITED])
    LSg = np.full(len(GROUND), shifts['groundShiftMHz'])
    return dict(S=S, B=B, Eg=Eg, Ee=Ee, Zg=Zg, Ze=Ze, LSg=LSg, LSe=LSe, mg=np.array([m for _, m in GROUND]), me=np.array([m for _, m in EXCITED]))

@nb.njit(inline='always')
def emission_pi(state):
    """A unit vector in the π dipole pattern (sin²θ about z)."""
    while True:
        c = 2 * rnd(state) - 1
        if rnd(state) < 1 - c * c: break
    s = math.sqrt(max(0.0, 1 - c * c)); ph = 2 * math.pi * rnd(state)
    return s * math.cos(ph), s * math.sin(ph), c

@nb.njit(parallel=True)
def run_multi(variant, n_atoms, seed, T_uK, dt, n_steps, every, P, beams, S, B, Eg, Ee, Zg, Ze, LSg, LSe, mg, me, g0,
              out_v, out_alive, out_g):
    """variant: 0 the rate model; 1 the same with the z motion frozen (v_z damped at 1/µs: any cooling acting on z alone
    can do no better); 2 the same with no Doppler shift from v_z (no velocity-dependent force along z: what the beams'
    z friction would be if a sub-Doppler force cancelled it). beams[b] = [k̂x, k̂y, k̂z, q, s, ν (MHz from |2⟩→|3'⟩), k]."""
    gamma, w0, zR, U0, hbar_m, uK = P[0], P[1], P[2], P[3], P[4], P[5]
    kT = T_uK * uK; vth = math.sqrt(kT * hbar_m)
    wR = math.sqrt(4 * U0 * hbar_m / w0 ** 2); wA = math.sqrt(2 * U0 * hbar_m / zR ** 2)
    sig = np.array([1.5 * vth / wA, 1.25 * vth / wR, 1.25 * vth / wR])
    nb_ = beams.shape[0]; ng = S.shape[0]; ne = S.shape[1]; n_out = out_v.shape[0]; twopi = 2 * math.pi
    for a in nb.prange(n_atoms):
        st = np.zeros(1, np.uint64)
        z0 = np.uint64(seed) * np.uint64(0x9E3779B97F4A7C15) + np.uint64(a + 1) * np.uint64(0xBF58476D1CE4E5B9)
        z0 = (z0 ^ (z0 >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9); z0 = (z0 ^ (z0 >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        z0 = z0 ^ (z0 >> np.uint64(31)); st[0] = z0 if z0 != np.uint64(0) else np.uint64(1)
        for _ in range(4): rnd(st)
        rates = np.zeros(nb_ * ne); tb = np.zeros(nb_ * ne, np.int64); te = np.zeros(nb_ * ne, np.int64); dls = np.zeros(nb_ * ne)
        while True:
            x = sig[0] * gauss(st); y = sig[1] * gauss(st); z = sig[2] * gauss(st)
            g, _, _, _ = beam(x, y, z, w0, zR)
            if g < 0.05: continue
            logw = -U0 * (1 - g) / kT + 0.5 * ((x / sig[0]) ** 2 + (y / sig[1]) ** 2 + (z / sig[2]) ** 2)
            if math.log(max(rnd(st), 1e-300)) < logw: break
        vx = vth * gauss(st); vy = vth * gauss(st); vz = vth * gauss(st)
        gs = g0; alive = True; k_out = 0
        for step in range(n_steps + 1):
            if step % every == 0 and k_out < n_out:
                out_v[k_out, a, 0] = vx; out_v[k_out, a, 1] = vy; out_v[k_out, a, 2] = vz
                out_alive[k_out, a] = alive; out_g[k_out, a] = gs; k_out += 1
            if step == n_steps or not alive: continue
            ree = 0.0; nt = 0
            for half in range(2):
                g, gx, gy, gz = beam(x, y, z, w0, zR)
                # the transitions from the present ground state, their rates and their excited-minus-ground shift
                nt = 0; ree = 0.0; fsum = 0.0
                for b in range(nb_):
                    kv = beams[b, 0] * vx + beams[b, 1] * vy + (0.0 if variant == 2 else beams[b, 2] * vz)
                    for e in range(ne):
                        if S[gs, e] <= 0.0 or me[e] - mg[gs] != int(beams[b, 3]): continue
                        d = twopi * (beams[b, 5] - (Ee[e] - Eg[gs]) - (Ze[e] - Zg[gs]) - (LSe[e] - LSg[gs]) * g) - beams[b, 6] * kv
                        s = beams[b, 4] * S[gs, e]
                        r = 0.5 * gamma * s / (1.0 + s + 4.0 * d * d / (gamma * gamma))
                        rates[nt] = r; tb[nt] = b; te[nt] = e; dls[nt] = twopi * (LSe[e] - LSg[gs]); nt += 1
                        fsum += r / gamma * twopi * (LSe[e] - LSg[gs])
                # mean force: −∇U_g − Σ ρ_ee,t ∇ΔU_t (U_g = −U0 g for every ground sublevel: its hyperfine dependence 4 kHz)
                ax = hbar_m * (U0 - fsum) * gx; ay = hbar_m * (U0 - fsum) * gy; az = hbar_m * (U0 - fsum) * gz
                vx += 0.5 * dt * ax; vy += 0.5 * dt * ay; vz += 0.5 * dt * az
                if half == 0:
                    x += dt * vx; y += dt * vy; z += dt * vz
            # jumps within the step (Gillespie at the step's end point; the rates recomputed after each jump)
            trem = dt; jumps = 0
            while jumps < 64:
                R = 0.0
                for t in range(nt): R += rates[t]
                if R <= 0.0: break
                u = rnd(st)
                tau = -math.log(max(u, 1e-300)) / R
                if tau > trem: break
                trem -= tau; jumps += 1
                pick = rnd(st) * R; t = 0; acc = rates[0]
                while acc < pick and t < nt - 1:
                    t += 1; acc += rates[t]
                b = tb[t]; e = te[t]; kk = beams[b, 6] * hbar_m
                vx += kk * beams[b, 0]; vy += kk * beams[b, 1]; vz += kk * beams[b, 2]
                pick = rnd(st); acc = 0.0; gn = ng - 1
                for g2 in range(ng):
                    acc += B[g2, e]
                    if pick < acc:
                        gn = g2; break
                if me[e] == mg[gn]: ex, ey, ez = emission_pi(st)
                else: ex, ey, ez = emission(st)
                vx += kk * ex; vy += kk * ey; vz += kk * ez
                gs = gn
                # the new state's transitions at the same point
                g, _, _, _ = beam(x, y, z, w0, zR); nt = 0
                for b2 in range(nb_):
                    kv = beams[b2, 0] * vx + beams[b2, 1] * vy + (0.0 if variant == 2 else beams[b2, 2] * vz)
                    for e2 in range(ne):
                        if S[gs, e2] <= 0.0 or me[e2] - mg[gs] != int(beams[b2, 3]): continue
                        d = twopi * (beams[b2, 5] - (Ee[e2] - Eg[gs]) - (Ze[e2] - Zg[gs]) - (LSe[e2] - LSg[gs]) * g) - beams[b2, 6] * kv
                        s = beams[b2, 4] * S[gs, e2]
                        rates[nt] = 0.5 * gamma * s / (1.0 + s + 4.0 * d * d / (gamma * gamma)); tb[nt] = b2; te[nt] = e2; nt += 1
            if variant == 1: vz *= math.exp(-dt / 1.0)
            g, _, _, _ = beam(x, y, z, w0, zR)
            E = 0.5 * (vx * vx + vy * vy + vz * vz) / hbar_m - U0 * g
            if g < 0.05 and E > 0: alive = False
    return 0

MULTI = {'S2QR': (0, 'S2Q, multilevel rate model (Doppler, optical pumping, Zeeman, light shifts; no ground coherences)'),
         'S2QZ': (1, 'S2Q rate model with z frozen: the best any z-only cooling (sub-Doppler included) can do'),
         'S2QV': (2, 'S2Q rate model without z Doppler shifts: z friction cancelled'),
         'S2R': (0, 'S2 by the multilevel rate model (independent beams): checks S2K and gives the depumping')}

def scene_beams(name, p):
    """The beams of S2Q / S2 for the multilevel model: probe +z σ+, reflection −z (σ− about z with the λ/4 plate, σ+
    without), repump +x π at F=1 → F'=2 (6568.03 MHz above F=2 → F'=3), every one at s = I/I_sat(cycling)."""
    rep = HFS_EXCITED[2] - HFS_GROUND[1]
    q_back = -1 if name.startswith('S2Q') else 1
    return np.array([[0, 0, 1, 1, p['s1'], 0.0, p['kp']], [0, 0, -1, q_back, p['s1'], 0.0, p['kp']],
                     [1, 0, 0, 0, p['s1'], rep, p['kp'] * (1 + rep / 384230484.4685)]], dtype=np.float64)

def simulate_multi(name, p, mt, atoms, end_us, every_us, dt, seed=12345):
    variant, desc = MULTI[name]
    n_steps = int(round(end_us / dt)); every = int(round(every_us / dt)); n_out = n_steps // every + 1
    out_v = np.zeros((n_out, atoms, 3)); out_alive = np.zeros((n_out, atoms), np.bool_); out_g = np.zeros((n_out, atoms), np.int64)
    P = np.array([p['gamma'], p['w0'], p['zR'], p['U0'], p['hbar_m'], p['uK']])
    t0 = time.time()
    run_multi(variant, atoms, seed, 30.0, dt, n_steps, every, P, scene_beams(name, p), mt['S'], mt['B'], mt['Eg'], mt['Ee'], mt['Zg'], mt['Ze'],
              mt['LSg'], mt['LSe'], mt['mg'], mt['me'], GROUND.index((2, 2)), out_v, out_alive, out_g)
    r = summarize(name, desc, p, atoms, dt, time.time() - t0, np.arange(n_out) * every * dt, out_v, out_alive)
    pops = np.zeros((n_out, len(GROUND)))
    for k in range(n_out):
        held = out_alive[k]
        if held.sum(): pops[k] = np.bincount(out_g[k][held], minlength=len(GROUND)) / held.sum()
    r['groundStates'] = [f'ground-{F}-{m}' for (F, m) in GROUND]; r['groundPopulations'] = pops.round(4).tolist()
    return r

def depumping(p, mt):
    """S2 (σ+σ+ and the π repump) at the trap's centre, the atom at rest in |2,2⟩: the rate of every way out of the cycle
    (1/s), per excited state and final state, from the rate tables; plus the closed-form other paths."""
    G = p['gamma']; beams = scene_beams('S2R', p); gs = GROUND.index((2, 2)); out = []
    for b in range(3):
        for e, (Fp, mp) in enumerate(EXCITED):
            if mt['S'][gs, e] <= 0 or mp - 2 != int(beams[b, 3]): continue
            d = 2 * math.pi * (beams[b, 5] - (mt['Ee'][e] - mt['Eg'][gs]) - (mt['Ze'][e] - mt['Zg'][gs]) - (mt['LSe'][e] - mt['LSg'][gs]))
            s = beams[b, 4] * mt['S'][gs, e]; R = 0.5 * G * s / (1 + s + 4 * d * d / G ** 2) * 1e6
            for g2, (F2, m2) in enumerate(GROUND):
                if (F2, m2) != (2, 2) and mt['B'][g2, e] > 0:
                    out.append(dict(beam=['probe', 'reflection', 'repump'][b], excited=f'D2-{Fp}-{mp}', final=f'ground-{F2}-{m2}', detuningMHz=d / 2 / math.pi, ratePerS=R * mt['B'][g2, e]))
    return out

def summarize(name, desc, p, atoms, dt, wall, times, out_v, out_alive):
    n_out = len(times); held = out_alive.sum(1); loss = 1 - held / atoms
    T = np.full((n_out, 3), np.nan); se = np.full((n_out, 3), np.nan)
    for k in range(n_out):
        v = out_v[k][out_alive[k]]
        if len(v) > 10:
            var = v.var(0, ddof=1); T[k] = var / p['hbar_m'] / p['uK']
            m4 = ((v - v.mean(0)) ** 4).mean(0); se[k] = np.sqrt(np.maximum(m4 - var ** 2 * (len(v) - 3) / (len(v) - 1), 0) / len(v)) / p['hbar_m'] / p['uK']
    Ttot = T.mean(1); setot = np.sqrt((se ** 2).sum(1)) / 3
    f = lambda arr: [None if not np.isfinite(x) else round(float(x), 3) for x in arr]
    return dict(model=name, description=desc, atoms=atoms, dtUs=dt, wallS=wall, timesUs=times.tolist(), loss=loss.tolist(),
                lossSE=np.sqrt(np.maximum(loss * (1 - loss), 1 / atoms) / atoms).tolist(),
                T=[f(T[:, a]) for a in range(3)], TSE=[f(se[:, a]) for a in range(3)], Ttotal=f(Ttot), TtotalSE=f(setot))

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
    ap.add_argument('--atoms', type=int, default=4000); ap.add_argument('--end', type=float, default=5000.0)
    ap.add_argument('--every', type=float, default=50.0); ap.add_argument('--models', default='S1K,S1F,S2K,S2F,S2R,S2QR,S2QZ,S2QV')
    ap.add_argument('--dt', type=float, default=0.05); ap.add_argument('--out', default=os.path.join(HERE, 'model.json'))
    ap.add_argument('--summary', action='store_true')
    ap.add_argument('--seed', type=int, default=12345)
    ap.add_argument('--keep', action='store_true', help='keep the other models\' runs already in --out')
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
    mt = multilevel_tables(shifts)
    # the tables' checks: F' → F=1 branching (Steck: F'=2 1/2, F'=1 5/6, F'=0 1), |3',3'⟩'s shift, the cycling Zeeman shift
    toF1 = {Fp: sum(mt['B'][g, EXCITED.index((Fp, 0))] for g, (F, m) in enumerate(GROUND) if F == 1) for Fp in (0, 1, 2, 3)}
    print('  multilevel: branching to F=1 from F\' = 0..3 (m\'=0):', {k: round(v, 6) for k, v in toF1.items()},
          "|3',3'⟩ shift", round(mt['LSe'][EXCITED.index((3, 3))], 4), 'cycling Zeeman', round(mt['Ze'][EXCITED.index((3, 3))] - mt['Zg'][GROUND.index((2, 2))], 4))
    print('  excited shifts at the focus (MHz):', {f'D2-{Fp}-{m}': round(float(v), 3) for (Fp, m), v in zip(EXCITED, mt['LSe']) if Fp >= 2})
    dep = depumping(p, mt)
    for d in sorted(dep, key=lambda d: -d['ratePerS'])[:6]: print('  depumping S2:', {k: (round(v, 3) if isinstance(v, float) else v) for k, v in d.items()})
    if a.summary or a.selftest: return
    p['s1'] *= a.s_scale
    models = [m for m in a.models.split(',') if m]
    result = dict(generator='motion/refs/sanity/model.py', lightShifts={k: v for k, v in shifts.items() if not isinstance(v, dict)},
                  chart=CHART, parameters={k: v for k, v in p.items()}, analytic=an,
                  selftest=[{k: float(v) for k, v in t.items()} for t in tests],
                  multilevel=dict(branchingToF1=toF1, excitedShiftsMHz={f'D2-{Fp}-{m}': float(v) for (Fp, m), v in zip(EXCITED, mt['LSe'])},
                                  zeemanMHz={**{f'ground-{F}-{m}': float(v) for (F, m), v in zip(GROUND, mt['Zg'])}, **{f'D2-{Fp}-{m}': float(v) for (Fp, m), v in zip(EXCITED, mt['Ze'])}},
                                  depumpingS2=dep), runs={})
    if a.keep and os.path.exists(a.out):
        old = json.load(open(a.out)); result['runs'] = {k: v for k, v in old['runs'].items() if k not in models}
    for m in models:
        r = simulate_multi(m, p, mt, a.atoms, a.end, a.every, a.dt, a.seed) if m in MULTI else simulate(m, p, a.atoms, a.end, a.every, a.dt, a.seed)
        print(f"{m}: {r['wallS']:.1f} s; " + '; '.join(f"t {r['timesUs'][k]/1e3:.1f} ms T {r['Ttotal'][k]} ({'/'.join(str(r['T'][i][k]) for i in range(3))}) L {r['loss'][k]:.3f}"
                                               for k in range(0, len(r['timesUs']), max(1, len(r['timesUs']) // 10))))
        result['runs'][m] = r
    json.dump(result, open(a.out, 'w'))
    print('wrote', a.out)

if __name__ == '__main__':
    main()
