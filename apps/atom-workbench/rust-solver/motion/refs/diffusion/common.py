"""What the diffusion references share (motion3d_status/rewrite/diffusion.status.md): the example binary's modes as
functions, and R4_TW2's constants. Run the scripts from a scratch directory (they write their dumps and results there):
  cargo build -p atom-motion --example diffusion --release --target x86_64-pc-windows-msvc
  python <this dir>/uniform_ref.py closed 0.1,0.05,0.02          exact F(v), 2D(v), uniform configuration
  python <this dir>/uniform_engine.py 0.1,0.05,0.02 8192 400 0.3125   the engine against them
  python <this dir>/osc.py 0.068,0.159,0.256 4096 60 0.3125      the trap's oscillation: drift and the two quadratures
  python <this dir>/pde.py b 500 30 128 64 0.1                   the phase-space equation, second order (repeat to go on)
  python <this dir>/pde.py x 450 30 256 64 0.1 exact             the master equation in the position representation
  python <this dir>/compare.py b,x free_1d,free_3d               T_z(t) against the quantum result
  python <this dir>/identity.py 0.068 2048 40                    the path identity from the engine's own states
The two references of the set (expect/R4D_FORCE.json, judged; expect/R4D_TW2.json, reported): node ../r4d.mjs force | tz
(`free_1d.json`: `diffusion free R4_TW2 1d 30 0.3125 8192 1000 10 free_1d.json`.)
The transverse part (`diffusion account`: every change of an axis's kinetic energy booked to its part):
  diffusion account R4_TW2 1d|free:yz|free:xz|3d 30 0.3125 8192 1000 10 acc_<run>.json   (TEMPS=x,y,z: start temperatures)
  python <this dir>/account.py 500,1000 acc_yz.json      the parts;  atoms.py: per atom;  flow.py, split.py: against 1D
  TEMPS=30,30,10 diffusion account R1_TWEEZER free:yz 30 0.3125 8192 1000 10 off_yz.json;  python <this dir>/bare.py
                                                         the bare exchange against the classical motion (toy2d.py)
  python <this dir>/old_worker.py <old atom-lindblad.exe> old.json   the old engine's own worker on R4_TW2"""
import subprocess, json, os, sys, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ref
EX = os.path.normpath(os.path.join(HERE, '..', '..', '..', 'target', 'x86_64-pc-windows-msvc', 'release', 'examples', 'diffusion.exe'))
K = 8.05282506255877          # the cooling light's wave number (1/µm)
LAM = 2 * np.pi / K
HM = 0.0007307375228700508    # ħ/m (µm²/µs)
OMEGA = 0.6317695417004418    # the trap's angular frequency along z (rad/µs)
WIDE = 2000                   # the uniform tweezer's waist (µm)

def dump(kind, waist, z, kappa, tag):
    pts, out = f'pts_{tag}.json', f'dump_{tag}.json'
    ref.write_points(pts, z, kappa)
    subprocess.run([EX, 'dump', 'R4_TW2', str(waist), kind, pts, out], check=True, stderr=subprocess.DEVNULL)
    m = ref.Model(out)
    os.remove(out); os.remove(pts)
    return m

# The uniform tweezer's residual variation over a wavelength (10⁻⁵ of the light shifts) is not a harmonic of the path: the
# path's harmonics of 𝓛 below this share of the largest are left out, and the exact rates are uncertain by it.
TOL = 1e-4

def constant(kind, v, n=32, harmonics=None):
    """The uniform configuration at the constant velocity v: (model, Periodic)."""
    z = -LAM / 2 + LAM * np.arange(n) / n
    m = dump(kind, WIDE, z, v / HM, f'{kind}_v{v}')
    return m, ref.Periodic(m, K * v, harmonics or max(16, int(6 / (K * v))), tol=TOL)

def oscillation(kind, waist, amplitude, omega=OMEGA, centre=0.0, n=128, harmonics=48):
    """The oscillation z = centre + A cos(ωt): (model, Periodic, the weights −sin ωt and cos ωt at the samples)."""
    t = 2 * np.pi / omega * np.arange(n) / n
    z = centre + amplitude * np.cos(omega * t)
    v = -amplitude * omega * np.sin(omega * t)
    m = dump(kind, waist, z, v / HM, f'{kind}_a{amplitude}_w{waist}')
    return m, ref.Periodic(m, omega, harmonics), -np.sin(omega * t), np.cos(omega * t)

def engine(course, waist, tau, count, burn, span, every, tag, env=None):
    out = f'eng_{tag}.json'
    r = subprocess.run([EX, 'engine', 'R4_TW2', str(waist), course, str(tau), str(count), str(burn), str(span), str(every), out], check=True, capture_output=True, text=True, env={**os.environ, **(env or {})})
    print(r.stderr.strip().splitlines()[-1])
    d = json.load(open(out))
    return np.array(d['data']).reshape(d['count'], d['samples'], d['width']), d

def engine_rates(a, d, which=0):
    """From the engine's samples (`diffusion engine`, each step's own decomposition, step.rs `Step::momentum`): the drift
    and the variance's rate of the increments weighted by weight `which` (0: 1; 1, 2: the course's), with their standard
    errors: (drift, se, the steps' part -- the kick at the middle with the reference's flow --, the jumps' part, the flow's
    part, variance rate of the whole steps over the span, its se, the steps' and the jumps' variances and their
    covariance, jumps per µs)."""
    assert d.get('columns') == ['whole', 'jumps', 'flow'], 'a worker before the whole-step columns: rebuild the diffusion example'
    T = d['span']
    inc = a[:, -1, :] - a[:, 0, :]
    tot, jmp, flow = inc[:, 3 * which], inc[:, 3 * which + 1], inc[:, 3 * which + 2]
    det = tot - jmp
    n = len(tot)
    var = tot.var(ddof=1)
    return {'drift': tot.mean() / T, 'drift_se': tot.std(ddof=1) / np.sqrt(n) / T, 'steps': det.mean() / T, 'jumps': jmp.mean() / T, 'flow': flow.mean() / T,
            'var': var / T, 'var_se': var * np.sqrt(2 / (n - 1)) / T, 'var_steps': det.var(ddof=1) / T, 'var_jumps': jmp.var(ddof=1) / T,
            'cov': 2 * np.cov(det, jmp)[0, 1] / T, 'jump_rate': inc[:, 9].mean() / T, 'other': a[:, :, 10].mean()}
