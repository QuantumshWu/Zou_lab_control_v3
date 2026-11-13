"""Booked ≥ actual at rest (motion3d_status/rewrite/restforce.status.md): at random points of a path through a scene (AXIS,
ORIGIN as `diffusion` takes them), what the closed model books of its units' errors where the atom is held (`diffusion
engine ... rest:s` with BOOKED=<file>: each unit's mean error over the held states, the means summed as the solve sums
them) against what its folds actually change -- the closed model's exact steady state at rest against the explicit
model's, every near-resonant drive explicit (`diffusion dump closed|explicit`, ref.py `at_rest`) --: the largest
|p_closed − p_explicit| over the ground states and over the hyperfine levels' totals, which the booking must bound.
  python restforce_booked.py <exe> <request> <points> <seed> [count burn span tau]
Exits with 1 where some point's booking is below what the folds change."""
import sys, json, os, subprocess, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ref
exe, req, npts, seed = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
count, burn, span, tau = (sys.argv[5:9] if len(sys.argv) > 8 else ['32', '200', '100', '0.3'])
G = 'C:/Users/eadri/AppData/Local/Temp/claude/c0815e347c54866c/8308d66c-699b-4483-ab9b-7305d93a9f7e/scratchpad/rewrite/guard.sh'
os.makedirs('out', exist_ok=True)
here = lambda name: os.path.abspath(os.path.join('out', name))
s_all = np.sort(np.random.default_rng(seed).uniform(0.0, 0.4, npts))
json.dump([[float(s), 0.0] for s in s_all], open(here('booked_points.json'), 'w'))


def run(args, env=None):
    p = subprocess.run(['bash', G, '-MaxSeconds', '290', exe] + args, capture_output=True, text=True, env={**os.environ, **(env or {})})
    assert p.returncode == 0, p.stderr[-2000:]


def steady(kind):
    run(['dump', req, '0', kind, here('booked_points.json'), here(f'booked_{kind}.json')])
    m = ref.Model(here(f'booked_{kind}.json'))
    out = []
    for r in ref.at_rest(m):
        p = np.real(np.diag(m.mat(r['rho'])))
        out.append({n: p[i] for i, n in enumerate(m.names)})
    os.remove(here(f'booked_{kind}.json'))
    return out


closed, explicit = steady('closed'), steady('explicit')
failed = False
print('s | booked populations (T) | actual: largest |Δp| of a ground state, of a level (F=1, F=2 totals) | booked/actual')
for s, pc, pe in zip(s_all, closed, explicit):
    ground = [n for n in pe if n.startswith('ground')]
    states = max(abs(pc.get(n, 0.0) - pe[n]) for n in ground)
    levels = max(abs(sum(pc.get(n, 0.0) for n in ground if n.startswith(f'ground-{f}-')) - sum(pe[n] for n in ground if n.startswith(f'ground-{f}-'))) for f in (1, 2))
    actual = max(states, levels)
    run(['engine', req, '0', f'rest:{s}', tau, count, burn, span, '10', here('booked_engine.json')], env={'BOOKED': here('booked_errors.json')})
    b = json.load(open(here('booked_errors.json')))
    os.remove(here('booked_errors.json'))
    os.remove(here('booked_engine.json'))
    ok = b['population'] >= actual
    failed |= not ok
    print('%.4f | %.4f (%.4f) | %.4f, %.4f | %.2f %s' % (s, b['population'], b['temperature'], states, levels, b['population'] / actual if actual > 0 else float('inf'), '' if ok else 'BELOW'), flush=True)
os.remove(here('booked_points.json'))
sys.exit(1 if failed else 0)
