"""The engine's mean force and momentum diffusion at constant velocities in the uniform configuration, against the
master equation of its own model: python uniform_engine.py <v,...> <trajectories> <span µs> <τ µs> [TABLES 1|0] [rows.json]
(rows.json: per velocity the exact mean force, its numerical uncertainty, the engine's and its standard error)."""
import sys, time, json, numpy as np, ref
from common import *
count, span, tau = int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
env = {'TABLES': sys.argv[5]} if len(sys.argv) > 5 else None
rows = []
print('v | exact F, engine F ± se (steps, jumps) | exact 2D = local + slow | engine 2D ± se = Brownian + Var(steps+jumps) [Var steps, Var jumps, 2Cov] | jumps/µs, F=1 share (engine; exact)')
for v in [float(x) for x in sys.argv[1].split(',')]:
    m, p = constant('closed', v)
    o = p.moments()
    # A whole number of the light's periods: the periodic part of the force then adds the same to every start phase.
    whole = max(1, round(span * v / LAM)) * LAM / v
    a, d = engine(f'const:{v}', WIDE, tau, count, 150.0, whole, whole, f'c{v}', env)
    e = engine_rates(a, d)
    f1 = p.population(np.array([n.startswith('ground-1') for n in m.names]))
    rows.append({'v': v, 'exact': o['drift'], 'u': abs(o['drift']) * (TOL + p.tail), 'engine': e['drift'], 'se': e['drift_se'], 'trajectories': count, 'spanUs': whole,
                 'exact2D': o['rate'], 'engine2D': e['var'] + e['brownian'], 'engine2Dse': e['var_se']})
    print('%6.3f | %9.5f %9.5f ± %.5f (%9.5f %9.5f) | %7.3f = %6.3f + %7.3f | %7.3f ± %.3f = %6.3f + %7.3f [%7.3f %6.3f %6.3f] | %.4f %.4f; %.4f' % (
        v, o['drift'], e['drift'], e['drift_se'], e['steps'], e['jumps'], o['rate'], o['local'], o['slow'], e['var'] + e['brownian'], e['var_se'], e['brownian'], e['var'], e['var_steps'], e['var_jumps'], e['cov'], e['jump_rate'], e['other'], f1), flush=True)
if len(sys.argv) > 6:
    json.dump(rows, open(sys.argv[6], 'w'))
