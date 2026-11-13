"""The oscillation z = A cos(ωt + φ) at the trap's frequency in the scene's own tweezer: the energy's drift A ω ⟨w₁F⟩ and
the diffusion's heating (ħ/m)(R₁ + R₂)/2 (R_i the variance rates of ∫w_i dκ, w₁ = −sin, w₂ = cos), exact against engine."""
import sys, time, json, numpy as np, ref
from common import *
amps = [float(x) for x in sys.argv[1].split(',')]
count, periods, tau = int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4])
waist = float(sys.argv[5]) if len(sys.argv) > 5 else 0
T = 2 * np.pi / OMEGA
rows = []
print('A | exact: drift power, heating = local + slow, net | engine: drift power ± se (steps, jumps), heating ± se = Var, net | R1, R2 exact; engine | F=1 exact; engine | jumps/µs')
for A in amps:
    t0 = time.time()
    m, p, w1, w2 = oscillation('closed', waist, A)
    o1, o2 = p.moments(w1), p.moments(w2)
    ex = {'A': A, 'power': A * OMEGA * o1['drift'], 'local': HM / 2 * (o1['local'] + o2['local']), 'slow': HM / 2 * (o1['slow'] + o2['slow']), 'R1': o1['rate'], 'R2': o2['rate'],
          'f1': p.population(np.array([n.startswith('ground-1') for n in m.names])), 'band': p.band, 'tail': p.tail}
    a, d = engine(f'osc:{A}:{OMEGA}:0', waist, tau, count, 15 * T, periods * T, periods * T, f'o{A}_w{waist}')
    e1, e2 = engine_rates(a, d, 1), engine_rates(a, d, 2)
    en = {'power': A * OMEGA * e1['drift'], 'power_se': A * OMEGA * e1['drift_se'], 'steps': A * OMEGA * e1['steps'], 'jumps': A * OMEGA * e1['jumps'],
          'var': HM / 2 * (e1['var'] + e2['var']), 'var_se': HM / 2 * np.hypot(e1['var_se'], e2['var_se']),
          'R1': e1['var'], 'R2': e2['var'], 'f1': e1['other'], 'rate': e1['jump_rate']}
    rows.append({'exact': ex, 'engine': en})
    exh, enh = ex['local'] + ex['slow'], en['var']
    print('%.3f | %9.6f  %8.6f = %8.6f + %9.6f  net %9.6f | %9.6f ± %.6f (%9.6f %9.6f)  %8.6f ± %.6f  net %9.6f | %7.3f %7.3f; %7.3f %7.3f | %.4f; %.4f | %.4f  (band %d, tail %.0e, %.0fs)' % (
        A, ex['power'], exh, ex['local'], ex['slow'], ex['power'] + exh, en['power'], en['power_se'], en['steps'], en['jumps'], enh, en['var_se'], en['power'] + enh,
        ex['R1'], ex['R2'], en['R1'], en['R2'], ex['f1'], en['f1'], en['rate'], ex['band'], ex['tail'], time.time() - t0), flush=True)
json.dump(rows, open(sys.argv[6] if len(sys.argv) > 6 else 'osc_rows.json', 'w'))
