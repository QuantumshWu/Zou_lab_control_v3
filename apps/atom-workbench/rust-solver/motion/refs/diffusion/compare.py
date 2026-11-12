"""T_z(t): the quantum result, the phase-space equation, the engine free in 1D and 3D, and saved solver runs.
python compare.py <pde tags> <free runs> [name=saved.json ...]"""
import json, sys, os, numpy as np
from common import *
import ref, pde
exp = json.load(open(os.path.join(HERE, '..', 'expect', 'R4_TW2.json')))
q = next(s for s in exp['series'] if s['quantity'] == 'T')
qf = next(s for s in exp['series'] if s['quantity'] == 'group')
tq = 10.0 * np.array(q['indices'])
cols = {'quantum': (tq, np.array(q['values']), None)}
f1 = {'quantum': (tq, np.array(qf['values']))}
for tag in sys.argv[1].split(','):
    d = np.load(f'pde_{tag}.npz', allow_pickle=True)
    t, tk, tz, pf = pde.temperatures(d['series'])
    # the mean over the window of 10 µs that ends at each output time, as the engine's free rows
    w = [np.mean(tk[(t > x - 10 + 1e-9) & (t <= x + 1e-9)]) if x > 0 else tk[0] for x in tq if x <= t[-1] + 1e-9]
    cols[f'pde {tag}'] = (tq[:len(w)], np.array(w), None)
    f1[f'pde {tag}'] = (t, pf)
for name in sys.argv[2].split(','):
    d = json.load(open(f'{name}.json'))
    r = np.array(d['rows'])
    cols[name] = (r[:, 0], d['hbar_over_mass'] * (r[:, 1] - r[:, 3] ** 2) / ref.KT_PER_UK, d['hbar_over_mass'] * r[:, 2] / ref.KT_PER_UK)
    f1[name] = (r[:, 0], r[:, 5])
# Saved solver results (run_scene.mjs --save), name=file pairs: the worker's own T_z.
for pair in sys.argv[3:]:
    name, file = pair.split('=')
    m = json.load(open(file))['final']['result']['motion']
    cols[name] = (tq, np.array(m['temperatureUK']['z']), np.array(m['temperatureErrorUK']['z']) if 'temperatureErrorUK' in m else None)
names = list(cols)
print('t µs  ' + '  '.join('%-18s' % n for n in names) + '   (T_z µK, and its deviation from the quantum result)')
for x in [50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]:
    out = []
    ref_T = cols['quantum'][1][list(tq).index(x)]
    for n in names:
        t, v, se = cols[n]
        i = np.argmin(np.abs(t - x))
        if abs(t[i] - x) > 1e-6:
            out.append('%-18s' % '')
            continue
        out.append('%-18s' % ('%6.2f %+5.1f%%%s' % (v[i], 100 * (v[i] / ref_T - 1), '' if se is None else ' ±%.2f' % se[i])))
    print('%5d  ' % x + '  '.join(out))
print('F=1 share at 100, 300, 500 µs: ' + '; '.join('%s %s' % (n, ' '.join('%.4f' % v[np.argmin(np.abs(t - x))] for x in (100, 300, 500) if t[-1] >= x)) for n, (t, v) in f1.items()))
