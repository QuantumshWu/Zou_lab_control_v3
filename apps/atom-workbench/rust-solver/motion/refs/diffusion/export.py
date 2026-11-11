"""T_z(t) of the phase-space runs as the references keep it (refs/data/R4D_TW2.json): python export.py <out.json> from the
directory that holds pde_b.npz (the second-order equation) and pde_x.npz (the master equation in the position
representation), the two runs of common.py's header. Per output time of R4_TW2 (every 10 µs) the mean of the kinetic
T over the 10 µs that end there, as `diffusion free` gives the engine's rows."""
import json, sys, numpy as np
import pde
out = {'note': 'pde.py b (128 x 64, dt 0.1) and pde.py x (256 x 64, dt 0.1, exact) of R4_TW2 in 1D, 30 µK; the closed model of the line at 0491ae13', 'everyUs': 10.0}
for tag, name in (('b', 'secondOrder'), ('x', 'quantum')):
    d = np.load(f'pde_{tag}.npz', allow_pickle=True)
    t, tk, tz, pf = pde.temperatures(d['series'])
    times = np.arange(0.0, min(t[-1], 1000.0) + 1e-9, 10.0)
    out[name] = {'timesUs': times.tolist(), 'TzUK': [float(np.mean(tk[(t > x - 10 + 1e-9) & (t <= x + 1e-9)])) if x > 0 else float(tk[0]) for x in times],
                 'F1': [float(pf[np.argmin(np.abs(t - x))]) for x in times]}
json.dump(out, open(sys.argv[1], 'w'))
print({k: (len(v['timesUs']), v['TzUK'][10], v['TzUK'][50], v['TzUK'][-1]) for k, v in out.items() if isinstance(v, dict)})
