"""The account of an axis's kinetic energy (`diffusion account`): per run, at the times asked, the kinetic temperature
T_a = 2K_a/k_B and the work of each part of the engine on that axis since t = 0, in µK of T_a (2W/k_B).
python account.py <t1,t2,...> <run.json> [...]"""
import json, sys, numpy as np
KT = 1.380649e-23 / 1.054571817e-34 * 1e-12
PARTS = ['Ehrenfest', 'scattered', 'Brownian', 'jumps', 'flow']
times = [float(x) for x in sys.argv[1].split(',')]
for file in sys.argv[2:]:
    d = json.load(open(file))
    rows, n = d['rows'], d['parts']
    t = np.array([r['t'] for r in rows])
    m = np.array([r['means'] for r in rows])
    e = np.array([r['errors'] for r in rows])
    u = 2 / KT
    print(f"{file}: {d['dimension']}, start {d['temperatures']} µK, tau {d['tau']}, {d['count']} atoms")
    first = m[0]
    for x in times:
        i = int(np.argmin(np.abs(t - x)))
        for a, axis in ((1, 'y'), (2, 'z')):
            w = m[i, 3 + a * n:3 + (a + 1) * n] * u
            print('  t %5.0f  T_%s %6.2f ± %.2f | work since 0: %s | all %+7.2f | light (first four) %+7.2f, flow %+7.2f' % (
                t[i], axis, m[i, a] * u, e[i, a] * u, '  '.join('%s %+7.2f' % (PARTS[k], w[k]) for k in range(n)), w.sum(), w[:4].sum(), w[4]))
        mo = m[i, 4 + 3 * n:10 + 3 * n] / t[i]
        print('  t %5.0f  ⟨L_x²⟩ %.4g | per µs: jumps ΣΔκΔκ yy %.3f zz %.3f yz %+.3f; Brownian ∫ΛΛᵀ yy %.4f zz %.3f yz %+.4f; jumps %.4f' % (t[i], m[i, 3 + 3 * n], *mo, m[i, 10 + 3 * n] / t[i]))
