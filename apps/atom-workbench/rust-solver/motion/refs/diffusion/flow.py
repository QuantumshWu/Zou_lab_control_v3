import json, sys, numpy as np
KT = 1.380649e-23 / 1.054571817e-34 * 1e-12
u = 2 / KT
base = json.load(open('acc_1d.json'))
def at(d, x):
    rows = d['rows']; t = np.array([r['t'] for r in rows]); i = int(np.argmin(abs(t - x)))
    return np.array(rows[i]['means']) * u, np.array(rows[i]['errors']) * u
n = base['parts']
print('y start | at 500 µs and 1 ms: T_y, T_z ± se | z: flow beyond −½ light, less the 1D run\'s | y: flow beyond −½ light | z light work less 1D\'s')
for f in sys.argv[1:]:
    d = json.load(open(f)); out = []
    for x in (500, 1000):
        m, e = at(d, x); b, _ = at(base, x)
        wz, wy, wb = m[3 + 2 * n:3 + 3 * n], m[3 + n:3 + 2 * n], b[3 + 2 * n:3 + 3 * n]
        zt = (wz[4] + 0.5 * wz[:4].sum()) - (wb[4] + 0.5 * wb[:4].sum())
        yt = wy[4] + 0.5 * wy[:4].sum()
        out.append('%6.2f %6.2f ±%.2f | z %+6.2f | y %+6.2f | light %+6.2f' % (m[1], m[2], e[2], zt, yt, wz[:4].sum() - wb[:4].sum()))
    print('%6.1f  ' % d['temperatures'][1] + '   ||   '.join(out))
