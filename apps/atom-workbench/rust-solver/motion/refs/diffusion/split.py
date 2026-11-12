"""The transverse part split (`diffusion account` runs against the 1D one): T_z less the 1D run's, and its two parts --
the exchange, what the trap's flow moved out of z beyond turning half of the light's work back (less the same of the
1D run), and the light, what the light's own work on z changed (half of it stays in T_z, the flow turns the other half
into potential energy); for the released tight axis what its flow brought beyond the recoil. µK of T = 2K/k_B at 1 ms,
with standard errors over the atoms.
python split.py <1d.json> <run.json> ..."""
import json, sys, numpy as np
KT = 1.380649e-23 / 1.054571817e-34 * 1e-12
u = 2 / KT
def load(f):
    d = json.load(open(f)); a = np.array(d['atoms']) * u; last = np.array(d['rows'][-1]['means']) * u; err = np.array(d['rows'][-1]['errors']) * u
    return d, a, last, err
def mean(x):
    return x.mean(), x.std(ddof=1) / np.sqrt(len(x))
d0, a0, l0, e0 = load(sys.argv[1])
ex0, li0 = mean(a0[:, 5] + 0.5 * a0[:, 4]), mean(a0[:, 4])
print('1D: T_z %.2f ± %.2f; light on z %+.2f ± %.2f; flow beyond −½ light %+.2f ± %.2f' % (l0[2], e0[2], *li0, *ex0))
print('run | T_z − 1D | exchange part | light part (half the change of the light\'s work) | sum | released axes: T, light\'s work, brought by the flow beyond −½ light')
for f in sys.argv[2:]:
    d, a, l, e = load(f)
    ex, li = mean(a[:, 5] + 0.5 * a[:, 4]), mean(a[:, 4])
    dT = (l[2] - l0[2], np.hypot(e[2], e0[2]))
    exch = (ex[0] - ex0[0], np.hypot(ex[1], ex0[1]))
    light = (0.5 * (li[0] - li0[0]), 0.5 * np.hypot(li[1], li0[1]))
    n = d['parts']
    held = d['dimension']
    extra = []
    for axis, name in ((0, 'x'), (1, 'y')):
        if l[axis] > 0:
            w = l[3 + axis * n:3 + (axis + 1) * n]
            if axis == 1:
                fl = mean(a[:, 3] + 0.5 * a[:, 2])
                extra.append('%s: T %.2f, light %+.2f, flow brought %+.2f ± %.2f' % (name, l[axis], w[:4].sum(), *fl))
            else:
                extra.append('%s: T %.2f, light %+.2f, flow beyond −½ light %+.2f' % (name, l[axis], w[:4].sum(), w[4] + 0.5 * w[:4].sum()))
    print('%s (start %s) | %+.2f ± %.2f | %+.2f ± %.2f | %+.2f ± %.2f | %+.2f | %s' % (held, d['temperatures'], *dT, *exch, *light, exch[0] + light[0], '; '.join(extra)))
