import sys, time, numpy as np, ref
from common import *
kind = sys.argv[1]
print(f'{kind}: v, F, beta = -F hm/v, 2D, local, slow, hm D/beta (µK), band, tail')
for v in [float(x) for x in sys.argv[2].split(',')]:
    t = time.time()
    m, p = constant(kind, v)
    o = p.moments()
    beta = -o['drift'] * HM / v
    print('%7.4f %10.5f %9.5f %9.3f %8.3f %9.3f %8.2f  band %d tail %.0e (%.0fs)' % (v, o['drift'], beta, o['rate'], o['local'], o['slow'], HM * o['rate'] / 2 / beta / ref.KT_PER_UK, p.band, p.tail, time.time() - t), flush=True)
