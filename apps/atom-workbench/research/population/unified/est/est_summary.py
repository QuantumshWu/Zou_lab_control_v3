"""usage: python est_summary.py OUTPUT.jsonl ...

Summarize the Rust estimator harness output (JSON lines): per case and checkpoint the ratio estimate/actual
[5% 50% 95%] min, R2's Python ratio and the Rust-Python deviation, steady est/actual, and the per-column work."""
import json, sys
from collections import defaultdict

rows = defaultdict(list)
heads = {}
for path in sys.argv[1:]:
    for line in open(path):
        if not line.startswith('{'):
            continue
        r = json.loads(line)
        (heads if 'buildMs' in r else rows)[r['case']] = r if 'buildMs' in r else rows[r['case']] + [r]
for case, h in heads.items():
    print(f"== {case}: modes {h['modes']} ring {h['ring']} classes {h['classes']} build {h['buildMs'] / 1e3:.1f}s (kernels {h['kernelMs'] / 1e3:.2f}s, "
          f"classes {h['classMs'] / 1e3:.1f}s, eigenbasis {h['eigenbasisMs'] / 1e3:.2f}s) operator dev {h['operatorDeviation']:.1e} "
          f"entries {h['kernelEntries']} |X|max {[round(x, 3) for x in h['smallParameter']]}")
    n = h['modes']
    for r in rows[case]:
        q = [float('nan') if v is None else v for v in r.get('quantiles', [None] * 4)]
        py = r.get('python', {}).get('ent0.005_k0')
        pys = f" | py {py['ratio']:.2f} dev {py['deviation']:.0e}" if py else ''
        st = ''
        if r.get('steadyActual'):
            st = f" | ss {r['steadyEst']:.2e}/{r['steadyActual']:.2e} ({r['steadyEst'] / r['steadyActual']:.2f})"
        hz = ''
        if r.get('horizons') and r.get('horizonActual'):
            ha = [r['horizonActual'][k] for k in sorted(r['horizonActual'], key=float)]
            hz = ' | h ' + ' '.join(f'{e:.1e}/{a:.1e}' for e, a in zip(r['horizons'], ha))
        w = r['work']
        cols = max(w['columns'], 1)
        c = r.get('count', [0] * 5)
        print(f"  {r['solves']:3d} k {r['k']:4d} act {r.get('actualMax', float('nan')):.2e} ratio {(r.get('ratio') or float('nan')):.2f} "
              f"[{q[0]:.2f} {q[1]:.2f} {q[2]:.2f}] min {q[3]:.2f}{pys}{st}{hz} | per col: op {w['blockOperations'] / cols:.0f} solve {w['blockSolves'] / cols:.0f} "
              f"frame {w['frameBlocks'] / cols:.0f} ({r['columnMs'] / cols:.0f} ms) eval {r['evaluateMs']:.0f} ms steady {r['steadyMs']:.0f} ms")
