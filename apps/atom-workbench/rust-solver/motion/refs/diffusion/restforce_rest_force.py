# DIAGNOSTIC (restforce, 2026-11-12; motion3d_status/rewrite/restforce.status.md): not part of any acceptance.
"""The engine held at rest (diffusion engine ... rest:s) -- its mean force along the axis read three ways: the kick and the
jumps alone (columns 0 + 1, as rest_engine.py), the reference flow's part (column 8, DIAGNOSTIC), the whole step (column 5).
python rest_force.py <exe> <request> <s,s,...> <count> <burn> <span> <tau> <tag>   (env AXIS, TABLES passed through)"""
import sys, json, os, subprocess, numpy as np
exe, req, ss, count, burn, span, tau, tag = sys.argv[1], sys.argv[2], [float(x) for x in sys.argv[3].split(',')], int(sys.argv[4]), float(sys.argv[5]), float(sys.argv[6]), float(sys.argv[7]), sys.argv[8]
G = 'C:/Users/eadri/AppData/Local/Temp/claude/c0815e347c54866c/8308d66c-699b-4483-ab9b-7305d93a9f7e/scratchpad/rewrite/guard.sh'
os.makedirs('out', exist_ok=True)
rows = []
for s in ss:
    f = os.path.abspath(f'out/f_{tag}_{s}.json')
    env = dict(os.environ, BURN=str(burn))
    p = subprocess.run(['bash', G, '-MaxSeconds', '290', exe, 'engine', req, '0', f'rest:{s}', str(tau), str(count), str(burn), str(span), str(span), f], capture_output=True, text=True, env=env)
    if p.returncode != 0:
        print('failed', s, p.returncode, p.stderr[-2000:]); continue
    d = json.load(open(f)); os.remove(f)
    a = np.array(d['data']).reshape(d['count'], d['samples'], d['width'])
    inc = a[:, -1, :] - a[:, 0, :] if d['samples'] > 1 else a[:, 0, :]
    n = inc.shape[0]
    m = lambda c: (inc[:, c].mean() / span, inc[:, c].std(ddof=1) / np.sqrt(n) / span)
    kick, jump, whole, flow = m(0), m(1), m(5), m(8)
    kj = inc[:, 0] + inc[:, 1]
    row = dict(s=s, kick_jumps=kj.mean() / span, kick_jumps_se=kj.std(ddof=1) / np.sqrt(n) / span, kick=kick, jumps=jump, flow=flow, whole=whole, jumprate=inc[:, 9].mean() / span, other=a[:, -1, 10].mean(), stderr=p.stderr.strip().splitlines()[-1] if p.stderr.strip() else '')
    rows.append(row)
    print(f"s {s:.4f}: kick+jumps {row['kick_jumps']:8.2f} ± {row['kick_jumps_se']:.2f} | flow {flow[0]:8.2f} ± {flow[1]:.3f} | whole {whole[0]:8.2f} ± {whole[1]:.2f} | kick {kick[0]:.2f} jumps {jump[0]:.2f} | jumps/µs {row['jumprate']:.3f} | F=1 share at end {row['other']:.3f} | {row['stderr']}", flush=True)
json.dump(rows, open(f'out/force_{tag}.json', 'w'), indent=1)
