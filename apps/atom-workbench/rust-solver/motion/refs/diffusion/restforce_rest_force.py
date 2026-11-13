"""The engine held at rest (`diffusion engine ... rest:s`): its mean force and momentum diffusion along the axis from each
step's own decomposition (step.rs `Step::momentum`; motion3d_status/rewrite/restforce.status.md), against the exact rest
values of a reference file when one is given (refaxis.py / ref.py `at_rest` rows: s, force, twoD, local).
  python restforce_rest_force.py <exe> <request> <s,s,...> <count> <burn> <span> <tau> <tag> [rest.json]
(env AXIS, ORIGIN, TABLES passed through). The columns: the whole step, its jumps' part, the reference flow's part; the
kick's part is the whole less the two. The mean force is the whole's: the kick and the jumps alone leave out the trap's
pull, which the reference's flow carries."""
import sys, json, os, subprocess, numpy as np
exe, req, ss, count, burn, span, tau, tag = sys.argv[1], sys.argv[2], [float(x) for x in sys.argv[3].split(',')], int(sys.argv[4]), float(sys.argv[5]), float(sys.argv[6]), float(sys.argv[7]), sys.argv[8]
exact = {round(r['s'], 9): r for r in json.load(open(sys.argv[9]))['rows']} if len(sys.argv) > 9 else {}
G = 'C:/Users/eadri/AppData/Local/Temp/claude/c0815e347c54866c/8308d66c-699b-4483-ab9b-7305d93a9f7e/scratchpad/rewrite/guard.sh'
os.makedirs('out', exist_ok=True)
rows = []
print('s | F engine ± se = kick + jumps + flow | F exact | 2D engine ± se | 2D exact | jumps/µs')
for s in ss:
    f = os.path.abspath(f'out/f_{tag}_{s}.json')
    p = subprocess.run(['bash', G, '-MaxSeconds', '290', exe, 'engine', req, '0', f'rest:{s}', str(tau), str(count), str(burn), str(span), str(span), f], capture_output=True, text=True)
    if p.returncode != 0:
        print('failed', s, p.returncode, p.stderr[-2000:])
        continue
    d = json.load(open(f))
    os.remove(f)
    assert d.get('columns') == ['whole', 'jumps', 'flow'], 'a worker before the whole-step columns: rebuild the diffusion example'
    a = np.array(d['data']).reshape(d['count'], d['samples'], d['width'])
    inc = a[:, -1, :] - a[:, 0, :] if d['samples'] > 1 else a[:, 0, :]
    n = inc.shape[0]
    whole, jumps, flow = inc[:, 0], inc[:, 1], inc[:, 2]
    kick = whole - jumps - flow
    var = whole.var(ddof=1) / span
    ref = exact.get(round(s, 9), {})
    row = dict(s=s, force=whole.mean() / span, force_se=whole.std(ddof=1) / np.sqrt(n) / span, kick=kick.mean() / span, jumps=jumps.mean() / span, flow=flow.mean() / span,
               twoD=var, twoD_se=var * np.sqrt(2 / (n - 1)), jumprate=inc[:, 9].mean() / span, exact_force=ref.get('force'), exact_twoD=ref.get('twoD'))
    rows.append(row)
    fmt = lambda x: '%9.2f' % x if x is not None else '        -'
    print(f"{s:.4f} | {row['force']:8.2f} ± {row['force_se']:.2f} = {row['kick']:8.2f} + {row['jumps']:6.2f} + {row['flow']:8.2f} | {fmt(row['exact_force'])} | {row['twoD']:10.1f} ± {row['twoD_se']:8.1f} | {fmt(row['exact_twoD'])} | {row['jumprate']:.3f}", flush=True)
json.dump(rows, open(f'out/force_{tag}.json', 'w'), indent=1)
