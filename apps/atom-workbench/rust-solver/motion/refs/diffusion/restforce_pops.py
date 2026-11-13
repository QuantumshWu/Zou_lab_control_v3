# DIAGNOSTIC (restforce, 2026-11-12; motion3d_status/rewrite/restforce.status.md): not part of any acceptance.
"""The engine's time-averaged internal state at rest (diffusion engine with STATES=, every 10 µs after the burn) against
the closed model's exact steady state (ref.py on the closed dump): populations of the clusters and the ground coherences.
python pops.py <exe> <request> <closed dump> <point index in the dump> <count> <burn> <span> <tau>"""
import sys, json, os, subprocess, numpy as np
sys.path.insert(0, 'C:/Users/eadri/WorkCode/zlc_v3_rw_force/apps/atom-workbench/rust-solver/motion/refs/diffusion')
import ref
exe, req, dumpf, i, count, burn, span, tau = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), float(sys.argv[6]), float(sys.argv[7]), float(sys.argv[8])
m = ref.Model(dumpf)
row = ref.at_rest(m)[i]
rho = m.mat(row['rho'])
s = row['z']
G = 'C:/Users/eadri/AppData/Local/Temp/claude/c0815e347c54866c/8308d66c-699b-4483-ab9b-7305d93a9f7e/scratchpad/rewrite/guard.sh'
out, states = os.path.abspath(f'out/p_{i}.json'), os.path.abspath(f'out/st_{i}.json')
p = subprocess.run(['bash', G, '-MaxSeconds', '290', exe, 'engine', req, '0', f'rest:{s}', str(tau), str(count), str(burn), str(span), '10', out], capture_output=True, text=True, env=dict(os.environ, STATES=states))
print(p.stderr.strip().splitlines()[-1][-90:])
rows = json.load(open(states)); os.remove(states); os.remove(out)
off = m.offsets
acc = np.zeros((m.dim, m.dim), complex)
n = 0
per_traj = []
for r in rows:
    one = np.zeros((m.dim, m.dim), complex)
    for c, flat in r['states']:
        v = np.array(flat[0::2]) + 1j * np.array(flat[1::2])
        o = off[c]
        one[o:o + len(v), o:o + len(v)] += np.outer(v, v.conj())
    one /= len(r['states'])
    per_traj.append(one)
acc = np.mean(per_traj, axis=0)
se = np.std([np.real(np.diag(x)) for x in per_traj], axis=0, ddof=1) / np.sqrt(len(per_traj))
names = m.names
print(f's {s:.4f}: state | exact | engine ± se')
for k in range(m.dim):
    print(f'  {names[k]:12s} {rho[k, k].real:.4f} {acc[k, k].real:.4f} ± {se[k]:.4f}')
g = [k for k in range(m.dim) if names[k].startswith('ground-2')]
C_ex = rho[np.ix_(g, g)]; C_en = acc[np.ix_(g, g)]
print('F=1 total exact %.4f engine %.4f; excited exact %.4f engine %.4f' % (np.real(np.trace(rho[:3, :3])), np.real(np.trace(acc[:3, :3])), sum(rho[k, k].real for k in range(m.dim) if names[k].startswith('D')), sum(acc[k, k].real for k in range(m.dim) if names[k].startswith('D'))))
print('F=2 ground coherences |rho_ij| exact vs engine (largest 6):')
pairs = sorted([(abs(C_ex[a, b]), a, b) for a in range(5) for b in range(a + 1, 5)], reverse=True)[:6]
for v, a, b in pairs:
    print(f'  {names[g[a]]}-{names[g[b]]}: exact {C_ex[a, b]:.4f} engine {C_en[a, b]:.4f}')
