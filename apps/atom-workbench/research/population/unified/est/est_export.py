"""Export R2's dumped N1 spaces (r2n/d2_<name>, n2_dump.py format) as fixtures for the Rust estimator test
(rust-solver lattice_core_adapter::tests::lattice_estimator_against_r2_fixtures).

usage: python est_export.py NAME [NAME ...] [--index-only]     (names as in r2n/eval2.sh: pt_20, two_20, three_10000, ...)

Per case, OUT/<name>/:
  index.json  case, request path, P, Q, states, groups, times, horizons, dim, kmax and per checkpoint: solves, k, r,
              the actual error series (metric vs the reference), the reference's own change (dref, when a previous
              round exists), the comparison mask of n2_eval, err_steady / err_h, and R2's estimate series when an
              e2_<name>.json exists;
  W.bin, AW.bin  the final checkpoint's basis and images, column after column (dim complex128 each);
  cp<solves>.bin  X (k x r), S (r x r), B (k x r), G (r x r), c0, th, th_e, a, a_e, ker, u_inf (r): complex128,
              column-major, in this order (ker as 0/1; u_inf = the bordered reduced steady state, R2's SteadyGCRO.rhs).
"""
import sys, os, json, glob
import numpy as np

S = 'C:/Users/eadri/AppData/Local/Temp/claude/C--Users-eadri--claude-spares-spare-aYE3fY/5f28e5a8-891d-4b8f-bd56-7c0275f16fce/scratchpad/'
U = 'C:/Users/eadri/WorkCode/zlc_v3_uni_estimate/apps/atom-workbench/research/population/unified'
sys.path.insert(0, U)
from r1_common import metric_series, groups  # noqa: E402
import n1_common as C  # noqa: E402

R2N, R1, N1 = S + 'unified/r2n/', S + 'unified/r1/', S + 'unified/n1/'
OUT = os.environ.get('EST_FIXTURES', S + 'est_rust/fx/')
REFS = dict(pt_20=('pt', R1 + 'ex_pt_20.json', None), nl_20=('nl', R1 + 'ex_nl_20.json', None),
            two_20=('two', R1 + 'ex_twoNear_new_k1_20.json', None), two_10000=('two', R1 + 'ex_twoNear_new_k1_10000.json', None),
            two_10_2000=('two', N1 + 'ref_two_10_2000.json', N1 + 'ref_two_10_2000_traj_round3.npy'),
            three_20=('three', N1 + 'ref_three_20_200.json', N1 + 'ref_three_20_200_traj_round5.npy'),
            three_10000=('three', N1 + 'ref_three_10000_200.json', N1 + 'ref_three_10000_200_traj_round4.npy'),
            three_10_2000=('three', N1 + 'ref_three_10_2000.json', N1 + 'ref_three_10_2000_traj_round4.npy'),
            four_20=('four', N1 + 'ref_four_20_200.json', N1 + 'ref_four_20_200_traj_round5.npy'),
            four_10000=('four', N1 + 'ref_four_10000_200.json', N1 + 'ref_four_10000_200_traj_round4.npy'),
            fourN_20=('fourN', N1 + 'ref_fourN_20_200.json', None), fourN_10000=('fourN', N1 + 'ref_fourN_10000_200.json', None),
            five_20=('five', N1 + 'ref_five_20_200.json', N1 + 'ref_five_20_200_traj_round5.npy'),
            five_10000=('five', N1 + 'ref_five_10000_200.json', N1 + 'ref_five_10000_200_traj_round3.npy'),
            fiveN_20=('fiveN', N1 + 'ref_fiveN_20_200.json', N1 + 'ref_fiveN_20_200_traj_round5.npy'),
            fiveN_10000=('fiveN', N1 + 'ref_fiveN_10000_200.json', N1 + 'ref_fiveN_10000_200_traj_round3.npy'))


def clean(x):
    if isinstance(x, float) and not np.isfinite(x):
        return None
    if isinstance(x, list):
        return [clean(v) for v in x]
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    return x


def cbytes(a):
    a = np.asarray(a, dtype=complex)
    return np.asfortranarray(a).reshape(-1, order='F').astype('<c16').tobytes()


def export(name):
    case, refp, prevp = REFS[name]
    dump = R2N + 'd2_' + name
    run = json.load(open(R2N + 'run2_' + name + '.json'))
    ref = json.load(open(refp))
    files = sorted(glob.glob(dump + '/cp*.npz'))
    req = C.request_path(C.CASES.get(case, case))
    states = json.load(open(req))['model']['states']
    st = ref['states']
    assert st == states, 'reference states differ from the request'
    gr = list(groups(states).values())
    dref = metric_series(np.load(prevp), np.array(ref['pops']), st) if prevp else None
    ests = None
    for tag in ('e5_', 'e2_'):                       # R2's latest evaluation of this case
        if os.path.exists(R2N + tag + name + '.json'):
            ests = {r['solves']: r for r in json.load(open(R2N + tag + name + '.json'))['runs']}
            break
    cps = {cp['solves']: cp for cp in run['checkpoints']}
    out = OUT + name + '/'
    os.makedirs(out, exist_ok=True)
    last = np.load(files[-1])
    Wl, AWl = last['W'], last['AW']
    dim, kmax = Wl.shape
    n = int(round(np.sqrt(dim // (len(last['P']) + len(last['Q'])))))
    if not INDEX_ONLY:
        with open(out + 'W.bin', 'wb') as f:
            f.write(np.ascontiguousarray(Wl.T).astype('<c16').tobytes())
        with open(out + 'AW.bin', 'wb') as f:
            f.write(np.ascontiguousarray(AWl.T).astype('<c16').tobytes())
    rows_c = None
    index = dict(name=name, case=case, request=req.replace('\\', '/'), n=n, P=last['P'].tolist(), Q=last['Q'].tolist(),
                 states=states, groups=gr, times=last['times'].tolist(), horizons=last['horizons'].tolist(), dim=int(dim),
                 kmax=int(kmax), checkpoints=[])
    nP = len(last['P'])
    centre = [i for i, q in enumerate(last['P'].tolist()) if all(x == 0 for x in q)][0]
    rows = centre * n * n + np.arange(n) * (n + 1)
    for fpath in files:
        z = np.load(fpath)
        sol = int(z['solves'])
        k = z['W'].shape[1]
        prefix = max(np.abs(z['W'] - Wl[:, :k]).max(), np.abs(z['AW'] - AWl[:, :k]).max())
        assert prefix < 1e-12, f'{name} cp{sol}: basis not append-only ({prefix})'
        X = z['X'] if 'X' in z.files else np.eye(k)
        r = X.shape[1]
        G, c0 = z['G'], z['c0']
        tau = (Wl[:, :k] @ X)[rows].sum(axis=0)
        Mx = np.vstack([G, tau[None, :]])
        rhs = np.zeros(r + 1, complex); rhs[-1] = 1.
        us = np.linalg.lstsq(Mx, rhs, rcond=None)[0] * (tau @ c0)
        blob = b''.join(cbytes(a) for a in (X, z['S'], z['B'], G, c0, z['th'], z['th_e'], z['a'], z['a_e'],
                                               z['ker'].astype(float), us))
        if not INDEX_ONLY:
            with open(out + f'cp{sol:03d}.bin', 'wb') as f:
                f.write(blob)
        row = dict(solves=sol, k=int(k), r=int(r))
        cp = cps.get(sol)
        if cp is not None and 'traj' in cp:
            act = metric_series(np.array(cp['traj']), np.array(ref['pops']), st)
            times = z['times']
            sel = (act > max(1e-4, 1e-2 * act.max())) & (times > 0)
            if dref is not None:
                sel &= act > 3 * dref
            row.update(actual=act.tolist(), select=sel.astype(int).tolist(), err_steady=cp.get('err_steady'),
                       err_h=cp.get('err_h'), steady_pops=cp.get('steady_pops'))
        if ests is not None and sol in ests:
            e = ests[sol]
            row['python'] = {v: dict(series=x['series'], ratio=x['ratio'], q=x['q'], horizons=x.get('horizons'), work=x.get('work')) for v, x in e['est'].items()}
            row['solve_gemv'] = e.get('solve_gemv')
            if 'steady' in e:
                row['python_steady'] = e['steady']
            row['solve_blocks'] = e.get('solve_blocks')
        row['count'] = z['count'].tolist()
        index['checkpoints'].append(row)
        print(f'  {name} cp{sol}: k {k} r {r} prefix {prefix:.1e}' + (' (python)' if 'python' in row else ''), flush=True)
    if dref is not None:
        index['dref'] = dref.tolist()
    json.dump(clean(index), open(out + "index.json", "w"), allow_nan=False)


INDEX_ONLY = '--index-only' in sys.argv

if __name__ == '__main__':
    for nm in sys.argv[1:]:
        if not nm.startswith('--'):
            export(nm)
