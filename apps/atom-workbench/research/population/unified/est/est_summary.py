"""usage: python est_summary.py OUTPUT.jsonl ...

Round-3 estimator harness summary: per checkpoint estimate/actual at the maximum [5% 50% 95%] min, R2's Python 'rule'
ratio and the Rust-Python deviation, steady estimate/actual, the slow part, the gate, the work and its cost against
the solve; per case the cumulative estimate cost up to the first accepted checkpoint (gate and 1.25 x max(trajectory,
steady) <= 5e-3)."""
import json, sys
from collections import defaultdict

EPS = 5e-3
rows = defaultdict(list)
heads = {}
for path in sys.argv[1:]:
    for line in open(path):
        if not line.startswith('{'):
            continue
        r = json.loads(line)
        if 'buildMs' in r:
            heads[r['case']] = r
        else:
            rows[r['case']].append(r)


def f(v, fmt='.2f'):
    return 'nan' if v is None else format(v, fmt)


for case, h in heads.items():
    print(f"== {case}: modes {h['modes']} near {h['near']} ring {h['ring']} build {h['buildMs'] / 1e3:.1f}s "
          f"(kernels {h['kernelMs'] / 1e3:.2f}s, classes {h['classMs'] / 1e3:.1f}s, eigenbasis {h['eigenbasisMs'] / 1e3:.2f}s) "
          f"operator dev {h['operatorDeviation']:.1e} |X|max {max(h['smallParameter'] or [0]):.2f} fold {h['ledger']:.1e}")
    cum_ms, cum_gemv, accepted = 0., 0., None
    for r in rows[case]:
        q = r.get('quantiles') or [None] * 4
        py = (r.get('python') or {}).get('rule')
        pys = f" | py {f(py['ratio'])} dev {py['deviation']:.0e}" if py else ''
        ss = r.get('steadyActual')
        sss = f" | ss {r['steadyEst']:.2e}/{ss:.2e} ({r['steadyEst'] / ss:.2f})" if ss else f" | ss {r['steadyEst']:.1e}/-"
        w = r['work']
        c = r['count']; n = r['modes']
        solve_blocks = c[1] + c[2] + c[3] + c[4] * n
        # R2's GEMV units: a frame change 1, an operator or preconditioner application on P u Q 2N, a ring block solve 2
        # (the solve: 2 x its block operations); a step of the pipeline's bordered solve = one application and one
        # preconditioner. The ring's masked couplings (one or two edges per ring block) are left out, as in R2.
        gemv = w['frameBlocks'] + w['poolFrames'] + 2 * n * (w['poolApplications'] + w['poolPreconditions'] + 2 * w['pipelineSteps']) + 2 * w['ringSolves']
        r['gemv'] = gemv
        if r['gate']['open']:
            cum_ms += r['ms']; cum_gemv = cum_gemv + gemv
            if accepted is None and 1.25 * max(r['estMax'], r['steadyEst']) <= EPS:
                accepted = (r['solves'], cum_ms, solve_blocks, r.get('actualMax'), cum_gemv)
        else:
            pending_gemv = w['frameBlocks']          # deferred columns are processed at the next open checkpoint
            cum_gemv += pending_gemv
        print(f"  {r['solves']:3d} k {r['k']:3d} act {f(r.get('actualMax'), '.2e')} ratio {f(r.get('ratio'))} "
              f"[{f(q[0])} {f(q[1])} {f(q[2])}] min {f(q[3])}{pys}{sss} | slow {r['slowModes']}{'T' if r['slowTrajectory'] else ''} "
              f"lam {r['lambdaSlow']:.2g} | gate {'open' if r['gate']['open'] else 'shut'} | {r['ms']:.0f} ms: "
              f"+{w['columns']} col ({w['columnMs']:.0f} ms, {w['frameBlocks']} frames), ring {w['ringOperations']}+{w['ringSolves']}s, "
              f"pool {w['poolApplications']}A {w['poolPreconditions']}P {w['poolSteps']} steps {w['pipelineSolves']}x{w['pipelineSteps']} "
              f"({w['pipelineMs']:.0f} ms) ext {w['extendMs']:.0f} step {w['stepMs']:.0f} | {gemv:.0f} GEMV vs solve {2 * solve_blocks} ({gemv / (2 * solve_blocks):.0%})")
    if accepted:
        print(f"  accepted at {accepted[0]} solves (actual {f(accepted[3], '.2e')}): estimate {accepted[1]:.0f} ms, {accepted[4]:.0f} GEMV cumulative "
              f"= {accepted[4] / (2 * accepted[2]):.0%} of the solve ({2 * accepted[2]} GEMV)")
