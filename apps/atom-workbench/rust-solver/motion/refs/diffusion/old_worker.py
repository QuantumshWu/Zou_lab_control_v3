"""The old engine's own worker on R4_TW2 (its request without the fields it does not know, as oldfmt.mjs), with the options
given: python old_worker.py <old atom-lindblad.exe> <out.json> [key=value ...] (options.<key>). Prints trajectories, step, T_z.
The old engine (crate atom-motion-3d) is no longer in the tree: its worker is built from the frozen worktree
C:/Users/eadri/WorkCode/zlc_v3_motion3d (62d4d206), apps/atom-workbench/rust-solver/target/x86_64-pc-windows-msvc/release/atom-lindblad.exe.
The worker loads OpenBLAS from ATOM_OPENBLAS_DLL, set as scripts/build-rust.ps1 sets it, or from beside itself."""
import json, subprocess, sys, os, time
if len(sys.argv) < 3:
    sys.exit(__doc__)
WORKER, OUT = sys.argv[1], sys.argv[2]
SCENE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'scenes', 'R4_TW2.json')
s = json.load(open(SCENE))
del s['model']['motion']['outsideForms']
keep = ['id', 'name', 'direction', 'waveNumberPerUm', 'waistUm', 'outsideShift', 'intensityMWcm2', 'frequencyIndex', 'coupling', 'counterCoupling']
s['model']['motion']['beams'] = [{k: v for k, v in b.items() if k in keep} for b in s['model']['motion']['beams']]
for pair in sys.argv[3:]:
    k, v = pair.split('=')
    target = s['model']['motion'] if k.startswith('motion.') else s['options']
    target[k.split('.')[-1]] = json.loads(v)
clock = time.time()
p = subprocess.Popen([WORKER], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
p.stdin.write(json.dumps({**s, 'progress': False}) + '\n'); p.stdin.flush()
final = None
for line in p.stdout:
    if line.strip():
        msg = json.loads(line)
        if msg.get('progress') != 'checkpoint':
            final = msg
            break
p.stdin.close(); p.wait(timeout=30)
if final is None or 'error' in final:
    print('no result', final, p.stderr.read()[:500]); sys.exit(1)
m = final['result']['motion']
tz, ez = m['temperatureUK']['z'], m['temperatureErrorUK']['z']
json.dump({'trajectories': m['trajectories'], 'timeStepUs': m['timeStepUs'], 'T': m['temperatureUK'], 'se': m['temperatureErrorUK'], 'options': s['options']}, open(OUT, 'w'))
print('%.1f s; %d trajectories, step %.4f; T_z at 300, 610, 890 µs: %s' % (time.time() - clock, m['trajectories'], m['timeStepUs'], ', '.join('%.2f ± %.2f' % (tz[i], ez[i]) for i in (30, 61, 89))))
