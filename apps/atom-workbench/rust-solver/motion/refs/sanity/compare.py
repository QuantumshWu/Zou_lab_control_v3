# Entry-by-entry comparison of the app-built S3 request with coherent.py's first-principles construction:
#   python compare.py <S3 request JSON> <dir holding coherent.py>
import sys, os, json, math
import numpy as np
sys.path.insert(0, sys.argv[2]); sys.path.insert(1, 'C:/Users/eadri/WorkCode/zlc_v3_rw_gates/apps/atom-workbench/rust-solver/motion/refs/sanity')
import coherent as Cm
M = Cm.M

req = json.load(open(sys.argv[1], encoding='utf-8'))
mo = req['model']['motion']; ids = req['model']['states']; ids = [s['id'] if isinstance(s, dict) else s for s in ids]
n = 32
A = Cm.setup(req)                 # the request's beams
L = Cm.setup(req, lab=True)       # the lab's description alone
p = A['p']
print('== beams (request | lab): direction, s = I/I_sat, ε_q (q = -1, 0, +1)')
for a, l in zip(zip(A['names'], A['k'], A['s'], A['eps']), zip(L['names'], L['k'], L['s'], L['eps'])):
    print(f"  {a[0]}: k {np.round(a[1], 5).tolist()} | {np.round(l[1], 5).tolist()};  s {a[2]:.5f} | {l[2]:.5f};  ε {np.round(a[3], 5).tolist()} | {np.round(l[3], 5).tolist()}")
dk = max(np.abs(A['k'] - L['k']).max(), 0.0); ds = np.abs(A['s'] / L['s'] - 1).max()
# polarization: each beam's ε the lab's up to one phase; the phase between the beams (the mirror's round trip)
ph = []
for ea, el in zip(A['eps'], L['eps']):
    m = np.abs(el) > 1e-9
    r = ea[m] / el[m]
    ph.append(np.angle(r[0])); assert np.allclose(np.abs(ea), np.abs(el), atol=1e-6), (ea, el)
    print(f'  ε request/lab: |ratio| {np.round(np.abs(r), 6).tolist()}, phase {np.round(np.degrees(np.angle(r)), 4).tolist()} deg')
print(f'  max |Δk| {dk:.2e} 1/µm; max |Δs/s| {ds:.2e}; relative phase of the two beams, request less lab: {math.degrees(ph[1] - ph[0]):.4f} deg (the mirror\'s round trip: lab_beams takes 180° as the request\'s)')

print('== couplings X[e<-g] of each probe beam on F=2 -> F\'=3 (rad/µs): request against (Ω/2) ε_q <2 m; 1 q|3 m+q>, Ω = Γ√(s/2)')
gidx = {m: ids.index(f'ground-2-{m}') for m in range(-2, 3)}; eidx = {m: ids.index(f'D2-3-{m}') for m in range(-3, 4)}
probes = [b for b in mo['beams'] if b['name'] in A['names']]
for b in probes:
    X = np.array(b['coupling']['real']) + 1j * np.array(b['coupling']['imag'])
    X = X.reshape(n, n)
    j = A['names'].index(b['name']); Om = p['gamma'] * math.sqrt(A['s'][j] / 2)
    ratios = []
    for m in range(-2, 3):
        for q in (-1, 0, 1):
            if abs(m + q) > 3: continue
            mine = 0.5 * Om * A['eps'][j][q + 1] * A['C'][q + 1, m + 2]
            theirs = X[eidx[m + q], gidx[m]]
            if abs(mine) > 1e-6: ratios.append((m, q, theirs / mine))
    r = np.array([x[2] for x in ratios])
    print(f"  {b['name']}: request/mine: |ratio| {np.abs(r).min():.4f} .. {np.abs(r).max():.4f}, phase {np.degrees(np.angle(r)).min():.3f} .. {np.degrees(np.angle(r)).max():.3f} deg over {len(r)} entries")
    worst = max(ratios, key=lambda x: abs(abs(x[2]) - 1))
    print(f'    farthest from 1 in size: |2,{worst[0]}> q {worst[1]}: {abs(worst[2]):.4f} (the 3 G field mixes |3\',m\'> with |2\',m\'>; coherent.py keeps F\'=3 pure)')
    other = X.copy()
    for m in range(-2, 3):
        for q in (-1, 0, 1):
            if abs(m + q) <= 3: other[eidx[m + q], gidx[m]] = 0
    blk = [eidx[m] for m in range(-3, 4)]
    print(f"    the beam's largest coupling outside F=2 -> F'=3: {np.abs(other).max():.3f} rad/µs (to F'=2, 1, 0 and D1: left out of coherent.py, 267 MHz and more away)")

print('== H0 on F=2 and F\'=3 (MHz): the request\'s, less the tweezer\'s outside shift at the focus (its `outsideShift`, which the engine takes out of H0 and puts back as the forms of the local field), against g_F μ_B B m and the offset')
H0 = (np.array(mo['hamiltonian']['real']) + 1j * np.array(mo['hamiltonian']['imag'])).reshape(n, n)
tw_ = next(b for b in mo['beams'] if b['id'] == mo['trapBeamId'])
O = (np.array(tw_['outsideShift']['real']) + 1j * np.array(tw_['outsideShift']['imag'])).reshape(n, n)
for b in mo['beams']:
    if b['id'] != mo['trapBeamId']:
        O += (np.array(b['outsideShift']['real']) + 1j * np.array(b['outsideShift']['imag'])).reshape(n, n)
mine = np.diag(A['H0']).real
for lab, idx, sl in (('F=2', [gidx[m] for m in range(-2, 3)], slice(0, 5)), ("F'=3", [eidx[m] for m in range(-3, 4)], slice(5, 12))):
    theirs = np.array([(H0[i, i] - O[i, i]).real for i in idx]) / (2 * math.pi)
    me = mine[sl] / (2 * math.pi)
    print(f'  {lab}: request {np.round(theirs, 4).tolist()}\n         mine    {np.round(me, 4).tolist()}  (max diff {np.abs(theirs - me).max():.4f} MHz)')
print('  the outside shift at the focus taken out (MHz):', np.round([O[i, i].real / (2 * math.pi) for i in [gidx[2]] + [eidx[m] for m in range(-3, 4)]], 3).tolist())

print('== the tweezer\'s shifts at the focus (MHz): model.py\'s sum over states (polarizabilities, a.u.) and the app\'s chart')
sh = A['shifts']
print(f"  α(5S1/2) {sh['alphaGroundAU']:.1f} au -> ground {sh['groundShiftMHz']:.4f} (chart {M.CHART['groundShiftMHz']}); α(5P3/2, m_J 3/2, π) {sh['alphaExcitedAU']:.1f} au -> {sh['excitedShiftMHz']:.4f} (chart {M.CHART['excitedShiftMHz']}); m_J 1/2: {sh['excitedHalfShiftMHz']:.4f}")
print('  |3\',m\'> (Σ|<m_J m_I|3 m\'>|² δE(m_J)):', np.round(A['diagL'][5:] / (2 * math.pi), 3).tolist())
