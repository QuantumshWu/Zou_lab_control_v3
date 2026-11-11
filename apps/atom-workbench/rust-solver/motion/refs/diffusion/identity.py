"""The path identity as a measurement (diffusion.status.md §9.4). On the trap's oscillation the momentum's variance
rate with a weight w is, exactly,

  R_exact = local + E_ψ[r_w · (𝓕[|ψ⟩⟨ψ|] − F̄ ρ)],

r_w(θ₀) the row 2 ∫₀^∞ dτ w(θ₀) w(θ₀ + ωτ) Tr 𝓕(θ₀ + ωτ) e^{∫𝓛} (the regression theorem's propagation from the phase θ₀ on),
and E_ψ the mean over the engine's own internal states on the path. The engine realizes of 𝓕[|ψ⟩⟨ψ|] the operator

  𝓕_engine[ψ] = (−⟨G⟩ + Σ_c (1 − s_c) m_c) |ψ⟩⟨ψ| + Σ_c s_c b_c Σ_a J_a|ψ⟩⟨ψ|J_a†

(Ehrenfest force and the scattered light's remainder on the state it has, the bounded conditional mean with the
jump's landing state), and what it leaves out is X = 𝓕[|ψ⟩⟨ψ|] − 𝓕_engine[ψ], traceless: X_E = −½{G − ⟨G⟩, |ψ⟩⟨ψ|}, the
Ehrenfest mean's, and X_J, the rest (the absorbed momentum's correlation with the landing state beyond the class's
mean). (F̄ρ taken on each state itself, F̄|ψ⟩⟨ψ|: every source is then traceless, and the engine's ensemble need not be the
master equation's to the last per cent -- its F = 1 share is still filling at the burn-in.)
So  R_exact − R_engine = E[r·X_E] + E[r·X_J]: the right side computed here from the engine's states, the left side
from the engine's measured variance rate. (The engine's own part, E[r·(𝓕_engine − F̄ρ)], is not estimated from the
states: r is large along the slow population modes, and the mean of that term over some 10⁵ states moves by tens;
the traceless X terms do not have that component and repeat to 1 %.)

python identity.py <amplitude> <trajectories> <periods>"""
import sys, json, os, numpy as np, scipy.linalg as sl
from common import *
import ref

A, count, periods = float(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
N = 128
T = 2 * np.pi / OMEGA
m, p, w1, w2 = oscillation('closed', 0, A, n=N)
L, F, B = m.supers()
tr, nv = m.trace, m.nv
dt = T / N

# r_w(j0) by the propagation from j0 on: trapezoid in τ, the generator between two points their mean
U = np.array([sl.expm(0.5 * (L[j] + L[(j + 1) % N]) * dt) for j in range(N)])
rowF = np.einsum('v,jvw->jw', tr, F)                      # tr·𝓕 at each point
span = int(round(400.0 / dt))
def rows(weights):
    """r_w(j0) for each weight, on one propagation."""
    out = [np.zeros((N, nv), complex) for _ in weights]
    for j0 in range(N):
        P = np.eye(nv, dtype=complex)
        acc = [0.5 * w[j0] * rowF[j0] for w in weights]
        for s in range(1, span + 1):
            P = U[(j0 + s - 1) % N] @ P
            step = rowF[(j0 + s) % N] @ P
            for i, w in enumerate(weights):
                acc[i] = acc[i] + w[(j0 + s) % N] * step
        for i, w in enumerate(weights):
            out[i][j0] = 2 * dt * w[j0] * acc[i]
    return out
r1, r2 = rows([w1, w2])
r = {1: r1, 2: r2}

# check on the exact density matrix: the slow part of the regression theorem
t = dt * np.arange(N)
rho = np.array([sum(p.rho[p.pos[n]] * np.exp(1j * n * OMEGA * tj) for n in p.ns) for tj in t])
fbar = np.real(np.einsum('jv,jv->j', rowF, rho))
exact = {k: p.moments(w) for k, w in ((1, w1), (2, w2))}
for k in (1, 2):
    slow = np.mean([np.real(r[k][j] @ (F[j] @ rho[j] - fbar[j] * rho[j])) for j in range(N)])
    print('weight %d: slow part by r %.3f, by the harmonic balance %.3f; local %.3f' % (k, slow, exact[k]['slow'], exact[k]['local']))

# the engine on the same path, its states at the reference's points
STRIDE = 65                               # points of the path between two samples (5 µs; coprime with N)
every = STRIDE * dt
out, states = f'eng_id_{A}.json', f'states_{A}.json'
a, d = engine(f'osc:{A}:{OMEGA}:0', 0, 0.3125, count, 15 * T, periods * T, every, f'id_{A}', env={'PHASES': str(N), 'STATES': states, 'ENDS': '1'})
d['span'] = (len(json.load(open(states))[0]['states']) - 1) * every      # the span its samples cover
e = {k: engine_rates(a, d, k) for k in (1, 2)}
S = json.load(open(states))
os.remove(states)
os.remove(f'eng_id_{A}.json')

# classes of the jumps: keys (cluster, class, axis) → class index; the absorbed light's wave number per class
classes = sorted({(c, k) for (c, k, _) in m.keys})
of = np.array([classes.index((c, k)) for (c, k, _) in m.keys])
sums = {k: np.zeros(4) for k in (1, 2)}     # engine's operator, X_E, X_J, count
G = m.G
kabs = m.kabs_class(classes)
member = np.zeros((len(m.keys), len(classes)))
member[np.arange(len(m.keys)), of] = 1.0
bins = [[] for _ in range(N)]
diag = []
for traj in S:
    base = int(round(traj['phase'] * N))
    for k, (cluster, flat) in enumerate(traj['states']):
        g = np.zeros(m.dim, complex)
        amp = np.array(flat[0::2]) + 1j * np.array(flat[1::2])
        g[m.offsets[cluster]:m.offsets[cluster] + len(amp)] = amp
        bins[(base + STRIDE * k) % N].append(g)
for j in range(N):
    if not bins[j]:
        continue
    g = np.array(bins[j])                                           # (n, dim)
    rho_s = np.einsum('na,nb->nab', g, g.conj())
    flat_of = lambda X: X.reshape(len(g), -1)[:, m.idx]
    v = flat_of(rho_s)
    full = v @ F[j].T
    mean_G = np.real(np.einsum('na,ab,nb->n', g.conj(), G[j], g))
    xe = flat_of(-0.5 * (np.einsum('ab,nbc->nac', G[j], rho_s) + np.einsum('nab,bc->nac', rho_s, G[j])) + mean_G[:, None, None] * rho_s)
    u, du = np.einsum('qab,nb->nqa', m.J[j], g), np.einsum('qab,nb->nqa', m.dJ[j], g)
    weight = np.einsum('nqa,nqa->nq', u.conj(), u).real @ member    # (n, classes)
    mean = np.einsum('nqa,nqa->nq', u.conj(), du).imag @ member
    b = np.where(weight > 0, mean / np.where(weight > 0, weight, 1.0), 0.0)
    s_c = np.where(np.abs(b) > kabs[None, :], kabs[None, :] / np.where(np.abs(b) > 0, np.abs(b), 1.0), 1.0)
    landed = np.einsum('nq,nqa,nqb->nab', (s_c * b) @ member.T, u, u.conj())
    engine_op = flat_of((-mean_G + ((1 - s_c) * mean).sum(axis=1))[:, None, None] * rho_s + landed)
    xj = full - engine_op - xe
    vbar = v.mean(axis=0)
    diag.append((len(g), np.abs(vbar - rho[j]).max(), np.real(tr @ vbar), [np.real(r[wk][j] @ (F[j] @ vbar - fbar[j] * vbar)) for wk in (1, 2)], [np.real(r[wk][j] @ (F[j] @ rho[j] - fbar[j] * rho[j])) for wk in (1, 2)], np.real(np.diag(m.mat(vbar))), np.real(np.diag(m.mat(rho[j])))))
    for wk in (1, 2):
        row = r[wk][j]
        sums[wk] += [np.real((engine_op - fbar[j] * v) @ row).sum(), np.real(xe @ row).sum(), np.real(xj @ row).sum(), len(g)]
print(f'A = {A}: {count} trajectories × {periods} periods; states {int(sums[1][3])}')
for k, name in ((1, 'velocity quadrature'), (2, 'position quadrature')):
    eng, xe, xj, n = sums[k]
    print('%s: exact R %.3f = local %.3f + slow %.3f | engine measured R %.3f ± %.3f (Brownian %.3f + Var %.3f)' % (name, exact[k]['rate'], exact[k]['local'], exact[k]['slow'], e[k]['var'] + e[k]['brownian'], e[k]['var_se'], e[k]['brownian'], e[k]['var']))
    print("   from the engine's states: E[r·X_E] = %+.3f; E[r·X_J] = %+.3f" % (xe / n, xj / n))
    print('   exact − engine measured %+.3f against E[r·X_E] + E[r·X_J] = %+.3f' % (exact[k]['rate'] - e[k]['var'] - e[k]['brownian'], (xe + xj) / n))

# the engine's mean state per phase against the master equation's (the unravelling's own check)
print("largest |mean state − ρ| over the phases: %.4f (the states' sampling)" % max(x[1] for x in diag))
