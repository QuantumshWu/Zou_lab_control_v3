"""Per atom (`diffusion account`): what the trap's flow moved into y beyond turning half of the light's work on y into
potential energy, against the atom's own mean kinetic energies along z and y over the run (µK of T = 2K/k_B)."""
import json, sys, numpy as np
KT = 1.380649e-23 / 1.054571817e-34 * 1e-12
for f in sys.argv[1:]:
    d = json.load(open(f)); a = np.array(d['atoms']) * 2 / KT
    ky, kz = a[:, 0], a[:, 1]
    into_y = a[:, 3] + 0.5 * a[:, 2]
    out_z = a[:, 5] + 0.5 * a[:, 4]
    print(f, 'start', d['temperatures'], ': all atoms: into y %+.2f, z\'s flow beyond −½ light %+.2f' % (into_y.mean(), out_z.mean()))
    # by the start energies (half the harmonic energy: the same unit as a kinetic temperature): nothing of the run's own
    # outcome selects the atoms
    ey, ez = a[:, 6] / 2, a[:, 7] / 2
    diff = ez - ey
    order = np.argsort(diff); n = len(order)
    print("   by the atom's start E_z - E_y (eighths; µK of T): range | start E_z, E_y | run's mean T_z, T_y | into y ± se | z flow beyond -½ light")
    for q in range(8):
        i = order[q * n // 8:(q + 1) * n // 8]
        print('   %7.1f … %7.1f | %6.2f %6.2f | %6.2f %6.2f | %+6.2f ± %.2f | %+7.2f' % (diff[i].min(), diff[i].max(), ez[i].mean(), ey[i].mean(), kz[i].mean(), ky[i].mean(), into_y[i].mean(), into_y[i].std() / np.sqrt(len(i)), out_z[i].mean()))
    print('   distribution of the atoms\' mean T_z: median %.2f, mean %.2f, 90th %.2f, 99th %.2f; T_y: median %.2f, mean %.2f' % (np.median(kz), kz.mean(), np.percentile(kz, 90), np.percentile(kz, 99), np.median(ky), ky.mean()))
