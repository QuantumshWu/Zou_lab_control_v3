"""The bare exchange between the tweezer's two tight axes, light off: the engine (R1_TWEEZER, `diffusion account ...
free:yz`, y and z started at 30 and 10 µK) against the classical motion in the same Gaussian potential (toy2d, no
force but the trap's), same start law, 100 000 atoms."""
import json, numpy as np, toy2d
KT = 0.1309203
d = json.load(open('off_yz.json'))
omega = d['omega'][1]
c = toy2d.run(100000, 30, 10, True, span=1000.0, dt=0.05, power=0.0, twoD=0.0, recoil=0.0, omega=omega, waist=1.2)
print('t µs | engine T_y ± se, T_z ± se | classical T_y, T_z | differences')
for i in (0, 4, 9, 19, 29, 49, 74, 99):
    m, e = np.array(d['rows'][i]['means']) * 2 / KT, np.array(d['rows'][i]['errors']) * 2 / KT
    print('%5.0f | %6.2f ± %.2f  %6.2f ± %.2f | %6.2f  %6.2f | %+5.2f  %+5.2f' % (d['rows'][i]['t'], m[1], e[1], m[2], e[2], c[i, 1], c[i, 2], m[1] - c[i, 1], m[2] - c[i, 2]))
