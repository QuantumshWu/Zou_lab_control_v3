import numpy as np, pde, ref
from common import *
z=(np.arange(64)-32)*(0.96/64)
L,F,S,m=pde.operators(z)
print('q (µm) : largest Re eig of L - iqF - q²S over z, and where; of L - iqF alone')
for q in [0,0.01,0.02,0.04,0.06,0.08,0.1,0.15,0.2,0.3,0.4,0.54]:
    g=[np.linalg.eigvals(L[i]-1j*q*F[i]-q*q*S[i]).real.max() for i in range(len(z))]
    g1=[np.linalg.eigvals(L[i]-1j*q*F[i]).real.max() for i in range(len(z))]
    i=int(np.argmax(g)); print('%5.2f  %10.3e at z=%6.3f   first order alone %10.3e'%(q,g[i],z[i],max(g1)))
# eigenvalues of S's "Hermitian-preserving" action: is S a positive map generator? smallest Re eig of S
print('S eigen real parts min/max over z:',min(np.linalg.eigvals(S[i]).real.min() for i in range(len(z))),max(np.linalg.eigvals(S[i]).real.max() for i in range(len(z))))
