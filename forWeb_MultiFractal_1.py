# clear up version for Github, on 2020-11-28
import os
import matplotlib.pyplot as plt

import numpy as np
from scipy import interpolate
from scipy.stats import skewnorm
from collections import OrderedDict
from tools import forWeb_funs as funs
plt.rcParams.update({'font.size': 23})
cmaps = OrderedDict()
#-----------------------------------------------------------------------------------------------------------------------
hg0 = 3.992e8   #total volume
ntime = 4e4     #periods in years
#-----------------------------------------------------------------------------------------------------------------------
if True:
    current_path = os.path.dirname(__file__)
    out_fold = os.path.join(current_path, 'fractals')
    fn_list = ['mf_256_10.txt', 'mf_512_29.txt','mf_1024_35.txt','mf_2048_52.txt','mf_4096_31.txt',]  # ,
    fig1, ax1 = plt.subplots(1, 1, figsize=(17, 7))
    for idx0, fn0 in enumerate(fn_list):
        mf = np.loadtxt(os.path.join(out_fold, fn0))
        n_mf = len(mf)
        t_mf = np.linspace(0,n_mf,n_mf)
        LavaEmission = np.zeros(n_mf)
        for idx, mu in enumerate(mf):
            if mu > 1e-10:
                LavaEmission += mf[idx] * skewnorm.pdf(t_mf, a=3.5, loc=t_mf[idx], scale=3*2)
        f1d = interpolate.interp1d(np.arange(0, n_mf), LavaEmission, kind='linear', fill_value='extrapolate')
        tspan_LAVA = np.linspace(0, 2**idx0 * ntime, 2**idx0 * ntime)
        LavaEmission = f1d(np.linspace(0, n_mf, len(tspan_LAVA)))
        LavaEmission = hg0 * LavaEmission / np.sum(LavaEmission)
        f1d_lava = interpolate.interp1d(tspan_LAVA, LavaEmission, kind='linear', fill_value='extrapolate')
        ax1.plot(tspan_LAVA/1000, LavaEmission / 1000, lw=1.1, alpha=0.6, label=f'{int(2**idx0*ntime/1e3)} ka')
        ax1.set_xlabel('Time (ka)')
        ax1.set_ylabel('Flux (Gg/a)')
        print('sum of lava: ',idx0, tspan_LAVA.shape, tspan_LAVA[-5:], '\n', sum(LavaEmission), max(LavaEmission)/1e3)
    plt.grid()
    plt.legend()
    #plt.waitforbuttonpress()
    fig1.savefig(os.path.join(out_fold, 'Emission_' + f'{1}.png'), dpi=400)
    print('\n\n',mf,sum(mf),mf.shape,LavaEmission.shape)
plt.show()

