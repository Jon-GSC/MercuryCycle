#
import os
import numpy as np
import pandas as pd
import random

import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.stats import skewnorm, beta

plt.rcParams.update({'font.size': 21})
#-----------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------
legend_list = ['Atmosphere', 'Terr-F', 'Terr-S', 'Terr-A', 'Ocean-S', 'Ocean-M', 'Ocean-D']
legend_list0 = ['atmosphere', 'fast terrestrial', 'slow soil', 'armored soil', 'surface ocean', 'subsurface ocean','deep ocean','deep mineral']
color_list = ['yellow', 'yellowgreen', 'green', 'darkgreen', 'turquoise', 'blue', 'darkblue', 'silver']
idx_out0 = np.eye(7, 7)  # obtain out or in.
idx_in0 = np.full(7, 1) - idx_out0
npixel = 400

#-----------------------------------------------------------------------------------------------------------------------
#
def scan_files(folder0):
    current_path = os.path.dirname(__file__)
    fold0_path = os.path.join(current_path, folder0)

    filelist_df = pd.DataFrame(columns=[])
    filelist = []
    for file in sorted(os.listdir(fold0_path)):
        if file.endswith('.txt'):
            filelist.append(file)
    filelist_df['txt_name'] = filelist
    return filelist_df


def mod_fractal_1(tspan_LAVA,data_folder,dt,hg0,fn0):
    mf = np.loadtxt(os.path.join(data_folder, fn0))  # test load text file.,unpack=True
    n_mf = len(mf)
    t_mf = np.linspace(0, n_mf, n_mf)

    LavaEmission = np.zeros(n_mf)
    for idx, mu in enumerate(mf):
        if mu > 1e-10:
            LavaEmission += mf[idx] * skewnorm.pdf(t_mf, a=3.5, loc=t_mf[idx], scale=3 * 2)
    f1d = interpolate.interp1d(np.arange(0, n_mf), LavaEmission, kind='linear', fill_value='extrapolate')

    LavaEmission = f1d(np.linspace(0, n_mf, len(tspan_LAVA)))
    LavaEmission = hg0 * LavaEmission / np.sum(LavaEmission) / dt
    f1d_lava = interpolate.interp1d(tspan_LAVA, LavaEmission, kind='linear', fill_value='extrapolate')
    return LavaEmission,f1d_lava


def FractalInterpolation():
    npoint = 1000000
    x1 = [0,30,60,100]
    F = [10,50,40,10]
    d = [0.5*random.randrange(91,100)/100,-0.5*random.randrange(91,100)/100,0.23*random.randrange(91,100)/100]
    a = [0,0,0,0]; e = [0,0,0,0]; c = [0,0,0,0]; f = [0,0,0,0]
    for n in np.arange(1,4,1):
        b = x1[3] - x1[0]
        a[n] = (x1[n]-x1[n-1])/b
        e[n] = (x1[3]*x1[n-1]-x1[0]*x1[n])/b
        c[n] = (F[n]-F[n-1]-d[n-1]*(F[3]-F[0]))/b
        f[n] = (x1[3]*F[n-1]-x1[0]*F[n]-d[n-1]*(x1[3]*F[0]-x1[0]*F[3]))/b

    x = np.empty(3*npoint+1)
    y = np.empty(3*npoint+1)
    x0=0; y0=0
    for n in range(npoint):
        k = int(3*random.random()-0.0001)+1
        x[n] = x0; y[n] = y0
        newx = a[k]*x0+e[k]
        newy = c[k]*x0+d[k-1]*y0+f[k]
        x0 = newx; y0 = newy
    idx_sort = x.argsort()
    x = x[idx_sort[::1]]
    y = y[idx_sort[::1]]
    a0=x[y>15]; a0=a0-min(a0); b0=y[y>15]

    f1d = interpolate.interp1d(a0,b0, kind='linear',fill_value='extrapolate')
    x2 = np.linspace(0,a0[-1],2*len(a0))
    y2 = f1d(x2)
    return x2,y2


def Katsuura_Jon(D):
    def Txy(x, y):
        Tx = [x[0] + (x[1] - x[0]) / 3, x[0] + (x[1] - x[0]) * 2 / 3]
        Ty = [y[0] + 2 * (y[1] - y[0]) / 3, y[0] + (y[1] - y[0]) / 3]
        return Tx, Ty

    x0 = [0,1]
    y0 = [0,1]
    if D == 0:
        return x0, y0
    else:
        x_temp,y_temp = Katsuura_Jon(D-1)
        x_temp0 = x_temp.copy(); y_temp0 = y_temp.copy()
        for j in range(len(x_temp)-1):
            Tx,Ty = Txy((x_temp[j],x_temp[j+1]),(y_temp[j],y_temp[j+1]))
            x_temp0 = x_temp0[:3*j+1] + Tx + x_temp0[3*j+1:]
            y_temp0 = y_temp0[:3*j+1] + Ty + y_temp0[3*j+1:]
        return x_temp0,y_temp0


def gaussian(x, step, mu, sig):
    x0 = (x - mu)/step
    mu0 = 1/2
    if sig==0: sig = 0.01
    gauss0 = np.exp(-np.power(x0 - mu0, 2.) / (2 * np.power(sig, 2.)))
    return gauss0


def skewer(x,step,a):
    x0 = (x)/step
    skewer = skewnorm.pdf(x, a * (np.random.random(1) + 0.1))
    return skewer


def FractalInterpolation_0():
    npoint = 1000
    x1 = [0,30,60,100]
    F = [0,100,45,0]
    d = [0.5*random.randrange(31,100)/100,-0.65*random.randrange(31,100)/100,0.13*random.randrange(31,100)/100]
    a = [0,0,0,0]; e = [0,0,0,0]; c = [0,0,0,0]; f = [0,0,0,0]
    for n in np.arange(1,4,1):
        b = x1[3] - x1[0]
        a[n] = (x1[n]-x1[n-1])/b
        e[n] = (x1[3]*x1[n-1]-x1[0]*x1[n])/b
        c[n] = (F[n]-F[n-1]-d[n-1]*(F[3]-F[0]))/b
        f[n] = (x1[3]*F[n-1]-x1[0]*F[n]-d[n-1]*(x1[3]*F[0]-x1[0]*F[3]))/b
    x = np.empty(3*npoint+1)
    y = np.empty(3*npoint+1)
    x0=0; y0=0
    for n in range(npoint):
        k = int(3*random.random()-0.0001)+1
        x[n] = x0; y[n] = y0
        newx = a[k]*x0+e[k]
        newy = c[k]*x0+d[k-1]*y0+f[k]
        x0 = newx; y0 = newy
    idx_sort = x.argsort()
    x = x[idx_sort[::1]]
    y = y[idx_sort[::1]]
    return x,y


def Eruption_distr(x, mu=2, sigma0=0.1):
    n_sample = 30
    Eruption_rate = np.zeros(x.shape)
    x1,y1 = FractalInterpolation_0()
    f1d = interpolate.interp1d(x1, y1, kind='linear', fill_value='extrapolate')
    y = f1d(np.arange(x1[0],x1[-1],(x1[-1]-x1[0])/len(x)))
    for i in range(n_sample):
        step = int(len(y)/n_sample)
        Eruption_rate[i*step] = random.sample(population=list(y[i*step:(i+1)*step]),k=1)[0] * (np.random.random(1)*0.8+0.6)
    return y


def multifractal(n=10 ,a=1.0 ,b=1.0):
    if n== 1:
        w1 = beta.rvs(a, b, size=1)
        return np.concatenate((w1, 1 - w1), axis=None).reshape(-1)
    else:
        beta_n = beta.rvs(a, b, size=2 ** (n - 1))
        wi = np.concatenate((beta_n, 1 - beta_n), axis=None).reshape(2, -1).T
        return np.multiply(np.repeat(multifractal(n - 1).reshape(-1, 1), 2, axis=1), wi).reshape(-1)


def func_M(t,K,M,Rate_Hg_deep,f1d):
    f_M = np.dot(K,M)
    f_M[0] = f_M[0] + Rate_Hg_deep + f1d(t)
    return f_M


def ODE_func(out_fold,t0,K,M0,Rate_Hg_deep,iflag=0):
    dt = t0[1] - t0[0]
    t8 = t0.copy()
    M8 = np.empty((7,len(t8)))
    M8[:,0] = M0
    for i in range(1,len(t8)):
        M8[:,i] = (np.dot(K, M8[:,i-1]) + np.array([Rate_Hg_deep,0,0,0,0,0,0]))*dt + M8[:,i-1]

    dM = np.dot(K, M8)
    M0_4K = M8[:,-1]
    MAX_4K = np.max(M8,axis=1)

    if iflag==1:
        f1 = plt.figure(1)
        plt.plot(t8, M8.T)
        plt.title('Total M')
        plt.xlabel('Year')
        plt.ylabel('Mg')
        plt.grid()
        plt.legend(legend_list)

        f2 = plt.figure(2)
        plt.plot(t8, dM.T)
        plt.xlabel('Year')
        plt.ylabel('Mg/a')
        plt.title('dM/dt')
        plt.grid()
        plt.legend(legend_list)

        f1.savefig(os.path.join(out_fold,'totalHg.png'))
        f2.savefig(os.path.join(out_fold,'deltaHg.png'))
    return M0_4K,MAX_4K


def ODE_Fig6_func(out_fold,t0,K,M0,Rate_Hg_deep,iflag=0):
    dt = t0[1] - t0[0]
    pulse = np.array([100 if t0[i]<=1 else 0 for i in range(1,len(t0))])
    M1 = np.empty((7, len(t0)))
    M1[:, 0] = M0
    for i in range(1, len(t0)):
        M1[:, i] = (np.dot(K, M1[:, i - 1]) + np.array([Rate_Hg_deep, 0, 0, 0, 0, 0, 0])) * dt + M1[:, i-1]

    M2 = np.empty((7, len(t0)))
    M2[:, 0] = M0
    for i in range(1, len(t0)):
        M2[:, i] = (np.dot(K, M2[:, i - 1]) + np.array([Rate_Hg_deep+pulse[i-1], 0, 0, 0, 0, 0, 0])) * dt + M2[:, i-1]

    M3 = np.empty((7, len(t0)))
    M3[:, 0] = M0
    for i in range(1, len(t0)):
        M3[:, i] = (np.dot(K, M3[:, i - 1]) + np.array([Rate_Hg_deep, pulse[i-1], 0, 0, 0, 0, 0])) * dt + M3[:, i-1]

    M4 = np.empty((7, len(t0)))
    M4[:, 0] = M0
    for i in range(1, len(t0)):
        M4[:, i] = (np.dot(K, M4[:, i - 1]) + np.array([Rate_Hg_deep, 0, 0, 0, pulse[i-1], 0, 0])) * dt + M4[:, i-1]

    if iflag == 1:
        func_plot_fill(t0, M1, M2, out_fold, 'Atmospheric Pulse')
        func_plot_fill(t0, M1, M3, out_fold, 'Fast Terrestrial Pulse')
        func_plot_fill(t0, M1, M4, out_fold, 'Surface Ocean Pulse')


def ODE_4K_func(out_fold,t0,K,M0,Rate_Hg_deep,f1d,iflag=0):
    dt = t0[1] - t0[0]
    emission = f1d(t0)
    icase = 2
    if icase==1:
        pass
    elif icase==2:
        M1 = np.empty((7,len(t0)))
        M1[:,0] = M0
        for i in range(1,len(t0)):
            M1[:,i] = (np.dot(K, M1[:,i-1]) + np.array([Rate_Hg_deep+emission[i],0,0,0,0,0,0]))*dt + M1[:,i-1]

    dM = func_M(t0,K,M1,Rate_Hg_deep,f1d)
    dM_in = np.dot(np.multiply(K,idx_in0),M1)
    dM_out = np.dot(np.multiply(K,-idx_out0),M1)
    M0_5K = M1[:, -1]

    if iflag==1:
        plot_figures(out_fold, t0, emission, M1, dM, dM_in, dM_out)
    return M0_5K


def ODE_LAVA_func(out_fold,t0,K,M0,Rate_Hg_deep,f1d,iflag=0):
    dt = t0[1] - t0[0]
    M1 = np.empty((7, len(t0)))
    M1[:, 0] = M0
    for i in range(1, len(t0)):
        M1[:, i] = (np.dot(K, M1[:, i-1]) + np.array([Rate_Hg_deep, 0, 0, 0, 0, 0, 0])) * dt + M1[:, i-1]

    emission = f1d(t0)
    M2 = np.empty((7, len(t0)))
    M2[:, 0] = M0
    for i in range(1, len(t0)):
        M2[:, i] = (np.dot(K, M2[:, i-1]) + np.array([Rate_Hg_deep+emission[i], 0, 0, 0, 0, 0, 0])) * dt + M2[:, i-1]

    dM = func_M(t0,K,M2,Rate_Hg_deep,f1d)
    dM_in = np.dot(np.multiply(K,idx_in0),M2)
    dM_out = np.dot(np.multiply(K,-idx_out0),M2)
    M0_5K = M2[:, -1]
    MAX_5K = np.max(M2,axis=1)

    if iflag==1:
        plot_figures(out_fold, t0, emission, M2, dM, dM_in, dM_out)
        func_plot_fill(t0, M1, M2, out_fold, 'Atmospheric Pulse')
    return M0_5K, MAX_5K


def ODE_LAVA_func_1(t0,K,M0,Rate_Hg_deep,f1d):
    dt = t0[1] - t0[0]
    M1 = np.empty((7, len(t0)))
    M1[:, 0] = M0
    for i in range(1, len(t0)):
        M1[:, i] = (np.dot(K, M1[:, i-1]) + np.array([Rate_Hg_deep, 0, 0, 0, 0, 0, 0])) * dt + M1[:, i-1]

    emission = f1d(t0)
    M2 = np.empty((7, len(t0)))
    M2[:, 0] = M0
    for i in range(1, len(t0)):
        M2[:, i] = (np.dot(K, M2[:, i-1]) + np.array([Rate_Hg_deep+emission[i], 0, 0, 0, 0, 0, 0])) * dt + M2[:, i-1]

    dM = func_M(t0,K,M2,Rate_Hg_deep,f1d)
    dM_in = np.dot(np.multiply(K,idx_in0),M2)
    dM_out = np.dot(np.multiply(K,-idx_out0),M2)
    M0_5K = M2[:, -1]
    MAX_5K = np.max(M2,axis=1)
    return M1,M2,dM,dM_in,dM_out,M0_5K, MAX_5K


def plot_figures(out_fold,t0,emission,M8,dM,dM_in,dM_out):
    for i in range(7):
        f1 = plt.figure(figsize=(10,6))
        plt.plot(t0/1e3, M8[i,:]/1e3)
        plt.title('Hg mass ('+legend_list[i]+')')
        plt.xlabel('Time (k.a.)')
        plt.ylabel('Hg mass (Gg)')
        plt.grid()
        f1.savefig(os.path.join(out_fold, 'totalHg_'+legend_list[i]+'.png'))
        f1.clear()

    f2 = plt.figure(figsize=(10,6))
    plt.plot(t0/1e3, dM.T/1e3)
    plt.xlabel('Time (ka)')
    plt.ylabel('Flux (Gg/a)')
    plt.title('Hg emissions flux')
    plt.grid()
    plt.legend(legend_list)

    f3 = plt.figure(figsize=(10,6))
    plt.plot(t0/1e3, emission/1e3, 'g')
    plt.xlabel('Time (ka)')
    plt.ylabel('Hg flux (Gg/a)')
    plt.title('Hg Emission to Atmosphere')
    plt.grid()

    f4 = plt.figure(figsize=(10,6))
    plt.plot(t0/1e3, dM_in.T/1e3)
    plt.xlabel('Time (ka)')
    plt.ylabel('Hg flux (Gg/a)')
    plt.title('Hg emissions flux (in)')
    plt.grid()
    plt.legend(legend_list)

    f5 = plt.figure(figsize=(10,6))
    plt.plot(t0/1e3, dM_out.T/1e3)
    plt.xlabel('Time (ka)')
    plt.ylabel('Hg flux (Gg/a)')
    plt.title('Hg emissions flux (Deposition)')
    plt.grid()
    plt.legend(legend_list)

    f2.savefig(os.path.join(out_fold, 'deltaHg.png'),dpi=npixel)
    f3.savefig(os.path.join(out_fold, 'EmissionHg.png'),dpi=npixel)
    f4.savefig(os.path.join(out_fold, 'deltaHg_in.png'),dpi=npixel)
    f5.savefig(os.path.join(out_fold, 'deltaHg_out.png'),dpi=npixel)
    f2.clear();  f3.clear();  f4.clear();  f5.clear()


def func_plot_fill(t0,M1,M2,out_fold,title0):
    dM = M2 - M1
    dM = np.r_[np.zeros((1,len(M1[0,:]))), dM]
    dM = np.cumsum(dM, axis=0)
    dM = np.r_[dM, np.max(dM)*np.ones((1,len(M1[0,:])))]

    f88, ax = plt.subplots(figsize=(11,5))
    for idx,col in enumerate(color_list):
        ax.fill_between(t0, dM.T[:, idx], dM.T[:, idx+1], color=col)
    plt.xscale('log')
    plt.xlim(1,np.max(t0))
    plt.ylim(0,np.max(dM[:,8]))
    plt.title(title0)
    plt.xlabel('Time (a)')
    plt.ylabel('Fraction (%)')
    plt.gca().invert_yaxis()
    plt.legend(legend_list,loc='lower right')
    f88.savefig(os.path.join(out_fold, title0+'.png'),dpi=npixel)


def write_resutls(out_fold,M0,M0_4K,M0_5K,MAX_5K,i=0):
    fid = open(os.path.join(out_fold, 'Hg_Results_'+str(i)+'.txt'), 'w')
    fid.write('-------Results:-------\n\n')
    fid.write('---Atmosphere, Terrestrial(fast, slow, armored), Ocean(surface, middle, deep)---')
    fid.write('\nHg Present(mg): \n')
    fid.write('  '.join(str(elem) for elem in M0))
    fid.write('\n\nHg Natural(mg): \n')
    fid.write('  '.join(str(elem) for elem in M0_4K))
    fid.write('\n\nHg Final(mg): \n')
    fid.write('  '.join(str(elem) for elem in M0_5K))
    fid.write('\n\nHg Maximum(mg): \n')
    fid.write('  '.join(str(elem) for elem in MAX_5K))
    fid.write('\n\nEnrichment Factor(Max/Natural): \n')
    fid.write('  '.join(str(elem) for elem in MAX_5K / M0_4K))
    fid.write('\n\nEnrichment Factor(Max/Present): \n')
    fid.write('  '.join(str(elem) for elem in MAX_5K / M0))
    Cc_ocean = [MAX_5K[4]*1e6*1e9/(361.9*1e6*0.2*1e9*1e3), MAX_5K[5]*1e6*1e9/(361.9*1e6*2.8*1e9*1e3), MAX_5K[6]*1e6*1e9/(361.9*1e6*1.0*1e9*1e3)]
    fid.write('\n\nConcentration of Surface Ocean(ng/L): \n')
    fid.write('---Ocean(surface,   middle,   deep)---\n')
    fid.write('  '.join(str(elem) for elem in Cc_ocean))
    fid.close()
