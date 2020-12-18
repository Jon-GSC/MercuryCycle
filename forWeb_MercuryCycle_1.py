# By Jon, since 2019-08-21
'''
note:
   A concise version of modelling mercury cycle.

   20201121: Different flux K are listed, reader need to check which one is more suitable.

   Citation: Grasby, S.E., Liu, X., Yin, R., Ernst, R.E. & Chen, Z. 2020,
             "Toxic mercury pulses into Late Permian terrestrial and marine environments",
             Geology vol. 48, issue 8, p. 830-833; 10.1130/G47295.1.
'''
import os
import numpy as np
import time
from datetime import timedelta
from scipy import interpolate

import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
import seaborn as sns
sns.set_style("white")

from tools import forWeb_funs as funs
np.set_printoptions(precision=8)
np.set_printoptions(suppress=True)
plt.rcParams.update({'font.size': 17})
legend_list = ['Atmosphere', 'Terr-F', 'Terr-S', 'Terr-A', 'Ocean-S', 'Ocean-M', 'Ocean-D']
npixel = 400   # resolution of figure
current_path = os.path.dirname(__file__)
#-----------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------
icase = [   1,2,3    ]   # choose 1,2,3

period0 = 4e4   # smallest period
save_fold = 'model_1'   #
fn_list = ['mf_256_10.txt', 'mf_512_29.txt', 'mf_1024_35.txt', 'mf_2048_52.txt', 'mf_4096_31.txt', ]  # ,
legend_list0 = ('40','80','160','320','640')
#-----------------------------------------------------------------------------------------------------------------------
# present-day reservoirs/flux: Atmosphere, Terrestrial(fast,slow,armored), Ocean(surface,middle,deep).
Reservoir = np.array([ 5000, 9600, 35000, 190000, 2900, 130000, 220000 ])

Flux0_atmo_terr = 1500
FluxII_atmo_terr = 1500
Flux_atmo_ocea = 3900 + 40

Flux_terrF_atmo = 460 + 850 + 290
Flux_terrF_terrS = 330
Flux_terrF_terrA = 10
Flux_terrF_oceaS = 365

Flux_terrS_atmo = 250 + 8
Flux_terrS_terrF = 210
Flux_terrS_terrA = 0.35   # need check-check, no info in paper.
Flux_terrS_oceaS = 10

Flux_terrA_atmo = 25 + 4
Flux_terrA_terrF = 15
Flux_terrA_oceaS = 5

Flux_oceaS_atmo = 3000
Flux_oceaS_oceaM = 3300 + 5100

Flux_oceaM_oceaS = 7100
Flux_oceaM_oceaD = 480 + 340

Flux_oceaD_oceaM = 180
Flux_oceaD_deepM = 210

Ratio_atmo_terr = [0.5, 0.32, 0.18]  #ratio from atmo to terrFast, Slow, Armored.

# Deep mineral reservoir to atmosphere, unit Mg/year
Rate_Hg_deep = 90
# Deep mineral reservoir, unit Mg
Hg_deep = 3e11
# total Hg mass emmission in Siberian
hg0 = 3.992e8  # Mg  (check the unit from Steve's paper in 2015)

dt = 0.2
# present-day Atmosphere,Terrestrial(fast,slow,armored), Ocean(surface,middle,deep) for initial.(Mg)
M0 = np.array([ 5000, 9620, 34900, 193600, 2910, 134000, 220649 ])

#note: the results of natural reservoir is more close to the paper on 20190610
Rate_atmo_terrF = (Flux0_atmo_terr+FluxII_atmo_terr*Ratio_atmo_terr[0])/Reservoir[0]  # to fast
Rate_atmo_terrS = FluxII_atmo_terr*Ratio_atmo_terr[1]/Reservoir[0]  # slow
Rate_atmo_terrA = FluxII_atmo_terr*Ratio_atmo_terr[2]/Reservoir[0]    # to armored
Rate_atmo_oceaS = Flux_atmo_ocea/Reservoir[0]  # to surface ocean

# from Terrestrial, unit
Rate_terrF_atmo = Flux_terrF_atmo/Reservoir[1]  # to atmosphere
Rate_terrF_terrS = Flux_terrF_terrS/Reservoir[1]  # to fast
Rate_terrF_terrA = Flux_terrF_terrA/Reservoir[1]  # to armored
Rate_terrF_oceaS = Flux_terrF_oceaS/Reservoir[1]  # to surfaceocean

Rate_terrS_atmo = Flux_terrS_atmo/Reservoir[2]  # to atmosphere
Rate_terrS_terrF = Flux_terrS_terrF/Reservoir[2]  # to fast
Rate_terrS_terrA = Flux_terrS_terrA/Reservoir[2]  # to armored
Rate_terrS_oceaS = Flux_terrS_oceaS/Reservoir[2]  # to surfaceocean

Rate_terrA_atmo = Flux_terrA_atmo/Reservoir[3]   # to atmosphere
Rate_terrA_terrF = Flux_terrA_terrF/Reservoir[3]   # to fast
Rate_terrA_oceaS = Flux_terrA_oceaS/Reservoir[3]    # to slow

# from Ocean
Rate_oceaS_atmo = Flux_oceaS_atmo/Reservoir[4]   # to atmo
Rate_oceaS_oceaM = Flux_oceaS_oceaM/Reservoir[4]  # to middle ocean

Rate_oceaM_oceaS = Flux_oceaM_oceaS/Reservoir[5]  # to middle ocean
Rate_oceaM_oceaD = Flux_oceaM_oceaD/Reservoir[5]  # to deep ocean

Rate_oceaD_oceaM = Flux_oceaD_oceaM/Reservoir[6]  # to middle ocean
Rate_oceaD_deepM = Flux_oceaD_deepM/Reservoir[6]  # to deep mineral

K = np.array([  #order: atmo,  terr-F, terr-S, terr-A,  ocean-S, ocean-M, ocean-D,  on 20190603
    [-(Rate_atmo_terrF+Rate_atmo_terrS+Rate_atmo_terrA+Rate_atmo_oceaS),Rate_terrF_atmo,Rate_terrS_atmo,Rate_terrA_atmo,Rate_oceaS_atmo, 0, 0],
    [Rate_atmo_terrF,-(Rate_terrF_atmo+Rate_terrF_terrS+Rate_terrF_terrA+Rate_terrF_oceaS),Rate_terrS_terrF, Rate_terrA_terrF, 0, 0, 0],
    [Rate_atmo_terrS,Rate_terrF_terrS,-(Rate_terrS_atmo+Rate_terrS_terrF+Rate_terrS_terrA+Rate_terrS_oceaS),0, 0, 0, 0],
    [Rate_atmo_terrA,Rate_terrF_terrA,Rate_terrS_terrA,-(Rate_terrA_atmo+Rate_terrA_terrF+Rate_terrA_oceaS),0, 0, 0],
    [Rate_atmo_oceaS,Rate_terrF_oceaS,Rate_terrS_oceaS,Rate_terrA_oceaS,-(Rate_oceaS_atmo+Rate_oceaS_oceaM),Rate_oceaM_oceaS, 0],
    [0, 0, 0, 0, Rate_oceaS_oceaM, -(Rate_oceaM_oceaS+Rate_oceaM_oceaD),Rate_oceaD_oceaM],
    [0, 0, 0, 0, 0, Rate_oceaM_oceaD,-(Rate_oceaD_oceaM+Rate_oceaD_deepM)]
    ])    # 7x7

K0 = np.array([   #pre-anthropogenic era
   [-1.66, 0.013565488565489, 0.000716332378223, 0.000012913223140, 1.615120274914089, 0, 0],
   [0.45081, -0.121898175309592, 0.005873925501433, 0.000077479338843, 0, 0, 0],
   [0.09639, 0.033783783783784, -0.007138508849012, 0, 0, 0, 0],
   [0.05259, 0.000935550935551, 0.000014326647564, -0.000142904557889, 0, 0, 0],
   [1.06, 0.027545342133927, 0.000199788865917, 0.000019649436598, -4.508591065292096, 0.052985074626866, 0],
   [0, 0, 0, 0, 2.893470790378007, -0.059067164179104, 0.000793114856627],
   [0, 0, 0, 0, 0, 0.006082089552239, -0.00174485268458]])

K1 = np.array([   #anthropogenic era
   [-1.66, 0.043531964656965, 0.000933681948424, 0.000034289772727, 1.615120274914089, 0, 0],
   [0.45081, -0.151864651401068, 0.005873925501433, 0.000077479338843, 0, 0, 0],
   [0.09639, 0.033783783783784, -0.007355858419212, 0, 0, 0, 0],
   [0.05259, 0.000935550935551, 0.000014326647564, -0.000164281107475, 0, 0, 0],
   [1.06, 0.027545342133927, 0.000199788865917, 0.000019649436598, -4.508591065292096, 0.052985074626866, 0],
   [0, 0, 0, 0, 2.893470790378007, -0.059067164179104, 0.000793114856627],
   [0, 0, 0, 0, 0, 0.006082089552239, -0.00174485268458]])

AnthroEmission_4K = np.loadtxt(current_path+'/fractals/AnthroEmissAllTime_20120112_Hellen.txt')
f1d = interpolate.interp1d(AnthroEmission_4K[:, 0], AnthroEmission_4K[:, 1], kind='cubic')

LavaEmission = np.loadtxt(current_path+'/fractals/LavaEmissAllTime_1.txt')
f1d_lava = interpolate.interp1d(LavaEmission[:, 0], LavaEmission[:, 1], kind='linear')

tspan = np.arange(0, 5e4+dt, dt)
tspan_4K = np.arange(AnthroEmission_4K[0, 0], AnthroEmission_4K[-1, 0], dt)
tspan_LAVA = np.arange(LavaEmission[0, 0], LavaEmission[-1, 0], dt)

#-----------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------
def main():
    start_time = time.time()
    data_folder = os.path.join(current_path,'fractals')
    out_fold = os.path.join(current_path,save_fold)
    os.makedirs(out_fold, exist_ok=True)
    if 1 in icase:  # figure-2?
        M0_4K,MAX_4K = funs.ODE_func(out_fold, tspan_4K, K0, M0, Rate_Hg_deep, iflag=0)
        M0_5K = funs.ODE_4K_func(out_fold,tspan_4K,K1,M0_4K,Rate_Hg_deep,f1d,iflag=1)

    if 2 in icase:  # figure-6?
        funs.ODE_Fig6_func(out_fold, tspan, K, M0, Rate_Hg_deep, iflag=1)

    if 3 in icase:  # eruptions
        T_initial = 0  #for plotting
        nf= len(fn_list)

        M0_4K,MAX_4K = funs.ODE_func(out_fold, tspan, K, M0, Rate_Hg_deep, iflag=0)
        Enrich_Factor1 = np.empty((nf,7))
        Enrich_Factor2 = np.empty((nf,7))

        f1, ax1 = plt.subplots(1, 1, figsize=(10, 6))
        f2, ax2 = plt.subplots(1, 1, figsize=(10, 6))
        f3, ax3 = plt.subplots(1, 1, figsize=(10, 6))
        for idx0, fn0 in enumerate(fn_list):
            tspan_LAVA = np.arange(0, 2**idx0 * period0, dt)
            LavaEmission, f1d_lava = funs.mod_fractal_1(tspan_LAVA,data_folder,dt,hg0,fn0)
            M1, M2, dM, dM_in, dM_out, M0_5K, MAX_5K = funs.ODE_LAVA_func_1(tspan_LAVA, K, M0_4K, Rate_Hg_deep, f1d_lava)

            Enrich_Factor1[idx0,:] = MAX_5K/M0_4K
            Enrich_Factor2[idx0,:] = MAX_5K/M0

            ax1.plot(T_initial+tspan_LAVA/1e3, LavaEmission/1e3)   # unit Gg/a
            ax1.set_xlabel('Time (ka)')
            ax1.set_ylabel('Flux (Gg/a)')

            ax2.plot(T_initial+tspan_LAVA/1e3, M2[4,:]/1e3)  # plot Surface Ocean(Gg-ka)
            ax2.set_xlabel('Time (ka)')
            ax2.set_ylabel('Hg mass (Gg)')
            ax2.legend(legend_list0)

            ax3.plot(T_initial+tspan_LAVA/1e3, M2[4,:]*1e6*1e9/(72.38*1e6*1e9*1e3))   #unit: ng/L
            ax3.set_xlabel('Time (ka)')
            ax3.set_ylabel('Hg concentration (ng/L)')
            ax3.legend(legend_list0)

            funs.write_resutls(out_fold, M0, M0_4K, M0_5K, MAX_5K, idx0)
            del f1d_lava, tspan_LAVA, M1, M2, dM, dM_in, dM_out, M0_5K, MAX_5K

        ax1.grid();  ax2.grid();  ax3.grid()
        f4, ax4 = plt.subplots(1, 1, figsize=(10, 6))
        ax4.plot(np.arange(0,nf),Enrich_Factor1[:,4],)
        ax4.scatter(np.arange(0,nf),Enrich_Factor1[:,4],)
        plt.xlabel('Eruption period (k.a.)')
        plt.ylabel('Enrichment factor (times)')
        plt.title('Enrichment Factor of Different Eruption Period to Natural')
        plt.grid()
        ax4.xaxis.set(ticks=range(0,nf), ticklabels=int(period0/1e3)*np.logspace(0,nf-1,nf,base=2,dtype=int))  #

        f5, ax5 = plt.subplots(1, 1, figsize=(10, 6))
        ax5.plot(np.arange(0,nf),Enrich_Factor2[:,4],)
        ax5.scatter(np.arange(0,nf),Enrich_Factor2[:,4],)
        plt.xlabel('Eruption period (k.a.)')
        plt.ylabel('Enrichment factor (times)')
        plt.title('Enrichment Factor of Different Eruption Period to Present')
        plt.grid()
        ax5.xaxis.set(ticks=range(0,nf), ticklabels=int(period0/1e3)*np.logspace(0,nf-1,nf,base=2,dtype=int))

        f1.savefig(os.path.join(out_fold, f'EruptionFlux_v{0}.png'), dpi=npixel)
        f2.savefig(os.path.join(out_fold, f'SurfaceOceanFlux_v{0}.png'), dpi=npixel)
        f3.savefig(os.path.join(out_fold, f'SurfaceOceanConc_v{0}.png'), dpi=npixel)
        f4.savefig(os.path.join(out_fold, f'Enrichment Factor(toNatural)_v{0}.png'), dpi=npixel)
        f5.savefig(os.path.join(out_fold, f'Enrichment Factor(toPresent)_v{0}.png'), dpi=npixel)

    time_dif = time.time() - start_time
    print('\nTime Used: ' + str(timedelta(seconds=int(round(time_dif)))), ' seconds\n')


if __name__ == '__main__':
    main()
