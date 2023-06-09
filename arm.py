#
# Initialization of the ARM case
import sys
import matplotlib.pyplot as plt
import numpy as np 
import LES_analysis
from model import *


""" 
Create empty model_input and set up ARM-SGP case
"""
arm = model_input()

arm.dt         = 10.        # time step [s]
arm.runtime    = 12*3600    # total run time [s]

# mixed-layer input
arm.sw_ml      = True       # mixed-layer model switch
arm.sw_shearwe = False      # shear growth mixed-layer switch
arm.sw_fixft   = False      # Fix the free-troposphere switch
arm.h          = 140.       # initial ABL height [m]
arm.Ps         = 970e2      # surface pressure [Pa]
arm.divU       = 0.         # horizontal large-scale divergence of wind [s-1]
arm.fc         = 1e-4       # Coriolis parameter [m s-1]
arm.P_ref      = 970e2      # Reference pressure used to calculate Exner function [Pa]

arm.theta       = 301.4     # initial mixed-layer potential temperature [K]
arm.dtheta      = 0.4       # initial temperature jump at h [K]
arm.gammatheta  = None      # free atmosphere potential temperature lapse rate [K m-1]
arm.advtheta_ml = 0.        # advection of heat to the mixed-layer [K s-1]
arm.advtheta_ft = 0.        # advection of heat to the free-troposphere [K s-1]
arm.beta        = 0.15      # entrainment ratio for virtual heat [-]  
arm.wtheta      = None      # surface kinematic heat flux [K m s-1]

arm.q          = 15.3e-3   # initial mixed-layer specific humidity [kg kg-1]  # simplified 15.3
arm.dq         = -0.2e-3   # initial specific humidity jump at h [kg kg-1]
arm.gammaq     = None      # free atmosphere specific humidity lapse rate [kg kg-1 m-1]
arm.advq_ml    = 0.        # advection of moisture to the mixed-layer [kg kg-1 s-1]
arm.advq_ft    = 0.        # advection of moisture to the free-troposphere [kg kg-1 s-1]
arm.wq         = None      # surface kinematic moisture flux [kg kg-1 m s-1]

arm.CO2        = 422.      # initial mixed-layer CO2 [ppm]
arm.dCO2       = -44.      # initial CO2 jump at h [ppm]
arm.gammaCO2   = 0.        # free atmosphere CO2 lapse rate [ppm m-1]
arm.advCO2     = 0.        # advection of CO2 [ppm s-1]
arm.wCO2       = 0.        # surface kinematic CO2 flux [ppm m s-1]

arm.sw_wind    = False     # prognostic wind switch
arm.u          = 6.        # initial mixed-layer u-wind speed [m s-1]
arm.du         = 4.        # initial u-wind jump at h [m s-1]
arm.gammau     = 0.        # free atmosphere u-wind speed lapse rate [s-1]
arm.advu       = 0.        # advection of u-wind [m s-2]

arm.v          = -4.0      # initial mixed-layer u-wind speed [m s-1]
arm.dv         = 4.0       # initial u-wind jump at h [m s-1]
arm.gammav     = 0.        # free atmosphere v-wind speed lapse rate [s-1]
arm.advv       = 0.        # advection of v-wind [m s-2]

arm.sw_sl      = False     # surface layer switch
arm.ustar      = 0.3       # surface friction velocity [m s-1]
arm.z0m        = 0.02      # roughness length for momentum [m]
arm.z0h        = 0.002     # roughness length for scalars [m]

arm.sw_rad     = False     # radiation switch
arm.lat        = 51.97     # latitude [deg]
arm.lon        = -4.93     # longitude [deg]
arm.doy        = 268.      # day of the year [-]
arm.tstart     = 12.5         # time of the day [h UTC]
arm.cc         = 0.0       # cloud cover fraction [-]
arm.Q          = 400.      # net radiation [W m-2] 
arm.dFz        = 0.        # cloud top radiative divergence [W m-2] 

arm.sw_ls      = False     # land surface switch
arm.ls_type    = 'js'      # land-surface parameterization ('js' for Jarvis-Stewart or 'ags' for A-Gs)
arm.wg         = 0.21      # volumetric water content top soil layer [m3 m-3]
arm.w2         = 0.21      # volumetric water content deeper soil layer [m3 m-3]
arm.cveg       = 0.85      # vegetation fraction [-]
arm.Tsoil      = 285.      # temperature top soil layer [K]
arm.T2         = 286.      # temperature deeper soil layer [K]
arm.a          = 0.219     # Clapp and Hornberger retention curve parameter a
arm.b          = 4.90      # Clapp and Hornberger retention curve parameter b
arm.p          = 4.        # Clapp and Hornberger retention curve parameter c
arm.CGsat      = 3.56e-6   # saturated soil conductivity for heat

arm.wsat       = 0.472     # saturated volumetric water content ECMWF config [-]
arm.wfc        = 0.323     # volumetric water content field capacity [-]
arm.wwilt      = 0.171     # volumetric water content wilting point [-]

arm.C1sat      = 0.132     
arm.C2ref      = 1.8

arm.LAI        = 2.        # leaf area index [-]
arm.gD         = 0.0       # correction factor transpiration for VPD [-]
arm.rsmin      = 110.      # minimum resistance transpiration [s m-1]
arm.rssoilmin  = 50.       # minimun resistance soil evaporation [s m-1]
arm.alpha      = 0.25      # surface albedo [-]

arm.Ts         = 290.      # initial surface temperature [K]

arm.Wmax       = 0.0002    # thickness of water layer on wet vegetation [m]
arm.Wl         = 0.0000    # equivalent water layer depth for wet vegetation [m]

arm.Lambda     = 5.9       # thermal diffusivity skin layer [-]

arm.c3c4       = 'c3'      # Plant type ('c3' or 'c4')

arm.sw_cu      = False     # Cumulus parameterization switch
arm.dz_h       = 150.      # Transition layer thickness [m]
arm.phi_cu     = 0.4       # scaling factor (qtcc - q) = phi_cu * q2_h**0.5. value of 0.51 suggested by Van Stratum et al. (2014)
arm.wcld_prefact = 0.84    # scaling factor wcc = wcld_prefact * w_ml

# Time dependent surface variables; linearly interpolated by the model
# Note the time offset, as the mixed-layer model starts one hour later than LES!
time   = np.array([0., 4, 6.5,  7.5,  10, 12.5, 14.5]) - 1
H      = np.array([-30.,  90., 140., 140., 100., -10.,  -10])
LE     = np.array([  5., 250., 450., 500., 420., 180.,    0])
rho    = arm.Ps / (287. * arm.theta * (1. + 0.61 * arm.q))
wtheta = H  / (rho*1005.)
wq     = LE / (rho*2.5e6)


time  *= 3600.
arm.timedep    = {'wtheta': (time, wtheta),
                  'wq':     (time, wq)}

# Binned height dependent lapse rates 
z1         = np.array([0, 700, 5000])
gammatheta = np.array([3.4e-3, 5.7e-3])

z2         = np.array([0, 650, 1300, 5000])
gammaq     = np.array([-0.6e-6, -2e-6, -8.75e-6])  

arm.heightdep  = {'gammatheta': (z1, gammatheta),
                  'gammaq':     (z2, gammaq)}

arm.sw_rhtend = True  # diagnose RH_h tendency factors

arm.sw_plume = False       # diagnose vertical velocity at LFC using entraining plume model
arm.zmax = 4990            # maximum simulation height for plume model [m]
arm.n_pts  = 499           # maximum number of points
arm.ent_corr_factor = 0.7  # factor controlling lateral entrainment rate in plume model [-]

arm.sw_cin   = False       # use plume data to (potentially) reduce mass-flux
arm.sw_store = False       # use tropopheric storage module
arm.hstore = 1.5e3         # cloud depth [m]

arm.sw_acc_sikma = False   # use parametrization of cloud core fraction from Sikma and Ouwersloot (2015)

"""
Init and run the model
"""

# With cloud parameterisation
r1 = model(arm)
r1.run()

# # Without cloud parameterisation
arm.sw_cu = True
arm.sw_plume = False
r2 = model(arm)
r2.run()

arm.sw_acc_sikma = False
r4 = model(arm)
r4.run()
#%%
arm.sw_acc_sikma = True
arm.sw_plume = True
arm.sw_cin = True
arm.sw_store = True
r3 = model(arm)
r3.run()



#%% LES data extraction and plotting settings

simname = 'arm_simplified'
plot_r3 = True

save = True
plot_les = True

if plot_les:
    les_data_loc = 'LES_data'   # location of LES data directory
    data, z = LES_analysis.main(les_data_loc, bomex=False)
    data.t_les = data.time / 3600 + 11.5 
else:
    data = None
    
plt.close('all')
    
fsize = 14

plt.figure()

plt.plot(r1.out.t, r1.out.h, label='Simple')
if plot_les:
    plt.plot(data.t_les, data.h_ml, label='LES', color='k', linestyle='--')
plt.plot(r2.out.t, r2.out.h, label='Cu')
plt.plot(r4.out.t, r4.out.h, label='Cu+acc2')
if plot_r3:
    plt.plot(r3.out.t, r3.out.h, label='Cu+CIN+Sft')

plt.xlabel('time [h]', fontsize=fsize)
plt.ylabel('h [m]', fontsize=fsize)
plt.legend(fontsize=12)
if save:
    plt.savefig(fname=simname +'_hml.pdf')
plt.show()

plt.figure()
plt.plot(r1.out.t, r1.out.theta, label='Simple')
if plot_les:
    plt.plot(data.t_les, data.thml, label='LES', color='k', linestyle='--')
plt.plot(r2.out.t, r2.out.theta, label='Cu')
plt.plot(r4.out.t, r4.out.theta, label='Cu+acc2')
if plot_r3:
    plt.plot(r3.out.t, r3.out.theta, label='Cu+CIN+Sft')

plt.xlabel('time [h]', fontsize=fsize)
plt.ylabel(r'$\theta$ [K]', fontsize=fsize)
# plt.legend(fontsize=12)
if save:
    plt.savefig(fname=simname +'_thetaml.pdf')
plt.show()

plt.figure()
plt.plot(r1.out.t, r1.out.q*1000., label='Simple')
if plot_les:
    plt.plot(data.t_les, data.qtml*1000, label='LES', color='k', linestyle='--')
plt.plot(r2.out.t, r2.out.q*1000., label='Cu+acc2')


plt.xlabel('time [h]', fontsize=fsize)
plt.ylabel(r'q [g $\mathrm{kg^{-1}}$]', fontsize=fsize)

plt.plot(r2.out.t, r4.out.q*1000., label='Cu+acc2')
if plot_r3:
    plt.plot(r3.out.t, r3.out.q*1000., label='Cu+CIN+Sft')

# plt.legend(fontsize=12)
if save:
    plt.savefig(fname=simname +'_qml.pdf')
plt.show() 

#%%
plt.figure()
if plot_les:
    plt.plot(data.t_les, data.Mcc, label=r'LES', linestyle='--', color='k')
plt.plot(r1.out.t, r2.out.M, label='Cu', color='tab:orange')
plt.plot(r2.out.t, r4.out.M, label='Cu+acc2', color='tab:green')
if plot_r3:
    plt.plot(r1.out.t, r3.out.M, label='Cu+CIN+Sft', color='tab:red')

# plt.legend(fontsize=12)
plt.ylabel('M [$\mathrm{ms^{-1}}$]', fontsize=fsize)
plt.xlabel('t [h]', fontsize=fsize)
# plt.ylim([0, 0.2])
if save:
    plt.savefig(fname=simname +'_M.pdf')
plt.show()


plt.figure()
if plot_les:
    plt.plot(data.t_les, np.sqrt(data.q2_h)*1000, label='LES', color='k', linestyle='--')
plt.plot(r1.out.t, np.sqrt(r2.out.q2_h)*1000, label='Cu', color='tab:orange')
plt.plot(r2.out.t, np.sqrt(r4.out.q2_h)*1000., label='Cu+acc2', color='tab:green')
if plot_r3:
    plt.plot(r1.out.t, np.sqrt(r3.out.q2_h)*1000, label='Cu+CIN+Sft', color='tab:red')

plt.ylabel(r'$\sigma_q$ [$\mathrm{g kg^{-1}}$]', fontsize=fsize)
plt.xlabel('t [h]', fontsize=fsize)
# plt.legend(fontsize=fsize)
if save:
    plt.savefig(fname=simname +'_sigma_q.pdf')
plt.show()

plt.figure()
if plot_les:
    plt.plot(data.t_les, data.acc_max, label='LES', color='k', linestyle='--')
plt.plot(r1.out.t, r2.out.acc, label='Cu', color='tab:orange')
plt.plot(r1.out.t, r4.out.acc, label='Cu+acc2', color='tab:green')
if plot_r3:
    plt.plot(r1.out.t, r3.out.acc, label='Cu+CIN+Sft', color='tab:red')

plt.ylabel(r'$a_{\mathrm{cc}}$ [-]', fontsize=fsize)
plt.xlabel('t [h]', fontsize=fsize)
# plt.ylim([0, 0.2])
# plt.legend(fontsize=12)
if save:
    plt.savefig(fname=simname +'_acc.pdf')
plt.show()


#%%
if plot_les:
    plt.figure()
    
    plt.plot(data.t_les, (data.qtcc_accmax - data.qt_accmax)*1e3, label=r'$q_{t,cc}-\overline{q}_t$')
    plt.plot(data.t_les, 2.7 * np.sqrt(data.q2_h)*1e3, label=r'2.7$\sigma_q$')
    plt.ylabel('q (g/kg)', fontsize=fsize)
    plt.xlabel('t (h)', fontsize=fsize)
    plt.legend(fontsize=fsize)
    if save:
        plt.savefig(fname=simname +'_nu_qt.pdf')
    plt.show()
    it = np.where(data.time/3600>2)[0][0]
    
    plt.figure()
    plt.plot(data.t_les, data.thlcc_accmax - data.thl_accmax, label=r'$\theta_{l,cc} - \overline{\theta}_l$')
    plt.plot(data.t_les, -2.7 * np.sqrt(data.thl2_h), label=r'$-2.7\sigma_{\theta_l}$')
    plt.legend(fontsize=fsize)
    plt.xlabel('t (h)', fontsize=fsize)
    plt.ylabel(r'$\theta_l$ (K)', fontsize=fsize)
    if save:
        plt.savefig(fname=simname +'_nu_thl.pdf')
    plt.show()
    
    #%%
    plt.figure()
    plt.plot(data.t_les, (data.qtcc_accmax - data.qtml)*1e3, label=r'$q_{t,cc}-q_\mathrm{ml}$')
    plt.plot(data.t_les, 0.4 * np.sqrt(data.q2_h)*1e3, label=r'0.4$\sigma_q$')
    plt.ylabel('q (g/kg)', fontsize=fsize)
    plt.xlabel('t (h)', fontsize=fsize)
    plt.legend(fontsize=fsize)
    if save:
        plt.savefig(fname=simname +'_phi_cu.pdf')
    plt.show()
    it = np.where(data.time/3600>2)[0][0]
    
    #%%
    
    plt.figure()
    plt.plot(data.t_les, data.thlcc_accmax - data.thml, label=r'$\theta_{l,\mathrm{cc}} - \theta_\mathrm{ml}$')
    plt.plot(data.t_les, -0.25 * np.sqrt(data.thl2_h), label=r'$-0.25\sigma_{\theta_l}$')
    plt.legend(fontsize=fsize)
    plt.xlabel('t (h)', fontsize=fsize)
    plt.ylabel(r'$\theta_l$ (K)', fontsize=fsize)
    if save:
        plt.savefig(fname=simname +'_thlcc.pdf')
    plt.show()
    
    #%%
    
    def calc_pres(p0, T0, q0, gth, gq, z):
        """"Specify hydrostatic pressure profile based on constant gradient of virtual potential temp."""
        Tv0 = T0 * (1 + 0.61 * q0)
        gtv = (1 + 0.61 * q0) * (gth - 9.81 / cp) + 0.61 * T0 * gq
        pres = p0 * (1 + gtv * z / Tv0) ** ( - 9.81 / (Rd * gtv))
        exner = (pres / p0) ** (Rd / cp)
        return pres, exner
    def esat(T):
        return 0.611e3 * np.exp(17.2694 * (T - 273.16) / (T - 35.86))
    
    def calc_sat(temp, pref):
        Rd = 287
        Rv = 461.5
        esat = 610.78 * np.exp(17.2694 * (temp - 273.16)/(temp - 35.86))
        qsat = Rd / Rv * esat / (pref + (1 - Rd / Rv) * esat)
        # qsat = 0.622 * esat / pref  # approximation for mixed-layer 
        return qsat
    
    p0 = 1e5
    Rd = 287
    cp = 1005
    Lv = 2.5e6
    
    h_lfc = data.z[data.i_acc_max]
    h_lfc[data.i_acc_max == 0] = np.nan
    q_ft0 = arm.q + arm.dq - gammaq[0] * arm.h
    qprof = plume_model.calc_input_prof(data.z, q_ft0, arm.h, z2, gammaq, arm.q)
    q_lfc = np.interp(h_lfc, data.z, qprof)
    
    theta_ft0 = arm.theta + arm.dtheta - gammatheta[0] * arm.h
    thetaprof = plume_model.calc_input_prof(data.z, theta_ft0, arm.h, z1, gammatheta, arm.theta)
    theta_lfc = np.interp(h_lfc, data.z, thetaprof)
    
    
    pres_accmax =  data.pres[np.arange(data.time.size), data.i_acc_max]
    exner_accmax = (pres_accmax / p0) ** (Rd / cp)
    
    qsat_approx = calc_sat(exner_accmax * theta_lfc, pres_accmax)
    
    #%%
    
    
    it = 400
    
    theta = data.thl + Lv / cp * data.ql
    th_lfc = theta[np.arange(len(data.time)), data.i_acc_max]
    th_lfc[data.i_acc_max == 0] = np.nan
    
    temp_lfc = data.temp[np.arange(len(data.time)), data.i_acc_max]
    temp_lfc[data.i_acc_max == 0] = np.nan
    qsat_approx2 = calc_sat(exner_accmax * theta_lfc, pres_accmax) 
    
    Q2_ori = (data.qt_accmax - data.qsat_accmax) / np.sqrt(data.q2_accmax) 
    Q2 = (data.qt_accmax - data.qsat_accmax) / np.sqrt(data.q2_h) 
    Q2_alt2 = (data.qt_accmax - qsat_approx2) / np.sqrt(data.q2_h)
    Q2_alt3 = (q_lfc - qsat_approx2) / np.sqrt(data.q2_h)  
    
    acc_ori = 0.292 * Q2_ori ** -2
    acc_param = 0.292 * Q2 ** -2
    acc_param_alt2 = 0.292 * Q2_alt2 ** -2
    acc_param_alt3 = 0.292 * Q2_alt3 ** -2
    
    # acc_param_alt2[acc_param_alt2 > 0.5] = np.nan
    plt.figure()
    plt.plot(data.t_les, data.acc_max, label=r'$a_\mathrm{cc}$')
    plt.plot(data.t_les, acc_ori, label=r'$a_\mathrm{cc}$-param. 1')
    plt.plot(data.t_les, acc_param, label=r'$a_\mathrm{cc}$-param. 2')
    plt.plot(data.t_les, acc_param_alt2, label=r'$a_\mathrm{cc}$-param. 3')
    plt.plot(data.t_les, acc_param_alt3, label=r'$a_\mathrm{cc}$-param. 4')
    
    # plt.ylim([0,0.2])
    plt.xlabel('t (h)', fontsize=fsize)
    plt.ylabel(r'$a_{\mathrm{cc}}$ (-)', fontsize=fsize)
    plt.legend(fontsize=fsize)
    if True:
        plt.savefig(fname=simname +'_acc_param.pdf')
    plt.show()
    #%%
    data.i_acc_max[data.i_acc_max == 0] = -1
    
    plt.figure()
    plt.plot(data.t_les, 1e3*(-(2.7-0.4)*np.sqrt(data.q2_h) +2e-6 * (h_lfc - data.h_ml)))
    plt.ylim([-2, 0])
    plt.xlabel('t (h)', fontsize=fsize)
    plt.ylabel('$\Delta q$ (g/kg)', fontsize=fsize)
    if True:
        plt.savefig(fname=simname +'_dq_cu.pdf')
    plt.show()




