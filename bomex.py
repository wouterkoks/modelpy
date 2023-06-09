import matplotlib.pyplot as plt
import numpy as np
import copy
import os 
import LES_analysis

from model import *



#%%
""" 
Create empty model_input and set up BOMEX case
"""
bomex = model_input()

bomex.dt         = 1          # time step [s]
bomex.runtime    = 12 * 3600. # total run time [s]

# mixed-layer bomexut
bomex.sw_ml      = True      # mixed-layer model switch
bomex.sw_shearwe = False     # shear growth mixed-layer switch
bomex.sw_fixft   = False     # Fix the free-troposphere switch
bomex.h          = 520.      # initial ABL height [m]
bomex.Ps         = 1015e2    # surface pressure [Pa]
bomex.divU       = 0.65 / (100 * 1500)   # horizontal large-scale divergence of wind [s-1] 
bomex.fc         = 1e-4      # Coriolis parameter [m s-1]
bomex.P_ref      = 1000e2    # Reference pressure used to calculate Exner function [Pa] 

bomex.theta       = 298.7     # initial mixed-layer potential temperature [K]
bomex.dtheta      = 0.4       # initial temperature jump at h [K]
bomex.gammatheta  = None      # free atmosphere potential temperature lapse rate [K m-1]
bomex.advtheta_ml = -2 / (24 * 3600)  # advection of heat to mixed-layer [K s-1]
bomex.advtheta_ft = -2 / (24 * 3600)  # advection of heat to free-troposphere [K s-1]
bomex.beta        = 0.15      # entrainment ratio for virtual heat [-]  
bomex.wtheta      = 8e-3      # surface kinematic heat flux [K m s-1]

bomex.q          = 16.3e-3   # initial mixed-layer specific humidity [kg kg-1]  16.3
bomex.dq         = -0.2e-3   # initial specific humidity jump at h [kg kg-1]
bomex.gammaq     = None      # free atmosphere specific humidity lapse rate [kg kg-1 m-1]
bomex.advq_ml    = -1.2*1e-8 # advection of moisture to the mixed-layer [kg kg-1 s-1]
bomex.advq_ft    = 0         # advection of moisture to the frree-troposphere [kg kg-1 s-1]
bomex.wq         = 5.2e-5    # surface kinematic moisture flux [kg kg-1 m s-1]

bomex.CO2        = 422.      # initial mixed-layer CO2 [ppm]
bomex.dCO2       = -44.      # initial CO2 jump at h [ppm]
bomex.gammaCO2   = 0.        # free atmosphere CO2 lapse rate [ppm m-1]
bomex.advCO2     = 0.        # advection of CO2 [ppm s-1]
bomex.wCO2       = 0.        # surface kinematic CO2 flux [ppm m s-1]

bomex.sw_wind    = False     # prognostic wind switch
bomex.u          = -8.75        # initial mixed-layer u-wind speed [m s-1]
bomex.du         = 4.        # initial u-wind jump at h [m s-1]
bomex.gammau     = 0.        # free atmosphere u-wind speed lapse rate [s-1]
bomex.advu       = 0.        # advection of u-wind [m s-2]

bomex.v          = -4.0      # initial mixed-layer u-wind speed [m s-1]
bomex.dv         = 4.0       # initial u-wind jump at h [m s-1]
bomex.gammav     = 0.        # free atmosphere v-wind speed lapse rate [s-1]
bomex.advv       = 0.        # advection of v-wind [m s-2]

bomex.sw_sl      = False     # surface layer switch
bomex.ustar      = 0.3       # surface friction velocity [m s-1]
bomex.z0m        = 0.02      # roughness length for momentum [m]
bomex.z0h        = 0.002     # roughness length for scalars [m]

bomex.sw_rad     = False     # radiation switch
bomex.lat        = 51.97     # latitude [deg]
bomex.lon        = -4.93     # longitude [deg]
bomex.doy        = 268.      # day of the year [-]
bomex.tstart     = 0         # time of the day [h UTC]
bomex.cc         = 0.0       # cloud cover fraction [-]
bomex.Q          = 400.      # net radiation [W m-2] 
bomex.dFz        = 0.        # cloud top radiative divergence [W m-2] 

bomex.sw_ls      = False     # land surface switch
bomex.ls_type    = 'js'      # land-surface parameterization ('js' for Jarvis-Stewart or 'ags' for A-Gs)
bomex.wg         = 0.21      # volumetric water content top soil layer [m3 m-3]
bomex.w2         = 0.21      # volumetric water content deeper soil layer [m3 m-3]
bomex.cveg       = 0.85      # vegetation fraction [-]
bomex.Tsoil      = 285.      # temperature top soil layer [K]
bomex.T2         = 286.      # temperature deeper soil layer [K]
bomex.a          = 0.219     # Clapp and Hornberger retention curve parameter a
bomex.b          = 4.90      # Clapp and Hornberger retention curve parameter b
bomex.p          = 4.        # Clapp and Hornberger retention curve parameter c
bomex.CGsat      = 3.56e-6   # saturated soil conductivity for heat

bomex.wsat       = 0.472     # saturated volumetric water content ECMWF config [-]
bomex.wfc        = 0.323     # volumetric water content field capacity [-]
bomex.wwilt      = 0.171     # volumetric water content wilting point [-]

bomex.C1sat      = 0.132     
bomex.C2ref      = 1.8

bomex.LAI        = 2.        # leaf area index [-]
bomex.gD         = 0.0       # correction factor transpiration for VPD [-]
bomex.rsmin      = 110.      # minimum resistance transpiration [s m-1]
bomex.rssoilmin  = 50.       # minimun resistance soil evaporation [s m-1]
bomex.alpha      = 0.25      # surface albedo [-]

bomex.Ts         = 290.      # initial surface temperature [K]

bomex.Wmax       = 0.0002    # thickness of water layer on wet vegetation [m]
bomex.Wl         = 0.0000    # equivalent water layer depth for wet vegetation [m]

bomex.Lambda     = 5.9       # thermal diffusivity skin layer [-]

bomex.c3c4       = 'c3'      # Plant type ('c3' or 'c4')

bomex.sw_cu        = False     # Cumulus parameterization switch
bomex.dz_h         = 150.      # Transition layer thickness [m]
bomex.phi_cu       = 0.4           # scaling factor (qtcc - q) = phi_cu * q2_h**0.5. value of 0.51 suggested by Van Stratum et al. (2014)
bomex.wcld_prefact = 0.84    # scaling factor wcc = wcld_prefact * w_ml

# Time dependent surface variables; linearly interpolated by the model
# size of time dependent variables should be the same size as time

bomex.timedep    = {}        # example input: bomex.timedep = {'acc': (time, acc)} 


# Binned height dependent lapse rates 
# size of height dependent variables should be z.size - 1
z1         = np.array([0, 1480, 2000, 5500])
gammatheta = np.array([3.85e-3, 11.15e-3, 3.65e-3])

z2         = np.array([0, 1480, 2000, 5500])
gammaq     = np.array([-5.833e-6, -12.5e-6, -1.2e-6])

bomex.heightdep  = {'gammatheta': (z1, gammatheta),
                  'gammaq':     (z2, gammaq)}


bomex.sw_rhtend       = True       # diagnose RH_h tendency factors

bomex.sw_plume        = False       # diagnose vertical velocity at LFC using entraining plume model
bomex.zmax            = 4990            # maximum simulation height for plume model [m]
bomex.n_pts           = 499           # maximum number of points
bomex.ent_corr_factor = 0.7  # factor controlling lateral entrainment rate in plume model [-]

bomex.sw_cin          = False       # use plume data to (potentially) reduce mass-flux
bomex.sw_store        = False       # use tropopheric storage module
bomex.hstore          = 1.5e3         # cloud depth [m]

bomex.sw_acc_sikma    = False        # use parametrization of cloud core fraction from Sikma and Ouwersloot (2015)

"""
Init and run the model
"""
# With cloud parameterisation
r1 = model(bomex)
r1.run()

# Without cloud parameterisation
bomex.sw_cu = True
r2 = model(bomex)
r2.run()

bomex.sw_acc_sikma    = True
r4 = model(bomex)
r4.run()

bomex.sw_acc_sikma    = False
bomex.sw_plume = True
bomex.sw_cin = True
bomex.sw_store = True
r3 = model(bomex)
r3.run()

bomex.sw_acc_sikma    = True
r5 = model(bomex)
r5.run()

#%% Import LES data 


plot_r3 = True
save = True
simname = 'bomex'
simtitle = "BOMEX"
plot_les = True
plt.close('all')

fsize = 14


if plot_les:
    les_data_loc = 'LES_data'     # location of LES data
    data, z = LES_analysis.main(les_data_loc, bomex=True) 
    data.t_les = data.time / 3600 
else:
    data = None
    
plt.figure()
plt.plot(r1.out.t, r1.out.h, label='Simple')
if plot_les:
    plt.plot(data.t_les, data.h_ml, label='LES', color='k', linestyle='--')
plt.plot(r2.out.t, r2.out.h, label='Cu')
plt.plot(r4.out.t, r4.out.h, label='Cu+acc2')
if plot_r3:
    plt.plot(r3.out.t, r3.out.h, label='Cu+CIN+Sft')
plt.plot(r3.out.t, r5.out.h, label='Cu+CIN+Sft+acc2')    

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
plt.plot(r3.out.t, r5.out.theta, label='Cu+CIN+Sft+acc2')    
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
plt.plot(r3.out.t, r5.out.q*1000., label='Cu+CIN+Sft+acc2')
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
plt.plot(r1.out.t, r5.out.M, label='Cu+CIN+Sft+acc2', color='tab:purple')
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
plt.plot(r1.out.t, np.sqrt(r5.out.q2_h)*1000, label='Cu+CIN+Sft+acc2', color='tab:purple')
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
plt.plot(r1.out.t, r5.out.acc, label='Cu+CIN+Sft+acc2', color='tab:purple')
plt.ylabel(r'$a_{\mathrm{cc}}$ [-]', fontsize=fsize)
plt.xlabel('t [h]', fontsize=fsize)
# plt.ylim([0, 0.2])
# plt.legend(fontsize=12)
if save:
    plt.savefig(fname=simname +'_acc.pdf')
plt.show()


#%%
plt.figure()

plt.plot(data.t_les, (data.qtcc_accmax - data.qt_accmax)*1e3, label=r'$q_{t,cc}-\overline{q}_t$')
plt.plot(data.t_les, 2.7 * np.sqrt(data.q2_h)*1e3, label=r'2.7$\sigma_q$')
plt.ylabel('q (g/kg)', fontsize=fsize)
plt.xlabel('t (h)', fontsize=fsize)
# plt.title(simtitle, fontsize=fsize)
plt.legend(fontsize=fsize)
if save:
    plt.savefig(fname=simname +'_nu_qt.pdf')
plt.show()
#%%

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
plt.plot(data.t_les, (data.qtcc_accmax - data.qtml)*1e3, label=r'$q_{t,cc}-q_{ml}$')
plt.plot(data.t_les, 0.4 * np.sqrt(data.q2_h)*1e3, label=r'0.4$\sigma_q$')
plt.ylabel('q (g/kg)', fontsize=fsize)
plt.xlabel('t (h)', fontsize=fsize)
# plt.title(simtitle, fontsize=fsize)
plt.legend(fontsize=fsize)
if save:
    plt.savefig(fname=simname +'_phi_cu.pdf')
plt.show()
it = np.where(data.time/3600>2)[0][0]

#%%

plt.figure()
plt.plot(data.t_les, data.thlcc_accmax - data.thml, label=r'$\theta_{l,cc} - \overline{\theta}_l$')
plt.plot(data.t_les, -0.25 * np.sqrt(data.thl2_h), label=r'$-0.2\sigma_{\theta_l}$')
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
    esat = 610.78 * np.exp(17.2694 * (temp - 273.16)/(temp - 35.86))
    qsat = Rd / Rv * esat / (pref + (1 - Rd / Rv) * esat)
    # qsat = 0.622 * esat / pref  # approximation for mixed-layer 
    return qsat

p0 = 1e5
Rd = 287
Rv = 461.5
cp = 1005
Lv = 2.5e6

h_lfc = data.z[data.i_acc_max]
q_ft0 = bomex.q + bomex.dq + 5.833e-6 * bomex.h
qprof = plume.calc_input_prof(data.z, q_ft0, bomex.h, z2, gammaq, bomex.q)
q_lfc = np.interp(h_lfc, data.z, qprof)
theta_ft0 = bomex.theta + bomex.dtheta - 3.85e-3 * bomex.h
thetaprof = plume.calc_input_prof(data.z, theta_ft0, bomex.h, z1, gammatheta, bomex.theta)
theta_lfc = np.interp(h_lfc, data.z, thetaprof)


pres_accmax =  data.pres[np.arange(data.time.size), data.i_acc_max]

exner_accmax = (pres_accmax / p0) ** (Rd / cp)
qsat_approx = calc_sat(exner_accmax * theta_lfc, pres_accmax)


it = 400

theta = data.thl + Lv / cp * data.ql
th_lfc = theta[np.arange(len(data.time)), data.i_acc_max]
th_lfc[data.i_acc_max == 0] = np.nan

temp_lfc = data.temp[np.arange(len(data.time)), data.i_acc_max]
temp_lfc[data.i_acc_max == 0] = np.nan
qsat_approx2 = calc_sat(exner_accmax * theta_lfc, pres_accmax) 

plt.figure()
plt.plot(data.qsat[it], data.z)
plt.plot(data.qt[it], data.z)
plt.vlines(qsat_approx2[it], 0, 3e3)
# plt.vlines(qsat_approx3, 0, 3e3)
# plt.vlines(q_lfc[it], 0, 3e3)
plt.vlines(data.qsat_accmax[it], 0, 3e3)
plt.hlines(data.z[data.i_acc_max[it]], 0.0025, 0.0225)
plt.show()


#%%
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

