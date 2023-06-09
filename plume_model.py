# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 11:31:12 2023
Lower boundary is the surface.
Run calculations without arrays, only save later to arrays
@author: Wouter Koks

"""

import numpy as np 
import matplotlib.pyplot as plt
from numba import jit
    
    
class Env_sv:
    """Static environment profiles"""
    def __init__(self, l):
        self.z     = np.zeros(l)
        self.qsat  = np.zeros(l)
        self.thv   = np.zeros(l)
        self.pres  = np.zeros(l)
        self.exner = np.zeros(l)
        self.thl   = np.zeros(l)
        self.qt    = np.zeros(l)
        
    def remove_fromi(self, i_f):
        '''Remove unsimulated zeros above LNB from output'''
        self.z  = self.z[:i_f]
        self.qsat = self.qsat[:i_f]
        self.thv = self.thv[:i_f]
        self.pres = self.pres[:i_f]
        self.exner = self.exner[:i_f]
        self.qt = self.qt[:i_f]
        self.thl = self.thl[:i_f]
        

class Env:
    """Static environment profiles"""
    def __init__(self):
        self.z     = None
        self.qsat  = None
        self.thv   = None
        self.pres  = None
        self.exner = None
        self.thl   = None
        self.qt    = None
        
        self.dz    = None
    
    def obtain(self, sv, i):
        self.z = sv.z[i]
        self.qsat = sv.qsat[i]
        self.thv = sv.thv[i]
        self.pres = sv.pres[i]
        self.exner = sv.exner[i]
        self.thl = sv.thl[i]
        self.qt = sv.qt[i]
        


class Parcel_sv:
    """Properties/profiles of plume"""
    def __init__(self, l):
        self.thv = np.zeros(l)
        self.temp = np.zeros(l)
        self.qt = np.zeros(l)
        self.thl = np.zeros(l)
        self.qsat = np.zeros(l)
        self.B = np.zeros(l)
        self.w = np.zeros(l)
        self.ent = np.zeros(l)
        
        self.cin = 0
        self.T0 = None
        self.wstar = None
        
        self.never_buoy = False
        self.above_lfc = False   
        self.above_lcl = False
        self.moist_bool = False
        self.parcel_dry = False
        self.inhibited = False
        
    def save(self, parcel, i):
         self.thv[i]  = parcel.thv
         self.temp[i] = parcel.temp
         self.qt[i]   = parcel.qt
         self.thl[i]  = parcel.thl
         self.B[i]    = parcel.B
         self.w[i]    = parcel.w
         self.ent[i]  = parcel.ent
         
    def remove_fromi(self, i_f):
        '''Remove unsimulated zeros above LNB from output'''
        self.thv = self.thv[:i_f]
        self.temp = self.temp[:i_f]
        self.qt = self.qt[:i_f]
        self.thl = self.thl[:i_f]
        self.qsat = self.qsat[:i_f]
        self.w   = self.w[:i_f]
        self.B   = self.B[:i_f]
        self.ent = self.ent[:i_f]
         
class Parcel:
    """Properties/profiles of plume"""
    def __init__(self):
        self.thv  = None
        self.temp = None
        self.qt   = None
        self.thl  = None
        self.qsat = None
        self.B    = None
        self.w    = None
        
        self.ent  = 0
        self.cin = 0
        self.T0 = None
        self.wstar = None
        
        self.never_buoy = False
        self.above_lfc = False   
        self.above_lcl = False
        self.moist_bool = False
        self.parcel_dry = False
        self.inhibited = False
        
        
@jit(nopython=True, nogil=True, fastmath=True)   
def calc_pres(p0, p_ref, T0, q0, gth, gq, z):
    """"Specify hydrostatic pressure profile based on constant gradient of virtual potential temp."""
    Tv0 = T0 * (1 + tv_const * q0)
    gtv = (1 + tv_const * q0) * (gth - g / cp) + tv_const * T0 * gq
    pres = p0 * (1 + gtv * z / Tv0) ** ( - g / (Rd * gtv))
    exner = (pres / p_ref) ** (Rd / cp)
    return pres, exner
  
    
@jit(nopython=True, nogil=True, fastmath=True)   
def calc_pres_zdep_gradients(ps, p_ref, T0, q0, gth_arr, gq_arr, z_bins, z_arr):
    """"Specify hydrostatic pressure profile using height-dependent gammatheta and gammaq"""
    Tv0 = T0 * (1 + tv_const * q0)
    gtv_arr = (1 + tv_const * q0) * (gth_arr - g / cp) #+ tv_const * T0 * gq_arr  # currently neglect dependence of pressure on gradient of specific humidiy for simplicity. 
    dp_zbins = (1 + gtv_arr * np.diff(z_bins) / Tv0) ** ( - g / (Rd * gtv_arr))
    dp_zbins = np.append([ps], dp_zbins)
    pres_bins = np.cumprod(dp_zbins)
    index_z = np.searchsorted(z_bins, z_arr, side='right') - 1
    pres = pres_bins[index_z] * (1 + gtv_arr[index_z] * (z_arr - z_bins[index_z]) / Tv0) ** ( - g / (Rd * gtv_arr[index_z]))
    exner = (pres / p_ref) ** (Rd / cp)
    return pres, exner
    
        

@jit(nopython=True, nogil=True, fastmath=True)                
def calc_sat(temp, pref):
    esat = 610.78 * np.exp(17.2694 * (temp - 273.16)/(temp - 35.86))
    qsat = Rd / Rv * esat / (pref + (1 - Rd / Rv) * esat)
    # qsat = 0.622 * esat / pref  # approximation for mixed-layer 
    return qsat


@jit(nopython=True, nogil=True, fastmath=True)      
def func(T, exner, pres, thl, qt):
    ql = max(0, qt - calc_sat(T, pres))
    return T - exner * (thl + Lv * ql / cp)
    

@jit(nopython=True, nogil=True, fastmath=True)        
def calc_thermo(T_i, pres_i, qt_i, exner_i, thl_i):
    T_old = 0
    eps = 1e-8
    i = 0
    while np.abs(T_i - T_old) > 1e-6 and i < 100:
        T_old = T_i
        dfdT = (func(T_old + eps, exner_i, pres_i, thl_i, qt_i) - func(T_old - eps, exner_i, pres_i, thl_i, qt_i)) / (2 * eps)
        T_i -=  func(T_old, exner_i, pres_i, thl_i, qt_i) / dfdT
        i += 1
    if i > 98:
        raise ValueError("Error: plume temperature calculation did not converge!")
 
    qsat_i = calc_sat(T_i, pres_i)
    
    ql_i = max(0, qt_i - qsat_i)
    thv_i = (T_i / exner_i) * (1 + tv_const * qt_i - (1 + tv_const) * ql_i)
    return T_i, qsat_i, ql_i, thv_i


@jit(nopython=True, nogil=True, fastmath=True)      
def calc_input_prof(z_arr, phi0, h, z_bins, gammaphi_bins, phi_ml):
        diff_phi = np.cumsum(gammaphi_bins * np.diff(z_bins))
        phi_zdep = np.append([0], diff_phi) + phi0
        index_z = np.searchsorted(z_bins, z_arr, side='right') - 1
        phi = phi_zdep[index_z] + gammaphi_bins[index_z] * (z_arr - z_bins[index_z])
        
        # set phi value in mixed-layer
        ind_h = np.searchsorted(z_arr, h) 
        phi[:ind_h] = phi_ml
        return phi
    

def init_conditions(mlm):
    '''Use MLM data to initialize first height level of parcel and env objects'''
    # environmental profile initialization
    env = Env()
    env_sv = Env_sv(mlm.input.n_pts)
    env_sv.z = np.linspace(0, mlm.input.zmax, mlm.input.n_pts)
    env.dz = env_sv.z[1] - env_sv.z[0]

    ind_h = np.searchsorted(env_sv.z, mlm.h, side='right') 

    # calculate vertical profiles based on (possibly height dependent) gradients

    if 'gammatheta' in mlm.heightdep:
        z_bins = mlm.heightdep['gammatheta'][0]
        gammatheta_bins  = mlm.heightdep['gammatheta'][1] 
        
        if z_bins[-1] < mlm.input.zmax:
            print("Increase height range of height dependent variables.")
        #calc input profile
        env_sv.thl = calc_input_prof(env_sv.z, mlm.theta_ft0, mlm.input.h, z_bins, gammatheta_bins, mlm.input.theta)
    else: 
        env_sv.thl[ind_h:] = mlm.theta_ft0 + env_sv.z[ind_h:] * mlm.input.gammatheta

    env_sv.thl[:ind_h] = mlm.theta 
    
    if 'gammaq' in mlm.heightdep:
        z_bins = mlm.heightdep['gammaq'][0]
        gammaq_bins  = mlm.heightdep['gammaq'][1] 
        if z_bins[-1] < mlm.input.zmax:
            print("Increase height range of height dependent variables.")
        #calc input profile
        env_sv.qt = calc_input_prof(env_sv.z, mlm.q_ft0, mlm.input.h, z_bins, gammaq_bins, mlm.input.q)
    else:  
        env_sv.qt[ind_h:] = mlm.q_ft0 + env_sv.z[ind_h:] * mlm.input.gammaq
    
    env_sv.qt[:ind_h] = mlm.q
    env_sv.qt[env_sv.qt < 0] = 0
    
    if mlm.input.sw_store:
        ind_store = np.searchsorted(env_sv.z, mlm.h + mlm.hstore, side='right') 
        env_sv.thl[ind_h:ind_store] +=  mlm.Stheta / mlm.hstore
        env_sv.qt[ind_h:ind_store] += mlm.Sq / mlm.hstore
    
    if 'gammatheta' in mlm.input.heightdep and 'gammaq' in mlm.input.heightdep:
        env_sv.pres, env_sv.exner = calc_pres_zdep_gradients(mlm.input.Ps, mlm.input.P_ref, mlm.theta_ft0, mlm.q_ft0, mlm.input.heightdep['gammatheta'][1], mlm.input.heightdep['gammaq'][1], mlm.input.heightdep['gammatheta'][0], env_sv.z)  # make exner array
    else:
        env_sv.pres, env_sv.exner = calc_pres(mlm.input.Ps, mlm.input.P_ref, mlm.theta_ft0, mlm.q_ft0, mlm.input.gammatheta, mlm.input.gammaq, env_sv.z)  # make exner array
    temp_env = env_sv.exner * env_sv.thl
    env_sv.Tv = temp_env * (1 + tv_const * env_sv.qt)
    env_sv.thv = env_sv.thl * (1 + tv_const * env_sv.qt)
    env_sv.qsat = calc_sat(temp_env, env_sv.pres)
    
    env.obtain(env_sv, 0)  
    
    # parcel surface initialization
    parcel = Parcel()
    
    parcel.wstar = mlm.wstar
    parcel.w = mlm.wstar 
    exner_srf = (mlm.Ps / mlm.P_ref) ** (Rd / cp)
    parcel.temp = mlm.theta * exner_srf
    parcel.qt  = mlm.q + mlm.input.phi_cu * np.sqrt(mlm.q2_h)

    qsat_p = calc_sat(parcel.temp, mlm.input.Ps)
    qvp = min(parcel.qt, qsat_p)
    parcel.ql = max(parcel.qt - qvp, 0)
    parcel.qsat = qsat_p    
    parcel.thl = mlm.theta - Lv * parcel.ql / (exner_srf * cp) 
    parcel.thv = mlm.theta * (1 + tv_const * qvp - parcel.ql) 
    return parcel, env, env_sv


def check_if_stop(parcel, env, mlm):
    """Determine whether the simulation should stop or continue. Return list of stop conditions. """
    if not parcel.above_lcl:
        if parcel.ql > 0:
            parcel.above_lcl = True
            parcel.z_lcl = env.z
        
    parcel.parcel_dry = (env.z > 4e+3 and parcel.ql == 0) 
    
    if not parcel.above_lfc: 
        parcel.above_lfc = (parcel.above_lcl and parcel.thv > env.thv and env.z > mlm.h) 
        
    if parcel.w <= 0:
        parcel.w = 0
        parcel.inhibited = True
            
    cond_lst = [parcel.parcel_dry, parcel.inhibited, parcel.above_lfc]  # list of stop conditions
    return cond_lst 


def simulate(mlm):
    '''Run parcel simulation using entraining plume model. MLM variables are used as input.'''
    #initialization
    parcel, env, env_sv = init_conditions(mlm)  # initialize profiles and surface conditions
    c1 = 1 / 2  # constants from Simpson and Wiggert (1969), Jakob and Siebesma (2003)
    c2 = 1 / 3
    
    # initialize values that will be altered during the simulation
    stop_cond = 10
    i_final = mlm.input.n_pts - 1
    lnb = np.nan  
    parcel.i_lcl = np.nan

    if mlm.input.save_ent_plume: # intialize saving vertical profiles
        parcel_sv = Parcel_sv(mlm.input.n_pts)
        parcel_sv.save(parcel, 0)
    else:
        parcel_sv = None
        
    if parcel.thv < env.thv:
            print('Surface parcel is not buoyant (severe liquid water loading?)')

    for i in range(1,mlm.input.n_pts - 1):
        # entrainment
        if env.z > mlm.h: # start entrainment from h_ml, since I initialize the plume using properties at the mixing height.
            if parcel.above_lcl:
                parcel.ent = mlm.input.ent_corr_factor * 1.15 * env.qt / (env.qsat * ((env.z - parcel.z_lcl) + 300))  # parametrization by Lu et al. (2018)
                # the fitting parameter ent_corr_factor is used to fit LES cloud core data (not cloud updraft as in Lu et al. (2018))
                if parcel.ent > 3e-3:  # impose a limit on entrainment rate roughly based on LES results. 
                    parcel.ent = 3e-3
            else:
                parcel.ent = 3e-3  # prescribe a constant entrainment rate below LCL, since the parametrization by Lu (2018) does not provide a value here
        parcel.qt  -=  parcel.ent * env.dz * (parcel.qt - env.qt) 
        parcel.thl -=  parcel.ent * env.dz * (parcel.thl - env.thl) 
        
        env.obtain(env_sv, i) # set env object data to the pre-calculated data at level i

        # calculate parcel thermodynamic properties at new height level
        parcel.temp, parcel.qsat, parcel.ql, parcel.thv = calc_thermo(parcel.temp, env_sv.pres[i], parcel.qt, env_sv.exner[i], parcel.thl)
        
        if ((parcel.thv < env.thv) and env.z > mlm.h):
            parcel.B    = g * (parcel.thv - env.thv) / env.thv
            parcel.w   +=  env.dz * (- c1 * parcel.ent * parcel.w + c2 * parcel.B / parcel.w)
            parcel.cin -= parcel.B * env.dz 
        
        # some logics to determine whether the simulation should continue
        cond_lst = check_if_stop(parcel, env, mlm)
        
        if mlm.input.save_ent_plume:  
            parcel_sv.save(parcel, i)  # save vertical profiles
        
        if sum(cond_lst):  # if the simulation should stop because any of the stop conditions is true
            stop_cond = [i for i, val in enumerate(cond_lst) if val][0]  # returns the index of cond_lst, for analysing bugs
            i_final = i - 1
            if not parcel.above_lfc:
                parcel.w = 0
                parcel.cin = np.nan
            break
    
    return stop_cond, i_final, parcel, parcel_sv, env_sv


def main(mlm):
    '''Run the entraining plume model using data from a single timestep of the CLASS model and return the velocity at the LFC'''  
    # run the entraining plume model
    mlm.input.save_ent_plume = False  # do not save output of plume model when used with CLASS
    _, _, parcel, _, _ = simulate(mlm) 
    w_lfc = parcel.w
    return w_lfc


# globals which could also be imported from CLASS
g = 9.81
Rd = 287
Rv = 461.5
tv_const = Rv / Rd - 1
Lv = 2.5e+6
cp = 1.005e+3
rho = 1.2


# Example of how this script can be used independent of the input by CLASS. 
class Sv:
    def __init__(self):
        pass

if __name__ == '__main__':
    # Initialization

    mlm = Sv
    mlm.input = Sv
    
    mlm.input.P_ref = 1e5
    mlm.h = 400  # mixed-layer height
    mlm.input.zmax = 5e3  # max simulation height
    mlm.input.n_pts = 500
    mlm.input.Ps = 1e5
    mlm.theta_ft0 = 298.6
    mlm.theta = 300

    mlm.input.gammatheta = 3.7e-3
    mlm.dtheta = mlm.theta_ft0 - mlm.theta + mlm.input.gammatheta * mlm.h
    
    mlm.q_ft0 = 0.8 * calc_sat(mlm.theta_ft0, mlm.input.Ps)
    print(mlm.q_ft0)
    mlm.q = 15e-3
    mlm.input.gammaq = - 3.5e-6
    mlm.q2_h = 7e-6  # specific humidity variance at mixing height!
    mlm.dq = mlm.q_ft0 - mlm.q + mlm.input.gammaq * mlm.h
    mlm.input.sw_store = True  # modify env profiles with tropospheric storage
    mlm.Stheta = -0 * 3e3  
    mlm.hstore = 3e3  # height over which tropospheric scalars are stored
    mlm.Sq = 0
    mlm.thetav = mlm.theta * (1 + tv_const * mlm.q)
    
    mlm.input.c_ent = 0.004  # not used if a non-constant entrainment rate is used 
    mlm.input.phi_cu = 0.51
    mlm.input.wcld_prefact = 0.84  # prefactor to estimate w_hml from wstar 
    mlm.wstar = 4 # Deardorff velocity scale
    mlm.input.ent_corr_factor = 0.7
    mlm.input.save_ent_plume = True

    z1         = np.array([0, 700, 1000, 1500, 5200])
    gammatheta = np.array([3.7, 2.5, 4, 5]) * 1e-3
    
    z2         = np.array([0, 700, 1000, 1500, 5200])
    gammaq     = np.array([0.6, 2, 8.75, 4]) * -1e-6
    
    mlm.input.heightdep  = {'gammatheta': (z1, gammatheta),
                      'gammaq':     (z2, gammaq)}
    
    # run simulation
    stop_cond, i_final, parcel, parcel_sv, env_sv = simulate(mlm) 

    if mlm.input.save_ent_plume:
        parcel_sv.remove_fromi(i_final)
        env_sv.remove_fromi(i_final)
    else:
        env_sv = None

    w_lfc = parcel.w

    # plotting
    fsize = 12
    imax = 200
    plt.figure()
    plt.plot(parcel_sv.thv[:imax], env_sv.z[:imax], label=r'$\theta_{v,p}$')
    plt.plot(env_sv.thv[:imax], env_sv.z[:imax], color='grey', linestyle='--', label=r'$\overline{\theta}_v$')
    plt.xlabel(r'$\theta_v$ (K)', fontsize=fsize)
    plt.ylabel(r'$h$ (m)', fontsize=fsize)
    plt.legend(fontsize=fsize)
    plt.show()
    
    B = parcel_sv.B
    B[np.isnan(B)] = 0

    plt.figure()
    plt.plot(B, env_sv.z, label=r'$B$')
    plt.xlabel(r'$B$', fontsize=fsize)
    plt.ylabel(r'$h$ (m)', fontsize=fsize)
    plt.legend(fontsize=fsize)
    plt.show()
    
    plt.figure()
    plt.plot(parcel_sv.w, env_sv.z, label=r'$w$')
    plt.xlabel(r'$w$', fontsize=fsize)
    plt.ylabel(r'$h$ (m)', fontsize=fsize)
    plt.legend(fontsize=fsize)
    plt.show()
    
    



    
