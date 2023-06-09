# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 10:36:03 2021

@author: Wouter Koks
"""
import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
import os




class vars_save:
    def __init__(self, shape_prof):
        self.thl = np.zeros(shape_prof)  #t, z
        self.qt = np.zeros(shape_prof)
        self.ql = np.zeros(shape_prof)
        self.thv = np.zeros(shape_prof)
        self.qt2r = np.zeros(shape_prof)
        self.qt = np.zeros(shape_prof)
        self.zi = np.zeros(shape_prof)  
    
    
def tmean(var, t, d2t):
    if len(var) < (t + d2t):
        print('ERROR')
    var_r = var[t - d2t:t + d2t + 1]
    return np.mean(var_r, axis=0)


def initial_profs(data):
    it = 350
    print(data.time[it]/3600)
    dz = data.z[1] - data.z[0]


    plt.figure()
    plt.plot(data.thl[0, :], data.z)
    plt.plot(data.thl[it, :], data.z)

    plt.title('Initial profile')
    plt.ylabel('z (m)')
    plt.xlabel(r'$\theta_l$ (K)')
    plt.show()
    plt.figure()
    plt.plot(data.qt[0, :], data.z)
    plt.plot(data.qt[it, :], data.z)
    plt.show()
    return 

    

def make_fits(data, bomex=False):
    '''Calculate h_ml using fit or DALES' output, calculate thml and qvml using fit. '''
    
    thml = np.zeros_like(data.time)
    qtml = np.zeros_like(data.time)
    h_ml = np.zeros_like(data.time)
    if bomex: 
        hmax = 1e3
    else:
        hmax = 2e3
    imax = np.searchsorted(data.z, hmax)
    
    for it in range(data.time.size):
        # h_ml = data.zi[it]
        wthv = data.wthv[it, :imax]
        ih = np.argmin(wthv)
        h_ml[it] = data.z[ih]

        thml[it] = np.mean(data.thl[it, :ih]+Lv/cp*data.ql[it, :ih])
        qtml[it] = np.mean(data.qt[it, :ih])
        # wthv = data.wthv[it]
        # plt.plot(wthv[1:], z)
        # plt.hlines(h_ml[it], np.min(wthv), np.max(wthv))
        # plt.show()
        
    return thml, qtml, h_ml

        

def main(upper_dir, bomex=False, info=True):

    if bomex:
        default_dir = upper_dir + '/bomex.default.0000000.nc'
        core_dir = upper_dir + '/bomex.qlcore.0000000.nc'
        ql_dir = upper_dir + '/bomex.ql.0000000.nc'
    else:
        default_dir = upper_dir + '/arm.default.0000000.nc'
        core_dir = upper_dir + '/arm.qlcore.0000000.nc'
        ql_dir = upper_dir + '/arm.ql.0000000.nc'
    tstart = 0
    prof_test = nc.Dataset(default_dir)
    z = np.array(prof_test['z'])
    t = np.array(prof_test['time'])
    t_range = np.where(t >= tstart)
        
    shape_prof = (len(t[t_range]), len(z))
    data = vars_save(shape_prof)
    prof = nc.Dataset(default_dir)
    core = nc.Dataset(core_dir)
    ql = nc.Dataset(ql_dir)
    
    data.z = z
    data.time = t[t_range]
    data.pres = np.array(prof['thermo/phydro'])[t_range]

    data.thl = np.array(prof['thermo/thl'])[t_range]
    data.temp = np.array(prof['thermo/T'])[t_range]
    data.qt = np.array(prof['thermo/qt'])[t_range]
    data.ql = np.array(prof['thermo/ql'])[t_range]
    data.thv = np.array(prof['thermo/thv'])[t_range]
    data.wthv = np.array(prof['thermo/thv_flux'])[t_range]
    data.wthl = np.array(prof['thermo/thl_flux'])[t_range]
    data.wq = np.array(prof['thermo/qt_flux'])[t_range]
    data.qt_2 = np.array(prof['thermo/qt_2'])[t_range]
    data.thl_2 = np.array(prof['thermo/thl_2'])[t_range]
    # data.qt2r = np.array(prof['thermo/qt2r'])[t_range]
    data.zi = np.array(prof['thermo/zi'])[t_range]
    data.qsat = np.array(prof['thermo/qsat'])[t_range]
    data.acc = np.array(core['default/areah'])[t_range]
    data.wcc = np.array(core['default/w'])[t_range]
    data.qtcc = np.array(core['thermo/qt'])[t_range]
    data.thlcc = np.array(core['thermo/thl'])[t_range]
    data.thlcc[data.thlcc > 1e3] = np.nan
    data.wqM = np.array(core['thermo/qt_flux'])[t_range]
    data.wqM[data.wqM > 1] = np.nan
    data.wthetaM = np.array(core['thermo/thl_flux'])[t_range]
    data.wthetaM[data.wthetaM > 1e3] = np.nan

    
        # determine mixed-layer quantities based on h_ml=height at which wthv is minimized. 
    data.thml, data.qtml, data.h_ml = make_fits(data, bomex=bomex)
    
    ind = np.where(data.wthv[:, 0] > 0)
    
    # determine cloud core massflux as in Van Stratum 2014. 
    data.wstar = np.zeros(data.time.size)
    data.wstar[ind] = (9.81 * data.h_ml[ind] * data.wthv[ind, 0] / data.thv[ind, 0]) ** (1. / 3.)       
    data.i_acc_max = np.argmax(data.acc, axis=1)
    iacc = data.i_acc_max
    i_arr = np.arange(data.time.size)
    data.qsat_accmax = data.qsat[i_arr, data.i_acc_max]
    data.qsat_accmax[iacc == 0] = np.nan
    data.qsat_accmax[data.qsat_accmax > 1] = np.nan
    data.acc_max = np.max(data.acc, axis=1)
    data.Mcc = data.acc[i_arr, data.i_acc_max] * data.wcc[i_arr, data.i_acc_max]
    data.Mcc[data.Mcc > 1] = np.nan 
    data.Mcc[iacc == 0] = np.nan

    data.wqM_accmax = data.wqM[i_arr, data.i_acc_max]
    data.wqM_accmax[iacc == 0] = np.nan
    
    data.qtcc_accmax = data.qtcc[i_arr, data.i_acc_max]
    data.qtcc_accmax[iacc == 0] = np.nan
    data.qtcc_accmax[data.qtcc_accmax > 1] = np.nan
    
    data.qt_accmax = data.qt[i_arr, data.i_acc_max]
    data.qt_accmax[data.qtcc_accmax > 1] = np.nan
    data.qt_accmax[iacc == 0] = np.nan

    
    data.q2_accmax = data.qt_2[i_arr, data.i_acc_max]
    data.q2_accmax[data.q2_accmax > 1] = np.nan
    data.q2_accmax[iacc == 0] = np.nan 

    
    data.thlcc_accmax = data.thlcc[i_arr, data.i_acc_max]
    data.thlcc_accmax[data.thlcc_accmax > 1e3] = np.nan
    data.thlcc_accmax[iacc == 0] = np.nan 
    
    data.thl_accmax = data.thl[i_arr, data.i_acc_max]
    data.thl_accmax[data.thl_accmax > 1e3] = np.nan
    data.thl_accmax[iacc == 0] = np.nan 
    
    data.wthl_accmax = data.wthl[i_arr, data.i_acc_max]  
    
    data.q2_h = np.zeros_like(data.time)
    data.thl2_h = np.zeros_like(data.time)
    

    
    for it in range(len(data.time)):
        if not np.isnan(data.h_ml[it]):
            data.q2_h[it] = np.interp(data.h_ml[it], data.z, data.qt_2[it])  # find q2_h at h using interpolation
            data.thl2_h[it] = np.interp(data.h_ml[it], data.z, data.thl_2[it])  # find thl2_h at h using interpolation


    data.q2_h[np.logical_or(data.q2_h == 0, data.q2_h > 1e3)] = np.nan
    data.thl2_h[np.logical_or(data.thl2_h == 0, data.thl2_h > 1e3)] = np.nan
    
    return data, z
    

Rd= 287
Rv = 461
cp = 1.004e+3
Lv = 2.5e+6
tv_const = 0.608


