#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Handy MCMC scripts.

Test for the different fit method (mcmc, ptmcmc, minimizer).

Author:
    Dimitri Misiak (misiak@ipnl.in2p3.fr)
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import scipy.signal as sgl
from os import path
import scipy.optimize as op

from import_package import custom_import

custom_import()

# import personal package
import mcmc_red as mcr
import ethem as eth

# import varaibles from files
from config import evad
from plot_data import noise_dict, name_array, temp_array, res_array

# close all plots
plt.close('all')

plt.figure('PLOT FIT')

lpsd_dict = noise_dict

psd_dict = dict()
for k,v in lpsd_dict.iteritems():

    if temp_array[np.where(name_array==k)[0][0]] < 0.018:
        continue

    freq_array, lpsd_array = v
    psd_array = lpsd_array**2
    v_new = np.vstack((freq_array, psd_array))
    psd_dict[k] = v_new

    plt.loglog(freq_array, psd_array, label=k)

plt.legend()
plt.grid(True)

# System Simulation
ref_bath = eth.System.Capacitor_f

param_top = (eth.System.Resistor_ntd.temperature,
             eth.System.Resistor_ntd.resistivity
             )

param_bot = (ref_bath.capacity,
             ref_bath.i_a1,
             ref_bath.i_a2,
             ref_bath.i_a3,
             ref_bath.e_a1,
             ref_bath.e_a2,
             ref_bath.e_a3,
             )

param_full = param_top + param_bot

sys_noise_fun = eth.noise_tot_param(param_full, evad, ref_bath)


def sys_noise(param):

    sys_noise_dict = dict()
    for name, temp, res in zip(name_array, temp_array, res_array):

        p_top = (temp, res)
        p_full = p_top + tuple(param)

        freq_array = noise_dict[name][0]
        sys_noise_psd = sys_noise_fun(p_full)(freq_array)

        sys_noise_array = np.vstack((freq_array, sys_noise_psd))

        sys_noise_dict[name] = sys_noise_array

    return sys_noise_dict

p0 = [evad[p] for p in param_bot]
p0 = [ 5.61409929e-11,  5.32513485e-19,  6.25502917e-17,  1.12350002e-23,
        1.10098069e-09, -1.24148795e-08,  5.12412524e-08]
noise0 = sys_noise(p0)

fake_noise_dict = dict()
for k,v in noise0.iteritems():

    freq_array, psd_array = v

    sigma = psd_array / 120**0.5

    blob = np.random.normal(0, 1, psd_array.shape)

    blob_array = blob * sigma

    fake_noise_dict[k] = np.vstack((freq_array, psd_array+blob_array))


plt.figure('INITIAL')

for k,v in fake_noise_dict.iteritems():
    plt.loglog(v[0], v[1], label=k)
plt.legend()
plt.grid(True)

def chi2_dict(dict1, dict2):

    chi2_tot = 0

    for key, noise_array in dict1.iteritems():

        psd_array_1 = noise_array[1]
        psd_array_2 = dict2[key][1]

        sigma_array = psd_array_1 / 120**0.5

        chi2_tot += mcr.chi2_simple(psd_array_1, psd_array_2, sigma_array)

    return chi2_tot

def chi2(param):

    sim_noise_dict = sys_noise(param)

#    x2 = chi2_dict(psd_dict, sim_noise_dict)
    x2 = chi2_dict(fake_noise_dict, sim_noise_dict)

    print x2

    return x2


print "Chi2= ", chi2(p0)
print "Number of Samples= ", freq_array.shape[0] * len(temp_array)



#P0=p0
#P0 = [ 5.61409929e-11,  2.32513485e-18,  7.25502917e-17,  1.12350002e-22,
#        1.10098069e-09, -1.24148795e-08,  5.12412524e-08]

plt.close('all')

P0 = [ 5.61409929e-10,  5.32513485e-18,  6.25502917e-18,  1.12350002e-22,
        1.10098069e-10, -1.24148795e-09,  5.12412524e-10]

result = op.minimize(chi2, P0, method='nelder-mead')

popt = P0
popt = result.x

model_dict = sys_noise(P0)
for k,v in psd_dict.iteritems():
    plt.loglog(model_dict[k][0], model_dict[k][1], color='k')
    plt.loglog(v[0], v[1])

# XXX MCMC
# save directory
sampler_path = 'mcmc_sampler/autosave'

# running the mcmc analysis
bounds = [(p/100, p*100) for p in p0]
sampler = mcr.mcmc_sampler(chi2, bounds, nsteps=1000, path=sampler_path)

# loading the mcmc results
logd, chain, lnprob, acc = mcr.get_mcmc_sampler(sampler_path)

lab = [p.name for p in param_bot]

dim = int(logd['dim'])
xopt, inf, sup = mcr.mcmc_results(dim, chain, lnprob, acc, tuple(lab))

print xopt, inf, sup

### PLOT FIT

plt.loglog(freq_array, sys_noise(xopt), label='mcmc fit')

plt.grid(True)
plt.legend()


