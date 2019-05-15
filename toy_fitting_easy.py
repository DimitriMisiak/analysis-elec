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

import mcmc_red as mcr
import ethem as eth
from config import evad

# close all plots
plt.close('all')

L = 1.
fs = 1e3
N = int(fs*L)

freq_array = np.flip(np.arange(fs/2., 0., -L**-1), axis=0)


# System Simulation
ref_bath = eth.System.Capacitor_f

param_sym = (ref_bath.capacity,
             ref_bath.i_a1,
             ref_bath.i_a2,
             ref_bath.i_a3,
             ref_bath.e_a1,
             ref_bath.e_a2,
             ref_bath.e_a3,
             )

sys_noise_fun = eth.noise_tot_param(param_sym, evad, ref_bath)

def sys_noise(param):
    return sys_noise_fun(param)(freq_array)



ndim = len(param_sym)

# fake data
p0 = [evad[p] for p in param_sym]

noise0 = sys_noise(p0)

blob = np.random.normal(0, 1e-17, len(noise0))

noise0 += blob

# comparison function
def chi2(param):
    return mcr.chi2_simple(noise0, sys_noise(param), 1e-17)


# XXX MCMC
# save directory
sampler_path = 'mcmc_sampler/autosave'

# running the mcmc analysis
#bounds = ((1e5, 1e7),(1e-12, 1e-10),)
bounds = [(p/10, p*10) for p in p0]
sampler = mcr.mcmc_sampler(chi2, bounds, nsteps=1000, path=sampler_path)

# loading the mcmc results
logd, chain, lnprob, acc = mcr.get_mcmc_sampler(sampler_path)

lab = tuple(['$p${}'.format(i) for i in range(ndim)])

dim = int(logd['dim'])
xopt, inf, sup = mcr.mcmc_results(dim, chain, lnprob, acc, lab)

print(xopt, inf, sup)

### PLOT FIT
plt.figure('PLOT FIT')

plt.loglog(freq_array, noise0, label='data')

plt.loglog(freq_array, sys_noise(xopt), label='mcmc fit')

plt.grid(True)
plt.legend()


