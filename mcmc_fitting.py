#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Handy MCMC scripts.

Test for the different fit method (mcmc, ptmcmc, minimizer).

Author:
    Dimitri Misiak (misiak@ipnl.in2p3.fr)
"""

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import scipy.optimize as op


# importing ethem and mcmc-red package from custom paths
from import_package import custom_import
data_dir, output_dir = custom_import()
import mcmc_red as mcr

from config import get_eval_dict
from model import initiate_model, make_model_data, make_fake_data
from process_data import get_processed_data
from comparator import chi2_model

#### dirty !!! should be better with classes
#name_array, temp_array, res_array, exp_data = get_processed_data()
#
#plt.close('all')
#
evad = get_eval_dict()
param_top, param_bot, model_fun = initiate_model()
#
p0 = [evad[p] for p in param_bot]

#from tqdm import trange
#for i in trange(10):
#    A = chi2_model(p0)

#p1 = [evad[p]*0.9 for p in param_bot]
#
#model0 = make_model_data(p0)
#
#plt.figure('model0 vs Exp plot')
#for md, ed in zip(model0, exp_data):
#
#    mfreq, mpsd = md
#    efreq, epsd = ed
#
#    plt.loglog(efreq, epsd, lw=0.5)
#    plt.loglog(mfreq, mpsd,
#               path_effects = [pe.Stroke(linewidth=3, foreground='k'),
#                               pe.Normal()]
#               )
#

#### XXX MINIMIZER
#P0 = [ 5.61409929e-10,  5.32513485e-18
#      ,  6.25502917e-18,  1.12350002e-22,
#        1.10098069e-10, -1.24148795e-09,  5.12412524e-10]
#
#chi2_aux = lambda x: chi2_model(x, check_print=True)
#result = op.minimize(chi2_aux, P0, method='nelder-mead')
#
#plt.figure()
#model_opt = make_model_data(result.x)
#
#for md in model_opt:
#    plt.loglog(md[0], md[1])



# XXX MCMC
# save directory
sampler_path = output_dir + '/mcmc_sampler/autosave'

# running the mcmc analysis
bounds = [(p/100, p*100) for p in p0]
sampler = mcr.mcmc_sampler(chi2_model, bounds, nsteps=1000, path=sampler_path)


# loading the mcmc results
logd, chain, lnprob, acc = mcr.get_mcmc_sampler(sampler_path)

lab = [p.name for p in param_bot]

dim = int(logd['dim'])
xopt, inf, sup = mcr.mcmc_results(dim, chain, lnprob, acc, tuple(lab))

print(xopt, inf, sup)

#### PLOT FIT
#
#plt.loglog(freq_array, sys_noise(xopt), label='mcmc fit')
#
#plt.grid(True)
#plt.legend()


