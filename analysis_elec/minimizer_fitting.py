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
import numpy as np

# importing ethem and mcmc-red package from custom paths
from import_package import custom_import
custom_import()
import mcmc_red as mcr

from config import get_eval_dict
from model import initiate_model, make_model_data, make_fake_data
from process_data import get_processed_data
from comparator import chi2_model

#### dirty !!! should be better with classes
name_array, temp_array, res_array, exp_data = get_processed_data()

plt.close('all')
plt.figure('explore plot', figsize=(10,14))
for edata in exp_data:
    freq, psd = edata
    plt.loglog(freq, psd, 'k', lw=0.5)

evad = get_eval_dict()
param_top, param_bot, model_fun = initiate_model()

#p0 = [evad[p] for p in param_bot]

cmap = plt.get_cmap('jet')

param_ind = 1
coeff_range = 10**np.linspace(0, 1, 10)

color_range = cmap(np.arange(len(coeff_range)) / float(len(coeff_range)))

p1 = [2.36e-10, 1e-15, 5e-16, 1e-19, 1e-10, 1e-08, 1e-08, 2e-9]
#p1 = tuple([2.36e-10, 1e-16, 1e-17, 2e-18, 1e-09, 2e-08, 2e-08])
p1 = tuple([4.8e-11, 6.17e-18, 2.76e-19, 1.28e-18, 1.6e-9, 2.39e-8, 7.81e-9, 2e-9])
p1 =[ 6.90604978e-11, -7.17442202e-16,  2.53899131e-17,  3.48881489e-19,
        1.63630904e-09,  8.91374843e-09, -3.50057415e-08,  2.44146304e-08]

for coeff, color in zip(coeff_range, color_range):
    p0=list(p1)
    p0[param_ind] *= coeff
    model_data = make_model_data(p0)

    for i,mdata in enumerate(model_data):
        freq, psd = mdata
        plt.loglog(freq, psd, color=color,
                   path_effects=[pe.Stroke(linewidth=3, foreground='k'),
                                 pe.Normal()],
                   lw=2)
        if not i:
            plt.loglog(freq, psd, label=p0[param_ind], color=color)

plt.legend()
plt.grid(True)

model0 = make_model_data(p1)

plt.figure('model0 vs Exp plot')
for md, ed in zip(model0, exp_data):

    mfreq, mpsd = md
    efreq, epsd = ed

    plt.loglog(efreq, epsd, lw=0.5)
    plt.loglog(mfreq, mpsd,
               path_effects = [pe.Stroke(linewidth=3, foreground='k'),
                               pe.Normal()]
               )


### XXX MINIMIZER

chi2_aux = lambda x: chi2_model(x, check_print=True)
result = op.minimize(chi2_aux, p1, method='nelder-mead')

plt.figure()
model_opt = make_model_data(result.x)

for md in exp_data:
    plt.loglog(md[0], md[1])

for md in model_opt:
    plt.loglog(md[0], md[1])


