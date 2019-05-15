#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Defining the comparator function for the model fitting.

Author:
    Dimitri Misiak (misiak@ipnl.in2p3.fr)
"""

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

# importing ethem and mcmc-red package from custom paths
from import_package import custom_import
custom_import()
import mcmc_red as mcr

from config import get_eval_dict
from model import initiate_model, make_model_data, make_fake_data
from process_data import get_processed_data

### dirty !!! should be better with classes
name_array, temp_array, res_array, exp_data = get_processed_data()
norm = 120**-0.5

def chi2_list(data_1, data_2, sigma_coeff=norm):
    """ Return the summed chi2 between two given set of data.
    """
    chi2_tot = 0

    for d1, d2 in zip(data_1, data_2):

        f1, y1 = d1
        f2, y2 = d2

        sigma_array = y1 * sigma_coeff

        chi2_tot += mcr.chi2_simple(y1, y2, sigma_array)

    return chi2_tot


def chi2_model(param, check_print=False):
    """ Return the chi2 between the experimental processed data and
    the model evaluated for the given parameter set.
    """
    model_data = make_model_data(param)

    x2 = chi2_list(model_data, exp_data)

    if check_print is True:
        print(x2)

    return x2


if __name__ == "__main__":

    plt.close('all')

    evad = get_eval_dict()
    param_top, param_bot, model_fun = initiate_model()

    p0 = [evad[p] for p in param_bot]
    p1 = [evad[p]*0.9 for p in param_bot]

    model0 = make_model_data(p0)
    model1 = make_model_data(p1)

    fake0 = make_fake_data(model0, sigma_coeff=120**-0.5)

    ddf = 0
    for md in model0:
        freq, psd = md
        ddf += len(freq)

    self_chi2_model = chi2_list(model0, model0, sigma_coeff=120**-0.5)
    self_chi2_fake = chi2_list(fake0, fake0, sigma_coeff=120**-0.5)
    chi2_check_00 = chi2_list(model0, fake0, sigma_coeff=120**-0.5)
    chi2_check_01 = chi2_list(model0, model1, sigma_coeff=120**-0.5)
    chi2_check_0exp = chi2_list(model0, exp_data, sigma_coeff=120**-0.5)
    chi2_real = chi2_model(p0, check_print=True)

    print("Self Chi2 Model = ", self_chi2_model)
    print("Self Chi2 Fake = ", self_chi2_fake)
    print("Chi2 Check Model vs Fake = ", chi2_check_00)
    print("Chi2 Check Model0 vs Model1 = ", chi2_check_01)
    print("Chi2 Check Model0 vs Exp = ", chi2_check_0exp)
    print("Chi2 Model vs Exp = ", chi2_real)
    print("Total number of degrees of freedom = ", ddf)

    plt.figure('chi2_list check plot')
    for md, fd, md1 in zip(model0, fake0, model1):

        mfreq, mpsd = md
        ffreq, fpsd = fd
        mfreq1, mpsd1 = md1

        plt.loglog(ffreq, fpsd, lw=0.5)
        plt.loglog(mfreq, mpsd,
                   path_effects = [pe.Stroke(linewidth=3, foreground='k'),
                                   pe.Normal()]
                   )
        plt.loglog(mfreq1, mpsd1,
                   path_effects = [pe.Stroke(linewidth=3, foreground='gold'),
                                   pe.Normal()]
                   )



    plt.figure('chi2_model check plot')
    for md, ed in zip(model0, exp_data):

        mfreq, mpsd = md
        efreq, epsd = ed

        plt.loglog(efreq, epsd, lw=0.5)
        plt.loglog(mfreq, mpsd,
                   path_effects = [pe.Stroke(linewidth=3, foreground='k'),
                                   pe.Normal()]
                   )
