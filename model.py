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
import matplotlib.patheffects as pe

# importing ethem package
from import_package import custom_import
custom_import()
import ethem as eth

from config import get_eval_dict
from process_data import get_processed_data

def initiate_model():
    """ Select the parameters for the model.

    Return
    ======
    param_top: tuple
        Symbols of the primary parameters of the models.

    param_bot: tuple
        Symbols of the secondary parameters of the models.

    sys_noise_full: fun
        Simulation function from ethem.
    """

    # referance bath
    ref_bath = eth.System.Capacitor_f

    # "primary" parameters
    param_top = (eth.System.Resistor_ntd.temperature,
                 eth.System.Resistor_ntd.resistivity
                 )

    # "secondary" parameters
    param_bot = (ref_bath.capacity,
                 ref_bath.i_a1,
                 ref_bath.i_a2,
                 ref_bath.i_a3,
                 ref_bath.e_a1,
                 ref_bath.e_a2,
                 ref_bath.e_a3,
                 ref_bath.e_dac,
                 )

    param_full = param_top + param_bot

    # evaluation dictionnary
    evad = get_eval_dict()

    sys_noise_fun = eth.noise_tot_param(param_full, evad, ref_bath)

    return param_top, param_bot, sys_noise_fun

### dirty !!! should be better with classes
name_array, temp_array, res_array, data_list = get_processed_data()

def make_model_data(param):
    """ Return the model array from the given set of parameters.
    """

    param_top, param_bot, sys_noise_fun = initiate_model()

    sys_noise_list = list()
    for i in range(len(name_array)):

        temp = temp_array[i]
        res = res_array[i]
        freq_array, lpsd = data_list[i]

        p_top = (temp, res)
        p_full = p_top + tuple(param)

        sys_noise_psd = sys_noise_fun(p_full)(freq_array)

        sys_noise_array = np.vstack((freq_array, sys_noise_psd))

        sys_noise_list.append(sys_noise_array)

    return sys_noise_list


def make_fake_data(model_data, sigma_coeff=1.):
    """ Add a gaussian noise depending on the model_data.
    """
    fake_data = list()
    for md in model_data:

        freq, y_data = md

        sigma = y_data * sigma_coeff

        blob = np.random.normal(0,sigma,sigma.shape)

        noisy_data = y_data + blob

        fake_data.append(np.vstack((freq, noisy_data)))

    return fake_data


if __name__ == "__main__":

    param_top, param_bot, sys_noise_fun = initiate_model()
    evad = get_eval_dict()
    p0 = [evad[p] for p in param_bot]
#    p0 = [ 5.61409929e-12,  1.32513485e-19,  3.25502917e-17,  9.12350002e-23,
#            6.10098069e-09, -3.24148795e-08,  1.12412524e-08]

    model0 = make_model_data(p0)
    fake0 = make_fake_data(model0, sigma_coeff=120**-0.5)

    name_array, temp_array, res_array, data_list = get_processed_data()
    ind_sort = temp_array.argsort()

    cmap = plt.get_cmap('magma')

    plt.close('all')
    plt.figure('noise dict plot', figsize=(10,7))

    for i in ind_sort:
        name = name_array[i]
        temp = temp_array[i]
        res = res_array[i]
        freq, fake_psd = fake0[i]
        _, model_psd = model0[i]
        color = cmap((i+1.)/(len(ind_sort)+1))

        plt.loglog(
                freq, fake_psd,
                label='{}, {} K, {:.2e} $\Omega$'.format(name, temp, res),
                color=color,
                lw=1,
        )

        plt.loglog(
                freq, model_psd,
                label='{}, {} K, {:.2e} $\Omega$'.format(name, temp, res),
                color=color,
                lw=1,
                path_effects=[pe.Stroke(linewidth=3, foreground='k'),
                              pe.Normal()],
        )

    plt.legend()
    plt.grid(True)
