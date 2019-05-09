#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Plotting the experimental data
@author: misiak
"""

import matplotlib.pyplot as plt
import numpy as np

from import_package import custom_import


def get_raw_data():
    """ Return the raw noise data. For a different set of data, edit this
    function.

    Return
    ======
    name_array: array of stream names.

    temp_array: array of temperature.

    res_array: array of NTD resistance.

    xy_data_list: list of freq,lpsd arrays.
    """

    data_dir, output_dir = custom_import()

    data_dir_ac = ''.join((data_dir, '/Bruit_AC'))

    log_path = ''.join((data_dir_ac, '/log_data'))

    name_array, temp_list, res_list = np.loadtxt(log_path, dtype=str, unpack=True)

    temp_array = temp_list.astype(float)
    res_array = res_list.astype(float)

    xy_data_list = list()

    for i,name in enumerate(name_array):

        file_name = '{}_0_PSD_Corr.txt'.format(name)
        file_path = '/'.join((data_dir_ac, file_name))

        freq_array, lpsd_array = np.loadtxt(file_path, unpack=True)[:,1:]

        xy_data = np.vstack((freq_array, lpsd_array))
        xy_data_list.append(xy_data)

    return name_array, temp_array, res_array, xy_data_list

if __name__ == '__main__':

    name_array, temp_array, res_array, data_list = get_raw_data()

    ind_sort = temp_array.argsort()

    cmap = plt.get_cmap('magma')

    plt.close('all')
    plt.figure('noise dict plot', figsize=(10,7))

    for i in ind_sort:
        name = name_array[i]
        temp = temp_array[i]
        res = res_array[i]
        freq, lpsd = data_list[i]
        color = cmap((i+1.)/(len(ind_sort)+1))

        plt.loglog(freq, lpsd**2,
                   label='{}, {} K, {:.2e} $\Omega$'.format(name, temp, res),
                   color=color)

    plt.legend()
    plt.grid(True)