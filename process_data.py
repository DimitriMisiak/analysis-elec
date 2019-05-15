#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Plotting the experimental data
@author: misiak
"""

import matplotlib.pyplot as plt
import numpy as np

from import_package import custom_import
from raw_data import get_raw_data

data_dir, output_dir = custom_import()

def crop_parasite_signal(x_data, y_data, window_length):
    #etalon : très haute fréquence proche du 5000Hz
    LEN = len(x_data)
    start = LEN - window_length
    end = 20
    critic_ratio = 1.2
    mean_w = np.mean(y_data[start - window_length: start])
    y_data_pure = []
    x_data_pure = []
    mean_pure = []
    ratio_list = []
    LIST = []
    y = 0
    n = 0
    #print 'Starting with mean at :', mean_w
    for i in range(LEN):
        i_rev = LEN - i - 1
        LIST.insert(0,i_rev)
        if i_rev > start:
            y_data_pure.insert(0, y_data[i_rev])
            x_data_pure.insert(0, x_data[i_rev])
            mean_pure.insert(0, y_data[i_rev])
            ratio_list.insert(0, 1.0)
        elif i_rev < end:
            y_data_pure.insert(0, y_data[i_rev])
            x_data_pure.insert(0, x_data[i_rev])

            mean_pure.insert(0, y_data[i_rev])
            ratio_list.insert(0, 1.0)
        else:
            ratio = y_data[i_rev] / mean_w
            ratio_list.insert(0, ratio)
            if ratio < critic_ratio:
                y_data_pure.insert(0, y_data[i_rev])
                x_data_pure.insert(0, x_data[i_rev])
                mean_pure.insert(0, mean_w)
                L = y_data_pure[1 : 1 + window_length]
                mean_w = np.mean(L)
                y += 1
            else:
                n += 1
    print('Number of Rejected Frequencies =', n)
    return np.array(x_data_pure), np.array(y_data_pure), mean_pure, ratio_list

def process_data():
    """ Process the raw noise data. Create the processed data files in the
    specified directory. No return for this function.
    """
    name_array, temp_array, res_array, xy_data_list = get_raw_data()

    for name, xy in zip(name_array,xy_data_list):
        freq, lpsd = xy
        freq_process, lpsd_process,_,_ = crop_parasite_signal(freq, lpsd, 10)

        psd_process = lpsd_process**2

        cut_index = np.where(freq_process > 1e4)[0][0]

        sp = ''.join((
            data_dir,
            '/Bruit_AC/processed_data/{}_PSD_Processed.txt'.format(name)
        ))

        xy_process = np.vstack((freq_process, psd_process))[:, :cut_index].T
        np.savetxt(sp, xy_process)


def get_processed_data():
    """ Return the processed noise data. For a different set of data, edit this
    function.

    Return
    ======
    name_array: array of stream names.

    temp_array: array of temperature.

    res_array: array of NTD resistance.

    xy_data_list: list of freq,lpsd arrays.
    """

    data_dir, output_dir = custom_import()

    data_dir_ac = ''.join((data_dir, '/Bruit_AC/processed_data'))

    log_path = ''.join((data_dir_ac, '/log_data'))

    name_array, temp_list, res_list = np.loadtxt(log_path, dtype=str,
                                                 unpack=True)

    temp_array = temp_list.astype(float)
    res_array = res_list.astype(float)

    xy_data_list = list()
    ### MUCH DIRTY SO WOW !!! fix that properly, classes may do it, idk
    valid_ind = np.where(temp_array>=0.018)[0]

#    for i,name in enumerate(name_array):
    for i in valid_ind:

        file_name = '{}_PSD_Processed.txt'.format(name_array[i])
        file_path = '/'.join((data_dir_ac, file_name))

        freq_array, lpsd_array = np.loadtxt(file_path, unpack=True)

        xy_data = np.vstack((freq_array, lpsd_array))
        xy_data_list.append(xy_data)

    return (name_array[valid_ind], temp_array[valid_ind],
         res_array[valid_ind], xy_data_list)


if __name__ == '__main__':

    ### Decomment for processing
#    process_data()

    name_array, temp_array, res_array, raw_list = get_raw_data()
    name_array, temp_array, res_array, process_list = get_processed_data()

    ind_sort = temp_array.argsort()

    cmap = plt.get_cmap('magma')

    plt.close('all')
    fig, ax = plt.subplots(nrows=2, num='raw noise vs processed noise',
                           figsize=(10,14))

    for i in ind_sort:
        name = name_array[i]
        temp = temp_array[i]
        res = res_array[i]
        freq_process, psd_process = process_list[i]
        freq_raw, lpsd_raw = raw_list[i]
        color = cmap((i+1.)/(len(ind_sort)+1))

        ax[0].loglog(freq_raw, lpsd_raw**2,
                     label='{}, {} K, {:.2e} $\Omega$'.format(name, temp, res),
                     color=color)

        ax[1].loglog(freq_process, psd_process,
                   label='{}, {} K, {:.2e} $\Omega$'.format(name, temp, res),
                   color=color)

    ax[0].legend(title='Raw Data')
    ax[1].legend(title='Processed Data')

    for a in ax:
        a.grid(True)
