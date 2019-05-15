#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Update the sys.path in order to allow the import of the "ethem" and "mcmc-red"
packages.
Also extract the data directory path in the variable 'data_dir'.
Also extract the output directory path in the variable 'output_dir'.

@author: misiak
"""

import os
import sys
import configparser

def custom_import():
    """ Custom import function.
    Here, it is to import the ethem package and the mcmc-red package.
    Also return the path to the data directory.
    """
    cfd = os.path.dirname(os.path.realpath(__file__))
    config_path = cfd + '/config.ini'

    Config = configparser.ConfigParser()
    Config.read(config_path)

    ethem_path = Config.get('ethem', 'path')
    mcmc_path = Config.get('mcmc-red', 'path')
    
#    sys.path.append(ethem_path)    
    for dirp in (ethem_path, mcmc_path):
        if dirp not in sys.path:
            sys.path.append(dirp)

    data_dir = Config.get('Data', 'path')

    output_dir = Config.get('Output', 'path')

    return data_dir, output_dir


if __name__ == '__main__':
    
    print(custom_import())