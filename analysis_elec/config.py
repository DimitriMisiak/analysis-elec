#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Config file for the test detector using ntd technology.
Basically set up the simulation of the detector.

@author: misiak
"""

import sympy as sy

from import_package import custom_import
custom_import()

import ethem as eth

#==============================================================================
# SYSTEM
#==============================================================================
### Defining time and frequency variables
time, freq = eth.System.time, eth.System.freq

### Chassis ground
ground = eth.Voltstat('ground')
ground.voltage = 0

### Wire capacitance
capa = eth.Capacitor('f')

#### Load capacitance
#load = eth.Capacitor('f')

### NTD resistance
elntd = eth.Resistor(capa, ground, 'ntd')

#==============================================================================
# PHYSICAL RELATIONS AND ADDITIONNAL SYMBOLS
#==============================================================================
## NTD characteristics
#R0, T0 = sy.symbols('R0, T0')
#elntd.resistivity = eth.ntd_char(R0, T0, thntd.temperature)


#==============================================================================
# NOISE POWER
#==============================================================================
# Johnson noise for each
for resi in [elntd,]:
    john = eth.johnson_noise(resi.resistivity, resi.temperature)
    john = john**0.5 # to obtain the LPSD
    john /= resi.resistivity # to obtain the noise current
    resi.noise_flux['Johnson '+resi.label] = john

# amplifier current noise (impact the system, and so the observer)
i_a1, i_a2, i_a3 = sy.symbols('i_a1, i_a2, i_a3')
capa.i_a1 = i_a1
capa.i_a2 = i_a2
capa.i_a3 = i_a3
noise_current = (i_a1**2 + i_a2**2 *freq + i_a3**2 *freq**2)**0.5
capa.noise_sys['Ampli. Current'] = noise_current

# dac current noise (impact system and so observer)
e_dac = sy.symbols('e_dac')
capa.e_dac = e_dac
dac_current = e_dac * sy.I * 2 * sy.pi * freq * 10e-12
capa.noise_sys['DAC Current'] = dac_current

# amplifier voltage noise (impact the observer only)
e_a1, e_a2, e_a3 = sy.symbols('e_a1, e_a2, e_a3')
capa.e_a1 = e_a1
capa.e_a2 = e_a2
capa.e_a3 = e_a3
noise_voltage = (e_a1**2 + e_a2**2 /freq + e_a3**2 /freq**2)**0.5
capa.noise_obs['Ampli. voltage'] = noise_voltage

#==============================================================================
# UPDATING THE SYSTEM
#==============================================================================
eth.System.build_sym(savepath='output/build_sym')

#==============================================================================
# EVENT PERTURBATION
#==============================================================================
#energy, tau_therm, eps = sy.symbols('E, tau_th, eps')
#
#per = eth.Perturbation(energy,
#                       [1-eps, 0., eps, 0.],
#                       [tau_therm, tau_therm, tau_therm, tau_therm])

#==============================================================================
# EVALUATION DICT
#==============================================================================
evad_const = {'kB' : 1.3806485e-23, #J.K-1
              }

evad_sys = {
            elntd.temperature : 20e-3, #K
            elntd.resistivity : 3e6, #Ohms
            capa.capacity : 236e-12, # F

}

#evad_per = {
##            tau_therm : 1e-4, # s
#            tau_therm : 5e-3, # s
#            energy : 1e3 * 1.6e-19, # J
#            eps : 0.1, #fraction
#}

evad_noise = {e_a1: 5e-9,
              e_a2: 5e-8,
              e_a3: 1e-8,
              i_a1: 1e-15,
              i_a2: 5e-16,
              i_a3: 1e-17,
              e_dac: 2e-9}

evad = dict()
evad.update(evad_const)
evad.update(evad_sys)
#evad.update(evad_per)
evad.update(evad_noise)

def get_eval_dict():
    return evad

### checking the completeness of the evaluation dictionnary
# free symbols without evaluation
free_set = set(eth.System.phi_vect)|{time,freq}

# checking the electro-thermal equations
ete_free = eth.System.eteq.subs(evad).free_symbols
assert ete_free.issubset(free_set)

## checking the event perturbation
#per_free = per.matrix.subs(evad).free_symbols
#assert per_free.issubset(free_set)

# checking the noise power
for e in eth.System.elements_list:

    if isinstance(e, eth.RealBath):

        for v in e.noise_obs.values():
            noise_free = v.subs(evad).free_symbols
            assert noise_free.issubset(free_set)

        for v in e.noise_sys.values():
            noise_free = v.subs(evad).free_symbols
            assert noise_free.issubset(free_set)

    if isinstance(e, eth.Link):

        for v in e.noise_flux.values():
            noise_free = v.subs(evad).free_symbols
            assert noise_free.issubset(free_set)

if __name__ == '__main__':
    eth.sys_scheme(fp='output/scheme.png')
