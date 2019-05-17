#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Handy MCMC scripts.

Test for the different fit method (mcmc, ptmcmc, minimizer).

Author:
    Dimitri Misiak (misiak@ipnl.in2p3.fr)
"""

import matplotlib.pyplot as plt
import numpy as np
import corner 
import emcee

# importing ethem and mcmc-red package from custom paths
from import_package import custom_import
data_dir, output_dir = custom_import()
import mcmc_red as mcr

from model import initiate_model


def mcmc_plot(ndim, chain, lnprob, acc, labels, scale='linear', savedir=None):
    """ Plot the results of the mcmc analysis, and return these results.

    Parameters
    ----------
    .
    .
    .
    scale : str, optional
        Scale for the spreading of the initial markov chains.
        Can be either 'linear' or 'log'.
    savedir : str, optional
        If given, save the figure to the savedir. By default, set to None,
        figures are not saved.

    Returns
    -------
    xopt : numpy.ndarray
        Optimal parameters, minimising the loglikelihood.
    inf : numpy.ndarray
        Lower 1-sigma bounds for the optimal parameters.
    sup : numpy.ndarray
        Upper 1-sigma bounds for the optimal parameters.
    """
    # acceptance fraction cut
    tracc = (0.2, 0.8)
    ind = np.where(np.logical_or(acc < tracc[0], acc > tracc[1]))
    bam = chain[ind]
    chain = np.delete(chain, ind, axis=0)
    lnprob = np.delete(lnprob, ind, axis=0)

    print('shape chain:'), chain.shape
    print('shape lnprob:'), lnprob.shape

    fig_acceptance = plt.figure('ACCEPTANCE FRACTION')
    plt.bar(np.arange(acc.shape[0]), acc)
    for thresh in tracc:
        plt.axhline(thresh, color='r', ls='--')
    plt.xlabel('Marker Chain Index')
    plt.ylabel('Acceptance fraction')
    plt.ylim(0., 1.)
    plt.tight_layout()

    ### CONVERGENCE plot
    fig_convergence, ax = plt.subplots(ndim+1, 1, sharex=True, figsize=(7, 8),
                           num='CONVERGENCE')
    ax[-1].set_xlabel('Iterations')
    ax[-1].set_yscale('log')
    ax[-1].set_xscale('log')
    for a, l in zip(ax, labels + ('lnprob',)):

        a.set_ylabel(l)
        a.grid()

    # loop over the parameters
    for n in range(ndim):
        if scale == 'log' :
            ax[n].set_yscale('log')
        if len(bam) > 0:
            # plotting the chains discarded by the acceptance cut
            ax[n].plot(bam[:, :, n].T, color='r', lw=1., alpha=0.4)

#    # convergence cut with mean
#    lnlncut = np.mean(np.log10(-lnprob))
#    burnin_list = list()
#    for lnk in lnprob:
#        try:
#            burn = np.where(np.log10(-lnk) > lnlncut)[0][-1] + 100
#        except:
#            burn = 0
#        burnin_list.append(burn)
#
#    ax[-1].axhline(np.power(10,lnlncut), color='r')

    # convergence cut with best prob
#    lncut = 1.1 * lnprob.max()
    lncut = -56115
    burnin_list = list()
    for lnk in lnprob:
        try:
#            burn = np.where(lnk <  lncut)[0][-1] + 100
            burn = np.where(lnk <  lncut)[0][-1]
        except:
            print('Could not apply convergence cut properly')
            burn = 0
        burnin_list.append(burn)

    ax[-1].axhline(-lncut, color='r')

    # plotting the log10(-lnprob) array and the cut threshold
    ax[-1].plot(-lnprob.T, color='k')

    plt.figure('lnprob')
    plt.axhline(-lncut, color='r')
    plt.plot(-lnprob.T, color='k')

    chain_ok_list = list()
    # loop over the chains
    for chk, brn, lnk in zip(chain, burnin_list, lnprob):

        # iterations array
        ite = range(chk.shape[0])

        # converged chain and saving it
        ite_ok = ite[brn:]
        chk_ok = chk[brn:, :]
        lnk_ok = lnk[brn:]
        chain_ok_list.append(chk_ok)

        # not converged chain
        ite_no = ite[:brn]
        chk_no = chk[:brn, :]

        # loop over the parameters
        for n in range(ndim):

            # plotting the accepted chain and their respective burnin
            ax[n].plot(ite_ok, chk_ok[:,n].T, color='b', lw=1., alpha=1.)
            ax[n].plot(ite_no, chk_no[:,n].T, color='k', lw=1., alpha=0.4)
            ax[n].scatter([0], chk[0, n], color='r', marker='o')

        # plotting converged chain lnprob
        ax[-1].plot(ite_ok, -lnk_ok.T, color='b')
        plt.plot(ite_ok, -lnk_ok.T, color='b')

    fig_convergence.tight_layout(h_pad=0.0)

    # samples = reduce(lambda a,b: np.append(a,b, axis=0), chain_ok_list)
    samples = np.vstack(chain_ok_list)
    
    best_ind = np.unravel_index(lnprob.argmax(), lnprob.shape)
    best_chi2 = -2 * lnprob[best_ind]
    xopt = chain[best_ind]


    ### CORRELATION plot
    fig_correlation, ax = plt.subplots(2, sharex=True, num='CORRELATION')
    for a in ax:
        a.grid()
        a.set_xscale('log')
    ax[1].set_xlabel('Iterations')
    ax[1].set_ylabel('Corr p0')
    ax[0].set_ylabel('p0')

    for c in chain[:,:,0]:
        funk = emcee.autocorr.function(c)
        ax[0].plot(c)
        ax[1].plot(funk)


    ### CORNER plot
#    aux_fig, ax = plt.subplots(ndim,ndim,num='CORNER', figsize=(ndim,ndim))
    if scale == 'linear':
        fig_corner = corner.corner(
                samples,
                bins=50, smooth=1,
                labels=labels,
                quantiles=[0.16, 0.5, 0.84], show_titles=True,
                truths=xopt,
                title_kwargs={"fontsize": 12}
        )

    elif scale == 'log':
        fig_corner = corner.corner(
                np.log10(samples),
                bins=50, smooth=1,
                labels=['log({})'.format(l) for l in labels],
                quantiles=[0.16, 0.5, 0.84], show_titles=True,
                truths=np.log10(xopt),
                title_kwargs={"fontsize": 12}
        )

    fig_corner.tight_layout()

    # quantiles of the 1d-histograms
    inf, med, sup = np.percentile(samples, [16, 50, 84], axis=0)

    # Analysis end message
    print("MCMC results :")
    for n in range(ndim):
        print(labels[n]+'= {:.2e} + {:.2e} - {:.2e}'.format(
            med[n], sup[n]-med[n], med[n]-inf[n]
        ))
    for n in range(ndim):
        print(labels[n]+'\in [{:.3e} , {:.3e}] with best at {:.3e}'.format(
                inf[n], sup[n], xopt[n]
        ))
    if not np.all(np.logical_and(inf<xopt, xopt<sup)):
        print('Optimal parameters out the 1-sigma range ! Good luck fixing that :P')

    print('Chi2 = {}'.format(best_chi2))

    if savedir is not None:
        fig_acceptance.savefig(savedir+'/acceptance.png')
        fig_convergence.savefig(savedir+'/convergence.png')
        fig_correlation.savefig(savedir+'/correlation.png')
        fig_corner.savefig(savedir+'/corner.png')

    return xopt, inf, sup



if __name__ == "__main__":

    plt.close('all')
    
    param_top, param_bot, model_fun = initiate_model()
    
    # XXX MCMC
    # save directory
    sampler_path = '/home/misiak/projects/analysis-elec/mcmc_sampler/output_cc_2'
    
    # loading the mcmc results
    logd, chain, lnprob, acc = mcr.get_mcmc_sampler(sampler_path)
    
    lab = [p.name for p in param_bot]
    
    dim = int(logd['dim'])
    xopt, inf, sup = mcmc_plot(dim, chain, lnprob, acc, tuple(lab))
    
    print(xopt, inf, sup)


