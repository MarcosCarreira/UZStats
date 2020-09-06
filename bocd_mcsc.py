#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 21:16:31 2020

@author: marcoscscarreira
"""

# %% Imports

import numpy as np
import scipy.stats as st

# %% Models

class StudentT:
    def __init__(self, alpha, beta, kappa, mu):
        self.alpha0 = self.alpha = np.array([alpha])
        self.beta0 = self.beta = np.array([beta])
        self.kappa0 = self.kappa = np.array([kappa])
        self.mu0 = self.mu = np.array([mu])
        self.params = np.transpose(np.array([self.alpha, self.beta, self.kappa, self.mu]))

    def pdf(self, data):
        return st.t.pdf(x=data, 
                           df=2*self.alpha,
                           loc=self.mu,
                           scale=np.sqrt(self.beta * (self.kappa+1) / (self.alpha *
                               self.kappa)))

    def update(self, data):
        muT0 = np.concatenate((self.mu0, (self.kappa * self.mu + data) / (self.kappa + 1)))
        kappaT0 = np.concatenate((self.kappa0, self.kappa + 1.))
        alphaT0 = np.concatenate((self.alpha0, self.alpha + 0.5))
        betaT0 = np.concatenate((self.beta0, self.beta + (self.kappa * (data -
            self.mu)**2) / (2. * (self.kappa + 1.))))
            
        self.mu = muT0
        self.kappa = kappaT0
        self.alpha = alphaT0
        self.beta = betaT0

# %% Function

def bocd(data, lamb=300, threshold=0.90, delay=15, params0=[[0.1, 1, 1, 0]]):
    obs_lik = StudentT(params0[0])
    params = params0.copy()
    chgpt_list = np.array([[1, 1]])
    t = 0
    growth_probs = np.array([[1]])
    for t, x in enumerate(data):
        pred_probs = obs_lik.pdf(x)
        haz = np.full(len(params), 1/lamb)
        chgpt_probs = np.sum(growth_probs[-1] * pred_probs * haz)
        newgr_probs = growth_probs[-1] * pred_probs * (1 - haz)
        newgr_probs = newgr_probs / np.sum(newgr_probs)
        mask = newgr_probs >= threshold
        chgpt_loc = np.compress(mask, arange(len(newgr_probs)))
        chgpt_prob = np.compress(mask, newgr_probs)
        if chgpt_loc >= delay:
            last_chgpt = t + 1 - chgpt_loc
            if last_chgpt > chgpt_list[-1][0]:
                print((last_chgpt, chgpt_prob))
                chgpt_list = np.concatenate((chgpt_list, np.array([last_chgpt, chgpt_prob])), axis=0)
                params = params[:chgpt_loc]
                newgr_probs = newgr_probs[:chgpt_loc+1] / np.sum(newgr_probs[:chgpt_loc+1])
        growth_probs = np.concatenate((growth_probs, newgr_probs))
        obs_lik.prune(chgpt_loc)
        obs_lik.update(x)
        params = obs_lik.params
        params = np.concatenate((params0.copy(), params))
        if t%100 == 0:
            print((t, len(newgr_probs), len(params)))
    return chgpt_list