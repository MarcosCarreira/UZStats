# The Robert and Rosenbaum Uncertainty Zones model
# Implementation by
# Marcos Costa Santos Carreira
# École Polytechnique - CMAP

# %% Imports

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functools import partial
import scipy.stats as st
from scipy.special import kv
from scipy import optimize
import numba as numba
from math import erfc

# %% Basic path functions

# %% Random vector

def frndn(nsteps=1, seed=None):
    '''frndn(nsteps=1, seed=None) returns a NumPy array of normally
    distributed numbers'''
    np.random.seed(seed)
    return np.random.randn(nsteps)

# %% Monte Carlo Path

def MCPath(psdrnd, vol=0.015, S0=100., t=1., drift=0.):
    '''MCPath(psdrnd, vol, S0, t, drift) returns a MC Path
    given the array of random (normal) numbers psdrnd,
    the volatility, the initial spot S0, the time to maturity
    and the drift'''
    steps=len(psdrnd)
    dt=t/steps
    logret=(drift-vol**2/2)*dt+vol*np.sqrt(dt)*psdrnd
    acclogret=np.add.accumulate(logret)
    path = np.insert(S0*np.exp(acclogret),0,S0)
    return path

# %% Times range

def atrange(nsteps=1, t=1.0):
    '''atrange(nsteps=1, t=1.0) returns a NumPy array for a time series
    [0,dt,...,t-dt,t]'''
    tsindex = np.arange(nsteps+1)*t/nsteps
    return tsindex

# %% Riffle

def riffle(a, b):
    c = np.empty((a.size + b.size,), dtype=a.dtype)
    c[0::2] = a
    c[1::2] = b
    return c

# %% Brownian Bridge - next iteration

def nextBBpath(trange, nrange, vol, indk, wk):
    '''nextBBpath(trange, nrange, vol, indk, wk) returns
    the next iteration of a Brownian Bridge
    given:
    trange (array of times, dimension 2**n+1)
    nrange (array of Z prepended by zero, dimension 2**n+1)
    vol: scalar (or array of volatilities, dimension 2**n+1)
    indk (indices of previous points)
    wk (values of previous points)'''
    tk = trange[indk]
    indk1 = (np.array(indk[1:])+np.array(indk[:-1]))//2
    tk1 = trange[indk1]
    zk1 = nrange[indk1]
    wk1 = ((tk[1:]-tk1)*wk[:-1]+(tk1-tk[:-1])*wk[1:])/\
        (tk[1:]-tk[:-1])+vol*np.sqrt((tk[1:]-tk1)*\
        (tk1-tk[:-1])/(tk[1:]-tk[:-1]))*zk1
    indkk1 = riffle(indk, indk1)
    wkk1 = riffle(wk, wk1)
    return [indkk1, wkk1]

# %% Brownian Bridge - all iterations

def mcBBPaths(psdrnd, vol=0.015, S0=100., ST=100., t=1.):
    '''mcBBPath(psdrnd, vol, S0, ST, t) returns all MC BB 
    Paths given the array of random (normal) numbers psdrnd,
    the volatility, the initial spot S0, the final Spot ST
    and the time to maturity t'''
    n = len(psdrnd)
    m = np.log2(n)
    indinit = np.array([0, n])
    tlist = np.linspace(0, t, n+1)
    tinit = tlist[indinit]
    nrange = np.insert(psdrnd, 0, 0)
    winit = np.array([0., np.log(ST/S0)])
    runlist = [[indinit, winit]]
    for j in range(0, np.floor(m).astype(int)):
        newpts = nextBBpath(tlist, nrange, vol,\
            runlist[-1][0], runlist[-1][1])
        runlist = runlist + [newpts]
    for j in range(0, len(runlist)):
        runlist[j][0] = tlist[runlist[j][0]]
        runlist[j][1] = S0*np.exp(runlist[j][1])
    return runlist

# %% Brownian Bridge - final path

def mcBBPath(psdrnd, vol=0.015, S0=100., ST=100., t=1.):
    '''mcBBPath(psdrnd, vol, S0, ST, t) returns the final
    MC BB Path given the array of random (normal) numbers
    psdrnd, the volatility, the initial spot S0, the final
    Spot ST and the time to maturity t'''
    n = len(psdrnd)
    m = np.log2(n)
    indinit = np.array([0, n])
    tlist = np.linspace(0, t, n+1)
    tinit = tlist[indinit]
    nrange = np.insert(psdrnd, 0, 0)
    winit = np.array([0., np.log(ST/S0)])
    runlist = [[indinit, winit]]
    for j in range(0, np.floor(m).astype(int)):
        newpts = nextBBpath(tlist, nrange, vol,\
            runlist[-1][0], runlist[-1][1])
        runlist = [newpts]
    runlist[0][0] = tlist[runlist[0][0]]
    runlist[0][1] = S0*np.exp(runlist[0][1])
    return pd.Series(runlist[0][1], index=runlist[0][0])


# %% Brownian Bridge - final path as array of prices

def mcBBPathS(psdrnd, vol=0.015, S0=100., ST=100., t=1.):
    '''mcBBPathS(psdrnd, vol, S0, ST, t) returns the final
    MC BB Path given the array of random (normal) numbers
    psdrnd, the volatility, the initial spot S0, the final
    Spot ST and the time to maturity t'''
    n = len(psdrnd)
    m = np.log2(n)
    indinit = np.array([0, n])
    tlist = np.linspace(0, t, n+1)
    tinit = tlist[indinit]
    nrange = np.insert(psdrnd, 0, 0)
    winit = np.array([0., np.log(ST/S0)])
    runlist = [[indinit, winit]]
    for j in range(0, np.floor(m).astype(int)):
        newpts = nextBBpath(tlist, nrange, vol,\
            runlist[-1][0], runlist[-1][1])
        runlist = [newpts]
    return S0*np.exp(runlist[0][1])

# %% Discretization functions

# Increment (in ticks) on last traded price
# alpha: Tick value
# eta: Microstructure parameter
# X(t): Efficient price (assumed here to be a Geometric Brownian Motion)
# P(t): Traded price (in a grid of multiples of $\alpha$)

@numba.jit(nopython=True)
def Li(alpha, eta, Pt, xt1):
    '''Li(alpha, eta, Pt, xt1) returns the increment (in ticks)
    of the transacion price P: P(t+1) - P(t) given the efficient
    price X(t+1), the tick value alpha and the parameter eta'''
    Lt = max(0, np.int(np.floor(abs(xt1 - Pt) / alpha + 0.5 - eta)))
    return Lt

# %% Updating the last traded price

@numba.jit(nopython=True)
def updtrpr(alpha,eta,Pt,xt1):
    '''updtrpr(alpha, eta, Pt, xt1) returns the next transacion
    price P(t+1) given the efficient price X(t+1), the previous
    transacion price P(t), the tick value alpha and the
    parameter eta'''
    return Pt+Li(alpha,eta,Pt,xt1)*np.sign(xt1 - Pt)*alpha

# %% Calculation of the traded prices given the efficient prices

@numba.jit(nopython=True)
def trprpath(alpha,eta,path):
    '''trprpath(alpha, eta, path) returns the last traded price P(t)
    path given the efficient price X(t) path, the tick value alpha 
    and the parameter eta'''
    trpath=path.copy()
    # this is much more efficient than appending the values
    for k in range(1,len(trpath)):
		# Assuming X(0) is a valid transaction price
        trpath[k]=updtrpr(alpha,eta,trpath[k-1],path[k])
        # trpath[k-1] was already changed from X to P
        # It's path-dependent (P(t) depends on P(t-1))
    # return trpath.transpose()[0]
    return trpath

@numba.jit(nopython=True)
def trprpathk(alpha, eta, path):
    '''trprpathk(alpha, eta, path) returns the last traded price P(t)
    path given the efficient price X(t) path, the tick value alpha 
    and the parameter eta'''
    trpath=path.copy()
    # this is much more efficient than appending the values
    for k in range(1, len(trpath)):
		# Assuming X(0) is a valid transaction price
        trpath[k] = updtrpr(alpha[k-1], eta[k-1], trpath[k-1], path[k])
        # trpath[k-1] was already changed from X to P
        # It's path-dependent (P(t) depends on P(t-1))
    # return trpath.transpose()[0]
    return trpath

# %% Durations, changes and effective prices

def diff_prices_df_X(tmseries, alpha, eta):
    '''diff_prices_df_X(tmseries, alpha, eta) returns a data frame
    with traded prices Pt and effective prices Xtj corresponding to 
    the times tj in which there is a price change'''
    tmsvl = tmseries[tmseries.values != tmseries.shift().values]
    dfdiff = pd.DataFrame(tmsvl, columns=['Ptj'])
    dfdiff.index.name = 'tj'
    dfdiff.reset_index(inplace=True)
    dfdiff['dPtj'] = dfdiff['Ptj'].diff()
    dfdiff['sign'] = np.sign(dfdiff['dPtj'])
    dfdiff['Li'] = np.abs(np.round(dfdiff['dPtj']/alpha))
    dfdiff['dtj'] = dfdiff['tj'].diff()
    dfdiff['Co'] = dfdiff['sign'].diff()==0
    dfdiff['Al'] = dfdiff['sign'].diff().abs()==2
    dfdiff['Xtj'] = dfdiff['Ptj']-alpha*(0.5-eta)*(\
        dfdiff['sign'].fillna(0))
    dfdiff.set_index('tj',inplace=True)
    return dfdiff

def diff_prices_df_group(tmseries, alpha):
    '''diff_prices_df_group(tmseries, alpha) returns a data frame
    with traded prices Ptj corresponding to the times tj in which
    there is a price change, either at a later time stamp or based on
    the last price of a big trade given a time series of times and
    prices'''
    tmsg = tmseries.groupby(level = 0).last()
    tmsg.name = 'Ptj'
    tmsvl = tmsg[tmsg.values != tmsg.shift().values]
    dfdiff = pd.DataFrame(tmsvl)
    dfdiff.index.name = 'tj'
    dfdiff.reset_index(inplace=True)
    dfdiff['dPtj'] = dfdiff['Ptj'].diff()
    dfdiff['sign'] = np.sign(dfdiff['dPtj'])
    dfdiff['Li'] = np.abs(np.round(dfdiff['dPtj']/alpha))
    dfdiff['dtj'] = dfdiff['tj'].diff()
    dfdiff['Co'] = dfdiff['sign'].diff()==0
    dfdiff['Al'] = dfdiff['sign'].diff().abs()==2
    dfdiff.set_index('tj',inplace=True)
    return dfdiff

def diff_prices_df_nogroup(tmseries, alpha):
    '''diff_prices_df_group(tmseries, alpha) returns a data frame
    with traded prices Ptj corresponding to the times tj in which
    there is a price change, either at a later time stamp or based on
    the last price of a big trade given a time series of times and
    prices'''
    tmsg = tmseries.copy()
    tmsg.name = 'Ptj'
    tmsvl = tmsg[tmsg.values != tmsg.shift().values]
    dfdiff = pd.DataFrame(tmsvl)
    dfdiff.index.name = 'tj'
    dfdiff.reset_index(inplace=True)
    dfdiff['dPtj'] = dfdiff['Ptj'].diff()
    dfdiff['sign'] = np.sign(dfdiff['dPtj'])
    dfdiff['Li'] = np.abs(np.round(dfdiff['dPtj']/alpha))
    dfdiff['dtj'] = dfdiff['tj'].diff()
    dfdiff['Co'] = dfdiff['sign'].diff()==0
    dfdiff['Al'] = dfdiff['sign'].diff().abs()==2
    dfdiff.set_index('tj',inplace=True)
    return dfdiff

def dur(S, alpha, eta, vol):
    '''dur(S,alpha,eta,vol) returns the estimated duration of traded price
    changes given sopt price S, tick value alpha, microstructure parameter
    eta and volatility vol'''
    return 2*eta*(alpha/(S*vol))**2

# %% Processing

# %% Traded price paths

def read_trd_path(pathf, j, vol, alpha, eta, filename='trdpaths'):
    return pd.read_hdf(pathf+filename+'_'+str(j)+'.h5')\
        .loc[:, (vol, alpha, eta)]

def read_trd_path_drift(pathf, j, vol, mu, alpha, eta, filename='mutrdpaths'):
    return pd.read_csv(pathf+filename+'_'+str(j)+'_'+str(vol)+'_'+str(mu)+\
        '_'+str(alpha)+'_'+str(eta)+'.csv', header=None)[0]

# %% Reduce to price changes

def loop_px_ch(pathf, npaths, vollist, alphalist, etalist, filename='trdpxs',\
    filenamein='trdpaths'):
    for j in range(npaths):
        for v in vollist:
            for a in alphalist:
                for e in etalist:
                    diff_prices_df_group(read_trd_path(pathf, j, v, a, e,\
                        filenamein), a).to_csv(pathf+filename+'_'+str(j)+'_'+\
                        str(v)+'_'+str(a)+'_'+str(e)+'.csv', index=False)
                    print((j, v, a, e))

def loop_px_ch_drift(pathf, npaths, vollist, mulist, alphalist, etalist, filename='mutrdpxs',\
    filenamein='mutrdpaths'):
    for j in range(npaths):
        for v in vollist:
            for mu in mulist:
                for a in alphalist:
                    for e in etalist:
                        diff_prices_df_group(read_trd_path_drift(pathf, j, v, mu, a, e,\
                            filenamein), a).to_csv(pathf+filename+'_'+str(j)+'_'+\
                            str(v)+'_'+str(float(mu))+'_'+str(a)+'_'+str(e)+'.csv', index=False)
                        print((j, v, mu, a, e))

def read_px_path(pathf, j, vol, alpha, eta, filename='trdpxs'):
    return pd.read_csv(pathf+filename+'_'+str(j)+'_'+str(v)+'_'+str(a)+'_'+\
        str(e)+'.csv')

def read_px_path_drift(pathf, j, vol, mu, alpha, eta, filename='trdpxs'):
    return pd.read_csv(pathf+filename+'_'+str(j)+'_'+str(v)+'_'+str(float(mu))+'_'+\
        str(a)+'_'+str(e)+'.csv')

# %% Statistics and estimations

# %% UZ statistics

def uz_coal_byk(data_frame_trades):
    '''uz_coal_byk(data_frame_trades) returns the uncertainty zones
    data frame for the different values of k (price changes in ticks)'''
    k_array = np.sort(data_frame_trades['Li'].dropna().unique())
    data_frame_k = pd.DataFrame(k_array, columns=['Li'])
    data_frame_k.set_index('Li', drop=False, inplace=True)
    coal_group_co = data_frame_trades['Co'].groupby(data_frame_trades['Li'])
    coal_group_al = data_frame_trades['Al'].groupby(data_frame_trades['Li'])
    data_frame_k['lamb'] = (coal_group_al.count()/\
                np.sum(coal_group_al.count()))
    data_frame_k['Co'] = coal_group_co.sum()
    data_frame_k['Al'] = coal_group_al.sum()
    data_frame_k['u'] = (data_frame_k['Li']*(\
                (data_frame_k['Co']/data_frame_k['Al']).fillna(0)-1)+1)/2
    data_frame_k['etas'] = data_frame_k['lamb']*data_frame_k['u']
    data_frame_k.reset_index(drop=True, inplace=True)
    return data_frame_k

# %% Realized volatility

def rlzvollog(prices):
    """rlzvollog(prices) calculates the realized volatility of 
    a time series of prices using logreturns"""
    pxs = np.log(prices/prices.shift(1))
    return np.sqrt(np.sum(pxs*pxs))

# %% UZ statistics and durations

def stats_trpath(j, vol, alpha, eta, dt, tpath):
    path_stats = uz_coal_byk(tpath)
    ndfpr = np.float(len(tpath))-1
    etas = max(0,min(1,np.dot(path_stats['lamb'],path_stats['u'])))
    eta1 = path_stats['etas'][path_stats['Li'] == 1][0]
    rvp = rlzvollog(tpath['Ptj'])
    rvxe = rvp*np.sqrt(2*eta1)
    mean_dur_cont = np.mean(tpath[tpath['Co']]['dtj'])
    mean_dur_alt = np.mean(tpath[tpath['Al']]['dtj'])
    mean_dur_all = np.mean(tpath['dtj'].dropna())
    median_dur_cont = np.median(tpath[tpath['Co']]['dtj'])
    median_dur_alt = np.median(tpath[tpath['Al']]['dtj'])
    median_dur_all = np.median(tpath['dtj'].dropna())
    chgavg = np.mean(np.abs(tpath['dPtj']))
    st_values = np.array([[np.float(j), vol, alpha, eta, dt, etas, eta1, rvp,\
        rvxe, mean_dur_cont, mean_dur_alt, mean_dur_all, median_dur_cont,\
        median_dur_alt, median_dur_all, chgavg, ndfpr]])
    st_header = ['j', 'vol', 'alpha', 'eta', 'dt', 'etas', 'eta1', 'rvp',\
        'rvxe', 'mean_dur_cont', 'mean_dur_alt', 'mean_dur_all',\
        'median_dur_cont', 'median_dur_alt', 'median_dur_all', 'chgavg',\
        'ndfpr']
    return pd.DataFrame(st_values, columns=st_header)


def stats_trpath_drift(j, vol, spot, alpha, eta, dt, tpath):
    path_stats = uz_coal_byk(tpath)
    ndfpr = np.float(len(tpath))-1
    reta = ((tpath['dPtj'].dropna()).sum())/alpha
    etas = max(0,min(1,np.dot(path_stats['lamb'],path_stats['u'])))
    eta1 = path_stats['etas'][path_stats['Li'] == 1][0]
    rvp = rlzvollog(tpath['Ptj'])
    rvxe = rvp*np.sqrt(2*etas)
    mean_dur_cont = np.mean(tpath[tpath['Co']]['dtj'])
    mean_dur_alt = np.mean(tpath[tpath['Al']]['dtj'])
    mean_dur_all = np.mean(tpath['dtj'].dropna())
    median_dur_cont = np.median(tpath[tpath['Co']]['dtj'])
    median_dur_alt = np.median(tpath[tpath['Al']]['dtj'])
    median_dur_all = np.median(tpath['dtj'].dropna())
    chgavg = np.mean(np.abs(tpath['dPtj']))
    st_values = np.array([[np.float(j), vol, spot, alpha, eta, dt, reta, \
        ndfpr, etas, eta1, rvp, rvxe, mean_dur_cont, mean_dur_alt,\
        mean_dur_all, median_dur_cont, median_dur_alt, median_dur_all, chgavg]])
    st_header = ['j', 'vol', 'lastspot', 'alpha', 'eta', 'dt', 'ret_ticks',\
        'ndfpr', 'etas', 'eta1', 'rvp', 'rvxe', 'mean_dur_cont', 'mean_dur_alt',\
        'mean_dur_all', 'median_dur_cont', 'median_dur_alt', 'median_dur_all',\
        'chgavg']
    return pd.DataFrame(st_values, columns=st_header)

def pathaestats(df,j,v,dtosec):
    params=df.columns.values
    nparam=len(params)
    stlist=[]
    for k in range(nparam):
            alpha, eta = params[k]
            stlist=stlist+[pathstats(j,v,alpha,eta,df[alpha,eta])]    
    dfst=pd.DataFrame(stlist,columns=
        ['j','vol','alpha','eta','etas','eta1','rvp','rvx',\
         'DTcont','DTalt','DTavg','chgavg','ndfpr'])
    dfst['deta1']=dfst['eta1']-dfst['eta']
    dfst['dalpha']=dfst['chgavg']-dfst['alpha']
    return dfst

# %% Limit of conditional CDFs - Numba

# def isign(x):
#     return (x > 0) - (x < 0)

@numba.jit(nopython=True)
def probc(ud: int, sud: int, s: float,
          α: float, η: float, σ: float, μ: float) -> float:
    '''probc(ud: int, sud: int, s: float, α0: float, η0: float,
          α: float, η: float, σ: float, μ: float) -> float)
          returns a float between 0 and 1 corresponding to the
          limit of P(t) for the given pair ud, sud.
          ud, sud = (-1, +1)
          s, α, η, σ: floats > 0
          μ = float'''
    ω = μ - 0.5 * (σ ** 2)
    x0 = np.log(s + sud * (0.5 - η) * α)
    b1 = np.log(s + ud * (0.5 + η) * α)
    b2 = np.log(s - ud * (0.5 + η) * α)
#   x0 should be between b1 and b2 (or b2 and b1)
    if ω == 0:
        return (b2 - x0) / (b2 - b1)
    else:
        ke = 2 * np.abs(ω) * ud / (σ ** 2)
        if np.exp(ke * (b2 - b1)) == 1:
            return (b2 - x0) / (b2 - b1)
        else:
            e0 = (1 - np.sign(ω) * ud) * ω * (b1 - x0) / (σ ** 2)
            return np.exp(e0) * (1 - np.exp(ke * (b2 - x0))) / (1 - np.exp(ke * (b2 - b1)))

# %% Limit probabilities of Markov Chain

def four_ps(s, α, η, σ, μ):
    pT_Al_Up = max(0, probc(+1, +1, s, α, η, σ, μ))
    pT_Al_Down = max(0, probc(-1, -1, s, α, η, σ, μ))
    if (pT_Al_Up + pT_Al_Down) == 0:
        p1_Al_Up = 1e-20
        p1_Al_Down = 1e-20
        p1_Co_Up = max(1e-20, (1 - pT_Al_Down) / 2)
        p1_Co_Down = max(1e-20, (1 - pT_Al_Up) / 2)
    else:
        p1_Al_Up = max(1e-20, (pT_Al_Up * pT_Al_Down) / (pT_Al_Up + pT_Al_Down))
        p1_Al_Down = max(1e-20, (pT_Al_Up * pT_Al_Down) / (pT_Al_Up + pT_Al_Down))
        p1_Co_Up = max(1e-20, (pT_Al_Up * (1 - pT_Al_Down)) / (pT_Al_Up + pT_Al_Down))
        p1_Co_Down = max(1e-20, ((1 - pT_Al_Up) * pT_Al_Down) / (pT_Al_Up + pT_Al_Down))
    return (p1_Al_Up, p1_Al_Down, p1_Co_Up, p1_Co_Down)

# %% Conditional CDFs (unscaled) - Numba, within epsilon (while)

@numba.jit(nopython=True)
def cdftc(ud, sud, s, α, η, σ, μ, t):
    if t == 0:
        ans = 0
    else:
        ω = μ - 0.5 * σ ** 2
        x0 = np.log(s + sud * (0.5 - η) * α)
        b1 = np.log(s + ud * (0.5 + η) * α)
        b2 = np.log(s - ud * (0.5 + η) * α)
        y1 = ud * (x0 - b1)
        z1 = ud * (2 * b2 - b1 - x0)
        tyzn = 0
        n = 0
        tyznn = 1
        eps=1e-16
        while tyznn > eps:
            n += 1
            dyzn = ud * 2 * (n - 1) * (b2 - b1)
            yn = dyzn + y1
            zn = dyzn + z1
            tyznn = (
                np.exp( (ω * yn / (σ ** 2)) + 
                    np.log(
                        erfc(-(yn + ω * t) / (σ * np.sqrt(2 * t))) -
                        erfc(-(zn + ω * t) / (σ * np.sqrt(2 * t))) *
                        np.exp((ω / (σ ** 2)) * ud * 2 * (b2 - x0))
                    )
                ) +
                
                np.exp( (- ω * yn / (σ ** 2)) + 
                    np.log(
                        erfc(-(yn - ω * t) / (σ * np.sqrt(2 * t))) -
                        erfc(-(zn - ω * t) / (σ * np.sqrt(2 * t))) *
                        np.exp((- ω / (σ ** 2)) * ud * 2 * (b2 - x0))
                    )
                )
            )/2
            # tyznn =\
            #     (np.exp(ω * yn / (σ ** 2)) * erfc(
            #     -(yn + ω * t) / (σ * np.sqrt(2 * t))) -
            #     np.exp(ω * zn / (σ ** 2)) * erfc(
            #     -(zn + ω * t) / (σ * np.sqrt(2 * t))) +
            #     np.exp(-ω * yn / (σ ** 2)) * erfc(
            #     -(yn - ω * t) / (σ * np.sqrt(2 * t))) -
            #     np.exp(-ω * zn / (σ ** 2)) * erfc(
            #     -(zn - ω * t) / (σ * np.sqrt(2 * t)))) / 2
            tyzn += tyznn
        ans = min(probc(ud, sud, s, α, η, σ, μ), np.exp(ω * (b1 - x0) / (σ ** 2)) * tyzn)
    return ans

# %% Adjusted conditional CDFs

@numba.jit(nopython=True)
def cdftT(ud, sud, s, α, η, σ, μ, t):
    if t == 0:
        return 0
    else:
        return cdftc(ud, sud, s, α, η, σ, μ, t) / probc(ud, sud, s, α, η, σ, μ)

# %% Conditional PDFs (unscaled):

@numba.jit(nopython=True)
def pdftc(ud, sud, s, α, η, σ, μ, t, δt=0.0001):
    if t == 0:
        return 0
    else:
        pc = probc(ud, sud, s, α, η, σ, μ)
        if t * δt * pc == 0:
            return 0
        else:
            return ((cdftc(ud, sud, s, α, η, σ, μ, t * (1 + δt)) -
                    cdftc(ud, sud, s, α, η, σ, μ, t * (1 - δt))) /
                    (2 * t * δt * pc))

# %% Inverse CDF

def t_from_p(ud, sud, s, α, η, σ, μ, p):
    '''t_from_p(ud, sud, s, α, η, σ, μ) solves the equation
    Pup(t) / Pup(inf) (or Pdown(t) / Pdown(inf))  = p;
    so for a random p such that 0 <= p <= 1 we find
    the expected time of a price change for the sign given'''
    cdft = partial(cdftc, ud, sud, s, α, η, σ, μ)
    cdfinf = probc(ud, sud, s, α, η, σ, μ)
    if cdfinf == 0:
        print('probc == 0')
        print((ud, sud, s, α, η, σ, μ))
    def groot(t):
        return np.round(cdft(t)/cdfinf - p, decimals=8)
    sol = optimize.root_scalar(groot, bracket=[0, 1], method='brentq')
    return sol.root

# %% Multinomial Likelihood

@numba.jit(nopython=True)
def multinom_likel(prior, hist):
    log1 = np.sum(np.array([np.log(y) for y in range(1, np.sum(hist) + 1)]))
    log2 = np.sum(np.array([np.sum(np.array([np.log(y) for y in range(1, xj + 1)])) for xj in hist]))
    log3 = np.sum(np.array([hist[j] * np.log(prior[j]) for j in range(len(prior))]))
    return np.exp(log1 - log2 + log3)

# %% Class for bayesian estimation

class Process():
    
    def __init__(self, ηs, σs, μs, hours=16, cutoff=(5e-3), roll_window=20, update_freq=20):
        self.__ηs = list(ηs.keys())
        self.__σs = list(σs.keys())
        self.__μs = list(μs.keys())
        self.__rw = roll_window
        self.__uf = update_freq
        self.__ηv = np.array(list(ηs.values()))
        self.__σv = np.array(list(σs.values()))
        self.__μv = np.array(list(μs.values()))
        self.__ηn = len(ηs)
        self.__σn = len(σs)
        self.__μn = len(μs)
        self.__j = 0
        self.__durs = pd.Series(dtype=float)
        self.__pxs = pd.Series(dtype=float)
        self.__ticks = pd.Series(dtype=float)
        self.__signs = pd.Series(dtype=int)
        self.__ks = pd.Series(dtype=int)
        self.__Al_Up = pd.Series(dtype=int)
        self.__Co_Up = pd.Series(dtype=int)
        self.__Al_Down = pd.Series(dtype=int)
        self.__Co_Down = pd.Series(dtype=int)
        self.__probs = np.multiply.outer(
            self.__ηv, np.multiply.outer(
                self.__σv, self.__μv))
        self.__probs_μ = self.__probs.view()
        self.__likel = self.__probs.view()
        self.__likel_μ = self.__probs.view()
        self.__prob_size = self.__ηn * self.__σn * self.__μn
        self.__prob_floor = 1e-5 / (self.__ηn * self.__σn * self.__μn)
        self.__count = [0, [0, 0, 0, 0]]
        self.__count_roll = [0, [0, 0, 0, 0]]
        self.__ηm = pd.Series(dtype=float)
        self.__σm = pd.Series(dtype=float)
        self.__μm = pd.Series(dtype=float)
        self.__sumCo = pd.Series(dtype=float)
        self.__sumAl = pd.Series(dtype=float)
        self.__sumAlk = pd.Series(dtype=float)
        self.__sumH = pd.Series(dtype=float)
        self.__cutoffms = cutoff * 1000
        self.__mindt = cutoff / (hours * 3600)
        self.__fast = pd.Series(dtype=float)
        self.__FIG_SIZE_1 = (9, 12)
        self.__Y_FIG_1 = 0.96
        self.__FIG_SIZE_2 = (9, 16)

        
    @property
    def display(self):
        return [self.__count, self.__count_roll]

    @property
    def show_probs(self):
        return self.__probs

    @property
    def show_sumH(self):
        return self.__sumH

    @property
    def show_fast(self):
        return self.__fast

    @property
    def show_sumcutoff(self):
        return np.sum(self.__fast) / len(self.__fast)

    @property
    def show_marginals(self):
        ηmrg = pd.Series(np.sum(self.__probs, axis=(1, 2)), index=self.__ηs)
        σmrg = pd.Series(np.sum(self.__probs, axis=(0, 2)), index=self.__σs)
        μmrg = pd.Series(np.sum(self.__probs, axis=(0, 1)), index=self.__μs)  
        return [ηmrg, σmrg, μmrg]      

    @property
    def marginals_means(self):
        return [self.__ηm, self.__σm, self.__μm]      

    @property
    def plot_marginals(self):   
        fig, axs = plt.subplots(3, 1, figsize=self.__FIG_SIZE_1)
        fig.suptitle('Marginals: ' + str(self.__count[0]) +
            ' \n [Al_Up, Al_Down, Co_Up, Co_Down] : \n ' +
            str(self.__count[1]), y=self.__Y_FIG_1)
        axs[0].plot(self.__ηs, np.sum(self.__probs, axis=(1, 2)), color = 'b', marker='o')
        axs[0].set_title('η = ' + '{:.3f}'.format(np.dot(self.__ηs, np.sum(self.__probs, axis=(1, 2)))))
        axs[1].plot(self.__σs, np.sum(self.__probs, axis=(0, 2)), color = 'b', marker='o')
        axs[1].set_title('σ = ' + '{:.5f}'.format(np.dot(self.__σs, np.sum(self.__probs, axis=(0, 2)))))
        axs[2].plot(self.__μs, np.sum(self.__probs, axis=(0, 1)), color = 'b', marker='o')
        axs[2].set_title('μ = ' + '{:.3e}'.format(np.dot(self.__μs, np.sum(self.__probs, axis=(0, 1)))))
 
    @property
    def plot_marginals_μ(self):   
        fig, axs = plt.subplots(3, 1, figsize=self.__FIG_SIZE_1)
        fig.suptitle('Marginals for μ update: ' + str(self.__count_roll[0]) +
            ' \n [Al_Up, Al_Down, Co_Up, Co_Down] : \n ' +
            str(self.__count_roll[1]), y=self.__Y_FIG_1)
        axs[0].plot(self.__ηs, np.sum(self.__probs_μ, axis=(1, 2)), color = 'b', marker='o')
        axs[0].set_title('η = ' + '{:.3f}'.format(np.dot(self.__ηs, np.sum(self.__probs_μ, axis=(1, 2)))))
        axs[1].plot(self.__σs, np.sum(self.__probs_μ, axis=(0, 2)), color = 'b', marker='o')
        axs[1].set_title('σ = ' + '{:.5f}'.format(np.dot(self.__σs, np.sum(self.__probs_μ, axis=(0, 2)))))
        axs[2].plot(self.__μs, np.sum(self.__probs_μ, axis=(0, 1)), color = 'b', marker='o')
        axs[2].set_title('μ = ' + '{:.3e}'.format(np.dot(self.__μs, np.sum(self.__probs_μ, axis=(0, 1)))))

    @property
    def plot_marginals_means(self):   
        fig, axs = plt.subplots(4, 1, figsize=self.__FIG_SIZE_2)
        fig.suptitle('Marginals (means): ' + str(self.__count[0]) +
            ' \n [Al_Up, Al_Down, Co_Up, Co_Down] : \n ' +
            str(self.__count[1]) + '\n Fast: ' + str(np.sum(self.__fast)) +
             ' Cutoff (ms): ' + str(self.__cutoffms), y=self.__Y_FIG_1)
        axs[0].plot(self.__durs.cumsum().values, self.__sumH.values, color = 'r', marker='o', drawstyle='steps-post')
        axs[0].set_title('H = ' + '{:.3e}'.format(self.__sumH.values[-1]))
        axs[1].plot(self.__durs.cumsum().values, self.__ηm.values, color = 'r', marker='o', drawstyle='steps-post')
        axs[1].set_title('η = ' + '{:.3f}'.format(np.dot(self.__ηs, np.sum(self.__probs, axis=(1, 2)))))
        axs[2].plot(self.__durs.cumsum().values, self.__σm.values, color = 'r', marker='o', drawstyle='steps-post')
        axs[2].set_title('σ = ' + '{:.5f}'.format(np.dot(self.__σs, np.sum(self.__probs, axis=(0, 2)))))
        axs[3].plot(self.__durs.cumsum().values, self.__μm.values, color = 'r', marker='o', drawstyle='steps-post')
        axs[3].set_title('μ = ' + '{:.3e}'.format(np.dot(self.__μs, np.sum(self.__probs, axis=(0, 1)))))

    def calc_update_freq(self):
        s = self.__pxs.iloc[-1]
        α = self.__ticks.iloc[-1]
        # η0 = self.__ηm.iloc[-1]
        for ηi in range(self.__ηn):
            η = self.__ηs[ηi]
            for σi in range(self.__σn):
                σ = self.__σs[σi]
                for μi in range(self.__μn):
                    μ = self.__μs[μi]
                    # Use counts to update μ
                    self.__likel_μ[ηi, σi, μi] = np.exp(0.25 * min(self.__count_roll[0])) *\
                        multinom_likel(four_ps(s, α, η, σ, μ), np.array(self.__count_roll[1]))  # η0
                    self.__probs_μ[ηi, σi, μi] = self.__prob_floor + self.__probs[ηi, σi, μi] * self.__likel_μ[ηi, σi, μi]
        self.__probs_μ = self.__probs_μ / np.sum(self.__probs_μ)

    def __calc_update_freq(self):
        s = self.__pxs.iloc[-1]
        α = self.__ticks.iloc[-1]
        # η0 = self.__ηm.iloc[-1]
        for ηi in range(self.__ηn):
            η = self.__ηs[ηi]
            for σi in range(self.__σn):
                σ = self.__σs[σi]
                for μi in range(self.__μn):
                    μ = self.__μs[μi]
                    # Use counts to update μ
                    self.__likel_μ[ηi, σi, μi] = np.exp(0.25 * min(self.__count_roll[0])) *\
                        multinom_likel(four_ps(s, α, η, σ, μ), np.array(self.__count_roll[1]))  # η0
                    self.__probs_μ[ηi, σi, μi] = self.__prob_floor + self.__probs[ηi, σi, μi] * self.__likel_μ[ηi, σi, μi]
        self.__probs_μ = self.__probs_μ / np.sum(self.__probs_μ)

    def update_freq(self):
        self.__probs = self.__probs_μ.view()
        self.__ηm.iloc[-1] = np.dot(self.__ηs, np.sum(self.__probs, axis=(1, 2)))
        self.__σm.iloc[-1] = np.dot(self.__σs, np.sum(self.__probs, axis=(0, 2)))
        self.__μm.iloc[-1] = np.dot(self.__μs, np.sum(self.__probs, axis=(0, 1)))

    def __update_freq(self):
        self.__probs = self.__probs_μ.view()
        self.__ηm.iloc[-1] = np.dot(self.__ηs, np.sum(self.__probs, axis=(1, 2)))
        self.__σm.iloc[-1] = np.dot(self.__σs, np.sum(self.__probs, axis=(0, 2)))
        self.__μm.iloc[-1] = np.dot(self.__μs, np.sum(self.__probs, axis=(0, 1)))

    def init_px(self, trade):
        ud, al, k, s, α, t = trade
        self.__durs.loc[0] = 0
        self.__pxs.loc[0] = s
        self.__ticks.loc[0] = α
        self.__ηm.loc[0] = np.dot(self.__ηs, np.sum(self.__probs, axis=(1, 2)))
        self.__σm.loc[0] = np.dot(self.__σs, np.sum(self.__probs, axis=(0, 2)))
        self.__μm.loc[0] = np.dot(self.__μs, np.sum(self.__probs, axis=(0, 1)))
        self.__sumH.loc[0] = 0.5
        self.__fast.loc[0] = False
        self.__j += 1
        
    def init_px_chg(self, trade):
        ud, al, k, s, α, t = trade
        self.__durs.loc[self.__j] = t
        s0 = self.__pxs.loc[self.__j - 1]
        α0 = self.__ticks.loc[self.__j - 1]
        # η0 = self.__ηm.loc[self.__j - 1]
        self.__pxs.loc[self.__j] = s
        self.__ticks.loc[self.__j] = α
        self.__signs.loc[self.__j] = ud
        self.__ks.loc[self.__j] = k
        self.__Al_Up.loc[self.__j] = 0
        self.__Co_Up.loc[self.__j] = 0
        self.__Al_Down.loc[self.__j] = 0
        self.__Co_Down.loc[self.__j] = 0
        self.__count = [self.__j, [0, 0, 0, 0]]
        self.__sumCo.loc[self.__j] = 0
        self.__sumAl.loc[self.__j] = 0
        self.__sumAlk.loc[self.__j] = 0
        self.__sumH.loc[self.__j] = 0.5
        # Use durations to update η, σ, μ at every step if duration is higher than cutoff
        # (assumes continuation at j=1)
        if t > self.__mindt:
            for ηi in range(self.__ηn):
                η = self.__ηs[ηi]
                for σi in range(self.__σn):
                    σ = self.__σs[σi]
                    for μi in range(self.__μn):
                        μ = self.__μs[μi]
                        αk = k * α
                        ηk = η / k
                        # Use durations to update η, σ, μ
                        self.__probs[ηi, σi, μi] = self.__prob_floor + self.__probs[ηi, σi, μi] *\
                            pdftc(ud, -ud, s0, αk, ηk, σ, μ, t)  # η0  # continuation: sud -> -ud
            self.__fast.loc[self.__j] = False
        else:
            self.__fast.loc[self.__j] = True
        self.__probs = self.__probs / np.sum(self.__probs)
        self.__ηm.loc[self.__j] = np.dot(self.__ηs, np.sum(self.__probs, axis=(1, 2)))
        self.__σm.loc[self.__j] = np.dot(self.__σs, np.sum(self.__probs, axis=(0, 2)))
        self.__μm.loc[self.__j] = np.dot(self.__μs, np.sum(self.__probs, axis=(0, 1)))
        self.__j += 1
                
    def update_with_counts(self, trade):
        ud, al, k, s, α, t = trade
        co = 1 - al
        sud = ud * (2 * al - 1)
        self.__durs.loc[self.__j] = t
        s0 = self.__pxs.loc[self.__j - 1]
        α0 = self.__ticks.loc[self.__j - 1]
        η0 = self.__ηm.loc[self.__j - 1]
        self.__pxs.loc[self.__j] = s
        self.__ticks.loc[self.__j] = α
        self.__signs.loc[self.__j] = ud
        self.__ks.loc[self.__j] = k
        self.__Al_Up.loc[self.__j] = self.__Al_Up.loc[self.__j - 1]\
            + al * (1 + ud) // 2
        self.__Co_Up.loc[self.__j] = self.__Co_Up.loc[self.__j - 1]\
            + k * co * (1 + ud) // 2 + (k - 1) * al * (1 + ud) // 2
        self.__Al_Down.loc[self.__j] = self.__Al_Down.loc[self.__j - 1]\
            + al * (1 - ud) // 2
        self.__Co_Down.loc[self.__j] = self.__Co_Down.loc[self.__j - 1]\
            + k * co * (1 - ud) // 2 + (k - 1) * al * (1 - ud) // 2
        self.__count = [self.__j,
            [self.__Al_Up.loc[self.__j], self.__Al_Down.loc[self.__j],
            self.__Co_Up.loc[self.__j], self.__Co_Down.loc[self.__j]]]
        self.__sumCo.loc[self.__j] = self.__sumCo.loc[self.__j - 1] + co * k
        self.__sumAl.loc[self.__j] = self.__sumAl.loc[self.__j - 1] + al * k
        self.__sumAlk.loc[self.__j] = self.__sumAlk.loc[self.__j - 1] + al * (k - 1)
        if self.__sumAl.loc[self.__j] == 0:
            self.__sumH.loc[self.__j] = 0.5
        else:
            self.__sumH.loc[self.__j] = (self.__sumCo.loc[self.__j] - self.__sumAlk.loc[self.__j]) /\
                self.__sumAl.loc[self.__j]
        # Make a rolling count as well, use it for better μ estimation
        self.__count_roll = [[self.__j, self.__rw],
            [self.__Al_Up.loc[self.__j] - self.__Al_Up.loc[max(1, self.__j - self.__rw)],
            self.__Al_Down.loc[self.__j] - self.__Al_Down.loc[max(1, self.__j - self.__rw)],
            self.__Co_Up.loc[self.__j] - self.__Co_Up.loc[max(1, self.__j - self.__rw)],
            self.__Co_Down.loc[self.__j] - self.__Co_Down.loc[max(1, self.__j - self.__rw)]]]
        # Use durations to update η, σ, μ at every step if duration is higher than cutoff
        if t > self.__mindt:
            for ηi in range(self.__ηn):
                η = self.__ηs[ηi]
                for σi in range(self.__σn):
                    σ = self.__σs[σi]
                    for μi in range(self.__μn):
                        μ = self.__μs[μi]
                        if al == 1:  # Alternation
                            αk = α
                            ηk = η + (k - 1) / 2
                        else:  # Continuation
                            αk = k * α
                            ηk = η / k
                        self.__likel[ηi, σi, μi] = pdftc(ud, sud, s0, αk, ηk, σ, μ, t)  # η0
                        self.__probs[ηi, σi, μi] = self.__prob_floor + self.__probs[ηi, σi, μi] * self.__likel[ηi, σi, μi]
            self.__probs = self.__probs / np.sum(self.__probs)
            self.__fast.loc[self.__j] = False
        else:
            self.__fast.loc[self.__j] = True
        self.__ηm.loc[self.__j] = np.dot(self.__ηs, np.sum(self.__probs, axis=(1, 2)))
        self.__σm.loc[self.__j] = np.dot(self.__σs, np.sum(self.__probs, axis=(0, 2)))
        self.__μm.loc[self.__j] = np.dot(self.__μs, np.sum(self.__probs, axis=(0, 1)))
        # Use counts to update η, σ, μ  every update_freq steps  
        if (self.__j % self.__uf) == 0:
            self.__calc_update_freq()
            self.__update_freq()
        self.__j += 1


# %% Default quantile definitions
lq = 0.10
uq = 0.90
npts = 9
qtls_def = np.linspace(lq, uq, npts)

# %% QQ Plot

λ0 = 1

def qq_plots(data, α, η, σ, μ, hours, λ=λ0, qtls=qtls_def):
    pxs = data['Ptj'].values[:-1]
    dts = data['dtj'].values[1:]
    s = np.dot(pxs, dts)/np.sum(dts)
    data_Up = data[data['sign'] == +1]
    data_Down = data[data['sign'] == -1]
    data_Al = data[data['Al']]
    data_Co = data[data['Co']]
    data_Al_Up = data_Up[data_Up['Al']]
    data_Co_Up = data_Up[data_Up['Co']]
    data_Al_Down = data_Down[data_Down['Al']]
    data_Co_Down = data_Down[data_Down['Co']]
    n = len(data) - 2
    n_Al = len(data_Al) / n
    n_Co = len(data_Co) / n
    n_Al_Up = len(data_Al_Up) / n
    n_Co_Up = len(data_Co_Up) / n
    n_Al_Down = len(data_Al_Down) / n
    n_Co_Down = len(data_Co_Down) / n
    h = (n_Co) / (2 * n_Al)
    rvp = rlzvollog(data['Ptj']) * np.sqrt((hours * 3600) / data['dtj'].sum())
    rvxe = rvp * np.sqrt(2 * h)

    data_Al_Up_qtls = data_Al_Up['dtj'].quantile(qtls).values
    data_Co_Up_qtls = data_Co_Up['dtj'].quantile(qtls).values
    data_Al_Down_qtls = data_Al_Down['dtj'].quantile(qtls).values
    data_Co_Down_qtls = data_Co_Down['dtj'].quantile(qtls).values

    param_Al_Up_qtls = t_from_ps(+1, +1, s, α, η, σ, μ, qtls) * hours * 3600
    param_Co_Up_qtls = t_from_ps(+1, -1, s, α, η, σ, μ, qtls) * hours * 3600
    param_Al_Down_qtls = t_from_ps(-1, -1, s, α, η, σ, μ, qtls) * hours * 3600
    param_Co_Down_qtls = t_from_ps(-1, +1, s, α, η, σ, μ, qtls) * hours * 3600

    pT_Al_Up = nprobw(+1, s + (+1) * (0.5 - η) * α, s, α, η, σ, μ, 1)
    pT_Co_Up = nprobw(+1, s + (-1) * (0.5 - η) * α, s, α, η, σ, μ, 1)
    pT_Al_Down = nprobw(-1, s + (-1) * (0.5 - η) * α, s, α, η, σ, μ, 1)
    pT_Co_Down = nprobw(-1, s + (+1) * (0.5 - η) * α, s, α, η, σ, μ, 1)

    qtls_Al_Up_params = nprobsw(+1, +1, s, α, η, σ, μ, data_Al_Up_qtls / (hours * 3600)) / pT_Al_Up
    qtls_Co_Up_params = nprobsw(+1, -1, s, α, η, σ, μ, data_Co_Up_qtls / (hours * 3600)) / pT_Co_Up
    qtls_Al_Down_params = nprobsw(-1, -1, s, α, η, σ, μ, data_Al_Down_qtls / (hours * 3600)) / pT_Al_Down
    qtls_Co_Down_params = nprobsw(-1, +1, s, α, η, σ, μ, data_Co_Down_qtls / (hours * 3600)) / pT_Co_Down

    print(pd.DataFrame({'Al_Up': np.round(qtls_Al_Up_params, 2), 'Co_Up': np.round(qtls_Co_Up_params, 2),
                        'Al_Down': np.round(qtls_Al_Down_params, 2), 'Co_Down': np.round(qtls_Co_Down_params, 2)},
                        index=qtls))

    weight_Al_Up = (pT_Al_Up * pT_Al_Down) / (pT_Al_Up + pT_Al_Down)
    weight_Co_Up = (pT_Al_Up * (1 - pT_Al_Down)) / (pT_Al_Up + pT_Al_Down)
    weight_Al_Down = (pT_Al_Up * pT_Al_Down) / (pT_Al_Up + pT_Al_Down)
    weight_Co_Down = ((1 - pT_Al_Up) * pT_Al_Down) / (pT_Al_Up + pT_Al_Down)

    dist_Al_Up = np.linalg.norm(qtls_Al_Up_params - qtls)
    dist_Co_Up = np.linalg.norm(qtls_Co_Up_params - qtls)
    dist_Al_Down = np.linalg.norm(qtls_Al_Down_params - qtls)
    dist_Co_Down = np.linalg.norm(qtls_Co_Down_params - qtls)


    dists_fw = np.linalg.norm(np.array([weight_Al_Up, weight_Co_Up, weight_Al_Down, weight_Co_Down]) -
        np.array([n_Al_Up, n_Co_Up, n_Al_Down, n_Co_Down]))

    dists = n_Al_Up * dist_Al_Up + n_Co_Up * dist_Co_Up\
        + n_Al_Down * dist_Al_Down + n_Co_Down * dist_Co_Down + dists_fw

    fig, axs = plt.subplots(2, 2, figsize=(11, 11))
    fig.suptitle('η= ' + str(np.round(η,3)) + ' , σ= ' + '{:.3e}'.format(σ) + ' , μ= ' + '{:.3e}'.format(μ)
        + ' , H= ' + str(np.round(h, 3))+ ' , σXe= ' + '{:.3e}'.format(rvxe)
        + ' , Σdists= ' + '{:.2e}'.format(dists) + ' , λ= ' + '{:.2e}'.format(λ))
    # fig =  plt.figure(figsize=(8,8))
    axs[0, 0].plot(param_Al_Up_qtls, data_Al_Up_qtls, color = 'r', marker='o')
    axs[0, 0].plot(param_Al_Up_qtls, param_Al_Up_qtls, color='b', marker='+')
    axs[0, 0].set_title('Al_Up: f= ' + '{:.1%}'.format(n_Al_Up)\
        + ' , w= ' + '{:.1%}'.format(weight_Al_Up) + ' , dist= ' + '{:.2e}'.format(dist_Al_Up))
    # fig =  plt.figure(figsize=(8,8))
    axs[1, 0].plot(param_Co_Up_qtls, data_Co_Up_qtls, color = 'r', marker='o')
    axs[1, 0].plot(param_Co_Up_qtls, param_Co_Up_qtls, color='b', marker='+')
    axs[1, 0].set_title('Co_Up: f= ' + '{:.1%}'.format(n_Co_Up)\
        + ' , w= ' + '{:.1%}'.format(weight_Co_Up) + ' , dist= ' + '{:.2e}'.format(dist_Co_Up))
    # fig =  plt.figure(figsize=(8,8))
    axs[1, 1].plot(param_Al_Down_qtls, data_Al_Down_qtls, color = 'r', marker='o')
    axs[1, 1].plot(param_Al_Down_qtls, param_Al_Down_qtls, color='b', marker='+')
    axs[1, 1].set_title('Al_Down: f= ' + '{:.1%}'.format(n_Al_Down)\
        + ' , w= ' + '{:.1%}'.format(weight_Al_Down) + ' , dist= ' +  '{:.2e}'.format(dist_Al_Down))
    # fig =  plt.figure(figsize=(8,8))
    axs[0, 1].plot(param_Co_Down_qtls, data_Co_Down_qtls, color = 'r', marker='o')
    axs[0, 1].plot(param_Co_Down_qtls, param_Co_Down_qtls, color='b', marker='+')
    axs[0, 1].set_title('Co_Down: f= ' + '{:.1%}'.format(n_Co_Down)\
        + ' , w= ' + '{:.1%}'.format(weight_Co_Down) + ' , dist= ' +  '{:.2e}'.format(dist_Co_Down))
    for ax in axs.flat:
        ax.set(xlabel='Parametric', ylabel='Data')
    for ax in axs.flat:
        ax.label_outer()

# %% Fit trio

def fit_trio(data, α, hours, λ=λ0, show_charts=False, qtls=qtls_def):
    pxs = data['Ptj'].values[:-1]
    dts = data['dtj'].values[1:]
    s = np.dot(pxs, dts)/np.sum(dts)
    data_Up = data[data['sign'] == +1]
    data_Down = data[data['sign'] == -1]
    data_Al = data[data['Al']]
    data_Co = data[data['Co']]
    data_Al_Up = data_Up[data_Up['Al']]
    data_Co_Up = data_Up[data_Up['Co']]
    data_Al_Down = data_Down[data_Down['Al']]
    data_Co_Down = data_Down[data_Down['Co']]
    n = len(data) - 2
    n_Al = len(data_Al) / n
    n_Co = len(data_Co) / n
    n_Al_Up = len(data_Al_Up) / n
    n_Co_Up = len(data_Co_Up) / n
    n_Al_Down = len(data_Al_Down) / n
    n_Co_Down = len(data_Co_Down) / n
    h = (n_Co) / (2 * n_Al)
    rvp = rlzvollog(data['Ptj']) * np.sqrt((hours * 3600) / data['dtj'].sum())
    rvxe = rvp * np.sqrt(2 * h)

    data_Al_Up_qtls = data_Al_Up['dtj'].quantile(qtls).values
    data_Co_Up_qtls = data_Co_Up['dtj'].quantile(qtls).values
    data_Al_Down_qtls = data_Al_Down['dtj'].quantile(qtls).values
    data_Co_Down_qtls = data_Co_Down['dtj'].quantile(qtls).values

    def dist_trio(x): #  x = [η, σ, μ];
        η = x[0]
        σ = x[1]
        μ = x[2]

        pT_Al_Up = nprobw(+1, s + (+1) * (0.5 - η) * α, s, α, η, σ, μ, 1)
        pT_Co_Up = nprobw(+1, s + (-1) * (0.5 - η) * α, s, α, η, σ, μ, 1)
        pT_Al_Down = nprobw(-1, s + (-1) * (0.5 - η) * α, s, α, η, σ, μ, 1)
        pT_Co_Down = nprobw(-1, s + (+1) * (0.5 - η) * α, s, α, η, σ, μ, 1)

        qtls_Al_Up_params = nprobsw(+1, +1, s, α, η, σ, μ, data_Al_Up_qtls / (hours * 3600)) / pT_Al_Up
        qtls_Co_Up_params = nprobsw(+1, -1, s, α, η, σ, μ, data_Co_Up_qtls / (hours * 3600)) / pT_Co_Up
        qtls_Al_Down_params = nprobsw(-1, -1, s, α, η, σ, μ, data_Al_Down_qtls / (hours * 3600)) / pT_Al_Down
        qtls_Co_Down_params = nprobsw(-1, +1, s, α, η, σ, μ, data_Co_Down_qtls / (hours * 3600)) / pT_Co_Down

        weight_Al_Up = (pT_Al_Up * pT_Al_Down) / (pT_Al_Up + pT_Al_Down)
        weight_Co_Up = (pT_Al_Up * (1 - pT_Al_Down)) / (pT_Al_Up + pT_Al_Down)
        weight_Al_Down = (pT_Al_Up * pT_Al_Down) / (pT_Al_Up + pT_Al_Down)
        weight_Co_Down = ((1 - pT_Al_Up) * pT_Al_Down) / (pT_Al_Up + pT_Al_Down)

        dist_Al_Up = np.linalg.norm(qtls_Al_Up_params - qtls)
        dist_Co_Up = np.linalg.norm(qtls_Co_Up_params - qtls)
        dist_Al_Down = np.linalg.norm(qtls_Al_Down_params - qtls)
        dist_Co_Down = np.linalg.norm(qtls_Co_Down_params - qtls)

        dists_fw = np.linalg.norm(np.array([weight_Al_Up, weight_Co_Up, weight_Al_Down, weight_Co_Down]) -
            np.array([n_Al_Up, n_Co_Up, n_Al_Down, n_Co_Down]))

        dists = n_Al_Up * dist_Al_Up + n_Co_Up * dist_Co_Up\
            + n_Al_Down * dist_Al_Down + n_Co_Down * dist_Co_Down + dists_fw * λ

        return dists


    x0 = np.array([h, rvxe, 0.])
    fit_pt = optimize.minimize(fun=dist_trio, x0=x0, method='Nelder-Mead', tol=1e-6,
                                options={'disp': False, 'maxiter': 2000, 'adaptive': True})
    # bounds_DE = [(0, 0.5), (1e-6, 4), (-100, +100)]
    # fit_pt = optimize.differential_evolution(dist_trio, bounds=bounds_DE, popsize=50, tol=1e-2, disp=True)
    η, σ, μ = fit_pt.x
    dists = fit_pt.fun

    pT_Al_Up = nprobw(+1, s + (+1) * (0.5 - η) * α, s, α, η, σ, μ, 1)
    pT_Co_Up = nprobw(+1, s + (-1) * (0.5 - η) * α, s, α, η, σ, μ, 1)
    pT_Al_Down = nprobw(-1, s + (-1) * (0.5 - η) * α, s, α, η, σ, μ, 1)
    pT_Co_Down = nprobw(-1, s + (+1) * (0.5 - η) * α, s, α, η, σ, μ, 1)

    qtls_Al_Up_params = nprobsw(+1, +1, s, α, η, σ, μ, data_Al_Up_qtls / (hours * 3600)) / pT_Al_Up
    qtls_Co_Up_params = nprobsw(+1, -1, s, α, η, σ, μ, data_Co_Up_qtls / (hours * 3600)) / pT_Co_Up
    qtls_Al_Down_params = nprobsw(-1, -1, s, α, η, σ, μ, data_Al_Down_qtls / (hours * 3600)) / pT_Al_Down
    qtls_Co_Down_params = nprobsw(-1, +1, s, α, η, σ, μ, data_Co_Down_qtls / (hours * 3600)) / pT_Co_Down

    param_Al_Up_qtls = t_from_ps(+1, +1, s, α, η, σ, μ, qtls) * hours * 3600
    param_Co_Up_qtls = t_from_ps(+1, -1, s, α, η, σ, μ, qtls) * hours * 3600
    param_Al_Down_qtls = t_from_ps(-1, -1, s, α, η, σ, μ, qtls) * hours * 3600
    param_Co_Down_qtls = t_from_ps(-1, +1, s, α, η, σ, μ, qtls) * hours * 3600

    weight_Al_Up = (pT_Al_Up * pT_Al_Down) / (pT_Al_Up + pT_Al_Down)
    weight_Co_Up = (pT_Al_Up * (1 - pT_Al_Down)) / (pT_Al_Up + pT_Al_Down)
    weight_Al_Down = (pT_Al_Up * pT_Al_Down) / (pT_Al_Up + pT_Al_Down)
    weight_Co_Down = ((1 - pT_Al_Up) * pT_Al_Down) / (pT_Al_Up + pT_Al_Down)

    dist_Al_Up = np.linalg.norm(qtls_Al_Up_params - qtls)
    dist_Co_Up = np.linalg.norm(qtls_Co_Up_params - qtls)
    dist_Al_Down = np.linalg.norm(qtls_Al_Down_params - qtls)
    dist_Co_Down = np.linalg.norm(qtls_Co_Down_params - qtls)

    if show_charts:

        fig, axs = plt.subplots(2, 2, figsize=(11, 11))
        fig.suptitle('H= ' + str(np.round(h, 3)) + ' , σXe= ' + '{:.3e}'.format(rvxe)
            + ' , η= ' + str(np.round(η, 3))+ ' , σ= ' + '{:.3e}'.format(σ) + ' , μ= ' + '{:.3e}'.format(μ)
            + ' , Σdists= ' + '{:.2e}'.format(dists) + ' , λ= ' + '{:.2e}'.format(λ))
        # fig =  plt.figure(figsize=(8,8))
        axs[0, 0].plot(param_Al_Up_qtls, data_Al_Up_qtls, color = 'r', marker='o')
        axs[0, 0].plot(param_Al_Up_qtls, param_Al_Up_qtls, color='b', marker='+')
        axs[0, 0].set_title('Al_Up: f= ' + '{:.1%}'.format(n_Al_Up)\
            + ' , w= ' + '{:.1%}'.format(weight_Al_Up) + ' , dist= ' + '{:.2e}'.format(dist_Al_Up))
        # fig =  plt.figure(figsize=(8,8))
        axs[1, 0].plot(param_Co_Up_qtls, data_Co_Up_qtls, color = 'r', marker='o')
        axs[1, 0].plot(param_Co_Up_qtls, param_Co_Up_qtls, color='b', marker='+')
        axs[1, 0].set_title('Co_Up: f= ' + '{:.1%}'.format(n_Co_Up)\
            + ' , w= ' + '{:.1%}'.format(weight_Co_Up) + ' , dist= ' + '{:.2e}'.format(dist_Co_Up))
        # fig =  plt.figure(figsize=(8,8))
        axs[1, 1].plot(param_Al_Down_qtls, data_Al_Down_qtls, color = 'r', marker='o')
        axs[1, 1].plot(param_Al_Down_qtls, param_Al_Down_qtls, color='b', marker='+')
        axs[1, 1].set_title('Al_Down: f= ' + '{:.1%}'.format(n_Al_Down)\
            + ' , w= ' + '{:.1%}'.format(weight_Al_Down) + ' , dist= ' +  '{:.2e}'.format(dist_Al_Down))
        # fig =  plt.figure(figsize=(8,8))
        axs[0, 1].plot(param_Co_Down_qtls, data_Co_Down_qtls, color = 'r', marker='o')
        axs[0, 1].plot(param_Co_Down_qtls, param_Co_Down_qtls, color='b', marker='+')
        axs[0, 1].set_title('Co_Down: f= ' + '{:.1%}'.format(n_Co_Down)\
            + ' , w= ' + '{:.1%}'.format(weight_Co_Down) + ' , dist= ' +  '{:.2e}'.format(dist_Co_Down))
        for ax in axs.flat:
            ax.set(xlabel='Parametric', ylabel='Data')
        for ax in axs.flat:
            ax.label_outer()

    results = pd.Series([h, rvxe, η, σ, μ, dists,
        n_Al_Up, weight_Al_Up, dist_Al_Up, n_Co_Up, weight_Co_Up, dist_Co_Up,
        n_Al_Down, weight_Al_Down, dist_Al_Down, n_Co_Down, weight_Co_Down, dist_Co_Down],
        index=['H', 'σXe', 'η', 'σ', 'μ', 'Σdists',
         'f_Al_Up', 'w_Al_Up', 'd_Al_Up', 'f_Co_Up', 'w_Co_Up', 'd_Co_Up',
          'f_Al_Down', 'w_Al_Down', 'd_Al_Down', 'f_Co_Down', 'w_Co_Down', 'd_Co_Down'])

    return results

# %% Stats window and plottting

min_window = 15
min_step = 1

def stats_window(data, α, hours, λ=λ0, window_min=min_window, time_step_min=min_step):
    min_range = range(min_window, int(hours * 60) + min_step, min_step)
    path_roll_min = [fit_trio(data.loc[((m - min_window) * 60):(m * 60)], α, hours, λ) for m in min_range]
    df_stats_roll = pd.concat(path_roll_min, axis=1).transpose()
    df_stats_roll.index = np.array(min_range) / 60
    path_cum_min = [fit_trio(data.loc[:(m * 60)], α, hours, λ) for m in min_range]
    df_stats_cum = pd.concat(path_cum_min, axis=1).transpose()
    df_stats_cum.index = np.array(min_range) / 60
    return [df_stats_roll, df_stats_cum]

def plot_stats(data_roll, data_cum, eta_ts, vol_ts, mu_ts, tmrst, hours, window_min=min_window):
    fig, axs = plt.subplots(4, 2, figsize=(18, 24))
    fig.suptitle('Window = ' + str(window_min) + 'min, t = ', y=0.90)
    
    data_index = data_roll.index
    
    eta_ts_ds = pd.Series(eta_ts, index=tmrst[:-1] * hours).loc[min_window / 60:]
    eta_plot = pd.Series([np.mean(eta_ts_ds.loc[t - 1e-5:t + 1e-5].values) for t in data_index],
                        index=data_index)
    vol_ts_ds = pd.Series(vol_ts, index=tmrst[:-1] * hours).loc[min_window / 60:]
    vol_plot = pd.Series([np.mean(vol_ts_ds.loc[t - 1e-5:t + 1e-5].values) for t in data_index],
                        index=data_index)
    mu_ts_ds = pd.Series(mu_ts, index=tmrst[:-1] * hours).loc[min_window / 60:]
    mu_plot = pd.Series([np.mean(mu_ts_ds.loc[t - 1e-5:t + 1e-5].values) for t in data_index],
                        index=data_index)
    
    axs[0, 0].plot(data_roll[['η', 'H']])
    axs[0, 0].plot(eta_plot, color='k')
    axs[0, 0].legend(['η', 'H'])
    axs[0, 0].set_title('η and H - Rolling')
    axs[0, 1].plot(data_cum[['η', 'H']])
    axs[0, 1].plot(eta_plot, color='k')
    axs[0, 1].legend(['η', 'H'])
    axs[0, 1].set_title('η and H - Cumulative')
    
    axs[1, 0].plot(data_roll[['σ', 'σXe']])
    axs[1, 0].plot(vol_plot, color='k')
    axs[1, 0].set_title('σ and σXe - Rolling')
    axs[1, 0].legend(['σ', 'σXe'])
    axs[1, 1].plot(data_cum[['σ', 'σXe']])
    axs[1, 1].plot(vol_plot, color='k')
    axs[1, 1].legend(['σ', 'σXe'])
    axs[1, 1].set_title('σ and σXe - Cumulative')
    
    axs[2, 0].plot(data_roll[['μ']])
    axs[2, 0].plot(mu_plot, color='k')
    axs[2, 0].legend(['μ'])
    axs[2, 0].set_title('μ - Rolling')
    axs[2, 1].plot(data_cum[['μ']])
    axs[2, 1].plot(mu_plot, color='k')
    axs[2, 1].legend(['μ'])
    axs[2, 1].set_title('μ - Cumulative')
    
    axs[3, 0].plot(data_roll[['f_Al_Up', 'f_Co_Up', 'f_Al_Down', 'f_Co_Down']])
    axs[3, 0].plot(eta_plot, color='k')
    axs[3, 0].legend(['f_Al_Up', 'f_Co_Up', 'f_Al_Down', 'f_Co_Down'])
    axs[3, 0].set_title('Frequencies - Rolling')
    axs[3, 1].plot(data_cum[['f_Al_Up', 'f_Co_Up', 'f_Al_Down', 'f_Co_Down']])
    axs[3, 1].plot(eta_plot, color='k')
    axs[3, 1].legend(['f_Al_Up', 'f_Co_Up', 'f_Al_Down', 'f_Co_Down'])
    axs[3, 1].set_title('Frequencies - Cumulative')

# def invprob(ud, sud, s, α, η, σ, μ, qtls=qtls_def, print_flag=False):
#     '''invprob(ud, s0, s, α, η, σ, μ, n) solves the equation
#     Pup(t) / Pup(1) (or Pdown(t) / Pdown(1))  = p;
#     so for a random p such that 0 <= p <= 1 we find
#     the expected time of a price change for the sign given'''
#     cont_flag =  - ud * sud
#     s0 = s + sud * (0.5 - η) * α
#     adjη = np.abs(ud + sud) * η + np.abs(ud - sud) / 2
#     scl = ((α / s) / σ)**2
#     adj = adjη * scl
#     mprobt = partial(nprobw, ud, s0, s, α, η, σ, μ)
#     mprobt1 = mprobt(1)
#     ts = []
#     for q in qtls:
#         def groot(t):
#             return np.round(mprobt(t)/mprobt1 - q, decimals=6)
#         sol = optimize.root_scalar(groot, bracket=[0, 1], method='brentq')
#         tp = sol.root
#         ts = ts + [tp]
#     ts = np.array(ts)
#     x0 = np.array([-0.5, 1., 1.])
#     def gig(x): #  x = [p, b, scale]; loc=0
#         return np.array([st.geninvgauss.cdf(t, x[0], x[1], 0, x[2] * adj) for t in ts])
#     def dist_gig(x):
#         return np.linalg.norm(qtls - gig(x))
#     fit_gig = optimize.minimize(fun=dist_gig, x0=x0, method='Nelder-Mead',
#                                 options={'disp': print_flag, 'maxiter': 2000, 'adaptive': True})
# #     bounds_DE = [(-1, 1), (1e-6, 4), (0.1, 10)]
# #     fit_gig_DE = optimize.differential_evolution(dist_gig, bounds=bounds_DE, popsize=50, tol=1e-2, disp=print_flag)
#     p, b, scale_adj = fit_gig.x
#     scale = scale_adj * adj
#     dist = fit_gig.fun
#     mean = scale * kv(1 + scale, b) / kv(scale, b)
#     adj_mean = mean / scl
#     fit_qtls = np.array([st.geninvgauss.ppf(q, p, b, 0, scale) for q in qtls])
#     if print_flag:
#         print(pd.DataFrame({'From P': ts, 'GIG': fit_qtls}, index=qtls))
#     return pd.Series([ud, sud, cont_flag, s0, s, α, η, σ, μ, scl, adj, dist, p, b, scale, scale_adj, mean, adj_mean],
#                         index=['ud', 'sud', 'Co/Al', 's0', 's', 'α', 'η', 'σ', 'μ', 'scl', 'adj', 'dist', 'p', 'b', 'scale', 'scale_adj', 'mean', 'mean_scl'])


# def invprobgig(ud, sud, s, α, p, b, scale, qtls=qtls_def, print_flag=False):
#     '''invprob(ud, s0, s, α, η, σ, μ, n) solves the equation
#     Pup(t) / Pup(1) (or Pdown(t) / Pdown(1))  = p;
#     so for a random p such that 0 <= p <= 1 we find
#     the expected time of a price change for the sign given'''
#     fit_qtls = np.array([st.geninvgauss.ppf(q, p, b, 0, scale) for q in qtls])
#     def pft(x): #  x = [η, σ, μ];
#         return np.array([nprobw(ud, s + sud * (0.5 - x[0]) * α, s, α, x[0], x[1], x[2], t) /
#                          nprobw(ud, s + sud * (0.5 - x[0]) * α, s, α, x[0], x[1], x[2], 1) for t in fit_qtls])
#     def dist_gig(x):
#         return np.linalg.norm(qtls - pft(x))
#     x0 = np.array([0.3, 0.01, 0.])
#     fit_gig = optimize.minimize(fun=dist_gig, x0=x0, method='Nelder-Mead',
#                                 options={'disp': print_flag, 'maxiter': 2000, 'adaptive': True})
# #     bounds_DE = [(-1, 1), (1e-6, 4), (0.1, 10)]
# #     fit_gig_DE = optimize.differential_evolution(dist_gig, bounds=bounds_DE, popsize=50, tol=1e-2, disp=print_flag)
#     η, σ, μ = fit_gig.x
#     cont_flag =  - ud * sud
#     s0 = s + sud * (0.5 - η) * α
#     scl = ((α / s) / σ)**2
#     adjη = np.abs(ud + sud) * η + np.abs(ud - sud) / 2
#     adj = adjη * scl 
#     scale_adj = scale / adj
#     dist = fit_gig.fun
#     mean = scale * kv(1 + scale, b) / kv(scale, b)
#     adj_mean = mean / scl
#     if print_flag:
#         mprobt = partial(nprobw, ud, s0, s, α, η, σ, μ)
#         mprobt1 = mprobt(1)
#         ts = []
#         for q in qtls:
#             def groot(t):
#                 return np.round(mprobt(t)/mprobt1 - q, decimals=6)
#             sol = optimize.root_scalar(groot, bracket=[0, 1], method='brentq')
#             tp = sol.root
#             ts = ts + [tp]
#         ts = np.array(ts)
#         print(pd.DataFrame({'From P': ts, 'GIG': fit_qtls}, index=qtls))
#     return pd.Series([ud, sud, cont_flag, s0, s, α, η, σ, μ, scl, adj, dist, p, b, scale, scale_adj, mean, adj_mean],
#                         index=['ud', 'sud', 'Co/Al', 's0', 's', 'α', 'η', 'σ', 'μ', 'scl', 'adj', 'dist', 'p', 'b', 'scale', 'scale_adj', 'mean', 'mean_scl'])

# def param_from_data(data, α, qtls=qtls_def, print_flag=False):
#     pxs = data['Ptj'].values[:-1]
#     dts = data['dtj'].values[1:]
#     s = np.dot(pxs, dts)/np.sum(dts)
#     data_Up = data[data['sign'] == +1]
#     data_Down = data[data['sign'] == -1]
#     data_Al = data[data['Al']]
#     data_Co = data[data['Co']]
#     data_Al_Up = data_Up[data_Up['Al']]
#     data_Co_Up = data_Up[data_Up['Co']]
#     data_Al_Down = data_Down[data_Down['Al']]
#     data_Co_Down = data_Down[data_Down['Co']]
#     n = len(data) - 2
#     n_Al = len(data_Al) / n
#     n_Co = len(data_Co) / n
#     n_Al_Up = len(data_Al_Up) / n
#     n_Co_Up = len(data_Co_Up) / n
#     n_Al_Down = len(data_Al_Down) / n
#     n_Co_Down = len(data_Co_Down) / n
#     h = (n_Co) / (2 * n_Al)
#     rvp = rlzvollog(data['Ptj']) * np.sqrt((hours * 3600) / data['dtj'].sum())
#     rvxe = rvp * np.sqrt(2 * h)

#     data_Al_Up_qtls = data_Al_Up['dtj'].quantile(qtls).values
#     data_Co_Up_qtls = data_Co_Up['dtj'].quantile(qtls).values
#     data_Al_Down_qtls = data_Al_Down['dtj'].quantile(qtls).values
#     data_Co_Down_qtls = data_Co_Down['dtj'].quantile(qtls).values


#     cont_flag =  - ud * sud
#     s0 = s + sud * (0.5 - η) * α
#     adjη = np.abs(ud + sud) * η + np.abs(ud - sud) / 2
#     scl = ((α / s) / σ)**2
#     adj = adjη * scl
#     mprobt = partial(nprobw, ud, s0, s, α, η, σ, μ)
#     mprobt1 = mprobt(1)
#     ts = []
#     for q in qtls:
#         def groot(t):
#             return np.round(mprobt(t)/mprobt1 - q, decimals=6)
#         sol = optimize.root_scalar(groot, bracket=[0, 1], method='brentq')
#         tp = sol.root
#         ts = ts + [tp]
#     ts = np.array(ts)
#     x0 = np.array([-0.5, 1., 1.])
#     def gig(x): #  x = [p, b, scale]; loc=0
#         return np.array([st.geninvgauss.cdf(t, x[0], x[1], 0, x[2] * adj) for t in ts])
#     def dist_gig(x):
#         return np.linalg.norm(qtls - gig(x))
#     fit_gig = optimize.minimize(fun=dist_gig, x0=x0, method='Nelder-Mead',
#                                 options={'disp': print_flag, 'maxiter': 2000, 'adaptive': True})
# #     bounds_DE = [(-1, 1), (1e-6, 4), (0.1, 10)]
# #     fit_gig_DE = optimize.differential_evolution(dist_gig, bounds=bounds_DE, popsize=50, tol=1e-2, disp=print_flag)
#     p, b, scale_adj = fit_gig.x
#     scale = scale_adj * adj
#     dist = fit_gig.fun
#     mean = scale * kv(1 + scale, b) / kv(scale, b)
#     adj_mean = mean / scl
#     fit_qtls = np.array([st.geninvgauss.ppf(q, p, b, 0, scale) for q in qtls])
#     if print_flag:
#         print(pd.DataFrame({'From P': ts, 'GIG': fit_qtls}, index=qtls))
#     return pd.Series([ud, sud, cont_flag, s0, s, α, η, σ, μ, scl, adj, dist, p, b, scale, scale_adj, mean, adj_mean],
#                         index=['ud', 'sud', 'Co/Al', 's0', 's', 'α', 'η', 'σ', 'μ', 'scl', 'adj', 'dist', 'p', 'b', 'scale', 'scale_adj', 'mean', 'mean_scl'])

# def get_fit(data, ud, sud, α, hours, qtls=qtls_def):
#     p, b, loc, scale = st.geninvgauss.fit(data['dtj'].values / (hours * 3600), floc=0)
#     pxs = data['Ptj'].values[:-1]
#     dts = data['dtj'].values[1:]
#     s = np.dot(pxs, dts)/np.sum(dts)
#     pt_df = invprobgig(ud, sud, s, α, p, b, scale, qtls=qtls)
#     return pt_df

# def get_fits_all(data, α, hours, qtls=qtls_def):
#     pxs = data['Ptj'].values[:-1]
#     dts = data['dtj'].values[1:]
#     s = np.dot(pxs, dts)/np.sum(dts)
#     data_Up = data[data['sign'] == +1]
#     data_Down = data[data['sign'] == -1]
#     data_Al_Up = data_Up[data_Up['Al']]
#     data_Co_Up = data_Up[data_Up['Co']]
#     data_Al_Down = data_Down[data_Down['Al']]
#     data_Co_Down = data_Down[data_Down['Co']]
#     n = len(data) - 2
#     n_Al_Up = len(data_Al_Up) / n
#     n_Co_Up = len(data_Co_Up) / n
#     n_Al_Down = len(data_Al_Down) / n
#     n_Co_Down = len(data_Co_Down) / n
#     fit_Al_Up = get_fit(data_Al_Up, +1, +1, α, hours, qtls)[['η', 'σ', 'μ']]
#     fit_Al_Up.index = ['η_Al_Up', 'σ_Al_Up', 'μ_Al_Up']
#     fit_Co_Up = get_fit(data_Co_Up, +1, -1, α, hours, qtls)[['η', 'σ', 'μ']]
#     fit_Co_Up.index = ['η_Co_Up', 'σ_Co_Up', 'μ_Co_Up']
#     fit_Al_Down = get_fit(data_Al_Down, -1, -1, α, hours, qtls)[['η', 'σ', 'μ']]
#     fit_Al_Down.index = ['η_Al_Down', 'σ_Al_Down', 'μ_Al_Down']
#     fit_Co_Down = get_fit(data_Co_Down, -1, +1, α, hours, qtls)[['η', 'σ', 'μ']]
#     fit_Co_Down.index = ['η_Co_Down', 'σ_Co_Down', 'μ_Co_Down']
#     h = (n_Co_Up + n_Co_Down) / (2 * (n_Al_Up + n_Al_Down))
#     rvp = rlzvollog(data['Ptj']) * np.sqrt((hours * 3600) / data['dtj'].sum())
#     rvxe = rvp * np.sqrt(2 * h)
#     ser_stats = pd.Series([n_Al_Up, n_Co_Up, n_Al_Down, n_Co_Down, h, rvp, rvxe],
#         index=['n_Al_Up', 'n_Co_Up', 'n_Al_Down', 'n_Co_Down', 'H', 'σP', 'σXe'])
#     return pd.concat([fit_Al_Up, fit_Co_Up, fit_Al_Down, fit_Co_Down, ser_stats])

# -----------------------------

# # Define t_from_p (for inverse CDF)
# def t_from_p(s0, s, α, η, σ, μ, p):
#     '''t_from_p(p, s0, s, α, η, σ, μ) solves the equation
#     Pup(t) + Pdown(t) = p; so for a random p such that 0 <= p <= 1 we find
#     the expected time of a price change'''
#     mprobtup = partial(nprobw, 1, s0, s, α, η, σ, μ)
#     mprobtdown = partial(nprobw, -1, s0, s, α, η, σ, μ)
#     def groot(t):
#         return np.round(mprobtup(t) + mprobtdown(t) - p, decimals=6)
#     sol = optimize.root_scalar(groot, bracket=[0, 1], method='brentq')
#     return sol.root

# Define t_from_pud (for inverse CDF)
# def t_from_pud(s0, s, α, η, σ, μ, ud, p):
#     '''t_from_pud(p, s0, s, α, η, σ, μ) solves the equation
#     Pup(t) / Pup(1) (or Pdown(t) / Pdown(1))  = p;
#     so for a random p such that 0 <= p <= 1 we find
#     the expected time of a price change for the sign given'''
#     mprobt = partial(nprobw, ud, s0, s, α, η, σ, μ)
#     mprobtT = nprobw(ud, s0, s, α, η, σ, μ, 1)
#     if mprobtT == 0:
#         print('mprobtT == 0')
#         print((s0, s, α, η, σ, μ, ud, p))
#     def groot(t):
#         return np.round(mprobt(t)/mprobtT - p, decimals=6)
#     sol = optimize.root_scalar(groot, bracket=[0, 1], method='brentq')
#     return sol.root

# # Define CDF
# def cdf(s0, s, α, η, σ, μ, npts=100+1):
#     grid = np.linspace(0., 0.999, npts)
#     pts = pd.Series(grid, index=[t_from_p(s0, s, α, η, σ, μ, p)
#                                  for p in grid])
#     pts.index.name = 't'
#     pts.name = 'CDF(t)'
#     return pts

# # Define CDFs
# def cdfs(s0, s, α, η, σ, μ, npts=100):
#     grid = np.linspace(0, 0.999, npts)
#     ts = [t_from_p(s0, s, α, η, σ, μ, p) for p in grid]
#     up = [nprobw(+1, s0, s, α, η, σ, μ, t) for t in ts]
#     up = np.minimum(up, grid)
#     pts = pd.DataFrame({'CDF(t)': grid, 'Up': up, 'Down': grid - up},
#                        index=ts)
#     pts.index.name = 't'
#     return pts

# # Define Quantiles
# def pquant(s, α, η, σ, μ, sup, ud, lg=0., ug=0.999, npts=100+1):
#     grid = np.linspace(lg, ug, npts)
#     prev_s = s + sup * (0.5 - η) * α
#     ts = [t_from_pud(prev_s, s, α, η, σ, μ, ud, p) for p in grid]
#     pts = pd.Series(grid, index=ts)
#     pts.index.name = 't'
#     pts.name = str((sup, ud))
#     return pts

# # Define Quantiles - inverted
# def pquantinv(s, α, η, σ, μ, sup, ud, lg=0., ug=0.999, npts=100+1):
#     grid = np.linspace(lg, ug, npts)
#     prev_s = s + sup * (0.5 - η) * α
#     ts = [t_from_pud(prev_s, s, α, η, σ, μ, ud, p) for p in grid]
#     pts = pd.Series(ts, index=grid)
#     pts.index.name = 'q'
#     pts.name = str((sup, ud))
#     return pts

# Define up_down
def up_down(s0, s, α, η, σ, μ, t, p):
    '''up_down(s0, s, α, η, σ, μ, t, p) finds Pup(t) and Pdown(t) with t
    given by solving t_from_p outside the function so that
    for a random p such that 0 <= p <= 1 we choose
    the expected sign (+1 or -1) of a price change'''
    pup = nprobw(1, s0, s, α, η, σ, μ, t)
    pdown = nprobw(-1, s0, s, α, η, σ, μ, t)
    return int(2 * np.heaviside(p - pdown / (pup + pdown), 1) - 1)

# Define next state
def next_state(α, η, σ, μ, state):
    sup = state[0]
    s = state[1]
    prev_s = s + sup * (0.5 - η) * α
    p = np.random.uniform(low=0., high=0.999999)
    up = up_down(prev_s, s, α, η, σ, μ, 1, p)
    p = np.random.uniform(low=0., high=0.999999)
    t = t_from_pud(prev_s, s, α, η, σ, μ, up, p)
    alt = sup == up
    cont = sup != up
    return [[t, s + up * α, up, alt, cont], [-up, s + up * α]]

# Full simulation with random 1st state
def simul(s, α, η, σ, μ, T):
    state = [np.random.choice([-1, +1]), s]
    df = pd.DataFrame({'dtj': 0, 'Ptj': state[1], 'sign': 0,
                       'Al': True, 'Co': False}, index=[0])
    n = 1
    while df['dtj'].sum() < T:
        row, state = next_state(α, η, σ, μ, state)
        df.loc[n] = row
        n += 1
    df['t'] = df['dtj'].cumsum()
    df.set_index('t', inplace=True)
    return df

