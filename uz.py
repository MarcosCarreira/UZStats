# The Robert and Rosenbaum Uncertainty Zones model
# Implementation by
# Marcos Costa Santos Carreira
# École Polytechnique - CMAP

# Imports

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functools import partial
from scipy import optimize
import numba as numba
from math import erfc

# Basic path functions

# Random vector

def frndn(nsteps=1, seed=None):
    '''frndn(nsteps=1, seed=None) returns a NumPy array of normally
    distributed numbers'''
    np.random.seed(seed)
    return np.random.randn(nsteps)

# Monte Carlo Path

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

# Times range

def atrange(nsteps=1, t=1.0):
    '''atrange(nsteps=1, t=1.0) returns a NumPy array for a time series
    [0,dt,...,t-dt,t]'''
    tsindex = np.arange(nsteps+1)*t/nsteps
    return tsindex

# Riffle

def riffle(a, b):
    c = np.empty((a.size + b.size,), dtype=a.dtype)
    c[0::2] = a
    c[1::2] = b
    return c

# Brownian Bridge - next iteration

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

# Brownian Bridge - all iterations

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

# Brownian Bridge - final path

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


# Brownian Bridge - final path as array of prices

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

# Discretization functions

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

# Updating the last traded price

@numba.jit(nopython=True)
def updtrpr(alpha,eta,Pt,xt1):
    '''updtrpr(alpha, eta, Pt, xt1) returns the next transacion
    price P(t+1) given the efficient price X(t+1), the previous
    transacion price P(t), the tick value alpha and the
    parameter eta'''
    return Pt+Li(alpha,eta,Pt,xt1)*np.sign(xt1 - Pt)*alpha

# Calculation of the traded prices given the efficient prices

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

# Durations, changes and effective prices

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

# Processing

# Traded price paths

def read_trd_path(pathf, j, vol, alpha, eta, filename='trdpaths'):
    return pd.read_hdf(pathf+filename+'_'+str(j)+'.h5')\
        .loc[:, (vol, alpha, eta)]

def read_trd_path_drift(pathf, j, vol, mu, alpha, eta, filename='mutrdpaths'):
    return pd.read_csv(pathf+filename+'_'+str(j)+'_'+str(vol)+'_'+str(mu)+\
        '_'+str(alpha)+'_'+str(eta)+'.csv', header=None)[0]

# Reduce to price changes

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

# Statistics and estimations

# UZ statistics

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

# Realized volatility

def rlzvollog(prices):
    """rlzvollog(prices) calculates the realized volatility of 
    a time series of prices using logreturns"""
    pxs = np.log(prices/prices.shift(1))
    return np.sqrt(np.sum(pxs*pxs))

# UZ statistics and durations

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

# Barrier CDF - Numba, within epsilon (while)

@numba.jit(nopython=True)
def nprobw(ud, s0, s, α, η, σ, μ, t):
    if t == 0:
        ans = 0
    else:
        ω = μ - 0.5 * σ ** 2
        x0 = np.log(s0)
        bu = np.log(s + (0.5 + η) * α)
        bd = np.log(s - (0.5 + η) * α)
        bpm = bu * (1 + ud) / 2 + bd * (1 - ud) / 2
        bmp = bu * (1 - ud) / 2 + bd * (1 + ud) / 2
        tyzn = 0
        n = 0
        tyznn = 1
        eps=1e-10
        while tyznn > eps:
            n += 1
            yn = ud * (2 * (n - 1) * bmp - (2 * n - 1) * bpm + x0)
            zn = ud * (2 * n * bmp - (2 * n - 1) * bpm - x0)
            tyznn =\
                (np.exp(ω * yn / (σ ** 2)) * erfc(
                -(yn + ω * t) / (σ * np.sqrt(2 * t))) -
                np.exp(ω * zn / (σ ** 2)) * erfc(
                -(zn + ω * t) / (σ * np.sqrt(2 * t))) +
                np.exp(-ω * yn / (σ ** 2)) * erfc(
                -(yn - ω * t) / (σ * np.sqrt(2 * t))) -
                np.exp(-ω * zn / (σ ** 2)) * erfc(
                -(zn - ω * t) / (σ * np.sqrt(2 * t)))) / 2
            tyzn += tyznn
        ans = np.exp(ω * (bpm - x0) / (σ ** 2)) * tyzn
    return ans

# Define t_from_p (for inverse CDF)
def t_from_p(s0, s, α, η, σ, μ, p):
    '''t_from_p(p, s0, s, α, η, σ, μ) solves the equation
    Pup(t) + Pdown(t) = p; so for a random p such that 0 <= p <= 1 we find
    the expected time of a price change'''
    mprobtup = partial(nprobw, 1, s0, s, α, η, σ, μ)
    mprobtdown = partial(nprobw, -1, s0, s, α, η, σ, μ)
    def groot(t):
        return np.round(mprobtup(t) + mprobtdown(t) - p, decimals=6)
    sol = optimize.root_scalar(groot, bracket=[0, 1], method='brentq')
    return sol.root

# Define t_from_pud (for inverse CDF)
def t_from_pud(s0, s, α, η, σ, μ, ud, p):
    '''t_from_pud(p, s0, s, α, η, σ, μ) solves the equation
    Pup(t) / Pup(1) (or Pdown(t) / Pdown(1))  = p;
    so for a random p such that 0 <= p <= 1 we find
    the expected time of a price change for the sign given'''
    mprobt = partial(nprobw, ud, s0, s, α, η, σ, μ)
    mprobtT = nprobw(ud, s0, s, α, η, σ, μ, 1)
    if mprobtT == 0:
        print('mprobtT == 0')
        print((s0, s, α, η, σ, μ, ud, p))
    def groot(t):
        return np.round(mprobt(t)/mprobtT - p, decimals=6)
    sol = optimize.root_scalar(groot, bracket=[0, 1], method='brentq')
    return sol.root

# Define CDF
def cdf(s0, s, α, η, σ, μ, npts=100+1):
    grid = np.linspace(0., 0.999, npts)
    pts = pd.Series(grid, index=[t_from_p(s0, s, α, η, σ, μ, p)
                                 for p in grid])
    pts.index.name = 't'
    pts.name = 'CDF(t)'
    return pts

# Define CDFs
def cdfs(s0, s, α, η, σ, μ, npts=100):
    grid = np.linspace(0, 0.999, npts)
    ts = [t_from_p(s0, s, α, η, σ, μ, p) for p in grid]
    up = [nprobw(+1, s0, s, α, η, σ, μ, t) for t in ts]
    up = np.minimum(up, grid)
    pts = pd.DataFrame({'CDF(t)': grid, 'Up': up, 'Down': grid - up},
                       index=ts)
    pts.index.name = 't'
    return pts

# Define Quantiles
def pquant(s, α, η, σ, μ, sup, ud, lg=0., ug=0.999, npts=100+1):
    grid = np.linspace(lg, ug, npts)
    prev_s = s + sup * (0.5 - η) * α
    ts = [t_from_pud(prev_s, s, α, η, σ, μ, ud, p) for p in grid]
    pts = pd.Series(grid, index=ts)
    pts.index.name = 't'
    pts.name = str((sup, ud))
    return pts

# Define Quantiles - inverted
def pquantinv(s, α, η, σ, μ, sup, ud, lg=0., ug=0.999, npts=100+1):
    grid = np.linspace(lg, ug, npts)
    prev_s = s + sup * (0.5 - η) * α
    ts = [t_from_pud(prev_s, s, α, η, σ, μ, ud, p) for p in grid]
    pts = pd.Series(ts, index=grid)
    pts.index.name = 'q'
    pts.name = str((sup, ud))
    return pts

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

# Default quantile definitions
lq = 0.10
uq = 0.90
npts = 9

# Default bounds
ηmin = 0.01
ηmax = 0.5
σmin = 0.0005
σmax = 0.1
μmin = -1.0
μmax = +1.0

# Defaults minimization
popsize=15
tol=1e-4

# Default λ
λreg = 0

# Quantiles durations from time series
def acup_qtl(df, lq=lq, uq=uq, npts=npts):
    df_aup = df[(df['sign'] == +1) & (df['Al'])]['dtj']
    df_cup = df[(df['sign'] == +1) & (df['Co'])]['dtj']
    df_ado = df[(df['sign'] == -1) & (df['Al'])]['dtj']
    df_cdo = df[(df['sign'] == -1) & (df['Co'])]['dtj']
    df_a = df[df['Al']]['dtj']
    df_c = df[df['Co']]['dtj']
    df_a_c = df['dtj'].copy()
    qtl_rng = np.linspace(lq, uq, npts)
    df_aupq = df_aup.quantile(qtl_rng).reset_index().set_index('dtj')
    df_cupq = df_cup.quantile(qtl_rng).reset_index().set_index('dtj')
    df_adoq = df_ado.quantile(qtl_rng).reset_index().set_index('dtj')
    df_cdoq = df_cdo.quantile(qtl_rng).reset_index().set_index('dtj')
    df_aq = df_a.quantile(qtl_rng).reset_index().set_index('dtj')
    df_cq = df_c.quantile(qtl_rng).reset_index().set_index('dtj')
    df_a_cq = df_a_c.quantile(qtl_rng).reset_index().set_index('dtj')
    df_aupq.name = 'p_Al_Up'
    df_cupq.name = 'p_Co_Up'
    df_adoq.name = 'p_Al_Do'
    df_cdoq.name = 'p_Co_Do'
    df_aq.name = 't_Al'
    df_cq.name = 't_Co'
    df_a_cq.name = 't_Al_Co'
    return [df_aupq, df_cupq, df_adoq, df_cdoq,  df_aq, df_cq, df_a_cq]

# Quantiles durations from time series - inverted
def acup_qtlinv(df, lq=lq, uq=uq, npts=npts):
    df_aup = df[(df['sign'] == +1) & (df['Al'])]['dtj']
    df_cup = df[(df['sign'] == +1) & (df['Co'])]['dtj']
    df_ado = df[(df['sign'] == -1) & (df['Al'])]['dtj']
    df_cdo = df[(df['sign'] == -1) & (df['Co'])]['dtj']
    df_a = df[df['Al']]['dtj']
    df_c = df[df['Co']]['dtj']
    df_a_c = df['dtj'].copy()
    qtl_rng = np.linspace(lq, uq, npts)
    df_aupq = df_aup.quantile(qtl_rng)
    df_cupq = df_cup.quantile(qtl_rng)
    df_adoq = df_ado.quantile(qtl_rng)
    df_cdoq = df_cdo.quantile(qtl_rng)
    df_aq = df_a.quantile(qtl_rng)
    df_cq = df_c.quantile(qtl_rng)
    df_a_cq = df_a_c.quantile(qtl_rng)
    df_aupq.name = 't_Al_Up'
    df_cupq.name = 't_Co_Up'
    df_adoq.name = 't_Al_Do'
    df_cdoq.name = 't_Co_Do'
    df_aq.name = 't_Al'
    df_cq.name = 't_Co'
    df_a_cq.name = 't_Al_Co'
    return [df_aupq, df_cupq, df_adoq, df_cdoq, df_aq, df_cq, df_a_cq]

def qq_plots(df_data, s, α, η, σ, μ, hours, lq=lq, uq=uq, npts=npts):
    dfq_data = pd.concat(acup_qtlinv(df_data, lq, uq, npts), axis=1)
    dfq_param = pd.concat([
        pquantinv(s, α, η, σ, μ, +1, +1, lq, uq, npts),
        pquantinv(s, α, η, σ, μ, -1, +1, lq, uq, npts),
        pquantinv(s, α, η, σ, μ, -1, -1, lq, uq, npts),
        pquantinv(s, α, η, σ, μ, +1, -1, lq, uq, npts)], axis=1)*hours*3600
    fig =  plt.figure(figsize=(8,8))
    fig.suptitle('Al_Up')
    plt.plot(dfq_param['(1, 1)'], dfq_data['t_Al_Up'], color = 'r', marker='o')
    plt.plot(dfq_param['(1, 1)'], dfq_param['(1, 1)'], color='b')
    fig =  plt.figure(figsize=(8,8))
    fig.suptitle('Co_Up')
    plt.plot(dfq_param['(-1, 1)'], dfq_data['t_Co_Up'], color = 'r', marker='o')
    plt.plot(dfq_param['(-1, 1)'], dfq_param['(-1, 1)'], color='b')
    fig =  plt.figure(figsize=(8,8))
    fig.suptitle('Al_Down')
    plt.plot(dfq_param['(-1, -1)'], dfq_data['t_Al_Do'], color = 'r', marker='o')
    plt.plot(dfq_param['(-1, -1)'], dfq_param['(-1, -1)'], color='b')
    fig =  plt.figure(figsize=(8,8))
    fig.suptitle('Co_Down')
    plt.plot(dfq_param['(1, -1)'], dfq_data['t_Co_Do'], color = 'r', marker='o')
    plt.plot(dfq_param['(1, -1)'], dfq_param['(1, -1)'], color='b')

def distance_dur(df_data, s, α, η, σ, μ, hours, lq=lq, uq=uq, npts=npts):
    n_aup = df_data[df_data['sign'] == +1]['Al'].sum() / len(df_data)
    n_cup = df_data[df_data['sign'] == +1]['Co'].sum() / len(df_data)
    n_ado = df_data[df_data['sign'] == -1]['Al'].sum() / len(df_data)
    n_cdo = df_data[df_data['sign'] == -1]['Co'].sum() / len(df_data)
    # print((n_cup + n_cdo) / (2 * (n_aup + n_ado)))
    dfq_data = pd.concat(acup_qtlinv(df_data, lq, uq, npts), axis=1)
    dfq_param = pd.concat([
        pquantinv(s, α, η, σ, μ, +1, +1, lq, uq, npts),
        pquantinv(s, α, η, σ, μ, -1, +1, lq, uq, npts),
        pquantinv(s, α, η, σ, μ, -1, -1, lq, uq, npts),
        pquantinv(s, α, η, σ, μ, +1, -1, lq, uq, npts)], axis=1)\
            * hours * 3600
    dist_Al_Up = np.linalg.norm(dfq_param['(1, 1)'] - dfq_data['t_Al_Up'])
    dist_Co_Up = np.linalg.norm(dfq_param['(-1, 1)'] - dfq_data['t_Co_Up'])
    dist_Al_Do = np.linalg.norm(dfq_param['(-1, -1)'] - dfq_data['t_Al_Do'])
    dist_Co_Do = np.linalg.norm(dfq_param['(1, -1)'] - dfq_data['t_Co_Do'])
    # print(np.array([n_aup, dist_Al_Up, n_cup, dist_Co_Up,
    #     n_ado, dist_Al_Do, n_cdo, dist_Co_Do]))
    return np.dot([n_aup, n_cup, n_ado, n_cdo],
        [dist_Al_Up, dist_Co_Up, dist_Al_Do, dist_Co_Do])

# def minde_dist(df_data, s, α, hours, λ=λreg, popsize=popsize, tol=tol, print_res=False,
#     ηmin=ηmin, ηmax=ηmax, σmin=σmin, σmax=σmax, μmin=μmin, μmax=μmax,
#     lq=lq, uq=uq, npts=npts):
#     n_aup = df_data[df_data['sign'] == +1]['Al'].sum() / len(df_data)
#     n_cup = df_data[df_data['sign'] == +1]['Co'].sum() / len(df_data)
#     n_ado = df_data[df_data['sign'] == -1]['Al'].sum() / len(df_data)
#     n_cdo = df_data[df_data['sign'] == -1]['Co'].sum() / len(df_data)
#     h = (n_cup + n_cdo) / (2 * (n_aup + n_ado))
#     pmax = df_data['Ptj'].max()
#     pmin = df_data['Ptj'].min()
#     tmax = df_data['Ptj'].idxmax() / (hours * 3600)
#     tmin = df_data['Ptj'].idxmin() / (hours * 3600)
#     μm = ((pmax - pmin) / s) / (tmax - tmin)
#     μmin0 = max(μmin, -np.abs(μm))
#     μmax0 = min(μmax, +np.abs(μm))
#     rvp = rlzvollog(df_data['Ptj']) * np.sqrt((hours * 3600) / df_data['dtj'].sum())
#     rvxe = rvp * np.sqrt(2 * h)
#     dfq_data = pd.concat(acup_qtlinv(df_data, lq, uq, npts), axis=1)
#     def sum_dist(x): #  x = [η, σ, μ]
#         dfq_param = pd.concat([
#             pquantinv(s, α, x[0], x[1], x[2], +1, +1, lq, uq, npts),
#             pquantinv(s, α, x[0], x[1], x[2], -1, +1, lq, uq, npts),
#             pquantinv(s, α, x[0], x[1], x[2], -1, -1, lq, uq, npts),
#             pquantinv(s, α, x[0], x[1], x[2], +1, -1, lq, uq, npts)], axis=1)\
#                 * hours * 3600
#         dist_Al_Up = np.linalg.norm(dfq_param['(1, 1)'] - dfq_data['t_Al_Up'])
#         dist_Co_Up = np.linalg.norm(dfq_param['(-1, 1)'] - dfq_data['t_Co_Up'])
#         dist_Al_Do = np.linalg.norm(dfq_param['(-1, -1)'] - dfq_data['t_Al_Do'])
#         dist_Co_Do = np.linalg.norm(dfq_param['(1, -1)'] - dfq_data['t_Co_Do'])
#         return np.dot([n_aup, n_cup, n_ado, n_cdo],
#             [dist_Al_Up, dist_Co_Up, dist_Al_Do, dist_Co_Do]) + λ * abs(x[2])
#     bounds = [(ηmin, ηmax), (σmin, σmax), (μmin0, μmax0)]
#     resg = optimize.differential_evolution(sum_dist, bounds, popsize=popsize, tol=tol)
#     if print_res:
#         print(resg)
#     return [h, resg.x[0], resg.x[1], rvxe, rvp, resg.x[2], μm, resg.fun]

def multfit_dist(df_data, s, α, hours, index_m, λ=λreg, popsize=popsize, tol=tol,
    ηmin=ηmin, ηmax=ηmax, σmin=σmin, σmax=σmax, μmin=μmin, μmax=μmax,
    lq=lq, uq=uq, npts=npts):
    n_Al_Co = df_data['Al'].sum() + df_data['Co'].sum()
    n_aup = df_data[df_data['sign'] == +1]['Al'].sum() / n_Al_Co
    n_cup = df_data[df_data['sign'] == +1]['Co'].sum() / n_Al_Co
    n_ado = df_data[df_data['sign'] == -1]['Al'].sum() / n_Al_Co
    n_cdo = df_data[df_data['sign'] == -1]['Co'].sum() / n_Al_Co
    n_a = df_data['Al'].sum() / n_Al_Co
    n_c = df_data['Co'].sum() / n_Al_Co
    h = n_c / (2 * n_a)
    pmax = df_data['Ptj'].max()
    pmin = df_data['Ptj'].min()
    tmax = df_data['Ptj'].idxmax() / (hours * 3600)
    tmin = df_data['Ptj'].idxmin() / (hours * 3600)
    μm = ((pmax - pmin) / s) / (tmax - tmin)
    μmin0 = max(μmin, -np.abs(μm))
    μmax0 = min(μmax, +np.abs(μm))
    bounds = [(ηmin, ηmax), (σmin, σmax), (μmin0, μmax0)]
    rvp = rlzvollog(df_data['Ptj']) * np.sqrt((hours * 3600) / df_data['dtj'].sum())
    rvxe = rvp * np.sqrt(2 * h)
    dfq_data = pd.concat(acup_qtlinv(df_data, lq, uq, npts), axis=1)
    def dist_Al_Up(x): #  x = [η, σ, μ]
        P_Al_Up = nprobw(+1, s + (0.5 - x[0]) * α, s, α, x[0], x[1], x[2], 1)
        P_Al_Do = nprobw(-1, s - (0.5 - x[0]) * α, s, α, x[0], x[1], x[2], 1)
        weight = (P_Al_Up * P_Al_Do) / (P_Al_Up + P_Al_Do)
        dfq_param = pquantinv(s, α, x[0], x[1], x[2], +1, +1, lq, uq, npts)\
            * hours * 3600
        return np.linalg.norm(weight * dfq_param - n_aup * dfq_data['t_Al_Up'])\
            + λ * abs(x[2])
    def dist_Co_Up(x):
        P_Al_Up = nprobw(+1, s + (0.5 - x[0]) * α, s, α, x[0], x[1], x[2], 1)
        P_Al_Do = nprobw(-1, s - (0.5 - x[0]) * α, s, α, x[0], x[1], x[2], 1)
        weight = (P_Al_Up * (1 - P_Al_Do)) / (P_Al_Up + P_Al_Do)
        dfq_param = pquantinv(s, α, x[0], x[1], x[2], -1, +1, lq, uq, npts)\
            * hours * 3600
        return np.linalg.norm(weight * dfq_param - n_cup * dfq_data['t_Co_Up'])\
            + λ * abs(x[2])
    def dist_Al_Do(x):
        P_Al_Up = nprobw(+1, s + (0.5 - x[0]) * α, s, α, x[0], x[1], x[2], 1)
        P_Al_Do = nprobw(-1, s - (0.5 - x[0]) * α, s, α, x[0], x[1], x[2], 1)
        weight = (P_Al_Up * P_Al_Do) / (P_Al_Up + P_Al_Do)
        dfq_param = pquantinv(s, α, x[0], x[1], x[2], -1, -1, lq, uq, npts)\
            * hours * 3600
        return np.linalg.norm(weight * dfq_param - n_ado * dfq_data['t_Al_Do'])\
            + λ * abs(x[2])
    def dist_Co_Do(x):
        P_Al_Up = nprobw(+1, s + (0.5 - x[0]) * α, s, α, x[0], x[1], x[2], 1)
        P_Al_Do = nprobw(-1, s - (0.5 - x[0]) * α, s, α, x[0], x[1], x[2], 1)
        weight = ((1 - P_Al_Up) * P_Al_Do) / (P_Al_Up + P_Al_Do)
        dfq_param = pquantinv(s, α, x[0], x[1], x[2], +1, -1, lq, uq, npts)\
            * hours * 3600
        return np.linalg.norm(weight * dfq_param - n_cdo * dfq_data['t_Co_Do'])\
            + λ * abs(x[2])
    def dist_Al(x): #  x = [η, σ, μ]
        P_Al_Up = nprobw(+1, s + (0.5 - x[0]) * α, s, α, x[0], x[1], x[2], 1)
        P_Al_Do = nprobw(-1, s - (0.5 - x[0]) * α, s, α, x[0], x[1], x[2], 1)
        weight_Up = (P_Al_Up * P_Al_Do) / (P_Al_Up + P_Al_Do)
        weight_Do = (P_Al_Up * P_Al_Do) / (P_Al_Up + P_Al_Do)
        dfq_param_Up = weight_Up * pquantinv(s, α, x[0], x[1], x[2], +1, +1, lq, uq, npts)\
            * hours * 3600
        dfq_param_Do = weight_Do * pquantinv(s, α, x[0], x[1], x[2], -1, -1, lq, uq, npts)\
            * hours * 3600
        return np.linalg.norm(dfq_param_Up - n_aup * dfq_data['t_Al_Up']) +\
            np.linalg.norm(dfq_param_Do - n_ado * dfq_data['t_Al_Do']) + λ * abs(x[2])
    def dist_Co(x): #  x = [η, σ, μ]
        P_Al_Up = nprobw(+1, s + (0.5 - x[0]) * α, s, α, x[0], x[1], x[2], 1)
        P_Al_Do = nprobw(-1, s - (0.5 - x[0]) * α, s, α, x[0], x[1], x[2], 1)
        weight_Up = (P_Al_Up * (1 - P_Al_Do)) / (P_Al_Up + P_Al_Do)
        weight_Do = ((1 - P_Al_Up) * P_Al_Do) / (P_Al_Up + P_Al_Do)
        dfq_param_Up = weight_Up * pquantinv(s, α, x[0], x[1], x[2], -1, +1, lq, uq, npts)\
            * hours * 3600
        dfq_param_Do = weight_Do * pquantinv(s, α, x[0], x[1], x[2], +1, -1, lq, uq, npts)\
            * hours * 3600
        return np.linalg.norm(dfq_param_Up - n_cup * dfq_data['t_Co_Up']) +\
            np.linalg.norm(dfq_param_Do - n_cdo * dfq_data['t_Co_Do']) + λ * abs(x[2])
    def dist_Al_Co(x): #  x = [η, σ, μ]
        P_Al_Up = nprobw(+1, s + (0.5 - x[0]) * α, s, α, x[0], x[1], x[2], 1)
        P_Al_Do = nprobw(-1, s - (0.5 - x[0]) * α, s, α, x[0], x[1], x[2], 1)
        weight_Al_Up = (P_Al_Up * P_Al_Do) / (P_Al_Up + P_Al_Do)
        weight_Co_Up = (P_Al_Up * (1 - P_Al_Do)) / (P_Al_Up + P_Al_Do)
        weight_Al_Do = (P_Al_Up * P_Al_Do) / (P_Al_Up + P_Al_Do)
        weight_Co_Do = ((1 - P_Al_Up) * P_Al_Do) / (P_Al_Up + P_Al_Do)
        dfq_param_Al_Up = weight_Al_Up * pquantinv(s, α, x[0], x[1], x[2], +1, +1, lq, uq, npts)\
            * hours * 3600
        dfq_param_Al_Do = weight_Al_Do * pquantinv(s, α, x[0], x[1], x[2], -1, -1, lq, uq, npts)\
            * hours * 3600
        dfq_param_Co_Up = weight_Co_Up * pquantinv(s, α, x[0], x[1], x[2], -1, +1, lq, uq, npts)\
            * hours * 3600
        dfq_param_Co_Do = weight_Co_Do * pquantinv(s, α, x[0], x[1], x[2], +1, -1, lq, uq, npts)\
            * hours * 3600
        return np.linalg.norm(dfq_param_Al_Up - n_aup * dfq_data['t_Al_Up']) +\
            np.linalg.norm(dfq_param_Al_Do - n_ado * dfq_data['t_Al_Do']) + \
            np.linalg.norm(dfq_param_Co_Up - n_cup * dfq_data['t_Co_Up']) +\
            np.linalg.norm(dfq_param_Co_Do - n_cdo * dfq_data['t_Co_Do']) + λ * abs(x[2])
    fit_Al_Up = optimize.differential_evolution(dist_Al_Up, bounds, popsize=popsize, tol=tol)
    fit_Co_Up = optimize.differential_evolution(dist_Co_Up, bounds, popsize=popsize, tol=tol)
    fit_Al_Do = optimize.differential_evolution(dist_Al_Do, bounds, popsize=popsize, tol=tol)
    fit_Co_Do = optimize.differential_evolution(dist_Co_Do, bounds, popsize=popsize, tol=tol)
    fit_Al = optimize.differential_evolution(dist_Al, bounds, popsize=popsize, tol=tol)
    fit_Co = optimize.differential_evolution(dist_Co, bounds, popsize=popsize, tol=tol)
    fit_Al_Co = optimize.differential_evolution(dist_Al_Co, bounds, popsize=popsize, tol=tol)
    ans = [n_aup, n_cup, n_ado, n_cdo, n_a, n_c, fit_Al_Co.fun,
        h, fit_Al_Up.x[0], fit_Co_Up.x[0], fit_Al_Do.x[0], fit_Co_Do.x[0],
        fit_Al.x[0], fit_Co.x[0], fit_Al_Co.x[0],
        rvp, rvxe, fit_Al_Up.x[1], fit_Co_Up.x[1], fit_Al_Do.x[1], fit_Co_Do.x[1],
        fit_Al.x[1], fit_Co.x[1], fit_Al_Co.x[1],
        μm, fit_Al_Up.x[2], fit_Co_Up.x[2], fit_Al_Do.x[2], fit_Co_Do.x[2],
        fit_Al.x[2], fit_Co.x[2], fit_Al_Co.x[2]]
    ans_cols = ['Al_Up', 'Co_Up', 'Al_Do', 'Co_Do', 'Al', 'Co', 'Al_Co_dist',
                'H', 'Al_Up_η', 'Co_Up_η', 'Al_Do_η', 'Co_Do_η',
                'Al_η', 'Co_η', 'Al_Co_η',
                'σP', 'σXe', 'Al_Up_σ', 'Co_Up_σ', 'Al_Do_σ', 'Co_Do_σ',
                'Al_σ', 'Co_σ', 'Al_Co_σ',
                'μmax', 'Al_Up_μ', 'Co_Up_μ', 'Al_Do_μ', 'Co_Do_μ',
                'Al_μ', 'Co_μ',  'Al_Co_μ']
    return pd.DataFrame([ans], columns=ans_cols, index=[index_m])

# Cumulative H
def cum_H(df):
    dfc = df.copy()
    dfc['Cum_Al'] = dfc['Al'].cumsum()
    dfc['Cum_Co'] = dfc['Co'].cumsum()
    dfc['H'] = dfc['Cum_Co'] / (2 * dfc['Cum_Al'])
    return dfc['H']