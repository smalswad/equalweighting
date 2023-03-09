# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 09:47:25 2023

@author: Alexander Swade
"""

import pandas as pd
import numpy as np

from quantfinlib.portfolio.measure import calc_perc

def run_analysis(data, configs):
    
    # Unpack configs
    start = configs['start']
    end = configs['end']    
    factors = data['factors']
    ret = data['ret']
    mcap = data['mcap']
    consti = data['constituents']
    sp500 = data['sp500']
    smbsp = data['smbsp']
    size_quintiles = data['size_q']
    
    # 1) Construct EW and VW portfolios (based on full universe)
    mcap_weights = calc_perc(mcap).shift(1)
    vw_ret = (ret*mcap_weights).sum(axis=1)
    ew_ret = ret.mean(axis=1)
    
    # # Returns for SP500 stocks
    sp500_vw, sp500_ew = constr_sp500_portf(ret, mcap, consti)  
    
    # 2) Construct regression data
    reg_dat = factors.join(sp500[['vwretd','ewretd']])
    reg_dat.rename(columns={'Mkt-RF':'MKT'}, inplace=True)
    reg_dat = reg_dat.loc[(reg_dat.index >= start) & (reg_dat.index <= end), :]
    reg_dat = reg_dat.join(pd.concat([vw_ret.rename('univ_vw'),
                                      ew_ret.rename('univ_ew'),
                                      sp500_vw,
                                      sp500_ew], axis=1))
    
    # Get actual market return (no excess return)
    reg_dat['MKT_noexc'] = reg_dat['MKT'] + reg_dat['RF']
    
    # Calculate performance differences between EW and VW portfolios
    reg_dat['EW_VW'] = reg_dat['ewretd'].astype(float) \
                            - reg_dat['vwretd'].astype(float)
    reg_dat['EW_VW_FU'] = reg_dat['univ_ew'].astype(float) \
                            - reg_dat['univ_vw'].astype(float)
    reg_dat['EW_VW_FU_RF'] = reg_dat['EW_VW_FU'] + reg_dat['RF']
    reg_dat['EW_VW_RF'] = reg_dat['EW_VW'] + reg_dat['RF']
    
    # Add SMBSP as well
    reg_dat = reg_dat.join(smbsp)
    
    # Add MKT_(-1) 
    reg_dat['MKT_lagged'] = reg_dat['MKT'].shift()

    # 3) Calculate portfolios with alternative rebalancing freq 
        
    ## Run calculations with different rebalancing frequencies
    sample_ret = ret.loc[(ret.index >= start) & (ret.index <= end), :]
    sample_mcap = mcap.loc[(mcap.index >= start) & (mcap.index <= end), :]

    rebal_portf, turnover = \
        calc_alt_rebal_portf(ret=sample_ret,
                             mcap=sample_mcap,
                             vw=reg_dat['univ_vw'],
                             freqs=configs['rebal_freq'],
                             rebal_fac=configs['rebal_factors'])    
    reg_dat = reg_dat.join(rebal_portf)

    # 4) Add quintile portfolios sorted by size (annual sorting, monthly 
    #    rebalancing)
    reg_dat = reg_dat.join(size_quintiles)
    
    # Clean spaces in column names
    reg_dat.columns = reg_dat.columns.str.replace(' ', '')
    
    
    # 5) Check for seasonality
    reg_dat['jan_dummy'] = np.where(reg_dat.index.month==1, 1, 0)
    reg_dat['nonjan_dummy'] = np.where(reg_dat.index.month==1, 0, 1)
    
    # 6) Calcualte cumulative performances
    sp_cum_ret = (reg_dat[['vwretd','ewretd','univ_vw','univ_ew']
                          ].astype('float') + 1).cumprod()
    
    # 7) Get descriptive data
    descr_data = get_descr_data(ret=reg_dat[['univ_vw','univ_ew','EW_VW_FU','SMB',
                                            'vwretd','ewretd','EW_VW','SMBSP']],
                                mcap=mcap,
                                sp500_val=sp500['usdval'],
                                ret_mkt=reg_dat[['RF','MKT_noexc']])
    
    return {'reg_dat': reg_dat,
            'cum_ret': sp_cum_ret,
            'descr_data': descr_data,
            'turnover': turnover}

def calc_alt_rebal_portf(ret, mcap, vw, freqs, rebal_fac): 
    '''
    Calculate EW-VW portfolio returns based on different rebalancing freqencies

    Parameters
    ----------
    ret : pd.DataFrame, shape(T,n)
        Return data for all CRSP stocks.
    mcap : od.DataFrame, shape(T,n)
        Market capitalization data for all cRSP stocks.
    vw : pd.Series, shape(T,)
        Value weighted return of all stocks.
    freq : list of strings
        Frequency names of rebalancing.
    rebal_fac : list of floats
        Factor for rebalancing.

    Returns
    -------
    ew_returns : pd.DataFrame, shape(T,n)
        EW returns based on different rebalancing frequencies.

    '''
    ew_ret_by_freq = list()
    turnover = list()
    for i, rebal in enumerate(freqs):
        rebal_freq = pd.Grouper(freq=rebal) 
        ew_weights = ret.groupby(rebal_freq).apply(_get_weights_grouper_func,
                                             mcap=mcap)
        ew_portf = ret.multiply(ew_weights.shift(1)).sum(axis=1)
        ew_ret_by_freq.append(ew_portf)
        if rebal == '1MS':
            sret = ret + 1
            wret = ew_weights.shift(1)*sret
            ewbh = wret.divide(wret.sum(axis=1),axis=0)
            w_diff = ewbh.fillna(0) - ew_weights.fillna(0)
            turnover.append(w_diff.abs().sum(axis=1).mean() * rebal_fac[i])
        else:     
            print(f'calculating {rebal}')
            turnover.append(calc_grouped_turnover(ew_weights, rebal_freq,
                                                  rebal_fac[i]))
    
    ew_returns = pd.concat(ew_ret_by_freq, axis=1, keys=freqs).subtract(
        vw, axis=0)
    ew_returns.columns = 'EW_VW_' + ew_returns.columns
    
    return ew_returns, turnover

def calc_drifting_weights(returns, init_weights):
    '''
    Calculate drifting weights for given return series. 

    Parameters
    ----------
    returns : pd.DataFrame, shape(T,N)
        Individual stock returns.
    init_weights : pd.Series, shape(N,)
        Initial portfolio weights.

    Returns
    -------
    weights : pd.DataFrame, shape(T,N)
        Drifting portfolio weights.

    '''
    returns.iloc[0] = 0
    cum_ret = (returns + 1).cumprod()
    weights = cum_ret.multiply(init_weights)
    weights = weights.divide(weights.sum(axis=1),axis=0)
    
    return weights

def calc_grouped_turnover(weights, grouper, factor):
    '''
    Helper function to calculate annualised turnover for given period.

    Parameters
    ----------
    weights : pd.DataFrame, shape(T,n)
        Individual stock portfolio weights.
    grouper : pd.Grouper object
        Grouper object to group weights with.
    factor : float
        Factor for rebalancing.

    Returns
    -------
    float
        Annualised turnover for the given period.

    '''
    starting_weights = weights.groupby(grouper).first()
    end_weights = weights.groupby(grouper).last()
    # Calculate differences between end of period weights and new weights 
    # for the following period
    weights_diff = starting_weights.fillna(0).shift() - end_weights.fillna(0)
    # Account for initial allocation in the first period
    weights_diff.iloc[0,:] = starting_weights.iloc[0,:]
    
    return weights_diff.abs().sum(axis=1).mean() * factor

def constr_sp500_portf(ret, mcap, consti):
    '''
    Construct SP500 portfolios based on constituents over time.

    Parameters
    ----------
    ret : pd.DataFrame, shape(T,n)
        Return data for all CRSP stocks.
    mcap : od.DataFrame, shape(T,n)
        Market capitalization data for all cRSP stocks.
    consti : pd.DataFrame, shape(k,4)
        Constituent information based on PERMNOs with start and end date in 
        SP500.

    Returns
    -------
    sp500_vw : pd.Series, shape(T,)
        SP500 VW return.
    sp500_ew : pd.Series, shape(T,)
        SP500 EW return.

    '''
    # Returns for SP500 stocks
    retSP500 = ret.loc[:, consti['permno'].unique()].copy()
    mcapSP500 = mcap.loc[:, consti['permno'].unique()].copy()
    
    # Create masks for constituents based on unique permno
    for p in consti['permno'].unique():
        p_info = consti[consti['permno'] == p]
        idx_ind = pd.Series(data=False, index=retSP500.index)
        
        for i in p_info.index:
            idx_ind.loc[(idx_ind.index >= p_info.loc[i, 'start']) &
                              (idx_ind.index <= p_info.loc[i, 'ending'])]\
                = True
    
        retSP500.loc[:, p].where(idx_ind, inplace=True)
        mcapSP500.loc[:, p].where(idx_ind, inplace=True)
    
    # Construct VW and EW SP500 portfolios 
    sp500_vw_weights = calc_perc(mcapSP500).shift(1)
    sp500_vw = (retSP500*sp500_vw_weights).sum(axis=1)
    sp500_ew = retSP500.mean(axis=1)
    
    return (sp500_vw.rename('sp500_vw'), sp500_ew.rename('sp500_ew'))

def get_descr_data(ret, mcap, sp500_val, ret_mkt):
    '''
    Function to generate relevant data to calculate some descriptive statistics
    (for exhibit 3).

    Parameters
    ----------
    ret : pd.DataFrame, shape(T,k)
        Return data for relevant time-series.
    mcap : pd.DataFrame, shape(T,n)
        Market capitalization for all individual stocks in CRSP.
    sp500_val : pd.Series, shape(T,)
        SP500 total market value.
    ret_mkt : pd.DataFrame, shape(T,2)
        Risk-free rate and market return (no excess).

    Returns
    -------
    descrip_data : pd.DataFrame with Multiindex
        Descriptive data used to calculate some descriptives.

    '''
    # Construct multi-index dataframe for descriptives
    descrip_lvl = 100*(ret+1).cumprod()
    descrip_mcap = pd.concat([mcap.sum(axis=1).rename('mcap'), sp500_val]*4,
                              axis=1).loc[ret.index].sort_index(axis=1)
    descrip_mcap.columns = ret.columns
    descrip_count = pd.concat([mcap.count(axis=1).rename('cnt'), sp500_val]*4,
                              axis=1).loc[ret.index].sort_index(axis=1)
    descrip_count.columns = descrip_mcap.columns
    
    descrip_data = pd.concat([ret,descrip_lvl, descrip_mcap, descrip_count],
                              axis=1, keys=['ret','level','mcap','count'],
                              names=['type','portf'])
    
    # Add MKT and RF
    new_midx = pd.MultiIndex.from_arrays([['ret', 'ret'], ['RF', 'MKT']])
    descrip_data = descrip_data.join(ret_mkt.set_axis(new_midx, axis=1))
    
    return descrip_data

def _get_init_weights(group_mcap, incl_const, method='EW'):
    '''
    Helper function to get inital constituent weights

    Parameters
    ----------
    group_mcap : pd.DataFrame, shape(T,N+X)
        Market capitalization time series.
    incl_const : pd.index, length N
        Column names of included constituents.
    method : str, optional
        Method to calculate inital weights with. The default is 'EW'.

    Returns
    -------
    init_weights : pd.Series, shape(N,)
        Portfolio weights of included constituents.

    '''
    # Catch method selection error
    methods = ['EW','VW']
    if method not in methods:
        raise ValueError("Invalid method type. Expected one of: %s" % methods)

    elif method == 'EW':
        count = len(incl_const)
        init_weights = pd.Series(1/count, index=incl_const)
    
    else:
        init_weights = calc_perc(group_mcap.loc[
            group_mcap.index[0], incl_const].to_frame().T).squeeze()

    return init_weights

def _get_weights_grouper_func(grp_ret, mcap):
    '''
    Helper function used to calculate weights for different rebalancing freq

    Parameters
    ----------
    grp_ret : pd.DataFrame, shape(tau,n)
        Stock returns for given period, 0 <= tau <= T.
    mcap : pd.DataFrame, shape(T,n)
        Stock mcap for whole period.

    Returns
    -------
    weights : pd.DataFrame, shape(tau,n)
        Individual stock portfolio weights.

    '''
    grp_mcap = mcap.loc[grp_ret.index]
    incl_const = identify_constituents(grp_ret)
    init_weights = _get_init_weights(grp_mcap, incl_const, method='EW')
    weights = calc_drifting_weights(grp_ret.loc[:,incl_const], init_weights)
    
    return weights

def identify_constituents(const_ret):
    '''
    Helper function to identify constituents which all have returns for the 
    given sample.

    Parameters
    ----------
    const_ret : pd.DataFrame. shape(T,n)
        Returns of constituents.

    Returns
    -------
    pd.index
        Column names of relevent constituents.

    '''
    return const_ret.dropna(how='any', axis=1).columns