# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 12:09:50 2021

@author: Alexander Swade
"""

# =============================================================================
# IMPORT SECTION
# =============================================================================

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from matplotlib import dates
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.nonparametric.smoothers_lowess import lowess
from sklearn.metrics import mean_squared_error
from pandas.tseries.offsets import MonthEnd
from math import sqrt
from tabulate import tabulate
from TimeSeriesFunctions import *
#import pyreadr
import pathlib
import itertools
import math

import MyClasses as mc
import utility_functions as utils


# =============================================================================
# FUNCTIONS
# =============================================================================

def get_perc(grp_obj):
    return grp_obj.divide(grp_obj.sum(axis=1), axis=0)

def calc_hhi(grp_obj):
    return (grp_obj**2).sum(axis=1)

def calc_descrip_stats(grp_obj, lvl='portf', freq='d', inclCAPM=True,
                       diff_dict=None):
    '''
    Parameters
    ----------
    grp_obj : df
        Multi-indexed dataframe containing all relevant data column-wise .
    lvl : string, optional
        Multi-index level to loop over. The default is 'portf'.
    freq : string, optional
        Indicator for data frequency ('d' or 'm'). The default is 'd'.
    inclCAPM : bool, optional
        Indicate if CAPM estimation should be included. The default is True.
    diff_dict : dict, optional
        Mapping dict for names of additional series to be included
        (mean + t-stat). The default is None. 

    Returns
    -------
    descrip_df : df
        Df containing all descriptive stats row-wise.

    '''
    print(f'Interval between {grp_obj.index.min()} and {grp_obj.index.max()}')
    
    # Get scaling frequency
    if freq=='d':
        scalar= 250
    elif freq=='m':
        scalar=12
    
    colnames = ['Ret','Std','Sharpe','maxDD_1M','AvgMcap','AvgConst'] 
    drop_names = ['MKT','RF']
    
    if diff_dict is not None:
        colnames = colnames + ['diff_ret', 'diff_tstat']
        drop_names = drop_names + list(set(diff_dict.values()))
        
    if (inclCAPM):
        colnames = colnames + ['alpha','alpha_t', 'beta', 'beta_t', 'rsqr']
    
    stat_list = list()
    mean_rf = grp_obj.xs(('ret','RF'), axis=1).mean()
    
    lvl_names = grp_obj.columns.unique(level=lvl).drop(drop_names, errors='ignore')
    
    for p in lvl_names:
        mean_ret = grp_obj.xs(('ret',p), axis=1).mean()
        std = grp_obj.xs(('ret',p), axis=1).std()        
        stats = [mean_ret*scalar*100,
                 std*np.sqrt(scalar)*100,
                 (mean_ret - mean_rf) / std,                                  
                 # grp_obj.xs(('level',p), axis=1).to_frame().apply(
                 #     mc.mDD, window=scalar)[0]*100,
                 grp_obj.xs(('ret',p), axis=1).min()*100,
                 grp_obj.xs(('mcap',p), axis=1).divide(
                     grp_obj.xs(('count',p), axis=1)).mean()/10**6,
                 round(grp_obj.xs(('count',p), axis=1).mean())]
        
        if diff_dict is not None:
            diff_ret = grp_obj.xs(('ret',diff_dict.get(p)), axis=1)
            diff_mean_ret = diff_ret.mean()
            stats = stats + [diff_mean_ret*100,
                             mc.tstat(diff_ret, diff_mean_ret, 0)]
            
        
        if (inclCAPM):            
            # Estimate Single Index model and add results to stats
            capm_data = grp_obj.loc[:, [('ret','RF'),('ret','MKT'), ('ret', p)]]
            alpha, alpha_t, beta, beta_t, rsqr, _, _, _, _ = \
                estimate_CAPM(capm_data.droplevel(0, axis=1), 'RF','MKT',p)
            stats = stats +  [alpha, alpha_t, beta, beta_t, rsqr*100]
            
        stat_list.append(stats)
    
    descrip_df = pd.DataFrame(stat_list, index=lvl_names, columns=colnames)
            
    return descrip_df

def calc_ols(data, formula, scale_alpha=100, orientation='horizontal'):
    '''
    Helper function to calculate ols results and save as pd.DataFrame

    Parameters
    ----------
    data : pd.DataFrame, shape(T,N)
        Regression data.
    formula : str
        OLS regression formula.
    scale_alpha : int, optional
        Scaling factor for alpha values. The default is 100.
    orientation : str, optional
        Define output orientation, i.e. params and tvalues horizontal or
        vertical. The default is horizontal. 

    Returns
    -------
    results : pd.DataFrame
        Regression coefficients, t-values and R².

    '''
     
    model = sm.formula.ols(formula=formula, data=data)
    model_results = model.fit()
    
    if orientation == 'horizontal':
        results = pd.DataFrame([model_results.params, model_results.tvalues],
                               index = ['param','tval'])
        results.loc['param', 'Intercept'] = \
            results.loc['param', 'Intercept'] * scale_alpha
        results.loc['param','adj.R2'] = model_results.rsquared_adj
    
    else:
        idx = pd.MultiIndex.from_product(
            [model_results.params.index, ['param','tval']],
            names=['variable', 'type'])
        values = utils.combine_alternating(model_results.params,
                                           model_results.tvalues)
        results = pd.DataFrame(values, index=idx)
        results.loc[('Intercept','param'), :] = \
            results.loc[('Intercept','param'), :] *scale_alpha
        results.loc[('adj.R2','param'), :] = model_results.rsquared_adj
        
    return results

def calc_portf_chars(portf, breaks, pnames, rf=None, freq=12, cvar=0.05, 
                     scale=True):
    '''
    Helper function to calculate various portfolio characteristis

    Parameters
    ----------
    portf : pd.DataFrame, shape(T,N)
        Dataframe containing N portfolio returns for T periods.
    breaks : list 
        List of date tuples functioning as breakpoints for different periods.
    pnames : list
        Period names.
    rf : pd.Series, shape(T,)
        Specify risk-free rate. If None, excess returns are assumed. 
        The default is None.
    freq : int, optional
        Data frequency scaling parameter. The default is 12.
    cvar : float, optional
        Confidence level of CVAR estimate. The default is 0.05.
    scale : boolean, optinal
        Indicate whether returns and std shall be scaled by 100, i.e. represent
        percentage points. The default is True.

    Returns
    -------
    df : pd.DataFrame with Multiindex
        Portfolio characteristics for different time-periods.

    '''
    if rf is None:
        rf = pd.Series(0, index= portf.index)
    
    if scale:
        scaling = 100
    else:
        scaling = 1
    
    #Create multi-indexed dataframe
    pn = portf.columns
    metric = ['Return p.a.', 'Std p.a.', 'Sharpe', #'MaxDD 1M',
               'MaxDD full period', 'Calmar ratio', 'Sortino ratio',
               'Expected shortfall']
    idx = pd.MultiIndex.from_product([pnames, pn],
                                     names=['Period', 'Portf'])
    df = pd.DataFrame(np.nan, idx, metric)
    
    for i, dat in enumerate(breaks):
        
        start, end  = dat
        ret_p = portf.loc[start:end]
        rf_p = rf.loc[start:end]
        
        #Returns
        df.loc[(pnames[i], pn), 'Return p.a.'] = \
            freq*ret_p.mean(axis=0).values * scaling
        
        #Volatility
        df.loc[(pnames[i], pn), 'Std p.a.'] = \
            np.sqrt(freq)*ret_p.std(axis=0).values * scaling
        
        #Sharpe ratio
        df.loc[(pnames[i], pn), 'Sharpe'] = ((ret_p.mean(axis=0)-rf_p.mean()) \
            / ret_p.std() * np.sqrt(freq)).values
            
        #Max drawdown
        #df.loc[(pnames[i], pn), 'MaxDD 1M'] = ret_p.min().values * scaling
        df.loc[(pnames[i], pn), 'MaxDD full period'] = \
            utils.max_drawdown(ret_p).values * scaling
        
        #Calmar ratio
        df.loc[(pnames[i], pn), 'Calmar ratio'] = \
            utils.calmar_ratio(ret_p, freq).values
        
        #Sortino ratio
        df.loc[(pnames[i], pn), 'Sortino ratio'] = \
            utils.sortino_ratio(ret_p, freq).values
                
        #Expected shortfall
        df.loc[(pnames[i], pn), 'Expected shortfall'] = \
            utils.expected_shortfall(ret_p, cvar).values * scaling
        
    return df

def estimate_CAPM(grp_obj, rf, ret_m, ret_stock, alpha_inc=True, hyp='x1=1',
                  scale_alpha=100):
    '''
    grp_obj     -   pandas group/df object
    rf          -   Colname for risk free rate
    ret_m       -   Colname for market return
    ret_stock   -   Colname(s) for stock return(s)
    '''
    grp_obj = grp_obj.rename(columns={rf:'x1'}, inplace=False)
    capm = mc.CAPM(grp_obj['x1'],
                   grp_obj[ret_m],
                   grp_obj[ret_stock])
    capm.evaluate(alpha_inc=alpha_inc, hyp=hyp)
    return [capm.alpha*scale_alpha, capm.alpha_tvalue, capm.beta,
            capm.beta_tvalue, capm.rsqr, capm.conf_int.loc['x1',0],
            capm.conf_int.loc['x1',1], capm.conf_int.loc['const',0]*scale_alpha,
            capm.conf_int.loc['const',1]*scale_alpha]
    
def estimate_FactorModel(grp_obj, dep_var, ind_var):
    '''
    Parameters
    ----------
    grp_obj : dataframe
        All the return data used to estimate the factor model.
    dep_var : string
        column name of dependant variable.
    ind_var : list
        List of column names of independant variables.

    Returns
    -------
    df      : Dataframe containing estimated model parameters
    rsqr    : Rsqr values of model estimation
    '''   
    
    #Estimate model
    fmodel = mc.MultiFactorModel(grp_obj, dep_var, ind_var)
    
    #Concat results
    df = pd.concat([fmodel.coef, fmodel.tvalues, fmodel.conf_int], axis=1)
    
    return [df, fmodel.rsqr]
    
def predict_AR(trainData, testData, window):
    ''' predict AR model including new timepoints
    trainData - data used for training purposes
    testData - data used for testing (and updating) purposes
    window - AR lag window
    
    returns: predictions - predicted values for testData
             rmse        - root mean squared error
    '''
    #Run AR model with specified lags
    model_fit = AutoReg(trainData, lags=window).fit()
    coef = model_fit.params
    
    #walk forward over time steps in test
    hist = trainData[len(trainData)-window:].tolist()
    predictions = list()
    
    for t in range(len(testData)):
        length = len(hist)
        lag = [hist[i] for i in range(length-window, length)]
        yhat = coef[0]
        for d in range(window):
            yhat += coef[d+1] * lag[window-d-1]
        obs = testData[t]
        predictions.append(yhat)
        hist.append(obs)
    
    rmse = sqrt(mean_squared_error(testData, predictions))
    
    return [predictions, rmse]
        


'''============================================================================
    MAIN PROGRAMM
============================================================================'''

# =============================================================================
# DATA LOADING
# =============================================================================

# =============================================================================
# # 1) Compustat & CRSP data by Ernst Thompson
# dd = ('C:\\Users\\Alexander Swade\\OneDrive - Lancaster University\\'
#       'PhD research\\201910_EWvsVW\\Data\\Ernst.Thompson.Miao\\')
# 
# rdata = pyreadr.read_r(dd+'SP500TimeSeries.RData')
# # print(rdata.keys())
# 
# # extract Rdata as single dfs and convert to intended data types
# comInfo = rdata["comInfo_df"].sort_values(by='PERMNO').drop_duplicates()
# 
# consti = rdata["consti"]
# consti[['start', 'ending']] = consti[['start', 'ending']].apply(pd.to_datetime, yearfirst=True)
# 
# sp500 = rdata["SP500"]
# sp500['caldt'] = pd.to_datetime(sp500['caldt'], yearfirst=True)
# 
# prct = rdata["PRCT"].set_index(sp500['caldt'])
# prct.columns = prct.columns.astype(int)
# 
# ret = rdata["RET"].set_index(sp500['caldt'])
# ret.columns = ret.columns.astype(int)
# 
# shrt = rdata["SHRT"].set_index(sp500['caldt'])
# shrt.columns = shrt.columns.astype(int)
# =============================================================================



# 2) FF5 Factors, i.e. MKT, SIZE, HM, CMA, RMW
# dd_ff = ('C:\\Users\\Alexander Swade\\OneDrive - Lancaster University\\'
#       'PhD research\\201910_EWvsVW\\Data\\EW-VW_FactorAnalysis\\')
# ff5 = pd.read_csv(dd_ff + 'F-F_Research_Data_5_Factors_2x3_daily.csv')

dd_ff = ('C:\\Users\\Alexander Swade\\OneDrive - Lancaster University\\'
      'PhD research\\201910_EWvsVW\\Data\\')
ff5 = pd.read_csv(dd_ff + 'F-F_Research_Data_5_Factors.csv', sep=';',
                  decimal=',')

#ff5['Date'] = pd.to_datetime(ff5.iloc[:,0], yearfirst=True, format="%Y%m%d")
ff5['Date'] = pd.to_datetime(ff5['Date'], yearfirst=True, format="%Y%m%d") + MonthEnd(0)
ff5 = ff5.iloc[:,1:].set_index('Date') / 100



# 3) Other factors, i.e. short-term & long-term reversals, momentum
ff_stRev = pd.read_csv(dd_ff + 'F-F_ST_Reversal_Factor.csv', sep=';', decimal=',')
ff_stRev['date'] = pd.to_datetime(ff_stRev['Date'], yearfirst=True, format="%Y%m%d") + MonthEnd(0)
ff_stRev = ff_stRev.iloc[:,2:].set_index('date') / 100

# ff_ltRev = pd.read_csv(dd_ff + 'F-F_LT_Reversal_Factor.csv', sep=';')
# ff_ltRev['date'] = pd.to_datetime(ff_ltRev['Date'], yearfirst=True, format="%Y%m%d") + MonthEnd(0)
# ff_ltRev = ff_ltRev.iloc[:,2:].set_index('date') / 100

ff_mom = pd.read_csv(dd_ff + 'F-F_Momentum_Factor.csv', sep=';', decimal=',')
ff_mom['date'] = pd.to_datetime(ff_mom['Date'], yearfirst=True, format="%Y%m%d") + MonthEnd(0)
ff_mom = ff_mom.iloc[:,2:].set_index('date') / 100

aqr_qmj = pd.read_excel(dd_ff+'Quality Minus Junk Factors Monthly.xlsx', 
                        sheet_name='QMJ Factors', skiprows=18)
aqr_qmj['date'] = pd.to_datetime(aqr_qmj['DATE'], yearfirst=False, format="%m/%d/%Y") + MonthEnd(0)
aqr_qmj = aqr_qmj[['date','USA']].set_index('date').rename({'USA':'QMJ'}, axis=1)

q5_fact = pd.read_csv(dd_ff+'q5_factors_monthly_2021.csv')
q5_fact['day'] = 28
q5_fact['date'] = pd.to_datetime(q5_fact[['year','month','day']]) + MonthEnd(0)
q5_fact = q5_fact[['date','R_ME','R_IA','R_ROE','R_EG']].set_index('date') /100


# Load EW factors
ff_equal_weighted =  pd.read_csv(dd_ff+'ff_equal_weighted.csv', sep=';',
                                 decimal=',')
ff_equal_weighted['date'] = \
    pd.to_datetime(ff_equal_weighted['Date'], yearfirst=True,
                   format="%Y%m%d") + MonthEnd(0)
ff_equal_weighted = ff_equal_weighted.iloc[:,2:].set_index('date') / 100


# Low Vol by Pim van Vliet
low_vol = pd.read_csv(dd_ff+'VanVliet_lowvol_factor.csv', sep=';', decimal=',')
low_vol['date'] = pd.to_datetime(low_vol['Date'], yearfirst=True,
                                 format="%Y%m%d") + MonthEnd(0)
low_vol = low_vol.iloc[:,1:].set_index('date')



# 4) Regulated sectors (based on Barclay & Smith, 1995; as used by Hou & Robinson, 2006)
regulated = pd.DataFrame({'SIC': [4011, 4210, 4213, 4512, 4812, 4813] + 
                          list(range(4900,4940)),
                          'start': pd.to_datetime(['1980-01-01', '1980-01-01',
                                                   '1980-01-01','1978-01-01',
                                                   '1982-01-01', '1982-01-01']+
                                                    40*['1925-12-31'], yearfirst=True),
                          'end': pd.to_datetime(['1980-12-31', '1980-12-31',
                                                 '1980-12-31','1978-12-31',
                                                 '1982-12-31', '1982-12-31']+
                                                  40*['2016-12-31'], yearfirst=True)
                          })


# 5) CCM data
dd5 = ('C:\\Users\\Alexander Swade\\OneDrive - Lancaster University\\'
      'PhD research\\201910_EWvsVW\\Data\\WRDS\\')

comInfo = pd.read_csv(dd5 + 'dseInfo.csv')
comInfo['hsiccd_2'] = (comInfo['hsiccd']/100).apply(np.floor)
comInfo['gic'] = (comInfo['gsubind']*10**-6).apply(np.floor)
comInfo = comInfo.drop_duplicates(subset=['permno']).astype({'gic':'Int64',
                                                             'permno':'Int64'})


# classifications as of
# https://www.spglobal.com/marketintelligence/en/documents/112727-gics-mapbook_2018_v3_letter_digitalspreads.pdf
gic_industries = {10:{'name':'Energy', 'color':'#919CA4'},
                  15:{'name':'Materials', 'color':'#5B88AE'},
                  20:{'name':'Industrials', 'color':'#AC15BD'},
                  25:{'name':'Cons. Discr.', 'color':'#F1ED17'},
                  30:{'name':'Cons. Stapl.', 'color':'#EFB217'},
                  35:{'name':'Health Care', 'color':'#CC2328'},
                  40:{'name':'Financials', 'color':'#14DE0D'},
                  45:{'name':'IT', 'color':'#0717F5'},
                  50:{'name':'Telecom', 'color':'#4B91DB'},
                  55:{'name':'Utilities', 'color':'#034452'},
                  60:{'name':'Real Estate', 'color':'#33E7CF'}
                  }


# 6) CRSP data
#h5file = dd5 + 'crsp_daily_df' + '.h5'
h5file = dd5 + 'crsp_monthly_df' + '.h5'
h5 = pd.HDFStore(path=h5file, mode='a')

#Load returns and drop duplicated indices
ret = h5['ret']
ret = ret[~ret.index.duplicated(keep='first')]
ret.index = ret.index + MonthEnd(0)
permnos_to_use = mc.same_items(ret.columns, comInfo['permno'])
ret = ret.loc[:, permnos_to_use]

#Load mcap and drop duplicated indices
mcap = h5['me']
mcap = mcap[~mcap.index.duplicated(keep='first')]
mcap.index = mcap.index + MonthEnd(0)
mcap = mcap.loc[:, permnos_to_use]

h5.close()

sp500 = pd.read_csv(dd5 + 'sp500_crsp_monthly.csv')
sp500['date'] = pd.to_datetime(sp500['caldt'], yearfirst=True) + MonthEnd(0)
sp500.set_index('date', inplace=True)

consti = pd.read_csv(dd5 + 'sp500_const.csv')
consti[['start', 'ending']] = consti[['start', 'ending']].apply(pd.to_datetime, yearfirst=True)

#remove constituents from SP500 for which there are no ret and me data
consti = consti[consti['permno'].isin(mcap.columns)]


# 7) Fund data by Bloomberg
bbg_data = pd.read_excel(dd_ff + 'small_cap_funds.xlsx')
bbg_data['date'] = pd.to_datetime(bbg_data['Dates'], yearfirst=True) + MonthEnd(0)
bbg_data.set_index('date', inplace=True, drop=True)
bbg_data = bbg_data.iloc[:,1:].pct_change().dropna()



# =============================================================================
# DATA PROCESSING
# =============================================================================


# Clean return data for SP500 stocks with abnormal high returns
# sp500_stock_ret = retSP500.copy()
# to_check = sp500_stock_ret[sp500_stock_ret.ge(0.5, axis=1).values].dropna(how='all', axis=1)
# to_check= to_check.loc[:, to_check.ge(2,axis=1).any().values]


# 1) Construct CNC factor

# # Calc market capitalization for each firm
# # mcap = (abs(prct) * shrt).sort_index(axis=1)

# # Get mcap per sector and year, based on unique SIC numbers - identifier HSICMG, i.e. 2-digit SIC code
# smcap = mcap.groupby(mcap.index.year).tail(1)
# d = dict(zip(smcap.columns, comInfo['hsiccd_2']))
# cols = pd.MultiIndex.from_arrays([smcap.columns.map(d.get), smcap.columns])
# smcap.set_axis(cols, axis=1, inplace=True)

# # Calculate sector weights
# sweights = smcap.groupby(axis=1, level=0).apply(get_perc)

# # Calculate sector concentration (averaged over 3 years)
# scons = sweights.groupby(axis=1, level=0).apply(calc_hhi).rolling(3).mean()

# # Sort industries based on concentration quintiles
# quintiles = scons.dropna(how='all').apply(lambda x: pd.qcut(x, 5, duplicates='drop',
#                           labels=False), axis=1) #labels=True, retbins=False

# quintiles = quintiles.reset_index().melt(id_vars='date', var_name='hsiccd_2',
#                                   value_name='quintile').sort_values(by=['date', 'hsiccd_2'])

# # Get constituents per quintile
# const_per_quint = comInfo[['permno','hsiccd_2']].merge(quintiles, left_on='hsiccd_2',
#                           right_on='hsiccd_2').sort_values(by=['date', 'permno']).\
#                           reset_index(drop=True)


# # Reset index of ret and include quintiles
# d = const_per_quint[['permno','quintile']].drop_duplicates().set_index('permno').to_dict()['quintile']
# cols = pd.MultiIndex.from_arrays([ret.columns.map(d.get), ret.columns])
# ret.set_axis(cols, axis=1, inplace=True)

# # Calculate quintile weights 
# qweights = mcap.copy()
# qweights.set_axis(cols, axis=1, inplace=True)
# qweights = qweights.groupby(axis=1, level=0).apply(get_perc)

# # Calculate quintile portfolios (value-weighted)
# quint_port = (ret*qweights).groupby(axis=1, level=0).sum()

# # Define CNC factor as: 5 (high concentration) - 1 (low concentration)
# quint_port['CNC'] = quint_port[4] - quint_port[0]


# # 2) Combine to df including all syncronized factors 
# factors = ff5.join(quint_port['CNC'])
# factors.to_csv(dd5 + 'factors_monthly.csv', index_label='date')


#Just load factors instead of constructing it again
factors = pd.read_csv(dd5 + 'factors_monthly.csv')
factors['date'] = pd.to_datetime(factors['date'], yearfirst=True)
factors.set_index('date', inplace=True)
factors.drop_duplicates(inplace=True)
factors = pd.concat([factors, ff_mom, ff_stRev, q5_fact, aqr_qmj],
                    axis=1)

#%%

# =============================================================================
# CALCULATIONS
# =============================================================================
start = '1963-07-01'
end = '2021-12-31'


# 0) Construct EW and VW portfolios (based on full universe)
mcap_weights = get_perc(mcap).shift(1)
vw_ret = (ret*mcap_weights).sum(axis=1)
ew_ret = ret.mean(axis=1)


# Returns for SP500 stocks
retSP500 = ret.loc[:, consti['permno'].unique()].copy()
mcapSP500 = mcap.loc[:, consti['permno'].unique()].copy()

# Create masks for constituents based on unique permno
for p in consti['permno'].unique():
    p_info = consti[consti['permno'] == p]
    idx_indicator = pd.Series(data=False, index=retSP500.index)
    
    for i in p_info.index:
        idx_indicator.loc[(idx_indicator.index >= p_info.loc[i, 'start']) &
                          (idx_indicator.index <= p_info.loc[i, 'ending'])] = True

    retSP500.loc[:, p].where(idx_indicator, inplace=True)
    mcapSP500.loc[:, p].where(idx_indicator, inplace=True)


# Construct VW and EW SP500 portfolios 
sp500_vw_weights = get_perc(mcapSP500).shift(1)
sp500_vw = (retSP500*sp500_vw_weights).sum(axis=1)
sp500_ew = retSP500.mean(axis=1)




# 1) Construct regression data
regDat = factors.join(sp500[['vwretd','ewretd']])
regDat.rename(columns={'Mkt-RF':'MKT'}, inplace=True)
regDat = regDat.loc[(regDat.index >= start) & (regDat.index <= end), :]
regDat = regDat.join(pd.concat([vw_ret.rename('univ_vw'),
                                ew_ret.rename('univ_ew'),
                                sp500_vw.rename('sp500_vw'),
                                sp500_ew.rename('sp500_ew')], axis=1))

#Get excess returns for all portfolios
regDat[['vwretd_exc','ewretd_exc','univ_vw_exc',
        'univ_ew_exc','sp500_vw_exc','sp500_ew_exc']] = \
    regDat[['vwretd','ewretd','univ_vw','univ_ew','sp500_vw','sp500_ew']].subtract(regDat['RF'], axis=0)
regDat['MKT_noexc'] = regDat['MKT'] + regDat['RF']

# Calculate performance differences between EW and VW portfolios
regDat['EW_VW'] = regDat['ewretd'].astype(float) - regDat['vwretd'].astype(float)
regDat['EW_VW_FU'] = regDat['univ_ew'].astype(float) - regDat['univ_vw'].astype(float)
regDat['EW_VW_mix'] = regDat['ewretd'].astype(float) - regDat['univ_vw'].astype(float)
regDat['SPW_MKT'] = regDat['ewretd'].astype(float) - regDat['MKT'].astype(float)
regDat['EW_MKT'] = regDat['univ_ew'].astype(float) - regDat['MKT'].astype(float)
regDat['EW_VW_FU_RF'] = regDat['EW_VW_FU'] + regDat['RF']
regDat['EW_VW_RF'] = regDat['EW_VW'] + regDat['RF']


# Add EW FF5 factors (incl. WML and STR)
regDat = pd.merge(regDat, ff_equal_weighted, how='left', on='date')

# Add low vol factor
regDat = pd.merge(regDat, low_vol, how='left', on='date')


# 2) Calculate SMB factor for SP500 universe
# In a similar vein as
#https://wrds-www.wharton.upenn.edu/pages/support/applications/python-replications/fama-french-factors-python/

# sp500_df = pd.merge(mcapSP500.melt(var_name='permno', value_name='mcap',
#                                     ignore_index=False).reset_index(),
#                     retSP500.melt(var_name='permno', value_name='ret',
#                                   ignore_index=False).reset_index(),
#                     on=['date','permno'])

# # sort by permno and date and also drop duplicates
# sp500_df = sp500_df.sort_values(by=['permno','date']).drop_duplicates()

# # keep December market cap
# sp500_df['year'] = sp500_df['date'].dt.year
# sp500_df['month'] = sp500_df['date'].dt.month
# decme = sp500_df[sp500_df['month']==12]
# decme = decme[['permno','date','mcap','year']].rename(columns={'mcap':'dec_me'})

# ### July to June dates
# sp500_df['ffdate'] = sp500_df['date'] + MonthEnd(-6)
# sp500_df['ffyear'] = sp500_df['ffdate'].dt.year
# sp500_df['ffmonth'] = sp500_df['ffdate'].dt.month
# sp500_df['1+retx'] = 1 + sp500_df['ret']
# sp500_df = sp500_df.sort_values(by=['permno','date'])

# # cumret by stock
# sp500_df['cumretx'] = sp500_df.groupby(['permno','ffyear'])['1+retx'].cumprod()

# # lag cumret
# sp500_df['lcumretx'] = sp500_df.groupby(['permno'])['cumretx'].shift(1)

# # lag market cap
# sp500_df['lme'] = sp500_df.groupby(['permno'])['mcap'].shift(1)

# # if first permno then use me/(1+retx) to replace the missing value
# sp500_df['count'] = sp500_df.groupby(['permno']).cumcount()
# sp500_df['lme'] = np.where(sp500_df['count']==0,
#                             sp500_df['mcap']/sp500_df['1+retx'],
#                             sp500_df['lme'])

# # baseline me
# mebase = sp500_df.loc[sp500_df['ffmonth']==1,
#                       ['permno','ffyear', 'lme']].rename(columns={'lme':'mebase'})

# # merge result back together
# sp500_df = pd.merge(sp500_df, mebase, how='left', on=['permno','ffyear'])
# sp500_df['wt'] = np.where(sp500_df['ffmonth']==1,
#                           sp500_df['lme'],
#                           sp500_df['mebase']*sp500_df['lcumretx'])

# decme['year'] = decme['year']+1
# decme = decme[['permno','year','dec_me']]

# # Info as of June
# sp500_df_jun = sp500_df[sp500_df['month']==6]

# sp500_jun = pd.merge(sp500_df_jun, decme, how='inner', on=['permno','year'])
# sp500_jun = sp500_jun[['permno','date','ret','mcap','wt','cumretx','mebase',
#                         'lme','dec_me']]
# sp500_jun = sp500_jun.sort_values(by=['permno','date']).drop_duplicates()

# #Calculate median
# sz_median = sp500_jun.groupby(['date'])['mcap'].median().\
#     to_frame().reset_index().rename(columns={'mcap':'sizemedn'})

# sp500_jun = pd.merge(sp500_jun, sz_median, how='left', on=['date'])

# #Function to assign size bucket
# def sz_bucket(row):
#     if np.isnan(row['mcap']):
#         value = ''
#     elif row['mcap']>=row['sizemedn']:
#         value = 'B'
#     else:
#         value = 'S'
    
#     return value

# sp500_jun['szport'] = sp500_jun.apply(sz_bucket, axis=1)

# # store portfolio assignment as of June
# june = sp500_jun.loc[:,['permno','date','szport']]
# june['ffyear'] = june['date'].dt.year

# # merge back with monthly records
# sp500_df = sp500_df.loc[:,['date','permno','ret','mcap','wt','cumretx','ffyear']]
# sp500_df = pd.merge(sp500_df, june[['permno','ffyear','szport']],
#                     how='left', on=['permno','ffyear'])    

# # function to calculate value weighted return
# def wavg(group, avg_name, weight_name):
#     d = group[avg_name]
#     w = group[weight_name]
#     try:
#         return (d * w).sum() / w.sum()
#     except ZeroDivisionError:
#         return np.nan

# vwret = sp500_df.groupby(['date','szport']).apply(wavg, 'ret','wt').to_frame().\
#     reset_index().rename(columns={0: 'vwret'})

# # transpose
# pseudo_ff_factor = vwret.pivot(index='date', columns='szport',
#                                 values='vwret').reset_index()

# pseudo_ff_factor['pseudo_SMB'] = pseudo_ff_factor['S'] - pseudo_ff_factor['B']
# pseudo_SMB = pseudo_ff_factor[['date', 'pseudo_SMB']].set_index('date')

# #save as csv
# pseudo_SMB.to_csv(dd5 + 'pseudo_SMB.csv')

#read pseudo SMB
pseudo_SMB = pd.read_csv(dd5 + 'pseudo_SMB.csv').set_index('date')
pseudo_SMB.index = pd.to_datetime(pseudo_SMB.index, yearfirst=True,
                                  format="%Y-%m-%d")

regDat = pd.merge(regDat, pseudo_SMB, how='left', on='date')
regDat = regDat.rename({'pseudo_SMB':'SMBSP'}, axis=1)


#Add MKT_(-1) 
regDat['MKT_lagged'] = regDat['MKT'].shift()




# 3) Calculate portfolios with alternative rebalancing freq


# Identify constituents which have all returns for given sample
def identify_constituents(const_ret):
    return const_ret.dropna(how='any', axis=1).columns

# Calculate initial constituent weights
def get_init_weights(group_mcap, incl_const, method='EW'):
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
        init_weights = get_perc(group_mcap.loc[
            group_mcap.index[0], incl_const].to_frame().T).squeeze()

    return init_weights

# Calc drifting weights
def calc_drifting_weights(returns, init_weights):
    '''
    Calculate drifting weights for given return series. FUNCTION IS VALIDATED!

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

def get_weights_grouper_func(grp_ret):
    grp_mcap = SAMPLE_MCAP.loc[grp_ret.index]
    incl_const = identify_constituents(grp_ret)
    init_weights = get_init_weights(grp_mcap, incl_const, method='EW')
    weights = calc_drifting_weights(grp_ret.loc[:,incl_const], init_weights)
    
    return weights


def calc_grouped_turnover(weights, grouper, factor):
    starting_weights = weights.groupby(grouper).first()
    end_weights = weights.groupby(grouper).last()
    # Calculate differences between end of period weights and new weights 
    # for the following period
    weights_diff = starting_weights.fillna(0).shift() - end_weights.fillna(0)
    # Account for initial allocation in the first period
    weights_diff.iloc[0,:] = starting_weights.iloc[0,:]
    
    return weights_diff.abs().sum(axis=1).mean() * factor

## Run calculations with different rebalancing frequencies
sample_ret = ret.loc[(ret.index >= start) & (ret.index <= end), :]
SAMPLE_MCAP = mcap.loc[(mcap.index >= start) & (mcap.index <= end), :]

rebal_freqs = ['1MS','3MS','6MS','1YS','3YS','5YS']
rebal_factors = [12, 4, 2, 1, 1/3, 1/5]
ew_ret_by_freq = list()
turnover = list()
for i, rebal in enumerate(rebal_freqs):
    rebal_freq = pd.Grouper(freq=rebal) 
    ew_weights = sample_ret.groupby(rebal_freq).apply(get_weights_grouper_func)
    ew_portf = sample_ret.multiply(ew_weights.shift(1)).sum(axis=1)
    ew_ret_by_freq.append(ew_portf)
    if rebal == '1MS':
        sret = sample_ret + 1
        wret = ew_weights.shift(1)*sret
        ewbh = wret.divide(wret.sum(axis=1),axis=0)
        w_diff = ewbh.fillna(0) - ew_weights.fillna(0)
        turnover.append(w_diff.abs().sum(axis=1).mean() * rebal_factors[i])
    else:     
        turnover.append(calc_grouped_turnover(ew_weights, rebal_freq,
                                              rebal_factors[i]))

ew_returns = pd.concat(ew_ret_by_freq, axis=1, keys=rebal_freqs).subtract(
    regDat['univ_vw'], axis=0)
ew_returns.columns = 'EW_VW_' + ew_returns.columns
regDat = regDat.join(ew_returns)




# 4) Construct quintile portfolios sorted by size (annual sorting, monthly rebal)

# size_df = pd.merge(mcap.melt(var_name='permno', value_name='mcap',
#                               ignore_index=False).reset_index(),
#                     ret.melt(var_name='permno', value_name='ret',
#                               ignore_index=False).reset_index(),
#                     on=['date','permno'])

# # # sort by permno and date and also drop duplicates
# size_df = size_df.sort_values(by=['permno','date']).drop_duplicates()

# # keep December market cap
# size_df['year'] = size_df['date'].dt.year
# size_df['month'] = size_df['date'].dt.month
# decme = size_df[size_df['month']==12]
# decme = decme[['permno','date','mcap','year']].rename(columns={'mcap':'dec_me'})

# ### July to June dates
# size_df['ffdate'] = size_df['date'] + MonthEnd(-6)
# size_df['ffyear'] = size_df['ffdate'].dt.year
# size_df['ffmonth'] = size_df['ffdate'].dt.month
# size_df['1+retx'] = 1 + size_df['ret']
# size_df = size_df.sort_values(by=['permno','date'])

# # cumret by stock
# size_df['cumretx'] = size_df.groupby(['permno','ffyear'])['1+retx'].cumprod()

# # lag cumret
# size_df['lcumretx'] = size_df.groupby(['permno'])['cumretx'].shift(1)

# # lag market cap
# size_df['lme'] = size_df.groupby(['permno'])['mcap'].shift(1)

# # if first permno then use me/(1+retx) to replace the missing value
# size_df['count'] = size_df.groupby(['permno']).cumcount()
# size_df['lme'] = np.where(size_df['count']==0,
#                           size_df['mcap']/size_df['1+retx'],
#                           size_df['lme'])

# # baseline me
# mebase = size_df.loc[size_df['ffmonth']==1,
#                       ['permno','ffyear', 'lme']].rename(columns={'lme':'mebase'})

# # merge result back together
# size_df = pd.merge(size_df, mebase, how='left', on=['permno','ffyear'])
# size_df['wt'] = np.where(size_df['ffmonth']==1,
#                           size_df['lme'],
#                           size_df['mebase']*size_df['lcumretx'])

# decme['year'] = decme['year']+1
# decme = decme[['permno','year','dec_me']]

# # Info as of June
# size_df_jun = size_df[size_df['month']==6]

# size_jun = pd.merge(size_df_jun, decme, how='inner', on=['permno','year'])
# size_jun = size_jun[['permno','date','ret','mcap','wt','cumretx','mebase',
#                       'lme','dec_me']]
# size_jun = size_jun.sort_values(by=['permno','date']).drop_duplicates()

# #Calculate median
# sz_quintile = size_jun.groupby(['date'])['mcap'].quantile([.2, .4, .6, .8]).\
#     reset_index().rename(columns={'level_1':'sz_breakp'})
    
# sz_quintile = sz_quintile.pivot(index='date', columns='sz_breakp',
#                                 values='mcap').reset_index()

# size_jun = pd.merge(size_jun, sz_quintile, how='left', on=['date'])

# #Function to assign size bucket
# def sz_bucket(row):
#     if np.isnan(row['mcap']):
#         value = ''
#     elif row['mcap']<=row[0.2]:
#         value = 'Q1'
#     elif (row['mcap']<=row[0.4] and row['mcap']>row[0.2]):
#         value = 'Q2'
#     elif (row['mcap']<=row[0.6] and row['mcap']>row[0.4]):
#         value = 'Q3'
#     elif (row['mcap']<=row[0.8] and row['mcap']>row[0.6]):
#         value = 'Q4'
#     else:
#         value = 'Q5'
    
#     return value

# size_jun['szport'] = size_jun.apply(sz_bucket, axis=1)

# # store portfolio assignment as of June
# june = size_jun.loc[:,['permno','date','szport']]
# june['ffyear'] = june['date'].dt.year

# # merge back with monthly records
# size_df = size_df.loc[:,['date','permno','ret','mcap','wt','cumretx','ffyear']]
# size_df = pd.merge(size_df, june[['permno','ffyear','szport']],
#                     how='left', on=['permno','ffyear'])    

# # function to calculate value weighted return
# def wavg(group, avg_name, weight_name):
#     d = group[avg_name]
#     w = group[weight_name]
#     try:
#         return (d * w).sum() / w.sum()
#     except ZeroDivisionError:
#         return np.nan

# vwret = size_df.groupby(['date','szport']).apply(wavg,'ret','wt').to_frame().\
#     reset_index().rename(columns={0: 'vwret'})

# ewret = size_df.groupby(['date','szport'])['ret'].mean().\
#     reset_index().rename(columns={'ret': 'ewret'})

# # merge ew and vw returns
# sz_ret = pd.merge(ewret, vwret, how='left', on=['date','szport'])
# sz_ret = sz_ret.drop(sz_ret[sz_ret['szport']==''].index)

# # calculate EW-VW spread
# sz_ret['ew_vw_ret'] = sz_ret['ewret'] - sz_ret['vwret']

# # transpose
# size_quint_ret = sz_ret.pivot(index='date', columns='szport',
#                                 values='ew_vw_ret').reset_index()

# #save as csv
# size_quint_ret.to_csv(dd5 + 'size_quintile_portfolios.csv')


#read size quintile portfolios
size_quint_ret = pd.read_csv(dd5 + 'size_quintile_portfolios.csv').set_index('date')
size_quint_ret.index = pd.to_datetime(size_quint_ret.index, yearfirst=True,
                                      format="%Y-%m-%d")

regDat = pd.merge(regDat, size_quint_ret, how='left', on='date')

# Replace spaces in column names
regDat.columns = regDat.columns.str.replace(' ', '')


# 5) Estimate single index model for non-overlapping periods, year by year
SIR_columns = ['alpha', 'alpha_t', 'beta', 'beta_t', 'rsqr', 'beta_lcb',
               'beta_ucb', 'alpha_lcb', 'alpha_ucb']

singleIndexResults = regDat.groupby(pd.Grouper(freq='1Y')).apply(
        estimate_CAPM, 'RF', 'vwretd', 'ewretd', alpha_inc=True)

singleIndexResults = pd.DataFrame(singleIndexResults.to_list(),
                              columns = SIR_columns,
                              index = singleIndexResults.index)

SIR_fullUniverse = regDat.groupby(pd.Grouper(freq='1Y')).apply(
        estimate_CAPM, 'RF', 'univ_vw', 'univ_ew', alpha_inc=True)

SIR_fullUniverse = pd.DataFrame(SIR_fullUniverse.to_list(),
                              columns = SIR_columns,
                              index = SIR_fullUniverse.index)

SIR_SPvsFullUniverse = regDat.groupby(pd.Grouper(freq='1Y')).apply(
        estimate_CAPM, 'RF', 'MKT_noexc', 'ewretd', alpha_inc=True)

SIR_SPvsFullUniverse = pd.DataFrame(SIR_SPvsFullUniverse.to_list(),
                              columns = SIR_columns,
                              index = SIR_SPvsFullUniverse.index)



# 6a) Check for seasonality (Jan vs Non-Jan returns)
regDat['EW_VW_jan'] = regDat['EW_VW'].where(regDat.index.month==1, regDat['RF'])
regDat['EW_VW_exjan'] = regDat['EW_VW'].where(regDat.index.month!=1, regDat['RF'])
regDat['SMBSP_jan'] = regDat['SMBSP'].where(regDat.index.month==1, regDat['RF'])
regDat['SMBSP_exjan'] = regDat['SMBSP'].where(regDat.index.month!=1, regDat['RF'])
regDat['EW_VW_FU_jan'] = regDat['EW_VW_FU'].where(regDat.index.month==1, regDat['RF'])
regDat['EW_VW_FU_exjan'] = regDat['EW_VW_FU'].where(regDat.index.month!=1, regDat['RF'])
regDat['SMB_jan'] = regDat['SMB'].where(regDat.index.month==1, regDat['RF'])
regDat['SMB_exjan'] = regDat['SMB'].where(regDat.index.month!=1, regDat['RF'])

regDat['jan_dummy'] = np.where(regDat.index.month==1, 1, 0)
regDat['nonjan_dummy'] = np.where(regDat.index.month==1, 0, 1)

# 6b) Cumulative returns & annual performance EW-VW
spCumRet = (regDat[['vwretd','ewretd','univ_vw','univ_ew','sp500_vw',
                    'sp500_ew','MKT_noexc','SMB','SMBSP','EW_VW',
                    'EW_VW_FU','EW_VW_jan','EW_VW_exjan','EW_VW_FU_jan',
                    'EW_VW_FU_exjan','SMBSP_jan','SMBSP_exjan','SMB_jan',
                    'SMB_exjan']].astype('float') + 1).cumprod()

annualDiff = regDat[['EW_VW','EW_VW_FU','EW_VW_mix','SMB','SMBSP']].groupby(
    pd.Grouper(freq='1Y')).apply(lambda x: (x+1).cumprod().iloc[-1,:]) - 1
    
annualMarket = regDat['MKT_noexc'].groupby(pd.Grouper(freq='1Y')).apply(lambda x:
     (x+1).cumprod()[-1]) -1

    
    
# 7) Estimate AR model for beta and alpha
# ar_model_beta = AutoReg(singleIndexResults['beta'], lags=1).fit() 

# testSize = 10
# lags = 2
# X = singleIndexResults['beta'].values
# trainData_beta, testData_beta = X[0:len(X)-testSize], X[len(X)-testSize:]
# predictedBeta, beta_rmse = predict_AR(trainData_beta, testData_beta, window=lags)

# X = singleIndexResults['alpha'].values
# trainData_alpha, testData_alpha = X[0:len(X)-testSize], X[len(X)-testSize:]
# predictedAlpha, alpha_rmse = predict_AR(trainData_alpha, testData_alpha, window=lags)



# 8) Conditional models based on concentration changes

# #Calculate market concentration
# #concSP500 = calc_hhi(sp500_vw_weights) #use lagged mcap weights
# concSP500 = calc_hhi(get_perc(mcapSP500)) # use end of day mcap weights
# concSP500 = concSP500[spCumRet.index]

# #Calculate rolling mean of concentration
# conc_roll_mean = concSP500.rolling(window=12*5).mean()
# conc_std = concSP500.std()

# #Estimate locally weighted regression smoothing of concentration 
# conc_loess_5 = lowess(concSP500.values, np.arange(len(concSP500.values)), frac=0.05)[:, 1]
# conc_loess_2_5 = lowess(concSP500.values, np.arange(len(concSP500.values)), frac=0.025)[:, 1]


# #Test for stationarity 
# adf_test = sm.tsa.stattools.adfuller(concSP500)
# print('ADF Statistic: %f' % adf_test[0])
# print('p-value: %f' % adf_test[1])
# print('Critical Values:')
# for key, value in adf_test[4].items():
# 	print('\t%s: %.3f' % (key, value))
    
# Construct AR(k) models for concentration



# 9) Regress data based on concentration changes

# Add new regressor: Change in concentration (CIC)
# regDat = regDat.join(concSP500.pct_change(1).rename('CIC'), how='inner')
# regDat['CIC_lagged']  = regDat['CIC'].shift()

# # Add concentration of full universe
# #concFU = calc_hhi(mcap_weights) #use lagged mcap_weights
# concFU = calc_hhi(get_perc(mcap)) # use actual mcap_weights
# concFU = concFU[spCumRet.index]
# regDat = regDat.join(concFU.pct_change(1).rename('CIC_FU'), how='inner')



# 10) Estimate factor model over time

# factor_model_results = regDat.groupby(pd.Grouper(freq='1Y')).apply(
#         estimate_FactorModel, 'EW_VW', ['MKT','SMB','HML','CIC'])

# fmr_rsqr = pd.DataFrame([x[1] for x in factor_model_results],
#                         index = factor_model_results.index, columns = ['rsqr'])

# fmr_df = pd.concat([x[0] for x in factor_model_results], axis=0,
#                    keys=factor_model_results.index)




# 11) Risk decomposition of given portfolios

# # CRSP
# print('\n---Calc CRSP decomposition---\n')
# # # Mcap weights
# # crsp_vw_decomp_dict = \
# #     calc_dynamic_decomp(ret.loc[regDat.index], 
# #                         regDat[['MKT','SMB','HML','Mom']],
# #                         mcap_weights, incl_PCA=True, how='rolling', model=BarraModel)
 
# # EW weights
# crsp_ew_weights = ret.mask(ret.notna(), 1/ret.notna().sum(axis=1), axis=0)
# # crsp_ew_decomp_dict = \
# #     calc_dynamic_decomp(ret.loc[regDat.index], 
# #                         regDat[['MKT','SMB','HML','Mom']],
# #                         crsp_ew_weights, how='rolling', model=BarraModel)

# # EW - VW spread
# crsp_ewvw_weights = crsp_ew_weights - mcap_weights
# # crsp_ewvw_decomp_dict = \
# #     calc_dynamic_decomp(ret.loc[regDat.index], 
# #                         regDat[['MKT','SMB','HML','Mom']],
# #                         crsp_ewvw_weights, how='rolling', model=BarraModel)

# # #Save as csv
# # crsp_vw_decomp_dict['pca_explVar'].to_csv(dd5 + 'crsp_monthly_PCA_explVar.csv')
# # crsp_vw_decomp_dict['pca_comp'].to_csv(dd5 + 'crsp_monthly_PCA_comp.csv')

# # crsp_vw_decomp_dict['risk_contr'].to_csv(dd5 + 'crsp_vw_monthly_riskContr.csv')
# # crsp_ew_decomp_dict['risk_contr'].to_csv(dd5 + 'crsp_ew_monthly_riskContr.csv')
# # crsp_ewvw_decomp_dict['risk_contr'].to_csv(dd5 + 'crsp_EWminusVW_monthly_riskContr.csv')



# # SP500
# print('\n---Calc SP500 decomposition---\n')
# # Mcap weights
# # sp500_vw_decomp_dict = \
# #     calc_dynamic_decomp(retSP500.loc[regDat.index], 
# #                         regDat[['MKT','SMB','HML','Mom']],
# #                         sp500_vw_weights, incl_PCA=True, how='rolling', model=BarraModel)

# # EW weights
# sp500_ew_weights = retSP500.mask(retSP500.notna(), 1/retSP500.notna().sum(axis=1), axis=0)
# # sp500_ew_decomp_dict = \
# #     calc_dynamic_decomp(retSP500.loc[regDat.index], 
# #                         regDat[['MKT','SMB','HML','Mom']],
# #                         sp500_ew_weights, how='rolling',model=BarraModel)
 
# # EW-VW spread
# sp500_ewvw_weights = sp500_ew_weights - sp500_vw_weights
# # sp500_ewvw_decomp_dict = \
# #     calc_dynamic_decomp(retSP500.loc[regDat.index], 
# #                         regDat[['MKT','SMB','HML','Mom']],
# #                         sp500_ewvw_weights, how='rolling', model=BarraModel)

# # #Save as csv
# # sp500_vw_decomp_dict['pca_explVar'].to_csv(dd5 + 'sp500_monthly_PCA_explVar.csv')
# # sp500_vw_decomp_dict['pca_comp'].to_csv(dd5 + 'sp500_monthly_PCA_comp.csv')

# # sp500_vw_decomp_dict['risk_contr'].to_csv(dd5 + 'sp500_vw_monthly_riskContr.csv')
# # sp500_ew_decomp_dict['risk_contr'].to_csv(dd5 + 'sp500_ew_monthly_riskContr.csv')
# # sp500_ewvw_decomp_dict['risk_contr'].to_csv(dd5 + 'sp500_EWminusVW_monthly_riskContr.csv')



# #Load PCA and risk decomp after initial calculation
# crsp_pca_explVar = pd.read_csv(dd5 + 'crsp_monthly_PCA_explVar.csv').set_index('date')
# crsp_pca_explVar.index = pd.to_datetime(crsp_pca_explVar.index, yearfirst=True, format="%Y-%m-%d")
# # crsp_pca_comp = pd.read_csv(dd5 + 'crsp_monthly_PCA_comp.csv')
# # crsp_pca_comp['date'] = pd.to_datetime(crsp_pca_comp['date'], yearfirst=True, format="%Y-%m-%d")
# # crsp_pca_comp = crsp_pca_comp.set_index(['date','portf'])
# crsp_vw_riskContr = pd.read_csv(dd5 + 'crsp_vw_monthly_riskContr.csv').set_index('date')
# crsp_vw_riskContr.index = pd.to_datetime(crsp_vw_riskContr.index, yearfirst=True, format="%Y-%m-%d")
# crsp_ew_riskContr = pd.read_csv(dd5 + 'crsp_ew_monthly_riskContr.csv').set_index('date')
# crsp_ew_riskContr.index = pd.to_datetime(crsp_ew_riskContr.index, yearfirst=True, format="%Y-%m-%d")
# crsp_ewvw_riskContr = pd.read_csv(dd5 + 'crsp_EWminusVW_monthly_riskContr.csv').set_index('date')
# crsp_ewvw_riskContr.index = pd.to_datetime(crsp_ewvw_riskContr.index, yearfirst=True, format="%Y-%m-%d")

# sp500_pca_explVar = pd.read_csv(dd5 + 'sp500_monthly_PCA_explVar.csv').set_index('date')
# sp500_pca_explVar.index = pd.to_datetime(sp500_pca_explVar.index, yearfirst=True, format="%Y-%m-%d")
# sp500_pca_comp = pd.read_csv(dd5 + 'sp500_monthly_PCA_comp.csv')
# sp500_pca_comp['date'] = pd.to_datetime(sp500_pca_comp['date'], yearfirst=True, format="%Y-%m-%d")
# sp500_pca_comp = sp500_pca_comp.set_index(['date','portf'])
# sp500_vw_riskContr = pd.read_csv(dd5 + 'sp500_vw_monthly_riskContr.csv').set_index('date')
# sp500_vw_riskContr.index = pd.to_datetime(sp500_vw_riskContr.index, yearfirst=True, format="%Y-%m-%d")
# sp500_ew_riskContr = pd.read_csv(dd5 + 'sp500_ew_monthly_riskContr.csv').set_index('date')
# sp500_ew_riskContr.index = pd.to_datetime(sp500_ew_riskContr.index, yearfirst=True, format="%Y-%m-%d")
# sp500_ewvw_riskContr = pd.read_csv(dd5 + 'sp500_EWminusVW_monthly_riskContr.csv').set_index('date')
# sp500_ewvw_riskContr.index = pd.to_datetime(sp500_ewvw_riskContr.index, yearfirst=True, format="%Y-%m-%d")



# 12) Calculate sector concentrations

# def calc_sector_weights(w_list, sector_info, gic_industries):
#     '''
#     Parameters
#     ----------
#     w_list : list
#         list of pd.DataFrames with portfolio weights.
#     sector_info : pd.Series
#         Permno as index, sector as value.
#     gic_industries : dict
#         {gic_code : name}

#     Returns
#     -------
#     sector_weights_list : list
#         List with sector contributions according to w_list.

#     '''
#     sector_weights_list = list()
#     for w in w_list:
        
#         d = sector_info.to_dict()
#         cols = pd.MultiIndex.from_arrays([w.columns.map(d.get), w.columns])
#         w = w.set_axis(cols, axis=1).rename_axis(['sector', 'permno'], axis=1)
        
#         sweights = w.groupby(axis=1, by='sector').sum()
#         sweights = sweights.rename(columns=gic_industries)
#         sweights['HHI'] = calc_hhi(sweights)
#         sector_weights_list.append(sweights)
    
#     return sector_weights_list
    

# gic_info = comInfo[['permno', 'gic']].set_index('permno').squeeze()
# gic_names = dict([(key, val['name']) for key, val in sorted(gic_industries.items())])

# #-- CRSP
# sector_weights_list_CRSP= \
#     calc_sector_weights([get_perc(mcap), crsp_ew_weights, crsp_ewvw_weights],
#                         gic_info, gic_names)

# #-- SP500
# sector_weights_list_SP500 = \
#     calc_sector_weights([get_perc(mcapSP500), sp500_ew_weights, sp500_ewvw_weights],
#                         gic_info, gic_names)






#%%

# =============================================================================
# OUTPUTS
# =============================================================================
# Create output folder
od = ('C:\\Users\\Alexander Swade\\OneDrive - Lancaster University\\'
      'PhD research\\201910_EWvsVW\\Output\\') + 'factorAnalysis_' + start + \
    '_' + end +'_monthly\\'

pathlib.Path(od + '//Graphics').mkdir(parents=True, exist_ok=True)

font = {'family': 'serif',
        'color':  'black',
        'weight': 'normal',
        'size': 11,
        }

col_palette = {'grey':'#1A4645','teal':'#266867', 'orange':'#F58800',
               'yellow':'#F8BC24', 'black':'#051821', 'red':'#E74C3C'}

col_diverging = ['#67001f', # dark red
                 '#b2182b', # red
                 '#d6604d', # bright red
                 '#f4a582', # bright orange
                 '#fddbc7', # beige
                 '#e0e0e0', # silver
                 '#bababa', # light grey
                 '#878787', # grey
                 '#4d4d4d', # dark grey
                 '#1a1a1a'] #very dark grey

col_gics = dict([(d['name'], d['color']) for _, d in sorted(gic_industries.items())])


col_gics_alt = ['#9e0142','#d53e4f','#f46d43','#fdae61','#fee08b','#ffffbf',
                '#e6f598','#abdda4','#66c2a5','#3288bd','#5e4fa2']

# 1) Regression results in LATEX format

# ----- CRSP universe decomposition

# Single Index model
reg_sim = ['univ_vw_exc ~ MKT',
           'univ_ew_exc ~ MKT',
           'EW_VW_FU ~ MKT'
           ]
reg_order = ['Intercept','jan_dummy','EW_VW_FU','EW_VW','MKT','MKT_lagged',
             'SMB','SMBSP','HML','Mom','RMW','CMA','ST_Rev','QMJ','low_vol', 
             'R_ME','R_IA','R_ROE','R_EG']
mod_names = [f'({x+1})\n{name}' for x, name in enumerate(['VW','EW','EW-VW'])]

mc.run_OLSregression(regDat.dropna(), reg_sim, od, reg_order=reg_order,
                     filename='regResults_SIM_CRSP', model_names=mod_names, 
                     float_fmt='%0.2f', toLatex=True, scale_alpha=12)
 
# Multi-factor model
regList1 = ['EW_VW_FU ~ MKT + MKT_lagged',
            'EW_VW_FU ~ SMB', 
            #'EW_VW_FU ~ MKT + SMB + HML',
            'EW_VW_FU ~ MKT + MKT_lagged + SMB + HML + RMW + CMA',
            'EW_VW_FU ~ MKT + MKT_lagged + R_ME + R_IA + R_ROE + R_EG',
            'EW_VW_FU ~ MKT + MKT_lagged + SMB + HML + QMJ',
            'EW_VW_FU ~ MKT + MKT_lagged + SMB + HML + RMW + CMA + Mom + ST_Rev + low_vol',   
            'EW_VW_FU ~ jan_dummy + MKT + MKT_lagged + SMB + HML + RMW + CMA + Mom + ST_Rev + low_vol',
            
            'EW_VW ~ MKT + MKT_lagged',
            'EW_VW ~ SMB', 
            #'EW_VW ~ MKT + SMB + HML',
            'EW_VW ~ MKT + MKT_lagged + SMB + HML + RMW + CMA',
            'EW_VW ~ MKT + MKT_lagged + R_ME + R_IA + R_ROE + R_EG',
            'EW_VW ~ MKT + MKT_lagged + SMB + HML + QMJ',
            'EW_VW ~ MKT + MKT_lagged + SMB + HML + RMW + CMA + Mom + ST_Rev + low_vol',   
            'EW_VW ~ jan_dummy + MKT + MKT_lagged + SMB + HML + RMW + CMA + Mom + ST_Rev + low_vol'
           ]
count_crsp_reg = 8
modNames = [f'({x})\nEWVW CRSP' for x in range(1,count_crsp_reg+1)] + \
    [f'({x})\nEWVW SP500' for x in range(count_crsp_reg+1,len(regList1)+1)]

mc.run_OLSregression(regDat, regList1, od, reg_order=reg_order,
                     filename='regResults_CRSP_and_SP500', model_names=modNames, 
                     float_fmt='%0.2f', stars=False, toLatex=True, scale_alpha=100)



# Multi-factor model with EW factors
regList_ew = \
    ['EW_VW_FU ~ MKT + MKT_lagged',
     'EW_VW_FU ~ SMB_ew', 
     'EW_VW_FU ~ MKT + MKT_lagged + SMB_ew + HML_ew + RMW_ew + CMA_ew',
     'EW_VW_FU ~ MKT + MKT_lagged + SMB_ew + HML_ew + RMW_ew + CMA_ew + WML_ew + STR_ew',   
     'EW_VW_FU ~ jan_dummy + MKT + MKT_lagged + SMB_ew + HML_ew + RMW_ew + CMA_ew + WML_ew + STR_ew',
     
     'EW_VW ~ MKT + MKT_lagged',
     'EW_VW ~ SMB_ew', 
     'EW_VW ~ MKT + MKT_lagged + SMB_ew + HML_ew + RMW_ew + CMA_ew',
     'EW_VW ~ MKT + MKT_lagged + SMB_ew + HML_ew + RMW_ew + CMA_ew + WML_ew + STR_ew',   
     'EW_VW ~ jan_dummy + MKT + MKT_lagged + SMB_ew + HML_ew + RMW_ew + CMA_ew + WML_ew + STR_ew',
           ]
count_ew_reg = 5
modNames = [f'({x})\nEWVW CRSP' for x in range(1,count_ew_reg+1)] + \
    [f'({x})\nEWVW SP500' for x in range(count_ew_reg+1,len(regList_ew)+1)]

mc.run_OLSregression(regDat, regList_ew, od, reg_order=reg_order,
                     filename='regResults_CRSP_and_SP500_EW_factors', model_names=modNames, 
                     float_fmt='%0.2f', stars=False, toLatex=True, scale_alpha=100)




# Extensive multi-factor model CRSP only
regList2 = ['EW_VW_FU ~ MKT',
            'EW_VW_FU ~ SMB', 
            'EW_VW_FU ~ MKT + SMB + HML',
            'EW_VW_FU ~ MKT + SMB + HML + Mom',
            'EW_VW_FU ~ MKT + SMB + HML + Mom + ST_Rev',
            'EW_VW_FU ~ MKT + SMB + HML + RMW + CMA',
            'EW_VW_FU ~ MKT + SMB + HML + RMW + CMA + Mom',
            'EW_VW_FU ~ MKT + SMB + HML + RMW + CMA + Mom + ST_Rev',
            'EW_VW_FU ~ MKT + SMB + HML + Mom + QMJ',
            'EW_VW_FU ~ MKT + SMB + RMW + Mom + ST_Rev',
            'EW_VW_FU ~ MKT + SMB + RMW + Mom + ST_Rev + jan_dummy',
            'EW_VW_FU ~ jan_dummy + MKT + MKT_lagged + SMB + HML + QMJ + Mom + ST_Rev + low_vol',
            'EW_VW_FU ~ MKT + R_ME + R_IA + R_ROE + R_EG',
            'EW_VW_FU ~ MKT + R_ME + R_IA + R_ROE + R_EG + Mom'
           ]
modNames = [f'({x})\nEWVW CRSP' for x in range(1,len(regList2)+1)]

mc.run_OLSregression(regDat, regList2, od, reg_order=reg_order,
                     filename='regResults_CRSP_extensive', model_names=modNames, 
                     float_fmt='%0.2f', stars=False, toLatex=False, scale_alpha=100)


# Single-factor model
regList3 = ['EW_VW_FU ~ MKT',
            'EW_VW_FU ~ SMB', 
            'EW_VW_FU ~ HML',
            'EW_VW_FU ~ Mom',
            'EW_VW_FU ~ ST_Rev',
            'EW_VW_FU ~ RMW',
            'EW_VW_FU ~ CMA',
            'EW_VW_FU ~ low_vol',
            'EW_VW_FU ~ QMJ',
            'EW_VW_FU ~ R_ME',
            'EW_VW_FU ~ R_IA',
            'EW_VW_FU ~ R_ROE',
            'EW_VW_FU ~ R_EG',
            'EW_VW ~ MKT',
            'EW_VW ~ SMB', 
            'EW_VW ~ SMBSP',
            'EW_VW ~ HML',
            'EW_VW ~ Mom',
            'EW_VW ~ ST_Rev',
            'EW_VW ~ RMW',
            'EW_VW ~ CMA',
            'EW_VW ~ low_vol',
            'EW_VW ~ QMJ',
            'EW_VW ~ R_ME',
            'EW_VW ~ R_IA',
            'EW_VW ~ R_ROE',
            'EW_VW ~ R_EG',
           ]
count_crsp_reg = 13
modNames = [f'({x})\nEWVW CRSP' for x in range(1,count_crsp_reg+1)] + \
    [f'({x})\nEWVW SP500' for x in range(count_crsp_reg+1,len(regList3)+1)]

mc.run_OLSregression(regDat, regList3, od, reg_order=reg_order,
                     filename='regResults_singleFactors', model_names=modNames, 
                     float_fmt='%0.2f', stars=False, toLatex=True, scale_alpha=100)


# SMB as in Hanauer and Blitz (2020)
regList_smb = ['SMB ~ MKT',
               'SMB ~ MKT + MKT_lagged', 
               'SMB ~ MKT + MKT_lagged + HML + Mom',
               'SMB ~ MKT + MKT_lagged + HML + Mom + RMW + CMA',
               'SMB ~ MKT + MKT_lagged + HML + Mom + QMJ + jan_dummy']
               # 'SMB ~ MKT + MKT_lagged + R_IA + R_ROE + R_EG'
               # ]
modNames = [f'({x})\nSMB' for x in range(len(regList_smb))] 

regressors = ['SMB','MKT','MKT_lagged','HML','Mom','RMW','CMA','QMJ', 'jan_dummy']
mc.run_OLSregression(regDat.loc[regDat.index.year<2020, regressors],
                     regList_smb, od, reg_order=reg_order,
                     filename='regResults_SMB_Hanauer.Blitz', model_names=modNames, 
                     float_fmt='%0.2f', stars=False, toLatex=True, scale_alpha=100)


# EW-VW monthly alpha
regList_smb = ['EW_VW_FU ~ MKT + MKT_lagged', 
               'EW_VW_FU ~ MKT + MKT_lagged + SMB + Mom',
               'EW_VW_FU ~ MKT + MKT_lagged + SMB + Mom + QMJ',
               'EW_VW ~ MKT + MKT_lagged', 
               'EW_VW ~ MKT + MKT_lagged + SMB + Mom',
               'EW_VW ~ MKT + MKT_lagged + SMB + Mom + QMJ',
               'EW_VW ~ MKT + MKT_lagged + SMBSP + Mom',
               'EW_VW ~ MKT + MKT_lagged + SMBSP + Mom + QMJ'
               ]

count_monthly_ewvw = 3
modNames = [f'({x})\nCRSP' for x in range(1,count_monthly_ewvw+1)] + \
    [f'({x})\nSP500' for x in range(count_monthly_ewvw+1,len(regList_smb)+1)]

regressors = ['MKT','MKT_lagged','SMB','SMBSP','Mom','QMJ','EW_VW_FU',
              'EW_VW']
mc.run_OLSregression(regDat.loc[:, regressors],
                     regList_smb, od, reg_order=reg_order,
                     filename='regResults_EWVW_premium', model_names=modNames, 
                     float_fmt='%0.2f', stars=False, toLatex=True, scale_alpha=100)

# SMB regressed against EW-VW
regList_smb_decomp = ['SMB ~ EW_VW_FU',
                      'SMB ~ EW_VW_FU + MKT',
                      'SMB ~ EW_VW_FU + MKT + HML + Mom',
                      'SMB ~ EW_VW_FU + MKT + HML + Mom + QMJ',
                      'SMB ~ EW_VW + MKT + HML + Mom + QMJ',
                      'SMBSP ~ EW_VW',
                      'SMBSP ~ EW_VW + MKT',
                      'SMBSP ~ EW_VW + MKT + HML + Mom',
                      'SMBSP ~ EW_VW + MKT + HML + Mom + RMW + CMA',
                      'SMBSP ~ EW_VW + MKT + HML + Mom + QMJ']

count_crsp = 5
modNames = [f'({x})\nSMB' for x in range(1,count_crsp+1)] + \
    [f'({x})\nSMBSP' for x in range(count_crsp+1,len(regList_smb_decomp)+1)]

mc.run_OLSregression(regDat,
                     regList_smb_decomp, od, reg_order=reg_order,
                     filename='regResults_SMB_decomp', model_names=modNames, 
                     float_fmt='%0.2f', stars=False, toLatex=True, scale_alpha=100)



# # Conditional regressions
# cond_reg = ['EW_VW ~ MKT + SMB + HML + CIC_lagged'] *4
# cond_subset = np.column_stack((np.where(regDat['EW_VW'] > 0, 1, 0),
#                                np.where(regDat['EW_VW'] > 0, 0, 1),
#                                np.where(regDat['MKT_noexc'] > 0, 1, 0),
#                                np.where(regDat['MKT_noexc'] > 0, 0, 1))).astype(bool)

# cond_names = ['(1)\nSPW-SPX\nposDiff', '(2)\nSPW-SPX\nnegDiff', 
#               '(3)\nSPW-SPX\nposMKT', '(4)\nSPW-SPX\nnegMKT']

# mc.run_OLSregression(regDat, cond_reg, od, subset=cond_subset, filename='condRegresults',
#                   model_names=cond_names, toLatex=False)




# 2) Descriptive stats


# ## OLD TABLE

# Construct multi-index dataframe for descriptives
descrip_ret = regDat[['univ_vw','univ_ew','EW_VW_FU','SMB',
                      'vwretd','ewretd','EW_VW','SMBSP']]
descrip_lvl = 100*(descrip_ret+1).cumprod()
descrip_mcap = pd.concat([mcap.sum(axis=1).rename('mcap'), sp500['usdval']]*4,
                          axis=1).loc[descrip_ret.index].sort_index(axis=1)
descrip_mcap.columns = descrip_ret.columns
descrip_count = pd.concat([mcap.count(axis=1).rename('cnt'), sp500['usdcnt']]*4,
                          axis=1).loc[descrip_ret.index].sort_index(axis=1)
descrip_count.columns = descrip_mcap.columns

descrip_data = pd.concat([descrip_ret,descrip_lvl, descrip_mcap, descrip_count],
                          axis=1, keys=['ret','level','mcap','count'],
                          names=['type','portf'])

# Add MKT and RF
new_midx = pd.MultiIndex.from_arrays([['ret', 'ret'], ['RF', 'MKT']])
descrip_data = descrip_data.join(regDat[['RF','MKT_noexc']].set_axis(new_midx, axis=1))




# =============================================================================
# Table - Descriptive Stats
# =============================================================================
descrip_table_ft = calc_descrip_stats(descrip_data, freq='m', inclCAPM=False).\
    reset_index(level='portf')
descrip_table_ft['universe'] = ['CRSP']*4 + ['SP500']*4
descrip_table_ft.replace({'univ_vw':'0_VW', 'univ_ew':'1_EW', 'EW_VW_FU':'2_EW-VW', 
                          'vwretd':'0_VW', 'ewretd':'1_EW', 'EW_VW':'2_EW-VW',
                          'SMBSP':'3_size', 'SMB':'3_size'}, inplace=True)
descrip_table_ft = descrip_table_ft.pivot(index='universe', columns='portf').\
    reset_index(level='universe')
    
# #Create grouping intervals 
grp_offset = pd.DateOffset(years=10)
bins = pd.date_range(pd.Timestamp('1960-12-31'), descrip_data.index[-1], freq=grp_offset)
grouper = pd.cut(descrip_data.index, bins=bins, labels=bins[1:]) 
descrip_table = descrip_data.groupby(grouper)\
    .apply(calc_descrip_stats, freq='m', inclCAPM=False).reset_index(level='portf')
descrip_table['universe'] = np.where(
    descrip_table['portf'].isin(['univ_vw','univ_ew','EW_VW_FU','SMB']),
    'CRSP', 'SP500')
descrip_table.replace({'univ_vw':'0_VW', 'univ_ew':'1_EW', 'EW_VW_FU':'2_EW-VW', 
                        'vwretd':'0_VW', 'ewretd':'1_EW', 'EW_VW':'2_EW-VW',
                        'SMBSP':'3_size', 'SMB':'3_size'}, inplace=True)
descrip_table = descrip_table.reset_index().pivot(
    index=['index','universe'], columns='portf').reset_index(level='universe')

print('-----------SUMMARY STATS------------')
idx = ['1963-2020']*2 + ['1963-1970']*2 + ['1971-1980']*2 + ['1981-1990']*2 + \
    ['1991-2000']*2 + ['2001-2010']*2 + ['2011-2020']*2
hds = ['port'] \
    + [i + '_' + p for i in ['Ret pa', 'Std pa', 'Sharpe', 'MaxDD'] \
        for p in ['VW','EW','EW-VW','size']]\
    + ['avgMcap[$Mrd]', 'avgConst']                 
fmt = [None]*2 + ['.1f']*8 + ['.2f']*4 + ['.1f']*5 + ['.0f']
table = np.concatenate([descrip_table_ft.values, descrip_table.values], axis=0)
table = np.delete(table, np.s_[18:24] , axis=1)
#print(tabulate(table, headers=hds, showindex=idx, floatfmt=fmt)) #tablefmt='latex_raw'
#Save as txt-file
yfile = od + 'descripStats.txt'
with open(yfile, 'w') as f:
    f.write(tabulate(table, headers=hds, showindex=idx, floatfmt=fmt ,tablefmt='latex_booktabs'
                      ))



# New table

pnames = ['0_FullSample', '1_January', '2_Non-Januaray', '3_Pre-publication',
          '4_Post-publication', '5_Pre-GFC', '6_Post-GFC']
portf_names = ['univ_vw', 'univ_ew', 'vwretd', 'ewretd', 'EW_VW_FU', 'EW_VW',
               'RF', 'MKT']
diff_dict = {'univ_vw':'EW_VW_FU', 'univ_ew':'EW_VW_FU', 'vwretd':'EW_VW',
             'ewretd':'EW_VW'}
period_idx = [regDat.index, #FullSample
              regDat.loc[regDat['jan_dummy']==1].index,
              regDat.loc[regDat['nonjan_dummy']==1].index,
              regDat.loc['1957-07-01':'1983-12-31'].index, #Pre-publication
              regDat.loc['1984-01-01':'1999-12-31'].index, #Post-publication
              regDat.loc['2000-01-01':'2009-12-31'].index, #Pre-GFC
              regDat.loc['2010-01-01':'2021-12-31'].index] #Post-GFC          
          
idx_slice = pd.IndexSlice
descrip_list = list()

for i, pidx in enumerate(period_idx):
    sample_data = descrip_data.loc[pidx, idx_slice[:, portf_names]]
    descrip = calc_descrip_stats(sample_data, freq='m', inclCAPM=False,
                                 diff_dict=diff_dict).reset_index(level='portf')
    descrip['universe'] = ['CRSP']*2 + ['SP500']*2
    descrip['sample'] = pnames[i]
    descrip_list.append(descrip)

descrip_tab = pd.concat(descrip_list, ignore_index=True)      
descrip_tab.replace({'univ_vw':'0_VW', 'univ_ew':'1_EW','vwretd':'0_VW',
                     'ewretd':'1_EW'}, inplace=True)
descrip_tab = descrip_tab.pivot(index=['universe','sample'], columns='portf').\
    reset_index(level='sample')

print('-----------SUMMARY STATS------------')
idx = ['CRSP']*7 + ['SP500']*7
hds = ['sample'] \
    + [i + '_' + p for i in ['Ret pa', 'Std pa', 'Sharpe', 'MaxDD1m'] \
        for p in ['VW','EW']]\
    + ['avgMcap[$Mrd]', 'avgConst']\
    + ['diff_ret','diff_t-stat']         
fmt = [None]*2 + ['.1f']*4 + ['.2f']*2 + ['.1f']*3 + ['.0f'] + ['.2f']*2
table = np.delete(descrip_tab.values, [9,11,13,15] , axis=1)
#print(tabulate(table, headers=hds, showindex=idx, floatfmt=fmt)) #tablefmt='latex_raw'
#Save as txt-file
yfile = od + 'descripStats_new.txt'
with open(yfile, 'w') as f:
    f.write(tabulate(table, headers=hds, showindex=idx, floatfmt=fmt ,tablefmt='latex_booktabs'
                      ))

# =============================================================================


# =============================================================================
# Table - Panel3_single_index_model.txt
# =============================================================================
sim_crsp = ['univ_vw', 'univ_ew'] + ['EW_VW_FU_RF']*5
sim_sp500 = ['vwretd','ewretd'] + ['EW_VW_RF']*5
sim_idx = [period_idx[0]]*3 + period_idx[3:]
pnames = ['0_FullSample','1_FullSample','2_FullSample','3_Pre-publication',
          '4_Post-publication', '5_Pre-GFC', '6_Post-GFC']

sim_crsp_results = pd.concat([
    pd.DataFrame(estimate_CAPM(regDat.loc[idx],'RF', 'MKT_noexc', k,
                               alpha_inc=True, hyp=hyp),
                 index=SIR_columns,
                 columns=[name]).T for k, hyp, idx, name in \
        zip(sim_crsp, ['x1=1']*2 + ['x1=0']*5, sim_idx, pnames)])

sim_sp500_results = pd.concat([
    pd.DataFrame(estimate_CAPM(regDat.loc[idx],'RF', 'MKT_noexc', k,
                               alpha_inc=True, hyp=hyp),
                 index=SIR_columns,
                 columns=[name]).T for k, hyp, idx, name in \
        zip(sim_sp500, ['x1=1']*2 + ['x1=0']*5, sim_idx, pnames)])
    
sim_table = pd.concat([sim_crsp_results, sim_sp500_results], axis=1, keys=['CRSP','SP500'])

print('-----------Panel 3: SIM results------------')
hds = [univ+ k for univ, k in itertools.product(
    [r'CRSP ',r'SP500 '], [r'alpha', r'$t(alpha)', r'beta',
                       r't(beta)', r'$R^2$'])]
fmt = [None] + ['.2f']*10
sim_tab = sim_table.loc[:, idx_slice[:,['alpha','alpha_t','beta','beta_t',
                                        'rsqr']]].values
                        
#print(tabulate(table, headers=hds, showindex=idx, floatfmt=fmt)) #tablefmt='latex_raw'
#Save as txt-file
yfile = od + 'panel3_single_index_model.txt'
with open(yfile, 'w') as f:
    f.write(tabulate(sim_tab, headers=hds, showindex=pnames, floatfmt=fmt,
                     tablefmt='latex_booktabs'))
    
# =============================================================================


# =============================================================================
# Figure - cumRet_alluniverses.png
# =============================================================================
fig, axs = plt.subplots(1,1, figsize=(12, 8))
fig.subplots_adjust(wspace=0.3, hspace=0.05) #defaults are 0.2 each


axs.plot(spCumRet.index, spCumRet.loc[:,'vwretd'], label='SPX', ls='--', 
            color=col_palette['black'])
axs.plot(spCumRet.index, spCumRet.loc[:,'ewretd'], label='SPW', ls='-',
            color=col_palette['black'])
axs.plot(spCumRet.index, spCumRet.loc[:,'univ_vw'], label='VW', ls='--',
            color=col_palette['orange'])
axs.plot(spCumRet.index, spCumRet.loc[:,'univ_ew'], label='EW', ls='-',
            color=col_palette['orange'])
axs.grid(visible=True, axis='both')
axs.legend(loc='upper center', ncol=2)
axs.set_ylabel('Cumulative return [log scale]')
axs.set_yscale('log')

#fig.suptitle('Historical performance of CRSP portfolios', fontweight='bold')
fig.savefig(od +'Graphics\\cumRet_alluniverses.png', bbox_inches='tight')

# =============================================================================



# =============================================================================
# Figure - factor_correlations.png
# =============================================================================
mc.plot_correlation(regDat[['EW_VW_FU','EW_VW','MKT', 'MKT_lagged', 'SMB',
                            'HML', 'Mom','ST_Rev','RMW','CMA','QMJ','low_vol',
                            'R_ME','R_IA', 'R_ROE', 'R_EG']],
                    cbar=False, labels=['EW-VW', 'SPW-SPX', 'MKT', 'MKT$_{t-1}$', 'SMB',
                                        'HML', 'WML', 'STR', 'RMW','CMA',
                                        'QMJ','VOL','ME','IA','ROE','EG'],
                    filepath=od+'Graphics\\factor_correlations.png')

# =============================================================================



# =============================================================================
# Table - seasonal_patterns.txt
# =============================================================================

season_crsp = 'EW_VW_FU ~ jan_dummy + MKT + MKT_lagged + SMB + HML + RMW + CMA + Mom + ST_Rev + low_vol'
season_sp500 = 'EW_VW ~ jan_dummy + MKT + + MKT_lagged + SMB + HML + RMW + CMA + Mom + ST_Rev + low_vol'
season_names = ['0_FullSample','1_Pre-publication','2_Post-publication',
                '3_Pre-GFC', '4_Post-GFC']
season_idx = [period_idx[0]] + period_idx[3:]

#Calculate regressions for both universes
season_patterns_crsp = pd.concat([calc_ols(
    regDat.loc[idx], season_crsp) for idx in season_idx], keys=season_names)

season_patterns_sp500 = pd.concat([calc_ols(
    regDat.loc[idx], season_sp500) for idx in season_idx], keys=season_names)

season_results = pd.concat([season_patterns_crsp, season_patterns_sp500],
                           keys = ['CRSP','SP500'])

# Adjust output format
season_results.loc[idx_slice[:,:,'param'],'jan_dummy'] = \
    season_results.loc[idx_slice[:,:,'param'],'jan_dummy']*100
season_results = season_results.applymap('{:,.2f}'.format)
season_results.loc[idx_slice[:,:,'tval'], :] = \
    season_results.loc[idx_slice[:,:,'tval'], :].applymap(
        lambda x: x if x=='nan' else '('+str(x)+')')


#Save as txt-file
yfile = od + 'seasonal_patterns.txt'
with open(yfile, 'w') as f:
    f.write(season_results.to_latex())
 
 
# =============================================================================






# 4) SMB vs EWVW

# # =============================================================================
# # Figure - cumRet_SMB_vs_EW-VW.png
# # =============================================================================
# fig, axs = plt.subplots(2,1, figsize=(12, 8), sharey=False, sharex=True,
#                         gridspec_kw={'height_ratios': [1,1,]})
# fig.subplots_adjust(wspace=0.3, hspace=0.05) #defaults are 0.2 each


# axs[0].plot(spCumRet.index, spCumRet.loc[:,'SMB'], label='SMB', ls='--',
#             color=col_palette['teal'])
# axs[0].plot(spCumRet.index, spCumRet.loc[:,'SMBSP'], label='SMBSP', ls='-',
#             color=col_palette['teal'])
# axs[0].plot(spCumRet.index, spCumRet.loc[:,'EW_VW_FU'], label='EW-VW', ls='--',
#             color=col_palette['orange'])
# axs[0].plot(spCumRet.index, spCumRet.loc[:,'EW_VW'], label='SPW-SPX', ls='-',
#             color=col_palette['black'])

# axs[0].grid(visible=True, axis='both')
# axs[0].legend(loc='upper center', ncol=2)
# axs[0].set_ylabel('Cumulative return')
# #axs[1].set_yscale('log')

# shift1 = pd.to_timedelta('5W')
# shift2 = pd.to_timedelta('12W')
# bar_width = 0.8
# line_width = 1.6
# spreads_to_plot = annualDiff.copy()
# spreads_to_plot.index = spreads_to_plot.index.shift(-12, freq='M') 
# axs[1].bar(spreads_to_plot.index - shift2, spreads_to_plot['EW_VW'], label='SPW-SPX',
#             color=col_palette['black'], edgecolor=col_palette['black'], 
#             width=bar_width, linewidth=line_width)
# axs[1].bar(spreads_to_plot.index - shift1, spreads_to_plot['EW_VW_FU'], label='EW-VW',
#             color=col_palette['orange'], edgecolor=col_palette['orange'], 
#             width=bar_width, linewidth=line_width)
# axs[1].bar(spreads_to_plot.index + shift1, spreads_to_plot['SMB'], label='SMB',
#             color=col_palette['teal'],edgecolor=col_palette['teal'], 
#             width=bar_width, linewidth=line_width)
# axs[1].bar(spreads_to_plot.index + shift2, spreads_to_plot['SMBSP'], label='SMBSP',
#             color=col_palette['yellow'],edgecolor=col_palette['yellow'], 
#             width=bar_width, linewidth=line_width)
# axs[1].set_axisbelow(True)
# axs[1].grid(visible=True, axis='both')
# axs[1].legend(loc='upper center', ncol=4)
# axs[1].set_ylabel('Annual return')

# #fig.suptitle('Historical performance of CRSP portfolios', fontweight='bold')
# fig.savefig(od +'Graphics\\cumRet_SMB_vs_EW-VW.png', bbox_inches='tight')

# # =============================================================================


# =============================================================================
# Table - SMB_EWVW_portf_characteristics.txt
# =============================================================================
'''TER ratios: 
    Invesco SP500 EW    0.200%
    iShares Russell2000 0.190%
    DFA small cap       0.270%
    DFA micro cap       0.410%
    Shorting costs      0.350%    
'''
tc_vec = np.array([0, 0, 0.0055, 0.0055, 0.0054, 0.0062, 0.0076])/12
pnames = ['FullSample', #'Pre-publication', 'Post-publication',
          'Pre-GFC', 'Post-GFC']
bbg_excess = bbg_data.subtract(regDat.loc[:,'vwretd'].rename(None), axis=0)  

portf = regDat.loc[:,['SMB','SMBSP','EW_VW_FU','EW_VW']].join(bbg_excess).dropna()
portf_net = portf.sub(tc_vec, axis=1)

breaks = [(portf.index[0], portf.index[-1]), #FullSample
          #('1957-07-01', '1983-12-31'), #Pre-publication
          #('1984-01-01', '1999-12-31'), #Post-publication
          ('2000-01-01', '2009-12-31'), #Pre-GFC
          ('2010-01-01', '2021-12-31')]  #Post-GFC

smb_portf_table = calc_portf_chars(portf_net, breaks, pnames)

print('-----------SUMMARY STATS FOR SP500------------')
idx = np.repeat(pnames, len(portf.columns)).tolist()
hds = ['Port'] + smb_portf_table.columns.to_list()
fmt = [None, None] + ['.2f']*len(smb_portf_table.columns)
table = smb_portf_table.reset_index(level='Portf').values
#print(tabulate(table, headers=hds, showindex=idx, floatfmt=fmt)) #tablefmt='latex_raw'
#Save as txt-file
yfile = od + 'SMB_EWVW_portf_characteristics.txt'
with open(yfile, 'w') as f:
    f.write(tabulate(table, headers=hds, showindex=idx, floatfmt=fmt ,
                     tablefmt='latex_booktabs'))

# =============================================================================


# =============================================================================
# Figure - size_correlations.png
# =============================================================================
mc.plot_correlation(portf, cbar=False,
                    labels=['SMB','SMBSP','EW-VW','SPW-SPX','R2000',
                            'DFA Small','DFA Micro'],
                    filepath=od+'Graphics\\size_correlations.png')

# =============================================================================



# =============================================================================
# Table - Alternative_portf_sorts.txt
# =============================================================================
pnames = ['FullSample', 'Pre-publication', 'Post-publication',
          'Pre-GFC', 'Post-GFC']
portf = regDat[[#'Q1','Q2','Q3','Q4','Q5',
                'EW_VW_FU','EW_VW_3MS','EW_VW_6MS','EW_VW_1YS','EW_VW_3YS',
                'EW_VW_5YS']]

breaks = [(portf.index[0], portf.index[-1]), #FullSample
          ('1957-07-01', '1983-12-31'), #Pre-publication
          ('1984-01-01', '1999-12-31'), #Post-publication
          ('2000-01-01', '2009-12-31'), #Pre-GFC
          ('2010-01-01', '2021-12-31')]  #Post-GFC

# Panel 1: Calculate annual returns
# alt_portf_ret = pd.DataFrame(np.nan, columns=portf.columns, index=pnames)

# for i, dat in enumerate(breaks):    
#     start, end  = dat
#     ret_p = portf.loc[start:end]      
#     alt_portf_ret.loc[pnames[i], portf.columns] = \
#         ret_p.mean(axis=0).values*100*12


# Alternative Panel 1: Calculate portfolio characteristics
alt_portf_ret = pd.DataFrame(np.nan, columns=portf.columns, index=[
    'Ret','Std','Sharpe','MDD','Turnover'])
alt_portf_ret.loc['Ret',:] = portf.mean().values*12*100
alt_portf_ret.loc['Std',:] = portf.std().values*np.sqrt(12)*100
alt_portf_ret.loc['Sharpe',:] = \
    alt_portf_ret.loc['Ret',:] / alt_portf_ret.loc['Std',:]
alt_portf_ret.loc['MDD',:] =  utils.max_drawdown(portf).values*100
alt_portf_ret.loc['Turnover',:] = turnover



# Panel 2: Regression coefficients
regList_size_quint = \
    [#'Q1 ~ jan_dummy + MKT + MKT_lagged + SMB + HML + RMW + CMA + Mom + ST_Rev',
     #'Q2 ~ jan_dummy + MKT + MKT_lagged + SMB + HML + RMW + CMA + Mom + ST_Rev',
     #'Q3 ~ jan_dummy + MKT + MKT_lagged + SMB + HML + RMW + CMA + Mom + ST_Rev',
     #'Q4 ~ jan_dummy + MKT + MKT_lagged + SMB + HML + RMW + CMA + Mom + ST_Rev',
     #'Q5 ~ jan_dummy + MKT + MKT_lagged + SMB + HML + RMW + CMA + Mom + ST_Rev',
     'EW_VW_FU ~ jan_dummy + MKT + MKT_lagged + SMB + HML + RMW + CMA + Mom + ST_Rev + low_vol',
     #'EW_VW_1MS ~ jan_dummy + MKT + MKT_lagged + SMB + HML + RMW + CMA + Mom + ST_Rev',
     'EW_VW_3MS ~ jan_dummy + MKT + MKT_lagged + SMB + HML + RMW + CMA + Mom + ST_Rev + low_vol',
     'EW_VW_6MS ~ jan_dummy + MKT + MKT_lagged + SMB + HML + RMW + CMA + Mom + ST_Rev + low_vol',
     'EW_VW_1YS ~ jan_dummy + MKT + MKT_lagged + SMB + HML + RMW + CMA + Mom + ST_Rev + low_vol',
     'EW_VW_3YS ~ jan_dummy + MKT + MKT_lagged + SMB + HML + RMW + CMA + Mom + ST_Rev + low_vol',
     'EW_VW_5YS ~ jan_dummy + MKT + MKT_lagged + SMB + HML + RMW + CMA + Mom + ST_Rev + low_vol'
     ]

alt_portf_reg = pd.concat([calc_ols(regDat, reg, orientation='vert') \
                           for reg in regList_size_quint],
                          axis=1, keys=portf.columns).droplevel(1, axis=1)

# Adjust output format
alt_portf_ret = alt_portf_ret.applymap('{:,.2f}'.format)    
    
alt_portf_reg.loc[('jan_dummy', 'param'), :] *= 100  
alt_portf_reg = alt_portf_reg.applymap('{:,.2f}'.format) 
alt_portf_reg.loc[idx_slice[:,'tval'], :] = \
    alt_portf_reg.loc[idx_slice[:,'tval'], :].applymap(
        lambda x: x if x=='nan' else '('+str(x)+')') 

alt_portf = pd.concat([alt_portf_ret, alt_portf_reg.droplevel(1)], axis=0)
    
#Save as txt-file
yfile = od + 'alternative_portf_sorts.txt'
with open(yfile, 'w') as f:
    f.write(alt_portf.to_latex())

# =============================================================================









# =============================================================================
# Figure - seasonality_performance.png
# =============================================================================
# fig, axs = plt.subplots(2,1, figsize=(10, 8), sharey=False, sharex=True,
#                         gridspec_kw={'height_ratios': [1,1]})
# fig.subplots_adjust(wspace=0.1, hspace=0.1)

# axs[0].plot(spCumRet.index, spCumRet.loc[:,'EW_VW_FU_jan'],
#             label='EW-VW January', ls='--', color=col_palette['orange'])
# axs[0].plot(spCumRet.index, spCumRet.loc[:,'EW_VW_FU_exjan'], 
#             label='EW-VW Non-January', ls='-', color=col_palette['orange'])
# axs[0].plot(spCumRet.index, spCumRet.loc[:,'SMB_jan'], label='SMB January',
#             ls='--', color=col_palette['teal'])
# axs[0].plot(spCumRet.index, spCumRet.loc[:,'SMB_exjan'],
#             label='SMB Non-January', ls='-', color=col_palette['teal'])
# axs[0].grid(visible=True, axis='both')
# axs[0].legend(loc='upper center', ncol=2)
# axs[0].set_ylabel('Cumulative return [log scale]')
# axs[0].set_yscale('log')
# axs[0].set_title('CRSP')


# axs[1].plot(spCumRet.index, spCumRet.loc[:,'EW_VW_jan'], label='EW-VW January',
#             ls='--', color=col_palette['orange'])
# axs[1].plot(spCumRet.index, spCumRet.loc[:,'EW_VW_exjan'],
#             label='EW-VW Non-January', ls='-', color=col_palette['orange'])
# axs[1].plot(spCumRet.index, spCumRet.loc[:,'SMBSP_jan'], label='SMBSP January',
#             ls='--', color=col_palette['teal'])
# axs[1].plot(spCumRet.index, spCumRet.loc[:,'SMBSP_exjan'],
#             label='SMBSP Non-January', ls='-', color=col_palette['teal'])
# axs[1].grid(visible=True, axis='both')
# axs[1].legend(loc='upper center', ncol=2)
# axs[1].set_ylabel('Cumulative return [log scale]')
# axs[1].set_yscale('log')
# axs[1].set_title('SP500')

# fig.savefig(od +'Graphics\\seasonality_performance.png', bbox_inches='tight')


# =============================================================================

# 5) Factor decomposition over time

# =============================================================================
# Figure - Risk_decomp.png
# =============================================================================
# fig, axs = plt.subplots(3,2, figsize=(10, 8), sharey=True, sharex=True,
#                         gridspec_kw={'height_ratios': [1, 1, 1]})
# fig.subplots_adjust(wspace=0.1, hspace=0.1)


# risk_list = [crsp_vw_riskContr, sp500_vw_riskContr, crsp_ew_riskContr, 
#              sp500_ew_riskContr, crsp_ewvw_riskContr, sp500_ewvw_riskContr]
# sub_y_labels = ['VW portfolio\nrel. variance',
#                 'EW portfolio\nrel. variance',
#                 'EW-VW spread\nrel. variance']
# sub_title = ['CRSP', 'SP500']
# label_counter = 0
# twin_axs = list()
# twin_ylim = [-0.02, 0.32]

# for i, (ax, df) in enumerate(zip(axs.reshape(-1), risk_list)):
    
#     #Get relevant factor data and split in positive and negative contributions
#     ndf = df[['MKT','SMB','HML','Mom','Idiosyncratic']]
#     df_neg, df_pos = ndf.clip(upper=0), ndf.clip(lower=0)
    
#     #Plot stackplot
#     ax.stackplot(df_pos.index, df_pos.T, colors=col_palette.values(),
#                  labels=df_pos.columns)
#     ax.stackplot(df_neg.index, df_neg.T, colors=col_palette.values())
#     ax.set_axisbelow(True)
#     ax.grid(True, which='major', linewidth=1)
    
#     #Create secondary axis
#     twin_axs.append(ax.twinx())
#     if i > 0:
#         twin_axs[0].get_shared_y_axes().join(twin_axs[0], twin_axs[i])        
#     twin_axs[i].plot(df.index, df['Portf_vola'], label='Vola p.a.',
#                      color='#e6352b')
    
#     #Label outer subplots & remove ticks from left column secondary yaxis
#     if i % 2:
#         #Odd subplots 
#         twin_axs[i].set_ylabel('Portfolio volatility')
#     else:
#         #Even subplots
#         ax.set_ylabel(sub_y_labels[label_counter])
#         plt.setp(twin_axs[i].get_yticklabels(), visible=False)
#         label_counter += 1

#     # format shared x-axis
#     if i > 3:
#         ax.set_xlim([df_pos.index.min(), df_pos.index.max()])
#         years = dates.YearLocator(10, month=1, day=1)
#         years1 = dates.YearLocator(2, month=1, day=1)
#         ax.xaxis.set_major_locator(years)
#         ax.xaxis.set_minor_locator(years1)
#         ax.xaxis.set_major_formatter(dates.DateFormatter('%Y'))
#         # ax.xaxis.set_minor_formatter(dates.DateFormatter('%y'))
#         ax.get_xaxis().set_tick_params(which='major', pad=5)
#         ax.set_xlabel('')
#         plt.setp(ax.get_xmajorticklabels(), rotation=0,ha='center')
        
#     #Add title to first two subplots
#     if i < 2:
#         ax.set_title(sub_title[i])

# #Rescale twin axis
# twin_axs[0].set_ylim(twin_ylim)
    
# # Create legend
# handles, labels = ax.get_legend_handles_labels()
# handles_twin, labels_twin = twin_axs[-1].get_legend_handles_labels()
# fig.legend(handles+handles_twin, labels+labels_twin, loc='lower center',
#            ncol=6)

# fig.savefig(od +'Graphics\Risk_decomp.png', bbox_inches='tight')

# =============================================================================



# =============================================================================
# Figure - SP500_sector_hhi.png
# =============================================================================

# def get_relative_sector_performance(returns, mcap_weights, sector_info, 
#                                     gic_industries, filepath):
#     '''
#     Calculate relative sector performance and concentration; plot results for 
#     each sector afterwards.
    
#     Parameters
#     ----------
#     ret : pd.DataFrame, shape(T,N)
#         Asset returns.
#     mcap_weights : pd.DataFrame, shape(T,N)
#         Market capitalization weights.
#     sector_info : pd.Series
#         Permno as index, sector as value.
#     gic_industries : dict
#         {gic_code : name}
#     filepath : str
#         Full filename incl. directory to save results to

#     Returns
#     -------
#     None.

#     '''
#     #https://www.spglobal.com/en/research-insights/articles/concentrating-on-technology
    
    
#     d = sector_info.to_dict()
#     cols = pd.MultiIndex.from_arrays([returns.columns.map(d.get), returns.columns])
#     returns = returns.set_axis(cols, axis=1).rename_axis(['sector', 'permno'], axis=1)
#     mcap_w = mcap_weights.set_axis(cols, axis=1).rename_axis(['sector','permno'], axis=1)
    
    
#     nrow, ncol = math.ceil(len(gic_industries)/3), 3
#     fig, axs = plt.subplots(nrow, ncol, figsize=(12, 10), sharey=False,
#                             sharex=True)
#     fig.subplots_adjust(wspace=0.4, hspace=0.2)
#     axs = axs.reshape(-1)
#     twin_axs = list()
    
#     #Bar plot settings
#     bar_width = 0.8
#     line_width = 0.8
    
#     for i, sector in enumerate(gic_names.keys()):
#         #sector=gic_names.get(10)
        
#         # Get assets in sector
#         idx=pd.IndexSlice
#         sector_ret = returns.loc[:, idx[sector,:]]
#         sector_w = mcap_w.loc[:, idx[sector,:]]
#         sector_w = sector_w.divide(sector_w.sum(axis=1).values, axis=0)
        
#         # Calc EW portfolio
#         ew_sect_portf = sector_ret.mean(axis=1)
        
#         # Calc VW portfolio
#         vw_sect_portf = (sector_ret*sector_w).sum(axis=1)
        
#         # Calc relative performance
#         rel_perf = (1 + (ew_sect_portf - vw_sect_portf)).cumprod()
        
#         # Calc (adjusted) HHI
#         hhi_ew = calc_hhi(sector_ret.mask(
#             sector_ret.notna(), 1/sector_ret.count(axis=1), axis=0))
#         adjusted_hhi = calc_hhi(sector_w) / hhi_ew
        
        
#         ###Plotting section###
#         twin_axs.append(axs[i].twinx())
#         twin_axs[i].bar(adjusted_hhi.index, adjusted_hhi, label='Adjusted HHI', 
#                         color='#c1e1ec', edgecolor='#c1e1ec', 
#                         width=bar_width, linewidth=line_width)
        
#         axs[i].plot(rel_perf.index, rel_perf, label='EW relative performance')
#         axs[i].set_title(gic_names.get(sector))
    
#         axs[i].set_zorder(twin_axs[i].get_zorder()+1)
#         axs[i].set_frame_on(False)
        
#     # format shared x-axis
#     for j in range((nrow-1)*ncol, nrow*ncol):        
#         axs[j].set_xlim([returns.index.min(), returns.index.max()])
#         years = dates.YearLocator(10, month=1, day=1)
#         years1 = dates.YearLocator(2, month=1, day=1)
#         axs[j].xaxis.set_major_locator(years)
#         axs[j].xaxis.set_minor_locator(years1)
#         axs[j].xaxis.set_major_formatter(dates.DateFormatter('%Y'))
#         # ax.xaxis.set_minor_formatter(dates.DateFormatter('%y'))
#         axs[j].get_xaxis().set_tick_params(which='major', pad=5)
#         axs[j].set_xlabel('')
#         plt.setp(axs[j].get_xmajorticklabels(), rotation=90, weight='bold',
#                  ha='center')
        
#     # Create legend
#     handles, labels = axs[0].get_legend_handles_labels()
#     handles_twin, labels_twin = twin_axs[0].get_legend_handles_labels()
#     fig.legend(handles+handles_twin, labels+labels_twin, loc='lower center',
#                ncol=2)
    
#     fig.text(0.05, 0.5,'EW relative performance', va='center', rotation='vertical')
#     fig.text(0.95, 0.5,'Adjusted HHI', va='center', rotation='vertical')

#     fig.savefig(filepath, bbox_inches='tight')


# #Run helper function
# get_relative_sector_performance(
#     retSP500.loc[(retSP500.index >= start) & (retSP500.index <= end)],
#     sp500_vw_weights.loc[(sp500_vw_weights.index >= start) & \
#                          (sp500_vw_weights.index <= end)],
#     gic_info,
#     gic_names,
#     od+'Graphics\SP500_sector_hhi.png')


# =============================================================================


### OUTDATED GRAPHICS ###


# =============================================================================
# Figure - CRSP_cumRet.png
# =============================================================================
# fig, axs = plt.subplots(2,1, figsize=(12, 8), sharey=False, sharex=True,
#                         gridspec_kw={'height_ratios': [3, 1,]})
# fig.subplots_adjust(wspace=0.3, hspace=0.05) #defaults are 0.2 each
# axs[0].plot(spCumRet.index, spCumRet.loc[:,'univ_vw'], label='VW', ls='-', 
#             color=col_palette['grey'])
# axs[0].plot(spCumRet.index, spCumRet.loc[:,'univ_ew'], label='EW', ls='-',
#             color=col_palette['orange'])
# # axs[0].plot(spCumRet.index, spCumRet.loc[:,'sp500_vw'], label='VW', ls='--',
# #             color=col_palette['black'])
# # axs[0].plot(spCumRet.index, spCumRet.loc[:,'sp500_ew'], label='EW', ls='--',
# #             color=col_palette['yellow'])
# axs[0].grid(b=True, axis='both')
# axs[0].legend(loc='upper center', ncol=2)
# axs[0].set_ylabel('Cumulative return [log scale]')
# axs[0].set_yscale('log')

# ml, sl, bl = axs[1].stem(annualDiff.index, annualDiff['EW_VW_FU'], linefmt='grey',
#                          markerfmt='o', basefmt='grey', use_line_collection=True)
# plt.setp(ml, 'color', col_palette['teal'])
# axs[1].grid(b=True, axis='both')
# axs[1].set_ylabel('EW-VW p.a.')

# #fig.suptitle('Historical performance of CRSP portfolios', fontweight='bold')
# fig.savefig(od +'Graphics\\CRSP_cumRet.png', bbox_inches='tight')

# =============================================================================



# =============================================================================
# Figure - SP500_cumRet.png
# =============================================================================
# fig, axs = plt.subplots(2,1, figsize=(12, 8), sharey=False, sharex=True,
#                         gridspec_kw={'height_ratios': [3, 1,]})
# fig.subplots_adjust(wspace=0.3, hspace=0.05) #defaults are 0.2 each
# axs[0].plot(spCumRet.index, spCumRet.loc[:,'vwretd'], label='SPX', ls='-', 
#             color=col_palette['grey'])
# axs[0].plot(spCumRet.index, spCumRet.loc[:,'ewretd'], label='SPW', ls='-',
#             color=col_palette['orange'])
# axs[0].plot(spCumRet.index, spCumRet.loc[:,'sp500_vw'], label='VW', ls='--',
#             color=col_palette['black'])
# axs[0].plot(spCumRet.index, spCumRet.loc[:,'sp500_ew'], label='EW', ls='--',
#             color=col_palette['yellow'])
# axs[0].grid(b=True, axis='both')
# axs[0].legend(loc='upper center', ncol=2)
# axs[0].set_ylabel('Cumulative return [log scale]')
# axs[0].set_yscale('log')

# ml, sl, bl = axs[1].stem(annualDiff.index, annualDiff['EW_VW'], linefmt='grey',
#                          markerfmt='o', basefmt='grey', use_line_collection=True)
# plt.setp(ml, 'color', col_palette['teal'])
# axs[1].grid(b=True, axis='both')
# axs[1].set_ylabel('SPW-SPX p.a.')

# #fig.suptitle('Historical performance of S&P500 portfolios', fontweight='bold')
# fig.savefig(od +'Graphics\\SP500_cumRet.png', bbox_inches='tight')

# =============================================================================


# =============================================================================
# Figure - SP500_SingleIndexModel.png
# =============================================================================
# fig, axs = plt.subplots(3,1, figsize=(12, 8), sharey=False, sharex=True,
#                         gridspec_kw={'height_ratios': [1, 1, 1,]})
# fig.subplots_adjust(wspace=0.3, hspace=0.05) #defaults are 0.2 each

# ml, sl, bl = axs[0].stem(annualDiff.index, annualDiff['EW_VW'], linefmt='grey',
#                          markerfmt='o', basefmt='grey', use_line_collection=True)
# plt.setp(ml, 'color', col_palette['teal'])
# axs[0].grid(b=True, axis='both')
# axs[0].set_ylabel('EW-VW p.a.')

# axs[1].fill_between(SIR_SPvsFullUniverse.index, SIR_SPvsFullUniverse.loc[:,'beta_lcb'],
#    SIR_SPvsFullUniverse.loc[:,'beta_ucb'], color=col_palette['yellow'], alpha=.15,
#    label='95% CI') 
# axs[1].plot(SIR_SPvsFullUniverse.index, SIR_SPvsFullUniverse.loc[:,'beta'],
#    linewidth=2, color=col_palette['yellow'], label=r'$\overline{\beta}$') 
# axs[1].axhline(y=1, color='black')
# axs[1].grid(b=True, which='both')
# axs[1].legend(loc='lower center', ncol = 2)
# axs[1].set_ylabel(r'Estimated $\beta$')

# axs[2].fill_between(SIR_SPvsFullUniverse.index, SIR_SPvsFullUniverse.loc[:,'alpha_lcb'],
#    SIR_SPvsFullUniverse.loc[:,'alpha_ucb'], color=col_palette['black'], alpha=.15,
#    label='95% CI') 
# axs[2].plot(SIR_SPvsFullUniverse.index, SIR_SPvsFullUniverse.loc[:,'alpha'],
#    linewidth=2, color=col_palette['black'], label=r'$\overline{\alpha}$') 
# axs[2].axhline(y=0, color='black')
# axs[2].grid(b=True, which='both')
# axs[2].legend(loc='lower center', ncol = 2)
# axs[2].set_ylabel(r'Estimated $\alpha$')

# fig.suptitle('Single Index Model Estimation', fontweight='bold')
# fig.savefig(od +'Graphics\\SP500_SingleIndexModel.png', bbox_inches='tight')

# =============================================================================


# =============================================================================
# Figure - CRSP_PCA_decomp.png
# =============================================================================
# fig, axs = plt.subplots(1,1, figsize=(12, 8))

# axs.stackplot(crsp_pca_explVar.index, crsp_pca_explVar.T, colors=col_diverging, 
#               labels=crsp_pca_explVar.columns)
# axs.legend(loc='upper center', ncol=5)
# axs.set_axisbelow(True)
# axs.grid(True, which='major', linewidth=1)

# #Format axis
# axs.set_ylim([0,1])
# axs.set_ylabel('Explained relative variance')
# axs.set_xlim([crsp_pca_explVar.index.min(), crsp_pca_explVar.index.max()])
# years = dates.YearLocator(10, month=1, day=1)
# years1 = dates.YearLocator(2, month=1, day=1)
# axs.xaxis.set_major_locator(years)
# axs.xaxis.set_minor_locator(years1)
# axs.xaxis.set_major_formatter(dates.DateFormatter('%Y'))
# axs.xaxis.set_minor_formatter(dates.DateFormatter('%y'))
# axs.get_xaxis().set_tick_params(which='major', pad=15)
# axs.set_xlabel('')
# plt.setp(axs.get_xmajorticklabels(), rotation=0, weight='bold', ha='center')

# fig.savefig(od +'Graphics\CRSP_PCA_decomp.png', bbox_inches='tight')

# =============================================================================


# =============================================================================
# Figure - SP500_PCA_decomp.png
# =============================================================================
# fig, axs = plt.subplots(1,1, figsize=(12, 8))

# axs.stackplot(sp500_pca_explVar.index, sp500_pca_explVar.T, colors=col_diverging, 
#               labels=sp500_pca_explVar.columns)
# axs.legend(loc='upper center', ncol=5)
# axs.set_axisbelow(True)
# axs.grid(True, which='major', linewidth=1)

# #Format axis
# axs.set_ylim([0,1])
# axs.set_ylabel('Explained relative variance')
# axs.set_xlim([sp500_pca_explVar.index.min(), sp500_pca_explVar.index.max()])
# years = dates.YearLocator(10, month=1, day=1)
# years1 = dates.YearLocator(2, month=1, day=1)
# axs.xaxis.set_major_locator(years)
# axs.xaxis.set_minor_locator(years1)
# axs.xaxis.set_major_formatter(dates.DateFormatter('%Y'))
# axs.xaxis.set_minor_formatter(dates.DateFormatter('%y'))
# axs.get_xaxis().set_tick_params(which='major', pad=15)
# axs.set_xlabel('')
# plt.setp(axs.get_xmajorticklabels(), rotation=0, weight='bold', ha='center')

# fig.savefig(od +'Graphics\SP500_PCA_decomp.png', bbox_inches='tight')

# =============================================================================



# =============================================================================
# Figure - CRSP_risk_decomp.png
# =============================================================================
# fig, axs = plt.subplots(3,1, figsize=(12, 8), sharey=False, sharex=True,
#                         gridspec_kw={'height_ratios': [1, 1, 1]})
# fig.subplots_adjust(wspace=0.3, hspace=0.1)

# # Prep risk contributions for plotting
# crsp_vw_relContr = crsp_vw_riskContr[['MKT','SMB','HML','Mom','Idiosyncratic']]
# df_vw_neg, df_vw_pos = crsp_vw_relContr.clip(upper=0), crsp_vw_relContr.clip(lower=0)
# crsp_ew_relContr = crsp_ew_riskContr[['MKT','SMB','HML','Mom','Idiosyncratic']]
# df_ew_neg, df_ew_pos = crsp_ew_relContr.clip(upper=0), crsp_ew_relContr.clip(lower=0)
# crsp_ewvw_relContr = crsp_ewvw_riskContr[['MKT','SMB','HML','Mom','Idiosyncratic']]
# df_ewvw_neg, df_ewvw_pos = crsp_ewvw_relContr.clip(upper=0), crsp_ewvw_relContr.clip(lower=0)

# ymin = min(df_vw_neg.sum(axis=1).min(), df_ew_neg.sum(axis=1).min())
# ymax = max(df_vw_pos.sum(axis=1).max(), df_ew_pos.sum(axis=1).max())
# ylim = [ymin - 0.02*(ymax-ymin), ymax + 0.02*(ymax-ymin)]
# twin_ylim = [-0.02, 0.32]

# #AX0: VW portfolio
# axs[0].stackplot(df_vw_pos.index, df_vw_pos.T, colors=col_palette.values(),
#                  labels=df_vw_pos.columns)
# axs[0].stackplot(df_vw_neg.index, df_vw_neg.T, colors=col_palette.values())
# axs[0].set_ylabel('VW Portfolio\nrel. variance')
# axs[0].set_ylim(ylim)
# axs[0].set_axisbelow(True)
# axs[0].grid(True, which='major', linewidth=1)

# axs0_twin = axs[0].twinx()
# axs0_twin.plot(crsp_vw_riskContr.index, crsp_vw_riskContr['Portf_vola'],
#               label='Vola p.a.', color='#e6352b')
# axs0_twin.set_ylim(twin_ylim)
# axs0_twin.set_ylabel('Portfolio volatility')

# #AX1: EW portfolio
# axs[1].stackplot(df_ew_pos.index, df_ew_pos.T, colors=col_palette.values(),
#                  labels=df_ew_pos.columns)
# axs[1].stackplot(df_ew_neg.index, df_ew_neg.T, colors=col_palette.values())
# axs[1].set_ylabel('EW Portfolio\nrel. variance')
# axs[1].set_ylim(ylim)
# axs[1].set_axisbelow(True)
# axs[1].grid(True, which='major', linewidth=1)

# axs1_twin = axs[1].twinx()
# axs1_twin.plot(crsp_ew_riskContr.index, crsp_ew_riskContr['Portf_vola'],
#               label='Vola p.a.', color='#e6352b')
# axs1_twin.set_ylim(twin_ylim)
# axs1_twin.set_ylabel('Portfolio volatility')

# #AX2: EW-VW spread
# axs[2].stackplot(df_ewvw_pos.index, df_ewvw_pos.T, colors=col_palette.values(),
#                  labels=df_ewvw_pos.columns)
# axs[2].stackplot(df_ewvw_neg.index, df_ewvw_neg.T, colors=col_palette.values())
# axs[2].set_ylabel('EW-VW Spread\nrel. variance')
# axs[2].set_ylim(ylim)
# axs[2].set_axisbelow(True)
# axs[2].grid(True, which='major', linewidth=1)

# axs2_twin = axs[2].twinx()
# axs2_twin.plot(crsp_ewvw_riskContr.index, crsp_ewvw_riskContr['Portf_vola'],
#               label='Vola p.a.', color='#e6352b')
# axs2_twin.set_ylim(twin_ylim)
# axs2_twin.set_ylabel('Portfolio volatility')

# # format shared x-axis
# axs[2].set_xlim([df_ew_pos.index.min(), df_ew_pos.index.max()])
# years = dates.YearLocator(10, month=1, day=1)
# years1 = mc.OffsetYearLocator(2, month=1, day=1, offset=0)
# axs[2].xaxis.set_major_locator(years)
# axs[2].xaxis.set_minor_locator(years1)
# axs[2].xaxis.set_major_formatter(dates.DateFormatter('%Y'))
# axs[2].xaxis.set_minor_formatter(dates.DateFormatter('%y'))
# axs[2].get_xaxis().set_tick_params(which='major', pad=15)
# axs[2].set_xlabel('')
# plt.setp(axs[2].get_xmajorticklabels(), rotation=0, weight='bold', ha='center')

# # Create legend
# handles, labels = axs[2].get_legend_handles_labels()
# handles_twin, labels_twin = axs2_twin.get_legend_handles_labels()
# fig.legend(handles+handles_twin, labels+labels_twin, loc='lower center', ncol=6)

# fig.savefig(od +'Graphics\crsp_risk_decomp.png', bbox_inches='tight')

# =============================================================================



# =============================================================================
# Figure - SP500_risk_decomp.png
# =============================================================================
# fig, axs = plt.subplots(3,1, figsize=(12,8), sharey=False, sharex=True,
#                         gridspec_kw={'height_ratios': [1, 1, 1]})
# fig.subplots_adjust(wspace=0.3, hspace=0.1)

# # Prep risk contributions for plotting
# sp500_vw_relContr = sp500_vw_riskContr[['MKT','SMB','HML','Mom','Idiosyncratic']]
# df_vw_neg, df_vw_pos = sp500_vw_relContr.clip(upper=0), sp500_vw_relContr.clip(lower=0)
# sp500_ew_relContr = sp500_ew_riskContr[['MKT','SMB','HML','Mom','Idiosyncratic']]
# df_ew_neg, df_ew_pos = sp500_ew_relContr.clip(upper=0), sp500_ew_relContr.clip(lower=0)
# sp500_ewvw_relContr = sp500_ewvw_riskContr[['MKT','SMB','HML','Mom','Idiosyncratic']]
# df_ewvw_neg, df_ewvw_pos = sp500_ewvw_relContr.clip(upper=0), sp500_ewvw_relContr.clip(lower=0)

# ymin = min(df_vw_neg.sum(axis=1).min(), df_ew_neg.sum(axis=1).min(),
#            df_ewvw_neg.sum(axis=1).min())
# ymax = max(df_vw_pos.sum(axis=1).max(), df_ew_pos.sum(axis=1).max(),
#            df_ewvw_pos.sum(axis=1).max())
# ylim = [ymin - 0.02*(ymax-ymin), ymax + 0.02*(ymax-ymin)]
# twin_ylim = [-0.02, 0.32]

# #AX0: VW portfolio
# axs[0].stackplot(df_vw_pos.index, df_vw_pos.T, colors=col_palette.values(),
#                  labels=df_vw_pos.columns)
# axs[0].stackplot(df_vw_neg.index, df_vw_neg.T, colors=col_palette.values())
# axs[0].set_ylabel('VW Portfolio\nrel. variance')
# axs[0].set_ylim(ylim)
# axs[0].set_axisbelow(True)
# axs[0].grid(True, which='major', linewidth=1)

# axs0_twin = axs[0].twinx()
# axs0_twin.plot(sp500_vw_riskContr.index, sp500_vw_riskContr['Portf_vola'],
#               label='Vola p.a.', color='#e6352b')
# axs0_twin.set_ylim(twin_ylim)
# axs0_twin.set_ylabel('Portfolio volatility')

# #AX1: EW portfolio
# axs[1].stackplot(df_ew_pos.index, df_ew_pos.T, colors=col_palette.values(),
#                  labels=df_ew_pos.columns)
# axs[1].stackplot(df_ew_neg.index, df_ew_neg.T, colors=col_palette.values())
# axs[1].set_ylabel('EW Portfolio\nrel. variance')
# axs[1].set_ylim(ylim)
# axs[1].set_axisbelow(True)
# axs[1].grid(True, which='major', linewidth=1)

# axs1_twin = axs[1].twinx()
# axs1_twin.plot(sp500_ew_riskContr.index, sp500_ew_riskContr['Portf_vola'],
#               label='Vola p.a.', color='#e6352b')
# axs1_twin.set_ylim(twin_ylim)
# axs1_twin.set_ylabel('Portfolio volatility')

# #AX2: EW-VW spread
# axs[2].stackplot(df_ewvw_pos.index, df_ewvw_pos.T, colors=col_palette.values(),
#                  labels=df_ewvw_pos.columns)
# axs[2].stackplot(df_ewvw_neg.index, df_ewvw_neg.T, colors=col_palette.values())
# axs[2].set_ylabel('EW-VW Spread\nrel. variance')
# axs[2].set_ylim(ylim)
# axs[2].set_axisbelow(True)
# axs[2].grid(True, which='major', linewidth=1)

# axs2_twin = axs[2].twinx()
# axs2_twin.plot(sp500_ewvw_riskContr.index, sp500_ewvw_riskContr['Portf_vola'],
#               label='Vola p.a.', color='#e6352b')
# axs2_twin.set_ylim(twin_ylim)
# axs2_twin.set_ylabel('Portfolio volatility')

# # format shared x-axis
# axs[2].set_xlim([df_ew_pos.index.min(), df_ew_pos.index.max()])
# years = dates.YearLocator(10, month=1, day=1)
# years1 = dates.YearLocator(2, month=1, day=1)
# axs[2].xaxis.set_major_locator(years)
# axs[2].xaxis.set_minor_locator(years1)
# axs[2].xaxis.set_major_formatter(dates.DateFormatter('%Y'))
# axs[2].xaxis.set_minor_formatter(dates.DateFormatter('%y'))
# axs[2].get_xaxis().set_tick_params(which='major', pad=15)
# axs[2].set_xlabel('')
# plt.setp(axs[2].get_xmajorticklabels(), rotation=0, weight='bold', ha='center')

# # Create legend
# handles, labels = axs[2].get_legend_handles_labels()
# handles_twin, labels_twin = axs2_twin.get_legend_handles_labels()
# fig.legend(handles+handles_twin, labels+labels_twin, loc='lower center', ncol=7)

# fig.savefig(od +'Graphics\SP500_risk_decomp.png', bbox_inches='tight')

# =============================================================================



# 3) Development of market concentration

# =============================================================================
# Figure - SP_concentration.png
# =============================================================================
# fig, axs = plt.subplots(3,1, figsize=(12, 8), sharey=False, sharex=True,
#                         gridspec_kw={'height_ratios': [2, 1, 1]})
# fig.subplots_adjust(wspace=0.3, hspace=0.05) #defaults are 0.2 each
# axs[0].plot(conc_roll_mean.index, conc_roll_mean, label='5y mean', color=col_palette['grey'])
# axs[0].fill_between(conc_roll_mean.index, conc_roll_mean - conc_std/2,
#                     conc_roll_mean + conc_std/2, color=col_palette['grey'],
#                     alpha=.15)
# axs[0].plot(concSP500.index, concSP500, label='HHI', ls='-', color=col_palette['orange'])
# axs[0].grid(b=True, axis='both')
# axs[0].legend(loc='upper center', ncol = 2)
# axs[0].set_ylabel('Concentration - HHI')

# ml, sl, bl = axs[1].stem(annualDiff.index, annualDiff['EW_VW'], linefmt='grey',
#                          markerfmt='o', basefmt='grey', use_line_collection=True)
# plt.setp(ml, 'color', col_palette['teal'])
# axs[1].grid(b=True, axis='both')
# axs[1].set_ylabel('SPW-SPX p.a.')

# axs[2].plot(concSP500.index, concSP500 - conc_roll_mean, label='ConcToMean', 
#             color=col_palette['yellow'])
# axs[2].grid(b=True, which='both')
# axs[2].set_ylabel(r'HHI-mean')

# fig.savefig(od +'Graphics\\SP_concentration_5yrolling.png', bbox_inches='tight')

# =============================================================================




# =============================================================================
# Figure - FU_concentration.png
# =============================================================================
# fig, axs = plt.subplots(3,1, figsize=(12, 8), sharey=False, sharex=True,
#                         gridspec_kw={'height_ratios': [2, 2, 1]})
# fig.subplots_adjust(wspace=0.3, hspace=0.05) #defaults are 0.2 each

# axs[0].plot(spCumRet.index, spCumRet.loc[:,'univ_vw'], label='VW', ls='-')
# axs[0].plot(spCumRet.index, spCumRet.loc[:,'univ_ew'], label='EW', ls='-')
# axs[0].grid(b=True, axis='both')
# axs[0].legend(loc='upper center', ncol=2)
# axs[0].set_ylabel('Cumulative return [log scale]')
# axs[0].set_yscale('log')

# axs[1].plot(concFU.index, concFU, label='HHI', ls='-')
# axs[1].grid(b=True, axis='both')
# axs[1].set_ylabel('Concentration - HHI')

# axsTwin = axs[1].twinx()
# axsTwin.plot(concFU.index, mcap.loc[concFU.index].count(axis=1),
#               label='ConstCount', color='#EDC051')
# axsTwin.set_ylabel('Number of constituents in full universe')

# lines, labels = axs[1].get_legend_handles_labels()
# lines2, labels2 = axsTwin.get_legend_handles_labels()
# axs[1].legend(lines+lines2, labels+labels2, loc='upper center', ncol=2)

# axs[2].stem(annualDiff.index, annualDiff.loc[:,'EW_VW_FU'], linefmt='grey', use_line_collection=True)
# axs[2].grid(b=True, axis='both')
# axs[2].set_ylabel('EW-VW p.a.')

# fig.savefig(od +'Graphics\\FU_concentration.png', bbox_inches='tight')

# =============================================================================



# =============================================================================
# Figure - SP500_asset_concentration.png
# =============================================================================
# fig, axs = plt.subplots(3,1, figsize=(12,8), sharey=False, sharex=True,
#                         gridspec_kw={'height_ratios': [1, 1, 1]})
# fig.subplots_adjust(wspace=0.3, hspace=0.1)

# hhi_sp500 = pd.concat([calc_hhi(x).loc[regDat.index] for x in \
#                        [get_perc(mcapSP500), sp500_ew_weights, sp500_ewvw_weights]],
#                       axis=1).rename(columns={0:'vw', 1:'ew', 2:'ewvw'})

# ylim =[0,0.023]
    
# #AX0: VW portfolio
# axs[0].plot(hhi_sp500.index, hhi_sp500['vw'], color=col_palette['black'])
# axs[0].fill_between(hhi_sp500.index, hhi_sp500['vw'], color=col_palette['black'],
#                     alpha=.65)
# axs[0].set_ylabel('VW Portfolio\nHHI')
# axs[0].set_ylim(ylim)
# axs[0].set_axisbelow(True)
# axs[0].grid(True, which='major', linewidth=1)

# #AX1: EW portfolio
# axs[1].plot(hhi_sp500.index, hhi_sp500['ew'], color=col_palette['grey'])
# axs[1].fill_between(hhi_sp500.index, hhi_sp500['ew'], color=col_palette['grey'],
#                     alpha=.65)
# axs[1].set_ylabel('EW Portfolio\nHHI')
# axs[1].set_ylim(ylim)
# axs[1].set_axisbelow(True)
# axs[1].grid(True, which='major', linewidth=1)

# #AX2: EW-VW spread
# axs[2].plot(hhi_sp500.index, hhi_sp500['ewvw'], color=col_palette['orange'])
# axs[2].fill_between(hhi_sp500.index, hhi_sp500['ewvw'], color=col_palette['orange'],
#                     alpha=.65)
# axs[2].set_ylabel('EW-VW Spread\nHHI')
# axs[2].set_ylim(ylim)
# axs[2].set_axisbelow(True)
# axs[2].grid(True, which='major', linewidth=1)

# # format shared x-axis
# axs[2].set_xlim([df_pos.index.min(), df_pos.index.max()])
# years = dates.YearLocator(10, month=1, day=1)
# years1 = dates.YearLocator(2, month=1, day=1)
# axs[2].xaxis.set_major_locator(years)
# axs[2].xaxis.set_minor_locator(years1)
# axs[2].xaxis.set_major_formatter(dates.DateFormatter('%Y'))
# axs[2].xaxis.set_minor_formatter(dates.DateFormatter('%y'))
# axs[2].get_xaxis().set_tick_params(which='major', pad=15)
# axs[2].set_xlabel('')
# plt.setp(axs[2].get_xmajorticklabels(), rotation=0, weight='bold', ha='center')

# # Create legend
# handles, labels = axs[2].get_legend_handles_labels()
# fig.legend(handles, labels, loc='lower center', ncol=6)

# fig.savefig(od +'Graphics\SP500_asset_concentration.png', bbox_inches='tight')

# =============================================================================



# =============================================================================
# Figure - concentration_lag_relation.png
# =============================================================================
# fig, axs = plt.subplots(3,1, figsize=(12, 8), sharey=False, sharex=False,
#                         gridspec_kw={'height_ratios': [1, 1, 1]})
# fig.subplots_adjust(wspace=0.3, hspace=0.4) #defaults are 0.2 each

# pd.plotting.lag_plot(regDat['CIC'], ax=axs[0])
# axs[0].set_title('First lag plot')

# sm.graphics.tsa.plot_acf(regDat['CIC'], ax=axs[1], lags=30, zero=False,
#                          auto_ylims=True)
# axs[1].set_title('Autocorrelation plot')

# sm.graphics.tsa.plot_pacf(regDat['CIC'], ax=axs[2], method='ywm', lags=30,
#                           zero=False, auto_ylims=True)
# axs[2].set_title('PAC plot')

# fig.suptitle(r'Concentration lag relation')
# fig.savefig(od +'Graphics\\concentration_lag_relation.png', bbox_inches='tight')

# =============================================================================



# =============================================================================
# Figure - AR_model_prediction.png
# =============================================================================
# fig, axs = plt.subplots(2,1, figsize=(12, 8), sharey=False, sharex=True,
#                         gridspec_kw={'height_ratios': [1, 1]})
# fig.subplots_adjust(wspace=0.3, hspace=0.1) #defaults are 0.2 each

# axs[0].plot(predictedBeta, label=r'Predicted $\beta$')
# axs[0].plot(testData_beta, label=r'Tested $\beta$')
# axs[0].text(0, max(testData_beta), r'RMSE: {:.3f}'.format(beta_rmse), fontdict=font)
# axs[0].legend(loc='lower center', ncol = 2)

# axs[1].plot(predictedAlpha, label=r'Predicted $\alpha$')
# axs[1].plot(testData_alpha, label=r'Tested $\alpha$')
# axs[1].text(0, max(testData_alpha), r'RMSE: {:.3f}'.format(alpha_rmse), fontdict=font)
# axs[1].legend(loc='lower center', ncol = 2)

# fig.suptitle(r'Predicted model parameters for AR({:})'.format(lags))
# fig.savefig(od + r'Graphics\\AR_{:}_model_prediction.png'.format(lags), bbox_inches='tight')

# =============================================================================



# =============================================================================
# Figure - Smoothed_concentration.png
# =============================================================================
# fig, axs = plt.subplots(3,1, figsize=(12, 8), sharey=False, sharex=True,
#                         gridspec_kw={'height_ratios': [1, 1, 1]})
# fig.subplots_adjust(wspace=0.3, hspace=0.3) #defaults are 0.2 each

# axs[0].plot(concSP500.index, concSP500, label='HHI', ls='-')
# axs[0].grid(b=True, axis='both')
# axs[0].set_ylabel('Concentration - HHI')
# axs[0].set_title('Original concentration')

# axs[1].plot(concSP500.index, conc_loess_5, label=r'Lowess smoothed 5%')
# axs[1].grid(b=True, axis='both')
# axs[1].set_ylabel('Concentration - HHI')
# axs[1].set_title('Lowess smoothed 5%')

# axs[2].plot(concSP500.index, conc_loess_2_5, label=r'Lowess smoothed 2.5%')
# axs[2].grid(b=True, axis='both')
# axs[2].set_ylabel('Concentration - HHI')
# axs[2].set_title('Lowess smoothed 2.5%')

# fig.suptitle(r'Market concentration')
# fig.savefig(od + r'Graphics\\smoothed_concentration', bbox_inches='tight')

# =============================================================================




# =============================================================================
# Figure - AR_model_prediction_conditional_concentration.png
# =============================================================================





# =============================================================================





# =============================================================================
# Figure - FullUniverse_extendedPeriod.png
# ============================================================================
# fig, axs = plt.subplots(4,1, figsize=(12, 8), sharey=False, sharex=True,
#                         gridspec_kw={'height_ratios': [2, 1, 1, 1]})
# fig.subplots_adjust(wspace=0.3, hspace=0.05) #defaults are 0.2 each
# axs[0].plot(spCumRet.index, spCumRet.loc[:,'vw_ret'], label='VW', ls='-')
# axs[0].plot(spCumRet.index, spCumRet.loc[:,'ew_ret'], label='EW', ls='-')
# axs[0].grid(b=True, axis='both')
# axs[0].legend(loc='upper center', ncol=2)
# axs[0].set_ylabel('Cumulative return [log scale]')
# axs[0].set_yscale('log')

# axs[1].stem(annualDiff.index, annualDiff.loc[:,'EW_VW_FU'], linefmt='grey', use_line_collection=True)
# axs[1].grid(b=True, axis='both')
# axs[1].set_ylabel('EW-VW p.a.')

# axs[2].fill_between(SIR_fullUniverse.index, SIR_fullUniverse.loc[:,'beta_lcb'],
#    SIR_fullUniverse.loc[:,'beta_ucb'], color='r', alpha=.15, label='95% CI') 
# axs[2].plot(SIR_fullUniverse.index, SIR_fullUniverse.loc[:,'beta'],
#    linewidth=2, color='r', label=r'EW $\beta$') 
# axs[2].axhline(y=1, color='black')
# axs[2].grid(b=True, which='both')
# axs[2].legend(loc='lower center', ncol = 2)
# axs[2].set_ylabel(r'Estimated $\beta$')

# axs[3].fill_between(SIR_fullUniverse.index, SIR_fullUniverse.loc[:,'alpha_lcb'],
#    SIR_fullUniverse.loc[:,'alpha_ucb'], color='r', alpha=.15, label='95% CI') 
# axs[3].plot(SIR_fullUniverse.index, SIR_fullUniverse.loc[:,'alpha'],
#    linewidth=2, color='b', label=r'EW $\alpha$') 
# axs[3].axhline(y=0, color='black')
# axs[3].grid(b=True, which='both')
# axs[3].legend(loc='lower center', ncol = 2)
# axs[3].set_ylabel(r'Estimated $\alpha$')

# fig.suptitle('Full universe portfolios', fontweight='bold')
# fig.savefig(od +'Graphics\\FullUniverse_extendedPeriod.png', bbox_inches='tight')

# =============================================================================



# =============================================================================
# Figure - SPvsFullUniverse_extendedPeriod.png
# ============================================================================
# fig, axs = plt.subplots(4,1, figsize=(12, 8), sharey=False, sharex=True,
#                         gridspec_kw={'height_ratios': [2, 1, 1, 1]})
# fig.subplots_adjust(wspace=0.3, hspace=0.05) #defaults are 0.2 each
# axs[0].plot(spCumRet.index, spCumRet.loc[:,'MKT'], label='MKT', ls='-')
# axs[0].plot(spCumRet.index, spCumRet.loc[:,'ewretd'], label='SPW', ls='-')
# axs[0].plot(spCumRet.index, spCumRet.loc[:,'vwretd'], label='SPX', ls='-')
# axs[0].grid(b=True, axis='both')
# axs[0].legend(loc='upper center', ncol=2)
# axs[0].set_ylabel('Cumulative return [log scale]')
# axs[0].set_yscale('log')

# axs[1].stem(annualDiff.index, annualDiff.loc[:,'EW_VW'], linefmt='grey', use_line_collection=True)
# axs[1].grid(b=True, axis='both')
# axs[1].set_ylabel('SPW-SPX p.a.')

# axs[2].fill_between(SIR_SPvsFullUniverse.index, SIR_SPvsFullUniverse.loc[:,'beta_lcb'],
#    SIR_SPvsFullUniverse.loc[:,'beta_ucb'], color='r', alpha=.15, label='95% CI') 
# axs[2].plot(SIR_SPvsFullUniverse.index, SIR_SPvsFullUniverse.loc[:,'beta'],
#    linewidth=2, color='r', label=r'SPW $\beta$') 
# axs[2].axhline(y=1, color='black')
# axs[2].grid(b=True, which='both')
# axs[2].legend(loc='lower center', ncol = 2)
# axs[2].set_ylabel(r'Estimated $\beta$')

# axs[3].fill_between(SIR_SPvsFullUniverse.index, SIR_SPvsFullUniverse.loc[:,'alpha_lcb'],
#    SIR_SPvsFullUniverse.loc[:,'alpha_ucb'], color='r', alpha=.15, label='95% CI') 
# axs[3].plot(SIR_SPvsFullUniverse.index, SIR_SPvsFullUniverse.loc[:,'alpha'],
#    linewidth=2, color='b', label=r'SPW $\alpha$') 
# axs[3].axhline(y=0, color='black')
# axs[3].grid(b=True, which='both')
# axs[3].legend(loc='lower center', ncol = 2)
# axs[3].set_ylabel(r'Estimated $\alpha$')

# fig.suptitle('SPW against full market', fontweight='bold')
# fig.savefig(od +'Graphics\\SPvsFullUniverse_extendedPeriod.png', bbox_inches='tight')

# =============================================================================



# =============================================================================
# Figure - beta_lag_relation.png
# =============================================================================
# fig, axs = plt.subplots(3,1, figsize=(12, 8), sharey=False, sharex=False,
#                         gridspec_kw={'height_ratios': [1, 1, 1]})
# fig.subplots_adjust(wspace=0.3, hspace=0.4) #defaults are 0.2 each

# pd.plotting.lag_plot(singleIndexResults['beta'], ax=axs[0])
# axs[0].set_title('First lag plot')

# sm.graphics.tsa.plot_acf(singleIndexResults['beta'], ax=axs[1], lags=10)
# axs[1].set_title('Autocorrelation plot')

# sm.graphics.tsa.plot_pacf(singleIndexResults['beta'], ax=axs[2], method='ywm', lags=10)
# axs[2].set_title('PAC plot')

# fig.suptitle(r'$\beta$ lag relation')
# fig.savefig(od +'Graphics\\beta_lag_relation.png', bbox_inches='tight')

# =============================================================================


# =============================================================================
# Figure - alpha_lag_relation.png
# =============================================================================
# fig, axs = plt.subplots(3,1, figsize=(12, 8), sharey=False, sharex=False,
#                         gridspec_kw={'height_ratios': [1, 1, 1]})
# fig.subplots_adjust(wspace=0.3, hspace=0.4) #defaults are 0.2 each

# pd.plotting.lag_plot(singleIndexResults['alpha'], ax=axs[0])
# axs[0].set_title('First lag plot')

# #pd.plotting.autocorrelation_plot(singleIndexResults['alpha'], ax=axs[1])
# sm.graphics.tsa.plot_acf(singleIndexResults['alpha'], ax=axs[1], lags=10)
# axs[1].set_title('Autocorrelation plot')

# sm.graphics.tsa.plot_pacf(singleIndexResults['alpha'], ax=axs[2], method='ywm', lags=10)
# axs[2].set_title('PAC plot')

# fig.suptitle(r'$\alpha$ lag relation')
# fig.savefig(od +'Graphics\\alpha_lag_relation.png', bbox_inches='tight')

# =============================================================================



# # =============================================================================
# # Figure - factorDecompositionOverTime.png
# # ============================================================================
# fig, axs = plt.subplots(5,1, figsize=(12, 8), sharey=False, sharex=True,
#                         gridspec_kw={'height_ratios': [1, 1, 1, 1, 1]})
# fig.subplots_adjust(wspace=0.3, hspace=0.2) #defaults are 0.2 each

# ml, sl, bl = axs[0].stem(annualDiff.index, annualDiff['EW_VW'], linefmt='grey',
#                          markerfmt='o', basefmt='grey', use_line_collection=True)
# plt.setp(ml, 'color', col_palette['teal'])
# axs[0].grid(b=True, axis='both')
# axs[0].set_ylabel('EW-VW p.a.')

# idx = fmr_df.index.get_level_values('date').unique()

# axs[1].fill_between(idx, fmr_df.xs('MKT', level=1).loc[:,'lb'],
#    fmr_df.xs('MKT', level=1).loc[:,'ub'], color=col_palette['black'],
#    alpha=.15, label='95% CI') 
# axs[1].plot(idx, fmr_df.xs('MKT', level=1).loc[:,'coef'],
#    linewidth=2, color=col_palette['black'], label=r'$\beta_{MKT}$') 
# axs[1].axhline(y=0, color='black')
# axs[1].set_ylim([-0.75, 0.75])
# axs[1].grid(b=True, which='both')
# axs[1].legend(loc='lower center', ncol = 2)
# axs[1].set_ylabel(r'MKT')

# axs[2].fill_between(idx, fmr_df.xs('SMB', level=1).loc[:,'lb'],
#    fmr_df.xs('SMB', level=1).loc[:,'ub'], color=col_palette['grey'], alpha=.15,
#    label='95% CI') 
# axs[2].plot(idx, fmr_df.xs('SMB', level=1).loc[:,'coef'],
#    linewidth=2, color=col_palette['grey'], label=r'$\beta_{SMB}$') 
# axs[2].axhline(y=0, color='black')
# axs[2].set_ylim([-0.75, 0.75])
# axs[2].grid(b=True, which='both')
# axs[2].legend(loc='lower center', ncol = 2)
# axs[2].set_ylabel(r'SMB')

# axs[3].fill_between(idx, fmr_df.xs('HML', level=1).loc[:,'lb'],
#    fmr_df.xs('HML', level=1).loc[:,'ub'], color=col_palette['teal'],
#    alpha=.15, label='95% CI') 
# axs[3].plot(idx, fmr_df.xs('HML', level=1).loc[:,'coef'],
#    linewidth=2, color=col_palette['teal'], label=r'$\beta_{HML}$') 
# axs[3].axhline(y=0, color='black')
# axs[3].set_ylim([-0.75, 0.75])
# axs[3].grid(b=True, which='both')
# axs[3].legend(loc='lower center', ncol = 2)
# axs[3].set_ylabel(r'HML')

# axs[4].fill_between(idx, fmr_df.xs('CIC', level=1).loc[:,'lb'],
#    fmr_df.xs('CIC', level=1).loc[:,'ub'], color=col_palette['orange'],
#    alpha=.15, label='95% CI') 
# axs[4].plot(idx, fmr_df.xs('CIC', level=1).loc[:,'coef'],
#    linewidth=2, color=col_palette['orange'], label=r'$\beta_{CIC}$') 
# axs[4].axhline(y=0, color='black')
# axs[4].set_ylim([-0.75, 0.75])
# axs[4].grid(b=True, which='both')
# axs[4].legend(loc='upper center', ncol = 2)
# axs[4].set_ylabel(r'CIC')

# fig.suptitle('Estimated factor loadings', fontweight='bold')
# fig.savefig(od +'Graphics\\FactorDecomposition.png', bbox_inches='tight')

# =============================================================================




# =============================================================================
# Figure - CRSP_sector_decomp.png
# =============================================================================
# fig, axs = plt.subplots(3,1, figsize=(12,8), sharey=False, sharex=True,
#                         gridspec_kw={'height_ratios': [1, 1, 1]})
# fig.subplots_adjust(wspace=0.3, hspace=0.1)

# vw_weights, ew_weights, ewvw_weights = \
#     [x.loc[regDat.index] for x in sector_weights_list_CRSP]

# ylim = [-0.05, 1.05]

# #AX0: VW portfolio
# axs[0].stackplot(vw_weights.index, vw_weights.drop(columns='HHI').T, 
#                  labels=vw_weights.columns, colors=col_gics_alt)
# axs[0].set_ylabel('VW Portfolio\nsector decomposition')
# axs[0].set_ylim(ylim)
# axs[0].set_axisbelow(True)
# axs[0].grid(True, which='major', linewidth=1)

# axs0_twin = axs[0].twinx()
# axs0_twin.plot(vw_weights.index, vw_weights['HHI'], label='HHI', color='#e6352b')
# axs0_twin.set_ylabel('Concentration HHI')

# #AX0: EW portfolio
# axs[1].stackplot(ew_weights.index, ew_weights.drop(columns='HHI').T,
#                  labels=ew_weights.columns, colors=col_gics_alt)
# axs[1].set_ylabel('EW Portfolio\nsector decomposition')
# axs[1].set_ylim(ylim)
# axs[1].set_axisbelow(True)
# axs[1].grid(True, which='major', linewidth=1)

# axs1_twin = axs[1].twinx()
# axs1_twin.plot(ew_weights.index, ew_weights['HHI'], label='HHI', color='#e6352b')
# axs1_twin.set_ylabel('Concentration HHI')

# #AX2: EW-VW spread
# axs[2].stackplot(ewvw_weights.index, ewvw_weights.drop(columns='HHI').T, 
#                  labels=ewvw_weights.columns, colors=col_gics_alt)
# axs[2].set_ylabel('EW-VW Spread\nsector decomposition')
# #axs[2].set_ylim(ylim)
# axs[2].set_axisbelow(True)
# axs[2].grid(True, which='major', linewidth=1)

# axs2_twin = axs[2].twinx()
# axs2_twin.plot(ewvw_weights.index, ewvw_weights['HHI'], label='HHI', color='#e6352b')
# axs2_twin.set_ylabel('Concentration HHI')

# # format shared x-axis
# axs[2].set_xlim([ew_weights.index.min(), ew_weights.index.max()])
# years = dates.YearLocator(10, month=1, day=1)
# years1 = dates.YearLocator(2, month=1, day=1)
# axs[2].xaxis.set_major_locator(years)
# axs[2].xaxis.set_minor_locator(years1)
# axs[2].xaxis.set_major_formatter(dates.DateFormatter('%Y'))
# axs[2].xaxis.set_minor_formatter(dates.DateFormatter('%y'))
# axs[2].get_xaxis().set_tick_params(which='major', pad=15)
# axs[2].set_xlabel('')
# plt.setp(axs[2].get_xmajorticklabels(), rotation=0, weight='bold', ha='center')

# # Create legend
# handles, labels = axs[2].get_legend_handles_labels()
# handles_twin, labels_twin = axs2_twin.get_legend_handles_labels()
# fig.legend(handles+handles_twin, labels+labels_twin, loc='upper center', ncol=6)

# fig.savefig(od +'Graphics\CRSP_sector_decomp.png', bbox_inches='tight')

# =============================================================================


# def rel_sector_contr(data):
#     sec_weights = data.drop(columns='HHI')
#     sec_weights = sec_weights.divide(sec_weights.abs().sum(axis=1), axis=0)
    
#     return sec_weights.join(data['HHI'])

# # =============================================================================
# # Figure - SP500_sector_decomp.png
# # =============================================================================
# fig, axs = plt.subplots(3,1, figsize=(12,8), sharey=False, sharex=True,
#                         gridspec_kw={'height_ratios': [1, 1, 1]})
# fig.subplots_adjust(wspace=0.3, hspace=0.1)

# vw_weights, ew_weights, ewvw_weights = \
#     [rel_sector_contr(x.loc[regDat.index]) for x in sector_weights_list_SP500]

# ewvw_neg, ewvw_pos = ewvw_weights.clip(upper=0), ewvw_weights.clip(lower=0)

# ylim = [-0.05, 1.05]

# #AX0: VW portfolio
# axs[0].stackplot(vw_weights.index, vw_weights.drop(columns='HHI').T, 
#                  labels=vw_weights.columns, colors=col_gics_alt)
# axs[0].set_ylabel('VW Portfolio\nsector decomposition')
# axs[0].set_ylim(ylim)
# axs[0].set_axisbelow(True)
# axs[0].grid(True, which='major', linewidth=1)

# axs0_twin = axs[0].twinx()
# axs0_twin.plot(vw_weights.index, vw_weights['HHI'], label='HHI', color='#e6352b')
# axs0_twin.set_ylabel('Concentration HHI')

# #AX0: EW portfolio
# axs[1].stackplot(ew_weights.index, ew_weights.drop(columns='HHI').T,
#                  labels=ew_weights.columns, colors=col_gics_alt)
# axs[1].set_ylabel('EW Portfolio\nsector decomposition')
# axs[1].set_ylim(ylim)
# axs[1].set_axisbelow(True)
# axs[1].grid(True, which='major', linewidth=1)

# axs1_twin = axs[1].twinx()
# axs1_twin.plot(ew_weights.index, ew_weights['HHI'], label='HHI', color='#e6352b')
# axs1_twin.set_ylabel('Concentration HHI')

# #AX2: EW-VW spread
# axs[2].stackplot(ewvw_neg.index, ewvw_neg.drop(columns='HHI').T, 
#                  labels=ewvw_weights.columns, colors=col_gics_alt)
# axs[2].stackplot(ewvw_pos.index, ewvw_pos.drop(columns='HHI').T, 
#                  colors=col_gics_alt)
# axs[2].set_ylabel('EW-VW Spread\nsector decomposition')
# #axs[2].set_ylim(ylim)
# axs[2].set_axisbelow(True)
# axs[2].grid(True, which='major', linewidth=1)

# axs2_twin = axs[2].twinx()
# axs2_twin.plot(ewvw_weights.index, ewvw_weights['HHI'], label='HHI', color='#e6352b')
# axs2_twin.set_ylabel('Concentration HHI')

# # format shared x-axis
# axs[2].set_xlim([ew_weights.index.min(), ew_weights.index.max()])
# years = dates.YearLocator(10, month=1, day=1)
# years1 = dates.YearLocator(2, month=1, day=1)
# axs[2].xaxis.set_major_locator(years)
# axs[2].xaxis.set_minor_locator(years1)
# axs[2].xaxis.set_major_formatter(dates.DateFormatter('%Y'))
# axs[2].xaxis.set_minor_formatter(dates.DateFormatter('%y'))
# axs[2].get_xaxis().set_tick_params(which='major', pad=15)
# axs[2].set_xlabel('')
# plt.setp(axs[2].get_xmajorticklabels(), rotation=0, weight='bold', ha='center')

# # Create legend
# handles, labels = axs[2].get_legend_handles_labels()
# handles_twin, labels_twin = axs2_twin.get_legend_handles_labels()
# fig.legend(handles+handles_twin, labels+labels_twin, loc='upper center', ncol=6)

# fig.savefig(od +'Graphics\SP500_sector_decomp.png', bbox_inches='tight')

# =============================================================================



# =============================================================================
# # Table - single_index_model_CRSP.txt
# # =============================================================================

# portf_names = ['univ_vw', 'univ_ew', 'EW_VW_FU_RF']
# sim_results = [pd.DataFrame(regDat.groupby(grouper).apply(
#     estimate_CAPM, 'RF', 'MKT_noexc', k, alpha_inc=True, hyp=hyp).tolist(),
#     columns=SIR_columns, index=grouper.categories)\
#         for k,hyp in zip(portf_names, ['x1=1','x1=1','x1=0'])]

# sim_table = pd.concat(sim_results, keys=portf_names, names=['portf', 'date']).\
#     sort_index(level=1, sort_remaining=False)

# sim_results_fp = [pd.DataFrame(estimate_CAPM(regDat, 'RF', 'MKT_noexc', k, alpha_inc=True, hyp=hyp),
#     index=SIR_columns, columns=[k])\
#     for k,hyp in zip(portf_names, ['x1=1','x1=1','x1=0'])]

# sim_results_fp = pd.concat(sim_results_fp, axis=1).T

# print('-----------SIM results for CRSP------------')
# idx = ['1963--2020']*3 + ['1963--1970']*3 + ['1971--1980']*3 + ['1981--1990']*3 + \
#     ['1991--2000']*3 + ['2001--2010']*3 + ['2011--2020']*3
# hds = [r'$\alpha$', r'$\alpha$ t-stat', r'$\beta', r'$\beta t-stat', r'$R^2$']
# fmt = [None, None, '.2f', '.2f', '.2f', '.2f', '.2f']
# table_sim = np.concatenate((sim_results_fp.iloc[:,:5].reset_index().values,
#                         sim_table.iloc[:,:5].reset_index(level='portf').values),
#                        axis=0)
                        
# #print(tabulate(table, headers=hds, showindex=idx, floatfmt=fmt)) #tablefmt='latex_raw'
# #Save as txt-file
# yfile = od + 'single_index_model_CRSP.txt'
# with open(yfile, 'w') as f:
#     f.write(tabulate(table_sim, headers=hds, showindex=idx, floatfmt=fmt ,tablefmt='latex_booktabs'
#                      ))
# # =============================================================================



# # =============================================================================
# # Table - single_index_model_combined.txt
# # =============================================================================


# portf_names = ['vwretd', 'ewretd', 'EW_VW_RF']
# sim_results_sp = [pd.DataFrame(regDat.groupby(grouper).apply(
#     estimate_CAPM, 'RF', 'MKT_noexc', k, alpha_inc=True, hyp=hyp).tolist(),
#     columns=SIR_columns, index=grouper.categories)\
#         for k,hyp in zip(portf_names, ['x1=1','x1=1','x1=0'])]

# sim_table_sp = pd.concat(sim_results_sp, keys=portf_names, names=['portf', 'date']).\
#     sort_index(level=1, sort_remaining=False)

# sim_results_fp_sp = [pd.DataFrame(estimate_CAPM(regDat, 'RF', 'MKT_noexc', k,
#                                                 alpha_inc=True, hyp=hyp),
#     index=SIR_columns, columns=[k])\
#     for k,hyp in zip(portf_names, ['x1=1','x1=1','x1=0'])]

# sim_results_fp_sp = pd.concat(sim_results_fp_sp, axis=1).T

# print('-----------SIM results for CRSP------------')
# idx = ['1963--2020']*3 + ['1963--1970']*3 + ['1971--1980']*3 + ['1981--1990']*3 + \
#     ['1991--2000']*3 + ['2001--2010']*3 + ['2011--2020']*3
# hds = [r'$\alpha$', r'$\alpha$ t-stat', r'$\beta', r'$\beta t-stat', r'$R^2$']*2
# fmt = [None, None] + ['.2f']*10
# table = np.concatenate((sim_results_fp_sp.iloc[:,:5].reset_index().values,
#                         sim_table_sp.iloc[:,:5].reset_index(level='portf').values),
#                        axis=0)
# table_combined = np.concatenate((table_sim, table[:,1:]), axis=1)
                        
# #print(tabulate(table, headers=hds, showindex=idx, floatfmt=fmt)) #tablefmt='latex_raw'
# #Save as txt-file
# yfile = od + 'single_index_model_combined.txt'
# with open(yfile, 'w') as f:
#     f.write(tabulate(table_combined, headers=hds, showindex=idx, floatfmt=fmt,
#                      tablefmt='latex_booktabs'))
# # =============================================================================
