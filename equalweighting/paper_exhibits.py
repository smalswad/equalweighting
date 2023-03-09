# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 20:50:04 2023

@author: Alexander Swade
"""
from tabulate import tabulate

import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

from quantfinlib.model.factormodel import CAPM
from quantfinlib.plots.correlation import plot_correlation
from quantfinlib.portfolio.measure import calmar_ratio
from quantfinlib.portfolio.measure import expected_shortfall
from quantfinlib.portfolio.measure import max_drawdown
from quantfinlib.portfolio.measure import sortino_ratio
from quantfinlib.statistic.regression import ols_to_table
from quantfinlib.statistic.tests import tstat
from quantfinlib.utils.list import combine_alternating

def generate_exhibits(direc, configs, res):
    
    # Unpack regression data only
    reg_dat = res['reg_dat']
    
    # Define specific periods    
    period_idx = [reg_dat.index, #FullSample
                  reg_dat.loc[reg_dat['jan_dummy']==1].index,
                  reg_dat.loc[reg_dat['nonjan_dummy']==1].index,
                  reg_dat.loc['1957-07-01':'1983-12-31'].index, #Pre-publication
                  reg_dat.loc['1984-01-01':'1999-12-31'].index, #Post-publication
                  reg_dat.loc['2000-01-01':'2009-12-31'].index, #Pre-GFC
                  reg_dat.loc['2010-01-01':'2021-12-31'].index] #Post-GFC
    
    # Generate all exhbits
    exhibit_1(direc, res['cum_ret'], configs['col_palette'])
    exhibit_2(direc, reg_dat, res['descr_data'], period_idx)
    exhibit_3(direc, reg_dat)
    exhibit_4(direc, reg_dat)
    exhibit_5(direc, reg_dat, period_idx)
    exhibit_6(direc, reg_dat, res['turnover'])
    exhibit_7(direc, reg_dat, res['bbg'])

def exhibit_1(direc, cum_ret, col_palette):
    '''    
    Function to generate Exhibit 1 of the final paper

    Parameters
    ----------
    direc : str
        Directory to save results to.
    cum_ret : pd.DataFrame, shape(T,4)
        Cumulative returns of CRSP and SP500 portfolios (each as VW and EW).
    col_palette : dict
        Color palette.

    Returns
    -------
    None.

    '''
    if direc[-1] != '/':
        direc = direc + '/'
        
    fig, axs = plt.subplots(1,1, figsize=(12, 8))
    fig.subplots_adjust(wspace=0.3, hspace=0.05) #defaults are 0.2 each
    
    
    axs.plot(cum_ret.index, cum_ret.loc[:,'vwretd'], label='SPX', ls='--', 
                color=col_palette['black'])
    axs.plot(cum_ret.index, cum_ret.loc[:,'ewretd'], label='SPW', ls='-',
                color=col_palette['black'])
    axs.plot(cum_ret.index, cum_ret.loc[:,'univ_vw'], label='VW', ls='--',
                color=col_palette['orange'])
    axs.plot(cum_ret.index, cum_ret.loc[:,'univ_ew'], label='EW', ls='-',
                color=col_palette['orange'])
    axs.grid(visible=True, axis='both')
    axs.legend(loc='upper center', ncol=2)
    axs.set_ylabel('Cumulative return [log scale]')
    axs.set_yscale('log')

    #fig.suptitle('Historical performance of CRSP portfolios', fontweight='bold')
    fig.savefig(direc +'exhibit1_cumRet.png', bbox_inches='tight')

def exhibit_2(direc, reg_dat, descr_data, period_idx):
    '''
    Function to generate Exhibit 2 of the final paper

    Parameters
    ----------
    direc : str
        Directory to save results to.
    reg_dat : pd.DataFrame, shape(T,k)
        Regression data (i.e. mostly factor and portfolio returns).
    descr_data : pd.DataFrame with Multiindex
        Descriptive data used to calculate some descriptives.
    period_idx : list of pd.index
        List of different subsamples.

    Returns
    -------
    None.

    '''
    if direc[-1] != '/':
        direc = direc + '/'
    
    # ------------- PANEL A + B: DESCRIPTIVES -----------------    
    
    pnames = ['0_FullSample','1_January','2_Non-Januaray','3_Pre-publication',
          '4_Post-publication','5_Pre-GFC', '6_Post-GFC']
    portf_names = ['univ_vw','univ_ew','vwretd','ewretd','EW_VW_FU','EW_VW',
                   'RF', 'MKT']
    diff_dict = {'univ_vw':'EW_VW_FU', 'univ_ew':'EW_VW_FU', 'vwretd':'EW_VW',
                 'ewretd':'EW_VW'}
          
    idx_slice = pd.IndexSlice
    descrip_list = list()
    
    for i, pidx in enumerate(period_idx):
        sample_data = descr_data.loc[pidx, idx_slice[:, portf_names]]
        descrip = calc_descrip_stats(
            sample_data, freq='m', inclCAPM=False, diff_dict=diff_dict). \
            reset_index(level='portf')
        descrip['universe'] = ['CRSP']*2 + ['SP500']*2
        descrip['sample'] = pnames[i]
        descrip_list.append(descrip)
    
    desrc_tab = pd.concat(descrip_list, ignore_index=True)      
    desrc_tab.replace({'univ_vw':'0_VW', 'univ_ew':'1_EW','vwretd':'0_VW',
                       'ewretd':'1_EW'}, inplace=True)
    desrc_tab = desrc_tab.pivot(index=['universe','sample'],
                                columns='portf').reset_index(level='sample')
    
    print('-----------SUMMARY STATS------------')
    idx = ['CRSP']*7 + ['SP500']*7
    hds = ['sample'] \
        + [i + '_' + p for i in ['Ret pa', 'Std pa', 'Sharpe', 'MaxDD1m'] \
            for p in ['VW','EW']]\
        + ['avgMcap[$Mrd]', 'avgConst']\
        + ['diff_ret','diff_t-stat']         
    fmt = [None]*2 + ['.1f']*4 + ['.2f']*2 + ['.1f']*3 + ['.0f'] + ['.2f']*2
    table = np.delete(desrc_tab.values, [9,11,13,15] , axis=1)

    #Save as txt-file
    yfile = direc + 'exhibit2_descripStats.txt'
    with open(yfile, 'w') as f:
        f.write(tabulate(table, headers=hds, showindex=idx, floatfmt=fmt,
                         tablefmt='latex_booktabs'))

    # ------------- PANEL C: SIM RESULTS -----------------
    
    sir_columns = ['alpha', 'alpha_t', 'beta', 'beta_t', 'rsqr', 'beta_lcb',
                   'beta_ucb', 'alpha_lcb', 'alpha_ucb']
    sim_crsp = ['univ_vw', 'univ_ew'] + ['EW_VW_FU_RF']*5
    sim_sp500 = ['vwretd','ewretd'] + ['EW_VW_RF']*5
    sim_idx = [period_idx[0]]*3 + period_idx[3:]
    pnames = ['0_FullSample','1_FullSample','2_FullSample','3_Pre-publication',
              '4_Post-publication', '5_Pre-GFC', '6_Post-GFC']
    
    sim_crsp_results = pd.concat([
        pd.DataFrame(estimate_CAPM(reg_dat.loc[idx],'RF', 'MKT_noexc', k,
                                   alpha_inc=True, hyp=hyp),
                     index=sir_columns,
                     columns=[name]).T for k, hyp, idx, name in \
            zip(sim_crsp, ['x1=1']*2 + ['x1=0']*5, sim_idx, pnames)])
    
    sim_sp500_results = pd.concat([
        pd.DataFrame(estimate_CAPM(reg_dat.loc[idx],'RF', 'MKT_noexc', k,
                                   alpha_inc=True, hyp=hyp),
                     index=sir_columns,
                     columns=[name]).T for k, hyp, idx, name in \
            zip(sim_sp500, ['x1=1']*2 + ['x1=0']*5, sim_idx, pnames)])
        
    sim_table = pd.concat([sim_crsp_results, sim_sp500_results], axis=1,
                          keys=['CRSP','SP500'])
    
    print('-----------Panel 3: SIM results------------')
    hds = [univ+ k for univ, k in itertools.product(
        [r'CRSP ',r'SP500 '], [r'alpha', r'$t(alpha)', r'beta',
                           r't(beta)', r'$R^2$'])]
    fmt = [None] + ['.2f']*10
    sim_tab = sim_table.loc[:, idx_slice[:,['alpha','alpha_t','beta','beta_t',
                                            'rsqr']]].values
                            
    #Save as txt-file
    yfile = direc + 'exhibit2_single_index_model.txt'
    with open(yfile, 'w') as f:
        f.write(tabulate(sim_tab, headers=hds, showindex=pnames, floatfmt=fmt,
                         tablefmt='latex_booktabs'))

def exhibit_3(direc, reg_dat):
    '''
    Function to generate Exhibit 3 of the final paper.

    Parameters
    ----------
    direc : str
        Directory to save results to.
    reg_dat : pd.DataFrame, shape(T,k)
        Regression data (i.e. mostly factor and portfolio returns).

    Returns
    -------
    None.

    '''
    if direc[-1] != '/':
        direc = direc + '/'
        
    plot_correlation(
        reg_dat[['EW_VW_FU','EW_VW','MKT', 'MKT_lagged', 'SMB', 'HML', 'Mom',
                 'ST_Rev','RMW','CMA','QMJ','low_vol', 'R_ME','R_IA', 'R_ROE',
                 'R_EG']],
        cbar=False, 
        labels=['EW-VW', 'SPW-SPX', 'MKT', 'MKT$_{t-1}$', 'SMB', 'HML', 'WML',
                'STR', 'RMW','CMA', 'QMJ','VOL','ME','IA','ROE','EG'],
        filepath=direc,
        filename='exhibit3_factor_correlations.png')

def exhibit_4(direc, reg_dat):
    '''
    Function to generate Exhibit 4 of the final paper.

    Parameters
    ----------
    direc : str
        Directory to save results to.
    reg_dat : pd.DataFrame, shape(T,k)
        Regression data (i.e. mostly factor and portfolio returns).

    Returns
    -------
    None.

    '''    
    if direc[-1] != '/':
        direc = direc + '/'
    
    reg_order = ['Intercept','jan_dummy','EW_VW_FU','EW_VW','MKT','MKT_lagged',
                 'SMB','SMBSP','HML','Mom','RMW','CMA','ST_Rev','QMJ','low_vol', 
                 'R_ME','R_IA','R_ROE','R_EG']
    
    # Multi-factor model
    regList1 = ['EW_VW_FU ~ MKT + MKT_lagged',
                'EW_VW_FU ~ SMB', 
                'EW_VW_FU ~ MKT + MKT_lagged + SMB + HML + RMW + CMA',
                'EW_VW_FU ~ MKT + MKT_lagged + R_ME + R_IA + R_ROE + R_EG',
                'EW_VW_FU ~ MKT + MKT_lagged + SMB + HML + QMJ',
                'EW_VW_FU ~ MKT + MKT_lagged + SMB + HML + RMW + CMA + Mom + ST_Rev + low_vol',   
                'EW_VW_FU ~ jan_dummy + MKT + MKT_lagged + SMB + HML + RMW + CMA + Mom + ST_Rev + low_vol',
                
                'EW_VW ~ MKT + MKT_lagged',
                'EW_VW ~ SMB', 
                'EW_VW ~ MKT + MKT_lagged + SMB + HML + RMW + CMA',
                'EW_VW ~ MKT + MKT_lagged + R_ME + R_IA + R_ROE + R_EG',
                'EW_VW ~ MKT + MKT_lagged + SMB + HML + QMJ',
                'EW_VW ~ MKT + MKT_lagged + SMB + HML + RMW + CMA + Mom + ST_Rev + low_vol',   
                'EW_VW ~ jan_dummy + MKT + MKT_lagged + SMB + HML + RMW + CMA + Mom + ST_Rev + low_vol'
               ]
    count_crsp_reg = 8
    modNames = [f'({x})\nEWVW CRSP' for x in range(1,count_crsp_reg+1)] + \
        [f'({x})\nEWVW SP500' for x in range(count_crsp_reg+1,len(regList1)+1)]
    
    ols_to_table(reg_dat, regList1, direc, reg_order=reg_order,
                 filename='exhibit4_regResults', model_names=modNames, 
                 float_fmt='%0.2f', stars=False, toLatex=True, scale_alpha=100)

def exhibit_5(direc, reg_dat, period_idx):
    '''
    Function to generate Exhibit 5 of the final paper.

    Parameters
    ----------
    direc : str
        Directory to save results to.
    reg_dat : pd.DataFrame, shape(T,k)
        Regression data (i.e. mostly factor and portfolio returns).
    period_idx : list of pd.index
        List of different subsamples.

    Returns
    -------
    None.

    '''    
    if direc[-1] != '/':
        direc = direc + '/'
        
    season_crsp = 'EW_VW_FU ~ jan_dummy + MKT + MKT_lagged + SMB + HML + RMW + CMA + Mom + ST_Rev + low_vol'
    season_sp500 = 'EW_VW ~ jan_dummy + MKT + + MKT_lagged + SMB + HML + RMW + CMA + Mom + ST_Rev + low_vol'
    season_names = ['0_FullSample','1_Pre-publication','2_Post-publication',
                    '3_Pre-GFC', '4_Post-GFC']
    season_idx = [period_idx[0]] + period_idx[3:]
    
    #Calculate regressions for both universes
    season_patterns_crsp = pd.concat([calc_ols(
        reg_dat.loc[idx], season_crsp) for idx in season_idx],
        keys=season_names)
    
    season_patterns_sp500 = pd.concat([calc_ols(
        reg_dat.loc[idx], season_sp500) for idx in season_idx],
        keys=season_names)
    
    season_results = pd.concat([season_patterns_crsp, season_patterns_sp500],
                               keys = ['CRSP','SP500'])
    
    # Adjust output format
    idx_slice = pd.IndexSlice
    season_results.loc[idx_slice[:,:,'param'],'jan_dummy'] = \
        season_results.loc[idx_slice[:,:,'param'],'jan_dummy']*100
    season_results = season_results.applymap('{:,.2f}'.format)
    season_results.loc[idx_slice[:,:,'tval'], :] = \
        season_results.loc[idx_slice[:,:,'tval'], :].applymap(
            lambda x: x if x=='nan' else '('+str(x)+')')
    
    
    #Save as txt-file
    yfile = direc + 'exhibit5_periodical_patterns.txt'
    with open(yfile, 'w') as f:
        f.write(season_results.to_latex())
    
def exhibit_6(direc, reg_dat, turnover):
    '''
    Function to generate Exhibit 6 of the final paper.

    Parameters
    ----------
    direc : str
        Directory to save results to.
    reg_dat : pd.DataFrame, shape(T,k)
        Regression data (i.e. mostly factor and portfolio returns).
    turnover : list
        Turnover for differently rebalanced portfolios.

    Returns
    -------
    None.

    '''
    if direc[-1] != '/':
        direc = direc + '/'
    
    portf = reg_dat[['EW_VW_FU','EW_VW_3MS','EW_VW_6MS','EW_VW_1YS',
                     'EW_VW_3YS','EW_VW_5YS']]
    
    # Panel 1: Calculate portfolio characteristics
    alt_portf_ret = pd.DataFrame(np.nan, columns=portf.columns, index=[
        'Ret','Std','Sharpe','MDD','Turnover'])
    alt_portf_ret.loc['Ret',:] = portf.mean().values*12*100
    alt_portf_ret.loc['Std',:] = portf.std().values*np.sqrt(12)*100
    alt_portf_ret.loc['Sharpe',:] = \
        alt_portf_ret.loc['Ret',:] / alt_portf_ret.loc['Std',:]
    alt_portf_ret.loc['MDD',:] =  max_drawdown(portf).values*100
    alt_portf_ret.loc['Turnover',:] = turnover

    # Panel 2: Regression coefficients
    regList_size_quint = \
        ['EW_VW_FU ~ jan_dummy + MKT + MKT_lagged + SMB + HML + RMW + CMA + Mom + ST_Rev + low_vol',
         'EW_VW_3MS ~ jan_dummy + MKT + MKT_lagged + SMB + HML + RMW + CMA + Mom + ST_Rev + low_vol',
         'EW_VW_6MS ~ jan_dummy + MKT + MKT_lagged + SMB + HML + RMW + CMA + Mom + ST_Rev + low_vol',
         'EW_VW_1YS ~ jan_dummy + MKT + MKT_lagged + SMB + HML + RMW + CMA + Mom + ST_Rev + low_vol',
         'EW_VW_3YS ~ jan_dummy + MKT + MKT_lagged + SMB + HML + RMW + CMA + Mom + ST_Rev + low_vol',
         'EW_VW_5YS ~ jan_dummy + MKT + MKT_lagged + SMB + HML + RMW + CMA + Mom + ST_Rev + low_vol'
         ]
    
    alt_portf_reg = pd.concat([calc_ols(reg_dat, reg, orientation='vert') \
                               for reg in regList_size_quint],
                              axis=1, keys=portf.columns).droplevel(1, axis=1)
    
    idx_slice = pd.IndexSlice
    
    # Adjust output format
    alt_portf_ret = alt_portf_ret.applymap('{:,.2f}'.format)    
        
    alt_portf_reg.loc[('jan_dummy', 'param'), :] *= 100  
    alt_portf_reg = alt_portf_reg.applymap('{:,.2f}'.format) 
    alt_portf_reg.loc[idx_slice[:,'tval'], :] = \
        alt_portf_reg.loc[idx_slice[:,'tval'], :].applymap(
            lambda x: x if x=='nan' else '('+str(x)+')') 
    
    alt_portf = pd.concat([alt_portf_ret, alt_portf_reg.droplevel(1)], axis=0)
        
    #Save as txt-file
    yfile = direc + 'exhibit6_alternative_rebal.txt'
    with open(yfile, 'w') as f:
        f.write(alt_portf.to_latex())    
    
def exhibit_7(direc, reg_dat, bbg_data):
    '''
    Function to generate Exhibit 7 of the final paper.

    Parameters
    ----------
    direc : str
        Directory to save results to.
    reg_dat : pd.DataFrame, shape(T,k)
        Regression data (i.e. mostly factor and portfolio returns).
    bbg_data : pd.DataFrame, shape(T,j)
        Fund return data.

    Returns
    -------
    None.

    '''
    if direc[-1] != '/':
        direc = direc + '/'
        
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
    bbg_excess = bbg_data.subtract(reg_dat.loc[:,'vwretd'].rename(None), axis=0)  
    
    portf = reg_dat.loc[:,['SMB','SMBSP','EW_VW_FU','EW_VW']].join(
        bbg_excess).dropna()
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
    yfile = direc + 'exhibit7_SMB_EWVW_portf_characteristics.txt'
    with open(yfile, 'w') as f:
        f.write(tabulate(table, headers=hds, showindex=idx, floatfmt=fmt ,
                         tablefmt='latex_booktabs'))
        
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
                             tstat(diff_ret, diff_mean_ret, 0)]
            
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
        values = combine_alternating(model_results.params, 
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
            max_drawdown(ret_p).values * scaling
        
        #Calmar ratio
        df.loc[(pnames[i], pn), 'Calmar ratio'] = \
            calmar_ratio(ret_p, freq).values
        
        #Sortino ratio
        df.loc[(pnames[i], pn), 'Sortino ratio'] = \
            sortino_ratio(ret_p, freq).values
                
        #Expected shortfall
        df.loc[(pnames[i], pn), 'Expected shortfall'] = \
            expected_shortfall(ret_p, cvar).values * scaling
        
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
    capm = CAPM(grp_obj['x1'],
                grp_obj[ret_m],
                grp_obj[ret_stock])
    capm.evaluate(alpha_inc=alpha_inc, hyp=hyp)
    
    return [capm.alpha*scale_alpha, capm.alpha_tvalue, capm.beta,
            capm.beta_tvalue, capm.rsqr, capm.conf_int.loc['x1',0],
            capm.conf_int.loc['x1',1], capm.conf_int.loc['const',0]*scale_alpha,
            capm.conf_int.loc['const',1]*scale_alpha]