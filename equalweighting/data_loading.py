# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 10:02:03 2023

@author: Alexander Swade
"""
import numpy as np
import pandas as pd
from pandas.tseries.offsets import MonthEnd

from quantfinlib.utils.list import list_intsect

# =============================================================================
# Factor data
# =============================================================================
def load_factors(dd_fac):
    '''
    Function to load all factor returns, standardize format and return combined
    data.
    
    Parameters
    ----------
    dd_fac : str
        Path to data directory.

    Returns
    -------
    factors : pd.DataFrame, shape(T,n)
        Factor return data for all factors used in the paper.

    '''
    if dd_fac[-1] != '/':
        dd_fac = dd_fac + '/' 
    
    # 1) Load FF5 factors, i.e. MKT, SMB, HML, CMA, RMW
    ff5 = pd.read_csv(dd_fac + 'F-F_Research_Data_5_Factors.csv', sep=';',
                      decimal=',')
    ff5['Date'] = pd.to_datetime(ff5['Date'], yearfirst=True,
                                 format="%Y%m%d") + MonthEnd(0)
    ff5 = ff5.iloc[:,1:].set_index('Date') / 100
    
    # 2) Other factors from K. French's website, i.e. short-term reversal,
    #    momentum
    ff_str = pd.read_csv(dd_fac + 'F-F_ST_Reversal_Factor.csv', sep=';',
                         decimal=',')
    ff_str['date'] = pd.to_datetime(ff_str['Date'], yearfirst=True,
                                    format="%Y%m%d") + MonthEnd(0)
    ff_str = ff_str.iloc[:,2:].set_index('date') / 100
    
    ff_mom = pd.read_csv(dd_fac + 'F-F_Momentum_Factor.csv', sep=';',
                         decimal=',')
    ff_mom['date'] = pd.to_datetime(ff_mom['Date'], yearfirst=True,
                                    format="%Y%m%d") + MonthEnd(0)
    ff_mom = ff_mom.iloc[:,2:].set_index('date') / 100
    
    # FF EW factos
    ff_equal_weighted =  pd.read_csv(dd_fac+'ff_equal_weighted.csv', sep=';',
                                     decimal=',')
    ff_equal_weighted['date'] = \
        pd.to_datetime(ff_equal_weighted['Date'], yearfirst=True,
                       format="%Y%m%d") + MonthEnd(0)
    ff_equal_weighted = ff_equal_weighted.iloc[:,2:].set_index('date') / 100
    
    # 3) AQR QMJ factor
    aqr_qmj = pd.read_excel(dd_fac+'Quality Minus Junk Factors Monthly.xlsx', 
                            sheet_name='QMJ Factors', skiprows=18)
    aqr_qmj['date'] = pd.to_datetime(aqr_qmj['DATE'], yearfirst=False,
                                     format="%m/%d/%Y") + MonthEnd(0)
    aqr_qmj = aqr_qmj[['date','USA']].set_index('date').rename(
        {'USA':'QMJ'}, axis=1)
    
    # 4) q5 factors
    q5_fact = pd.read_csv(dd_fac+'q5_factors_monthly_2021.csv')
    q5_fact['day'] = 28
    q5_fact['date'] = pd.to_datetime(q5_fact[['year','month','day']]) \
        + MonthEnd(0)
    q5_fact = \
        q5_fact[['date','R_ME','R_IA','R_ROE','R_EG']].set_index('date') /100
    
    # 5) Low Vol by Pim van Vliet
    low_vol = pd.read_csv(dd_fac+'VanVliet_lowvol_factor.csv', sep=';',
                          decimal=',')
    low_vol['date'] = pd.to_datetime(low_vol['Date'], yearfirst=True,
                                     format="%Y%m%d") + MonthEnd(0)
    low_vol = low_vol.iloc[:,1:].set_index('date')
    
    # Combine factor data
    factors = pd.concat([ff5, ff_mom, ff_str, q5_fact, aqr_qmj, low_vol,
                         ff_equal_weighted], axis=1)

    return factors

# =============================================================================
# WRDS data
# =============================================================================
def load_wrds(dd_wrds):
    '''
    Function to load all data from WRDS (preprocessed and stored locally)

    Parameters
    ----------
    dd_wrds : str
        Path to data directory.

    Returns
    -------
    ret : pd.DataFrame, shape(T,n)
        Return data for all CRSP stocks downloaded.
    mcap : pd.DataFrame, shape(T,n)
        Market capitalization data for all CRSP stocks downloaded.

    '''
    if dd_wrds[-1] != '/':
        dd_wrds = dd_wrds + '/' 
    
    # Company information by Compustat
    comInfo = pd.read_csv(dd_wrds + 'dseInfo.csv')
    comInfo['hsiccd_2'] = (comInfo['hsiccd']/100).apply(np.floor)
    comInfo['gic'] = (comInfo['gsubind']*10**-6).apply(np.floor)
    comInfo = comInfo.drop_duplicates(subset=['permno']).astype({'gic':'Int64',
                                                                 'permno':'Int64'})
    
    # Return and mcap data from CRSP
    h5file = dd_wrds + 'crsp_monthly_df' + '.h5'
    h5 = pd.HDFStore(path=h5file, mode='a')
    
    #Load returns and drop duplicated indices
    ret = h5['ret']
    ret = ret[~ret.index.duplicated(keep='first')]
    ret.index = ret.index + MonthEnd(0)
    permnos_to_use = list_intsect(ret.columns, comInfo['permno'])
    ret = ret.loc[:, permnos_to_use]
    
    #Load mcap and drop duplicated indices
    mcap = h5['me']
    mcap = mcap[~mcap.index.duplicated(keep='first')]
    mcap.index = mcap.index + MonthEnd(0)
    mcap = mcap.loc[:, permnos_to_use]
    
    h5.close()
    
    # Official SP500 returns from CRSP
    sp500 = pd.read_csv(dd_wrds + 'sp500_crsp_monthly.csv')
    sp500['date'] = pd.to_datetime(sp500['caldt'], yearfirst=True) + MonthEnd(0)
    sp500.set_index('date', inplace=True)
    
    # Official Sp500 constituents from CRSP
    consti = pd.read_csv(dd_wrds + 'sp500_const.csv')
    consti[['start', 'ending']] = consti[['start', 'ending']].apply(pd.to_datetime, yearfirst=True)
    
    #remove constituents from SP500 for which there are no ret and me data
    consti = consti[consti['permno'].isin(mcap.columns)]
    
    return (ret, mcap, consti, sp500)

# =============================================================================
# Fund data from BBG
# =============================================================================
def load_bbg(dd_bbg):
    '''
    Load Bloomberg data (levels) and return return data.

    Parameters
    ----------
    dd_bbg : str
        Path to data directory.

    Returns
    -------
    bbg_data : pd.DataFrame
        Returns of Bloomberg data.

    '''
    if dd_bbg[-1] != '/':
        dd_bbg = dd_bbg + '/' 
        
    bbg_data = pd.read_excel(dd_bbg + 'small_cap_funds.xlsx')
    bbg_data['date'] = pd.to_datetime(bbg_data['Dates'], yearfirst=True) \
        + MonthEnd(0)
    bbg_data.set_index('date', inplace=True, drop=True)
    bbg_data = bbg_data.iloc[:,1:].pct_change().dropna()

    return bbg_data

# =============================================================================
# Other data
# =============================================================================
def get_quintile_portf(direc, load=True, ret=None, mcap=None):
    '''
    
    Function to either load quintile portfolios sorted by size or to construct
    them from scratch.

    Parameters
    ----------
    direc : str
        Directory to either load or save return from/to.
    load : bool, optional
        Indicate whether return series should be loaded or calculated. 
        The default is True.
    ret : pd.DataFrame, optional
        Only relevant if load==False. Return data containing Sp500 returns only.
        The default is None.
    mcap : pd.DataFrame, optional
        Only relevant if load==False. Mcap data containing Sp500 mcap only.
        The default is None.

    Raises
    ------
    ValueError
        Raise error if ret or mcap are not correctly specified.

    Returns
    -------
    pd.DataFrame, shape(T,,5)
        Return of 5 quintile portfolios sorted by size.

    '''
    
    if load:
        # Load quintile portfolios
        if direc[-1] != '/':
            direc = direc + '/'
            
        size_quint_ret = pd.read_csv(direc + 'size_quintile_portfolios.csv'). \
            set_index('date', drop=True)
        size_quint_ret.index = pd.to_datetime(
            size_quint_ret.index, yearfirst=True, format="%Y-%m-%d") 
        size_quint_ret = size_quint_ret.iloc[:,1:]
    
    else:
        # Calculate quintile portfolios 
        size_df = pd.merge(mcap.melt(var_name='permno', value_name='mcap',
                                      ignore_index=False).reset_index(),
                            ret.melt(var_name='permno', value_name='ret',
                                      ignore_index=False).reset_index(),
                            on=['date','permno'])
        
        # # sort by permno and date and also drop duplicates
        size_df = size_df.sort_values(by=['permno','date']).drop_duplicates()
        
        # keep December market cap
        size_df['year'] = size_df['date'].dt.year
        size_df['month'] = size_df['date'].dt.month
        decme = size_df[size_df['month']==12]
        decme = decme[['permno','date','mcap','year']].rename(
            columns={'mcap':'dec_me'})
        
        ### July to June dates
        size_df['ffdate'] = size_df['date'] + MonthEnd(-6)
        size_df['ffyear'] = size_df['ffdate'].dt.year
        size_df['ffmonth'] = size_df['ffdate'].dt.month
        size_df['1+retx'] = 1 + size_df['ret']
        size_df = size_df.sort_values(by=['permno','date'])
        
        # cumret by stock
        size_df['cumretx'] = size_df.groupby(
            ['permno','ffyear'])['1+retx'].cumprod()
        
        # lag cumret
        size_df['lcumretx'] = size_df.groupby(['permno'])['cumretx'].shift(1)
        
        # lag market cap
        size_df['lme'] = size_df.groupby(['permno'])['mcap'].shift(1)
        
        # if first permno then use me/(1+retx) to replace the missing value
        size_df['count'] = size_df.groupby(['permno']).cumcount()
        size_df['lme'] = np.where(size_df['count']==0,
                                  size_df['mcap']/size_df['1+retx'],
                                  size_df['lme'])
        
        # baseline me
        mebase = size_df.loc[size_df['ffmonth']==1,
                              ['permno','ffyear', 'lme']].rename(
                                  columns={'lme':'mebase'})
        
        # merge result back together
        size_df = pd.merge(size_df, mebase, how='left', on=['permno','ffyear'])
        size_df['wt'] = np.where(size_df['ffmonth']==1,
                                  size_df['lme'],
                                  size_df['mebase']*size_df['lcumretx'])
        
        decme['year'] = decme['year']+1
        decme = decme[['permno','year','dec_me']]
        
        # Info as of June
        size_df_jun = size_df[size_df['month']==6]
        
        size_jun = pd.merge(size_df_jun, decme, how='inner',
                            on=['permno','year'])
        size_jun = size_jun[['permno','date','ret','mcap','wt','cumretx',
                             'mebase', 'lme','dec_me']]
        size_jun = size_jun.sort_values(by=['permno','date']).drop_duplicates()
        
        #Calculate median
        sz_quintile = size_jun.groupby(['date'])['mcap'].quantile(
            [.2, .4, .6, .8]).reset_index().rename(
                columns={'level_1':'sz_breakp'})
            
        sz_quintile = sz_quintile.pivot(index='date', columns='sz_breakp',
                                        values='mcap').reset_index()
        
        size_jun = pd.merge(size_jun, sz_quintile, how='left', on=['date'])
        
        #Function to assign size bucket
        def sz_bucket(row):
            if np.isnan(row['mcap']):
                value = ''
            elif row['mcap']<=row[0.2]:
                value = 'Q1'
            elif (row['mcap']<=row[0.4] and row['mcap']>row[0.2]):
                value = 'Q2'
            elif (row['mcap']<=row[0.6] and row['mcap']>row[0.4]):
                value = 'Q3'
            elif (row['mcap']<=row[0.8] and row['mcap']>row[0.6]):
                value = 'Q4'
            else:
                value = 'Q5'
            
            return value
        
        size_jun['szport'] = size_jun.apply(sz_bucket, axis=1)
        
        # store portfolio assignment as of June
        june = size_jun.loc[:,['permno','date','szport']]
        june['ffyear'] = june['date'].dt.year
        
        # merge back with monthly records
        size_df = size_df.loc[:,['date','permno','ret','mcap','wt','cumretx',
                                 'ffyear']]
        size_df = pd.merge(size_df, june[['permno','ffyear','szport']],
                            how='left', on=['permno','ffyear'])    
        
        # function to calculate value weighted return
        def wavg(group, avg_name, weight_name):
            d = group[avg_name]
            w = group[weight_name]
            try:
                return (d * w).sum() / w.sum()
            except ZeroDivisionError:
                return np.nan
        
        vwret = size_df.groupby(['date','szport']).apply(wavg,'ret','wt'). \
            to_frame().reset_index().rename(columns={0: 'vwret'})
        
        ewret = size_df.groupby(['date','szport'])['ret'].mean().\
            reset_index().rename(columns={'ret': 'ewret'})
        
        # merge ew and vw returns
        sz_ret = pd.merge(ewret, vwret, how='left', on=['date','szport'])
        sz_ret = sz_ret.drop(sz_ret[sz_ret['szport']==''].index)
        
        # calculate EW-VW spread
        sz_ret['ew_vw_ret'] = sz_ret['ewret'] - sz_ret['vwret']
        
        # transpose
        size_quint_ret = sz_ret.pivot(index='date', columns='szport',
                                        values='ew_vw_ret').reset_index()
        
        #save as csv
        size_quint_ret.to_csv(direc + 'size_quintile_portfolios.csv')
    
    return size_quint_ret

def get_smbsp(direc, load=True, ret=None, mcap=None):
    '''
    Function to either load SMB factor based on SP500 stocks or to construct
    it from scratch.

    Parameters
    ----------
    direc : str
        Directory to either load or save return from/to.
    load : bool, optional
        Indicate whether return series should be loaded or calculated. 
        The default is True.
    ret : pd.DataFrame, optional
        Only relevant if load==False. Return data containing Sp500 returns only.
        The default is None.
    mcap : pd.DataFrame, optional
        Only relevant if load==False. Mcap data containing Sp500 mcap only.
        The default is None.

    Raises
    ------
    ValueError
        Raise error if ret or mcap are not correctly specified.

    Returns
    -------
    pd.Series, shape(T,)
        Return series of new factor SMBSP.

    '''
    
    if load:
        if direc[-1] != '/':
            direc = direc + '/'
            
        pseudo_smb = pd.read_csv(direc + 'pseudo_SMB.csv').set_index('date')
        pseudo_smb.index = pd.to_datetime(pseudo_smb.index, yearfirst=True,
                                          format="%Y-%m-%d")        
        
    else:
        if (
            not isinstance(ret, pd.DataFrame) 
            or not isinstance(mcap, pd.DataFrame)
        ): 
            raise ValueError('Expected returns and mcap as input.')
        
        # Calculate SMB-like factor for SP500 universe only
        sp500_df = pd.merge(mcap.melt(var_name='permno', value_name='mcap',
                                            ignore_index=False).reset_index(),
                            ret.melt(var_name='permno', value_name='ret',
                                          ignore_index=False).reset_index(),
                            on=['date','permno'])
        
        # sort by permno and date and also drop duplicates
        sp500_df = sp500_df.sort_values(by=['permno','date']).drop_duplicates()
        
        # keep December market cap
        sp500_df['year'] = sp500_df['date'].dt.year
        sp500_df['month'] = sp500_df['date'].dt.month
        decme = sp500_df[sp500_df['month']==12]
        decme = decme[['permno','date','mcap','year']].rename(
            columns={'mcap':'dec_me'})
        
        ### July to June dates
        sp500_df['ffdate'] = sp500_df['date'] + MonthEnd(-6)
        sp500_df['ffyear'] = sp500_df['ffdate'].dt.year
        sp500_df['ffmonth'] = sp500_df['ffdate'].dt.month
        sp500_df['1+retx'] = 1 + sp500_df['ret']
        sp500_df = sp500_df.sort_values(by=['permno','date'])
        
        # cumret by stock
        sp500_df['cumretx'] = \
            sp500_df.groupby(['permno','ffyear'])['1+retx'].cumprod()
        
        # lag cumret
        sp500_df['lcumretx'] = sp500_df.groupby(['permno'])['cumretx'].shift(1)
        
        # lag market cap
        sp500_df['lme'] = sp500_df.groupby(['permno'])['mcap'].shift(1)
        
        # if first permno then use me/(1+retx) to replace the missing value
        sp500_df['count'] = sp500_df.groupby(['permno']).cumcount()
        sp500_df['lme'] = np.where(sp500_df['count']==0,
                                    sp500_df['mcap']/sp500_df['1+retx'],
                                    sp500_df['lme'])
        
        # baseline me
        mebase = sp500_df.loc[
            sp500_df['ffmonth']==1,
            ['permno','ffyear', 'lme']].rename(columns={'lme':'mebase'})
        
        # merge result back together
        sp500_df = pd.merge(sp500_df, mebase, how='left',
                            on=['permno','ffyear'])
        sp500_df['wt'] = np.where(sp500_df['ffmonth']==1,
                                  sp500_df['lme'],
                                  sp500_df['mebase']*sp500_df['lcumretx'])
        
        decme['year'] = decme['year']+1
        decme = decme[['permno','year','dec_me']]
        
        # Info as of June
        sp500_df_jun = sp500_df[sp500_df['month']==6]
        
        sp500_jun = pd.merge(sp500_df_jun, decme, how='inner',
                             on=['permno','year'])
        sp500_jun = sp500_jun[['permno','date','ret','mcap','wt','cumretx',
                               'mebase','lme','dec_me']]
        sp500_jun = sp500_jun.sort_values(
            by=['permno','date']).drop_duplicates()
        
        #Calculate median
        sz_median = sp500_jun.groupby(['date'])['mcap'].median().\
            to_frame().reset_index().rename(columns={'mcap':'sizemedn'})
        
        sp500_jun = pd.merge(sp500_jun, sz_median, how='left', on=['date'])
        
        #Function to assign size bucket
        def sz_bucket(row):
            if np.isnan(row['mcap']):
                value = ''
            elif row['mcap']>=row['sizemedn']:
                value = 'B'
            else:
                value = 'S'
            
            return value
        
        sp500_jun['szport'] = sp500_jun.apply(sz_bucket, axis=1)
        
        # store portfolio assignment as of June
        june = sp500_jun.loc[:,['permno','date','szport']]
        june['ffyear'] = june['date'].dt.year
        
        # merge back with monthly records
        sp500_df = sp500_df.loc[:,['date','permno','ret','mcap','wt','cumretx',
                                   'ffyear']]
        sp500_df = pd.merge(sp500_df, june[['permno','ffyear','szport']],
                            how='left', on=['permno','ffyear'])    
        
        # function to calculate value weighted return
        def wavg(group, avg_name, weight_name):
            d = group[avg_name]
            w = group[weight_name]
            try:
                return (d * w).sum() / w.sum()
            except ZeroDivisionError:
                return np.nan
        
        vwret = sp500_df.groupby(['date','szport']).apply(
            wavg, 'ret','wt').to_frame().reset_index().rename(
                columns={0: 'vwret'})
        
        # transpose
        pseudo_ff = vwret.pivot(index='date', columns='szport',
                                        values='vwret').reset_index()
        
        pseudo_ff['SMBSP'] = pseudo_ff['S'] - pseudo_ff['B']
        pseudo_smb = pseudo_ff[['date', 'SMBSP']].set_index('date')
        
        #save as csv
        pseudo_smb.to_csv(direc + 'pseudo_SMB.csv')

    return pseudo_smb














