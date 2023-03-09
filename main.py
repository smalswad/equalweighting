# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 20:45:18 2023

@author: Alexander Swade
"""

import os
import sys
import yaml
import glob
import pathlib


# Add utility library to sys path
sys.path.insert(1, 'C:/Users/Alexander Swade/OneDrive - Lancaster University/'
                'PhD research/quantfinlib')
from equalweighting.analysis import run_analysis
from equalweighting.data_loading import get_smbsp
from equalweighting.data_loading import get_quintile_portf
from equalweighting.data_loading import load_bbg
from equalweighting.data_loading import load_factors
from equalweighting.data_loading import load_wrds
from equalweighting.paper_exhibits import generate_exhibits

def main(config_file):
    
    #Load configs
    with open(config_file, 'r') as file: 
        configs = yaml.load(file, Loader=yaml.FullLoader)
        file.close()
    
    ### -------------- LOAD DATA --------------- ###
    cwd = os.getcwd()
    factors = load_factors(os.path.join(cwd, 'data', 'factors'))
    ret, mcap, consti, sp500 = load_wrds(os.path.join(cwd, 'data', 'wrds'))
    bbg = load_bbg(os.path.join(cwd, 'data', 'other'))
    smbsp = get_smbsp(os.path.join(cwd, 'data', 'other'))
    size_quintiles = get_quintile_portf(os.path.join(cwd, 'data', 'other'))
    
    data = {'factors': factors,
            'ret': ret,
            'mcap': mcap,
            'constituents': consti,
            'sp500': sp500,
            'smbsp': smbsp,
            'size_q': size_quintiles}
    
    ### -------------- RUN ANALYSIS --------------- ###
    analysis_results = run_analysis(data, configs)
    
    ### -------------- GENERATE EXHIBITS --------------- ###
    od = os.path.join(cwd, 'outputs')
    pathlib.Path(od).mkdir(parents=True, exist_ok=True)
    analysis_results['bbg'] = bbg
    generate_exhibits(od, configs, analysis_results)

# =============================================================================
# RUN MAIN
# =============================================================================
if __name__ == '__main__':
    files = glob.glob(os.path.join(os.getcwd(), 'configs', '*.yml'))
    
    for config in files:
        main(config)
