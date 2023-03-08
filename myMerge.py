# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 22:53:05 2022

@author: laukkara

output_... -kansiosta löytyy case-kansioita, kuten kernelridgesigmoid_bayessearchcv_...
- jos kaikki on mennyt hyvin, niin kansiosta löytyy log.txt, jossa on rivi: wall_clock_time, minutes
- jos tätä riviä ei löydy, niin sitten laskenta ei ole mennyt läpi ja tuloksia ei ole saatu tallennettua



"""

import os
import sys
import time
import itertools
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
import json

import myResults


if 'win' in sys.platform:
    # The output_folder_base contains the one or many "output_..." folders
    output_folder_base = r'C:\Users\laukkara\Data\ML_indoor_Narvi'
    folder_github = r'C:\Users\laukkara\github\ML_indoor'

elif 'lin' in sys.platform:
    output_folder_base = '/lustre/scratch/laukkara/ML_indoor/'
    folder_github = '/home/laukkara/github/ML_indoor'



folder_merged = os.path.join(output_folder_base,
                             'merged_results')
if not os.path.exists(folder_merged):
    os.makedirs(folder_merged)




create_combined = True
combine_combined = True





########################################

if create_combined:
    # Make sure that the combined.csv files exist and are complete
    # This function is intended as initialization, which makes the
    # combined.csv files for each subfolder and also creates the
    # plotting folder: '0output'.
    
    print('We are at create_combined', flush=True)
    
    for output_fold in os.listdir(output_folder_base):
        
        output_path = os.path.join(output_folder_base, output_fold)
        
        if os.path.isdir(output_path) and 'output' in output_fold:
            
            myResults.combine_results_files(output_path)





########################################




if combine_combined:
    # Read all combined.csv files to a single pandas DataFrame:
    
    print('We are at: combine_combined', flush=True)
    
    output_folders = [f for f in os.listdir(output_folder_base) if 'output_' in f]
    
    df_list = []
    
    for folder in output_folders:
        
        fname = os.path.join(output_folder_base, folder, 'combined.csv')
        print(f'Starting to read: {fname}', flush=True)
        try:
            if os.path.exists(fname):
                df_single = pd.read_csv(fname)
                df_list.append(df_single)
            else:
                print(f'File: {fname} does not exist!', flush=True)
        except:
            print(f'File: {fname} threw and error', flush=True)
    
    df_all = pd.concat(df_list, ignore_index=True)
    
    df_all['R2_mean_validate_test'] \
        = df_all.loc[:, ['R2_validate','R2_test']].mean(axis=1)
    
    df_all['RMSE_mean_validate_test'] \
        = df_all.loc[:, ['RMSE_validate','RMSE_test']].mean(axis=1)
    
    df_all['MAE_mean_validate_test'] \
        = df_all.loc[:, ['MAE_validate','MAE_test']].mean(axis=1)
    
    df_all.index.name = 'index'
    
    
    # Export to pickle file, with and without time stamp in file name
    time_str = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    fname = os.path.join(folder_merged,
                         'df_all_{}.pickle'.format(time_str))
    with open(fname, 'wb') as f:
        pickle.dump(df_all, f)
    
    fname = os.path.join(folder_merged,
                         'df_all.pickle')
    with open(fname, 'wb') as f:
        pickle.dump(df_all, f)

    
    # Export to xlsx file, with and without time stamp in file name
    fname = os.path.join(folder_merged,
                         'df_all_{}.xlsx'.format(time_str))
    df_all.to_excel(fname)

    fname = os.path.join(folder_merged,
                         'df_all.xlsx')
    df_all.to_excel(fname)

    # Export to csv file, with and without time stamp in file name
    fname = os.path.join(folder_merged,
                         'df_all_{}.csv'.format(time_str))
    df_all.to_csv(fname)

    fname = os.path.join(folder_merged,
                         'df_all.csv')
    df_all.to_csv(fname)





##########################################


print('df_all.shape:', flush=True)
print(df_all.shape, flush=True)


print('df_all.columns from myMerge:', flush=True)
print(df_all.columns, flush=True)


print('myMerge.py, END', flush=True)




