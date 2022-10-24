# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 22:53:05 2022

@author: laukkara

This file contains post-analysis and plotting of the results from the 
machine learning fitting prediction.
1: main.py
2: myPostAnalysis.py

"""

import os
import time
import itertools
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
import json

import myResults


# This folder contains the one or many "output_..." folders
# path_repo_root = r'C:\Local\laukkara\Data\ML_indoor_Narvi'
path_repo_root = '/lustre/scratch/laukkara/ML_indoor/'


# The results files don't need to be run every time. Set this is False,
# if the results data is already collected from the various "output_..."
# folders.
run_combiner = True # True, False

run_checker = True # True, False

run_make_sbatch_files = True # True, False



folder_merged = os.path.join(path_repo_root,
                             'merged_results')
if not os.path.exists(folder_merged):
    os.makedirs(folder_merged)




########################################

def run_combine_results_files(root_folder_repository):
    # This function is intended as initialization, which makes the
    # combined.csv files for each subfolder and also creates the
    # plotting folder: '0output'.
    
    for output_fold in os.listdir(root_folder_repository):
        
        output_path = os.path.join(root_folder_repository, output_fold)
        
        if os.path.isdir(output_path) and 'output' in output_fold:
            
            myResults.combine_results_files(output_path)




########################################



if run_combiner:

    print(f'We are at: run_combiner', flush=True)

    # Make sure that the combined.csv files exist and are complete
    run_combine_results_files(path_repo_root)
    
    
    # Read all combined.csv files to a single pandas DataFrame:
    output_folders = [f for f in os.listdir(path_repo_root) if 'output_' in f]
    
    df_list = []
    
    for folder in output_folders:
        
        fname = os.path.join(path_repo_root, folder, 'combined.csv')
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
    
    
    # Export to pickle file, with and without time stamp in file name
    time_str = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    fname = os.path.join(folder_merged,
                         'df_all_{}.pickle'.format(time_str))
    with open(fname, 'wb') as f:
        pickle.dump(df_all, f)
    
    fname = os.path.join(folder_merged,
                         'df_all.pickle'.format(time_str))
    with open(fname, 'wb') as f:
        pickle.dump(df_all, f)

    
    # Export to xlsx file, with and without file name
    fname = os.path.join(folder_merged,
                         'df_all_{}.xlsx'.format(time_str))
    df_all.to_excel(fname)

    fname = os.path.join(folder_merged,
                         'df_all.xlsx'.format(time_str))
    df_all.to_excel(fname)




if run_checker:
    # Check if all the listed files have been completed successfully
    # Add to a list of it hasn't been finished properly

    print('We are at: run_checker', flush=True)
    
    ## Older versio, where file name with time stamp was used
#    # Find the xlsx file with the newest time stamp
#    time_format = '%Y-%m-%d-%H-%M-%S'
#
#    dt_stamps = []
#
#    for xlsx_file in [f for f in os.listdir(folder_merged) if '.xlsx' in f]:
#        
#        dt_str = xlsx_file[7:-5]
#        dt_stamps.append(time.strptime(dt_str, time_format))
#
#
#    if len(dt_stamps) == 0:
#        print('xlsx file with suitable name does not exist')
#    elif len(dt_stamps) == 1:
#        time_str_newest = time.strftime(time_format, dt_stamps[0])
#    else:
#        dt_stamp_newest = dt_stamps[0]
#        for dt_stamp in dt_stamps[1:]:
#            if dt_stamp > dt_stamp_newest:
#                dt_stamp_newest = dt_stamp
#
#    time_str_newest = time.strftime(time_format, dt_stamp_newest)
#
#    fname = os.path.join(folder_merged,
#                         'df_all_{}.xlsx'.format(time_str_newest))

    ## New version, where the file: "df_all.xlsx" is always rewritten
    ## to contain the newest data
    fname = os.path.join(folder_merged, 'df_all.xlsx')
    df_all = pd.read_excel(fname)
    
    measurement_point_names = ['Espoo1', 'Espoo2', 'Tampere1', 'Tampere2', 'Valkeakoski']
    
    model_names = ['svrpoly',
                   'kernelridgesigmoid',
                   'kernelridgecosine',
                   'kernelridgepolynomial',
                   'nusvrpoly',
                    'dummyregressor',
                    'expfunc',
                    'piecewisefunc',
                    'linearregression',
                    'ridge',
                    'lasso',
                    'elasticnet',
                    'lars',
                    'lassolars',
                    'huberregressor',
                    'ransacregressor',
                    'theilsenregressor',
                    'kernelridgelinear',
                    'kernelridgerbf',
                    'kernelridgelaplacian',
                    'linearsvr',
                    'nusvrlinear',
                    'nusvrrbf',
                    'nusvrsigmoid',
                    'svrlinear',
                    'svrrbf',
                    'svrsigmoid',
                    'kneighborsregressoruniform',
                    'kneighborsregressordistance',
                    'decisiontreeregressorbest',
                    'decisiontreeregressorrandom',
                    'extratreeregressorbest',
                    'extratreeregressorrandom',
                    'adaboostdecisiontree',
                    'adaboostextratree',
                    'baggingdecisiontree',
                    'baggingextratree',
                    'extratreesregressorbootstrapfalse',
                    'extratreesregressorbootstraptrue',
                    'gradientboostingregressor',
                    'histgradientboostingregressor',
                    'randomforest',
                    'lgbgbdt',
                    'lgbgoss',
                    'lgbdart',
                    'lgbrf',
                    'xgbgbtree',
                    'xgbdart']
    
    optimization_methods = ['pso', 'randomizedsearchcv', 'bayessearchcv']
    
    X_lags = [0] # 0, 1
    y_lags = [0]    
    
    N_CVs = [3] # 3, 4, 5
    N_ITERs = [10, 20, 50, 100, 200, 500]
    N_CPUs = [1]
    
    n_tot = len(measurement_point_names) \
            * len(model_names) \
            * len(optimization_methods) \
            * len(X_lags) * len(y_lags) \
            * len(N_CVs) * len(N_ITERs) * len(N_CPUs)
    
    print(f'n_tot = {n_tot}', flush=True)
    
    path_to_main = '/home/laukkara/github/ML_indoor/main.py'

    reruns_list = []
    
    for mp in measurement_point_names:
        for opt in optimization_methods:
            for xlag in X_lags:
                for ylag in y_lags:
                    for ncv in N_CVs:
                        for niter in N_ITERs:
                            for ncpu in N_CPUs:
                                
                                arr_list = []
                                
                                for mod in model_names:
                                    
                                    # This is True for those rows that match
                                    idxs = (df_all['measurement_point_name'] == mp) \
                                        & (df_all['model_name'] == mod) \
                                        & (df_all['optimization_method'] == opt) \
                                        & (df_all['X_lag'] == xlag) \
                                        & (df_all['y_lag'] == ylag) \
                                        & (df_all['N_CV'] == ncv) \
                                        & (df_all['N_ITER'] == niter) \
                                        & (df_all['N_CPU'] == ncpu)
                                    
                                    # Get indexis from the True rows
                                    idxs_true = idxs.index[idxs].values
                                    
                                    
                                    if idxs_true.shape[0] == 0 \
                                        or df_all.loc[idxs_true, 'MAE_mean_validate_test'].min() >= 5.0:
                                        # There are no corresponding rows or
                                        # the best rows have been far off.
                                        
                                        # python indexing is 0-based, slurm array is 1-based
                                        idx_mod = model_names.index(mod) + 1
                                        arr_list.append(idx_mod)
                                    
                                    
                                ## continue
                                idx_mp = measurement_point_names.index(mp)
                                idx_opt = optimization_methods.index(opt)
                            
                                str_python = f"python3 {path_to_main} "\
                                            f"{idx_mp} {idx_mp+1} "\
                                            f"$SLURM_ARRAY_TASK_ID $(( $SLURM_ARRAY_TASK_ID + 1 )) "\
                                            f"{idx_opt} {idx_opt+1} "\
                                            f"{ncv} {niter} {ncpu} "\
                                            f"{xlag} {ylag}"
                                # print(s)
                                reruns_list.append({'arr': arr_list,
                                                    'str_python': str_python})
                            
    # reruns_list has now been filled
    print(f"len(reruns_list) = {len(reruns_list)}", flush=True)
    
    # Write file with timestamp in file name
    time_str = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    fname = os.path.join(folder_merged,
                         'reruns_{}.json'.format(time_str))    
    with open(fname, 'w') as f:
        json.dump(reruns_list, f, indent=4)


    # Write file without timestamp in file name
    fname = os.path.join(folder_merged,
                         'reruns.json')
    with open(fname, 'w') as f:
        json.dump(reruns_list, f, indent=4)




if run_make_sbatch_files:
    
    # Read in information for the cases that should be rerun
    print('We are at: run_make_sbatch_files', flush=True)

    fname = os.path.join(folder_merged,
                         'reruns.json')
    with open(fname, 'r') as f:
        reruns_list = json.load(f)
    
    # Read in the base case shell script
    sbatch_template = []
    fname = os.path.join(path_repo_root,
                         'ML_indoor_template.sh')
    with open(fname, 'r') as f:
        for line in f:
            sbatch_template.append(line.rstrip())
    
    
    # Make sure output folder exists
    folder_sbatch = os.path.join(folder_merged,
                                 'sbatch_files')
    if not os.path.exists(folder_sbatch):
        os.makedirs(folder_sbatch)
    
    
    # Write the new sbatch files
    filenames_list = []
    for item in reruns_list:
        
        str_helper = item['str_python'][48:] \
                .replace("$SLURM_ARRAY_TASK_ID $(( $SLURM_ARRAY_TASK_ID + 1 )) ",'') \
                .replace(' ', '_')
        
        sbatch_single_file = 'ML_indoor_{}.sh'.format(str_helper)
        
        fname = os.path.join(folder_sbatch,
                             sbatch_single_file)
        
        with open(fname, 'w') as f:
        
            for line_template in sbatch_template:
                if '#SBATCH --array' in line_template:
                    # sbatch array definition line
                    str_dummy = ','.join(str(x) for x in item['arr'])
                    s_to_write = '#SBATCH --array=' + str_dummy
                        
                    f.write(s_to_write + '\n')
                
                elif 'python3' in line_template:
                    # python code line
                    f.write(item['str_python'] + '\n')
                    
                else:
                    f.write(line_template + '\n')
        
        filenames_list.append(sbatch_single_file)
    
    
    # Write the single sbatch calls to a single shell script file
    fname = os.path.join(folder_sbatch,
                         'ML_indoor_rerun_all.sh')
    with open(fname, 'w') as f:
        for line in filenames_list:
            f.write('sbatch ' + line + '\n')
    
    
    
    

##########################################



print('df_all.columns from myMerge:', flush=True)
print(df_all.columns, flush=True)


print('myMerge.py, END', flush=True)




