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




figseiz = (5.5, 3.5)
dpi_val = 200
markers = itertools.cycle(['o', '^', 's', 'x', 'd'])

col_names_numeric = ['MAE_train', 'MAE_validate', 'MAE_test',
                    'RMSE_train','RMSE_validate', 'RMSE_test',
                    'R2_train', 'R2_validate', 'R2_test',
                    'wall_clock_time_minutes']


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




def plot_single_series(df, output_folder):
    
    for col in col_names_numeric:
        
        fig, ax = plt.subplots(figsize=figseiz)
        df.loc[:, col].plot(ax=ax)
        ax.set_ylabel(col)
        fname = os.path.join(output_folder,
                             f'{col}.png')
        fig.savefig(fname, dpi=dpi_val, bbox_inches='tight')
        plt.close(fig)
        



def plot_R2_vs_RMSE(df, output_folder):

    for key in ['train', 'validate', 'test']:
        # key = 'train'
        fig, ax = plt.subplots(figsize=figseiz)
        for label, grp in df.groupby(['measurement_point_name']):
            ax.scatter(x=f'RMSE_{key}', y=f'R2_{key}', \
                       s=12.0, marker=next(markers),
                       data=grp, label=label)
        ax.set_ylim((-0.1, 1.1))
        ax.set_xlim((-0.1, 3))
        ax.grid(True)
        ax.set_axisbelow(True)
        ax.legend()
        ax.set_xlabel(f'RMSE {key}')
        ax.set_ylabel(f'R$^2$ {key}')
        fname = os.path.join(output_folder,
                             f'{key} R2 vs RMSE.png')
        fig.savefig(fname, dpi=dpi_val, bbox_inches='tight')
        plt.close(fig)


def plot_R2_vs_MAE(df, output_folder):
    # There is a little bit more variation (jitter) in the R2 vs MAE figures,
    # when compared to R2 vs RMSE figures, but the difference is not that big.
    # The MAE values are little bit smaller than RMSE values, but the order 
    # of measurement_points seem to stay the same.
    
    for key in ['train', 'validate', 'test']:
        # key = 'train'
        fig, ax = plt.subplots(figsize=figseiz)
        for label, grp in df.groupby(['measurement_point_name']):
            ax.scatter(x=f'MAE_{key}', y=f'R2_{key}', \
                        s=12.0, marker=next(markers),
                        data=grp, label=label)
        ax.set_ylim((-0.1, 1.1))
        ax.set_xlim((-0.1, 3))
        ax.grid(True)
        ax.set_axisbelow(True)
        ax.legend()
        ax.set_xlabel(f'MAE {key}')
        ax.set_ylabel(f'R$^2$ {key}')
        fname = os.path.join(output_folder,
                              f'{key} R2 vs MAE.png')
        fig.savefig(fname, dpi=dpi_val, bbox_inches='tight')
        plt.close(fig)

########################################



if run_combiner:

    # Make sure that the combined.csv files exist and are complete
    run_combine_results_files(path_repo_root)
    
    
    # Read all combined.csv files to a single pandas DataFrame:
    output_folders = [f for f in os.listdir(path_repo_root) if 'output_' in f]
    
    df_list = []
    
    for folder in output_folders:
        
        fname = os.path.join(path_repo_root, folder, 'combined.csv')
        print(fname, flush=True)
        df_single = pd.read_csv(fname)
        df_list.append(df_single)
    
    df_all = pd.concat(df_list, ignore_index=True)
    
    df_all['R2_mean_validate_test'] \
        = df_all.loc[:, ['R2_validate','R2_test']].mean(axis=1)
    
    df_all['RMSE_mean_validate_test'] \
        = df_all.loc[:, ['RMSE_validate','RMSE_test']].mean(axis=1)
    
    df_all['MAE_mean_validate_test'] \
        = df_all.loc[:, ['MAE_validate','MAE_test']].mean(axis=1)
    
    
    # Export to pickle file
    time_str = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    fname = os.path.join(folder_merged,
                         'df_all_{}.pickle'.format(time_str))
    with open(fname, 'wb') as f:
        pickle.dump(df_all, f)

    
    # Export to xlsx file
    fname = os.path.join(folder_merged,
                         'df_all_{}.xlsx'.format(time_str))
    df_all.to_excel(fname)




if run_checker:
    # Check if all the listed files have been completed successfully
    # Add to a list of it hasn't been finished properly
    
    
    fname = os.path.join(path_repo_root,
                         'df_all_2022-10-21-21-52-37.xlsx')
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
    
    print(f'n_tot = {n_tot}')
    
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
    print(f"len(reruns_list) = {len(reruns_list)}")
    
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
    # reruns_list = []
    fname = os.path.join(folder_merged,
                         'reruns.json')
    with open(fname, 'r') as f:
        # for line in f:
        #     reruns_list.append(line.rstrip())
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

"""
fname = os.path.join(folder_merged,
                     'df_all.pickle')
with open(fname, 'rb') as f:
    df_all = pickle.load(f)
"""


print(df_all.columns)


print('END, myPostAnalysis.py', flush=True)




