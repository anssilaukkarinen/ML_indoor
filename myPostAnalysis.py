# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 22:53:05 2022

@author: laukkara

This file contains post-analysis and plotting of the results from the 
machine learning fitting prediction.
1: main.py
2: myPostAnalysis.py

This file analyses the contents of a single folder 
that contains a combined.csv file.

The code is developed here and when it is finished, it is moved into
'myPostAnalysis_helper.py' module.
"""

import os
import itertools
import matplotlib.pyplot as plt
import pandas as pd
import pickle

import myResults


# This folder contains the one or many "output_..." folders
# path_repo_root = r'C:\Local\laukkara\Data\ML_indoor_Narvi'
path_repo_root = '/lustre/scratch/laukkara/ML_indoor/'


# The results files don't need to be run every time. Set this is False,
# if the results data is already collected from the various "output_..."
# folders.
run_combiner = True




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
    fname = os.path.join(folder_merged,
                         'df_all.pickle')
    with open(fname, 'wb') as f:
        pickle.dump(df_all, f)

    
    # Export to xlsx file
    fname = os.path.join(folder_merged,
                         'df_all.xlsx')
    df_all.to_excel(fname)






##########################################

fname = os.path.join(folder_merged,
                     'df_all.pickle')
with open(fname, 'rb') as f:
    df_all = pickle.load(f)

print(df_all.columns)


print('END, myPostAnalysis.py', flush=True)




