# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 09:14:53 2022

@author: anssi
"""

import os
import itertools
import matplotlib.pyplot as plt

from main import combine_results_files_old
from main import combine_results_files


figseiz = (5.5, 3.5)
dpi_val = 200
markers = itertools.cycle(['o', '^', 's', 'x', 'd'])

col_names_numeric = ['MAE_train', 'MAE_validate', 'MAE_test',
                    'RMSE_train','RMSE_validate', 'RMSE_test',
                    'R2_train', 'R2_validate', 'R2_test',
                    'wall_clock_time_minutes']



####################################



def run_combine_results_files(root_folder_repository):
    # This function is intended as initialization, which makes the
    # combined.csv files for each subfolder and also creates the
    # plotting folder: '0output'.
    
    for item in os.listdir(root_folder_repository):
        
        subdir = os.path.join(root_folder_repository, item)
        
        if os.path.isdir(subdir) and 'output' in item:
            
            # Create combined.csv
            if 'säästä' in item:
                # Older format
                combine_results_files_old(subdir)
            
            elif 'save' in item:
                # Newer format, added for clarity, 
                # if especially important folder are marked with '_save'
                combine_results_files(subdir)
            
            else:
                # Default option
                combine_results_files(subdir)
            
        
            # Add output folder for plotting and possible results files
            output_folder = os.path.join(subdir,
                                       '0output')
            if not os.path.exists(subdir):
                os.makedirs(output_folder)


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