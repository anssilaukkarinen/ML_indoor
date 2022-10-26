# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 22:53:05 2022

@author: laukkara

"""

import os
import time
import itertools
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
import json



fname_xlsx_input = os.path.join(r'C:\Local\laukkara\Data\ML_indoor_Narvi',
                                'df_all_2022-10-25-22-55-35.xlsx')

df = pd.read_excel(fname_xlsx_input)


output_folder = os.path.join(r'C:\Local\laukkara\Data\github\ML_indoor',
                             'myPostAnalysis')
if not os.path.exists(output_folder):
    os.makedirs(output_folder)



figseiz = (5.5, 3.5)
dpi_val = 200
markers = itertools.cycle(['o', '^', 's', 'x', 'd'])

col_names_numeric = ['MAE_train', 'MAE_validate', 'MAE_test',
                    'RMSE_train','RMSE_validate', 'RMSE_test',
                    'R2_train', 'R2_validate', 'R2_test',
                    'wall_clock_time_minutes']




########################################

def plot_single_series(df, output_folder):
    
    # to_plot = ['MAE', 'RMSE', 'R2', 'clock']
    to_plot = ['MAE', 'RMSE', 'clock']
    
    for col in df.columns:
        
        if any(x in col for x in to_plot):
        
            fig, ax = plt.subplots(figsize=figseiz)
            df.loc[:, col].plot(ax=ax, style='.', ms=0.5)
            ax.set_ylabel(col)
            ax.set_ylim(bottom=-0.1, top=np.quantile(df.loc[:,col].values, 0.98))
            fname = os.path.join(output_folder,
                                 f'single_{col}.png')
            fig.savefig(fname, dpi=dpi_val, bbox_inches='tight')
            plt.close(fig)
        
plot_single_series(df, output_folder)






def plot_sorted(df, output_folder):
    
    linetypes = {'train': '-',
                 'validate': '--',
                 'test': '-.'}
    
    
    for key1 in ['MAE', 'RMSE', 'R2']:
        
        fig, ax = plt.subplots(figsize=figseiz)
        
        ylim_top = 0.0
        
        for key2 in ['train', 'validate', 'test']:
            
            col = key1 + '_' + key2
            
            df_single_sorted = df.loc[:, [col]].sort_values(by=col,
                                                            ascending=True,
                                                            ignore_index=True)
            
            ax.plot(df_single_sorted,
                    label=key2,
                    linestyle=linetypes[key2])
            
            # df_single_sorted.plot(label=key2,
            #                       style=linetypes[key2],
            #                       ax=ax)
            
            ylim_top = np.max((ylim_top,
                               np.quantile(df.loc[:,col].values, 0.98)))
            
        ax.set_ylabel(key1)
        ax.set_ylim(bottom=-0.1, top=ylim_top)
        ax.legend()
        fname = os.path.join(output_folder,
                             f'sorted_{key1}.png')
        fig.savefig(fname, dpi=dpi_val, bbox_inches='tight')
        plt.close(fig)

plot_sorted(df, output_folder)
    







def plot_R2_vs_RMSE(df, output_folder):

    for key in ['train', 'validate', 'test']:
        
        fig, ax = plt.subplots(figsize=figseiz)
        
        for label, grp in df.groupby(['measurement_point_name']):
            ax.scatter(x=f'RMSE_{key}', y=f'R2_{key}', \
                       s=5.0, marker=next(markers),
                       data=grp, label=label)
        ax.set_ylim((-0.1, 1.1))
        ax.set_xlim((-0.1, 3))
        ax.grid(True)
        ax.set_axisbelow(True)
        ax.legend()
        ax.set_xlabel(f'RMSE, {key}')
        ax.set_ylabel(f'R$^2$, {key}')
        fname = os.path.join(output_folder,
                             f'R2 vs RMSE {key}.png')
        fig.savefig(fname, dpi=dpi_val, bbox_inches='tight')
        plt.close(fig)

plot_R2_vs_RMSE(df, output_folder)






def plot_R2_vs_MAE(df, output_folder):

    for key in ['train', 'validate', 'test']:
        
        fig, ax = plt.subplots(figsize=figseiz)
        
        for label, grp in df.groupby(['measurement_point_name']):
            ax.scatter(x=f'MAE_{key}', y=f'R2_{key}', \
                        s=5.0, marker=next(markers),
                        data=grp, label=label)
        ax.set_ylim((-0.1, 1.1))
        ax.set_xlim((-0.1, 3))
        ax.grid(True)
        ax.set_axisbelow(True)
        ax.legend()
        ax.set_xlabel(f'MAE, {key}')
        ax.set_ylabel(f'R$^2$, {key}')
        fname = os.path.join(output_folder,
                              f'R2 vs MAE {key}.png')
        fig.savefig(fname, dpi=dpi_val, bbox_inches='tight')
        plt.close(fig)

plot_R2_vs_MAE(df, output_folder)









print('myMerge.py, END', flush=True)




