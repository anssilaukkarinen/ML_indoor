# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 22:53:05 2022

@author: laukkara

Note!
The input data was originally divided into three sections:
    training, validation and testing.
The training was eventually changed to cross-validation using only
training data, so the two other parts became available for evaluation.
The naming wasn't changed throughout the code, but only naming in
certain outputs are changed.


first set 192 shell scripts: 192*3*48*3 = 82 944
second set 48 shell scripts: 48*3*48*3 = 20 736
-> 103 680 in total

"""

import os
import time
import itertools
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
import json

pd.set_option('display.expand_frame_repr', True)

Narvi_folder = os.path.join(r'C:\Users\laukkara\Data\ML_indoor_Narvi')

#github_folder = r'C:\Local\laukkara\Data\github\ML_indoor'



font = {'family': 'arial',
        'size': 7}
matplotlib.rc('font', **font)

boxprops = dict(linewidth=0.4)
medianprops = dict(linewidth=0.4)
flierprops = dict(marker='.', markersize=1)
capprops = dict(linewidth=0.4)
whiskerprops = dict(linewidth=0.4)



R2_limit_min = -0.1

figseiz = (3.5, 2.5) # (3.5, 2.5), (5.5, 3.5)
dpi_val = 600
markers = itertools.cycle(['o', '^', 's', 'x', 'd', '.'])

colors_mp = ['C0','C1','C2','C3','C4','C5']


col_names_numeric = ['MAE_train', 'MAE_validate', 'MAE_test',
                    'RMSE_train','RMSE_validate', 'RMSE_test',
                    'R2_train', 'R2_validate', 'R2_test',
                    'wall_clock_time_minutes']


cols_dup = ['measurement_point_name', 'model_name', 'optimization_method', \
            'X_lag', 'y_lag', 'N_CV', 'N_ITER', 'N_CPU']




######################

# Read in the merger xlsx file from Narvi
fname_xlsx_input = os.path.join(Narvi_folder,
                                'merged_results',
                                'df_all.xlsx')

df = pd.read_excel(fname_xlsx_input, index_col=0)
df.sort_index(inplace=True)




# Make sure the output folder and output xlsx file exists
output_folder = os.path.join(Narvi_folder,
                             'myPostAnalysis')
if not os.path.exists(output_folder):
    os.makedirs(output_folder)


xlsx_output_file = os.path.join(output_folder,
                                'output.xlsx')

if not os.path.exists(xlsx_output_file):
    with pd.ExcelWriter(xlsx_output_file,
                        mode='w',
                        engine='openpyxl') as writer:
        # This didn't work first. You need to open a desktop excel program
        # and save an empty file with this name to this location,
        # and then the current code starts to work.
        pd.DataFrame(data=[0]).to_excel(writer, sheet_name='start')


fname_logfile = os.path.join(output_folder,
                             'logfile.txt')
log_file_handle = open(fname_logfile, mode='w')





    
# Are there duplicate rows?

n_duplicated = df.duplicated(subset=cols_dup,
                             keep='first').sum()
n_rows_total = df.shape[0]

print(f'Sum of duplicated rows: {n_duplicated}, ', \
      f'{100*n_duplicated/n_rows_total:.2f} %\n\n',
      file=log_file_handle)


    
df.drop_duplicates(subset=cols_dup,
                   keep='first',
                   ignore_index=True,
                   inplace=True)

print(f'df.shape after dropping duplicated: {df.shape}',
          file=log_file_handle)




# Convert t_start_local from object to datetime using pd.datetime()
df['t_start_local'] = pd.to_datetime(df['t_start_local'],
                                     format='%Y-%m-%d %H:%M:%S')


# Sort rows according to MAE_mean_validate_test for dummyregressor
# First measurement_point_names and then 

mps = df.loc[:,'measurement_point_name'].unique()

df_per_mp = []
df_MAE_mean_per_mp = []

for mp in mps:
    
    idxs = df['measurement_point_name'] == mp
    
    dummy = df.loc[idxs, :].copy()
    
    dummy.sort_values(by='MAE_mean_validate_test', inplace=True)
    
    idxs_for_mean = dummy['model_name'] == 'dummyregressor'
    dummy_mean = dummy.loc[idxs_for_mean, 'MAE_mean_validate_test'].mean()
    
    df_per_mp.append(dummy)
    df_MAE_mean_per_mp.append(dummy_mean)

idxs_sorted = np.argsort(df_MAE_mean_per_mp)
df_per_mp = [df_per_mp[idx] for idx in idxs_sorted]
df = pd.concat(df_per_mp, ignore_index=True)




########################################

print(df.columns)

print(df.columns, '\n\n',
      file=log_file_handle)






## Establish baseline
# This is in the manuscript

idxs = df['model_name'] == 'dummyregressor'
df_dummyregressor_sorted = df.loc[idxs, :] \
                            .reset_index(drop=True) \
                            .loc[:, ['measurement_point_name',
                                     'MAE_train',
                                     'MAE_validate',
                                     'MAE_test',
                                     'MAE_mean_validate_test']]



# Baseline, numerically

MAE_mean_validate_test_mp = df_dummyregressor_sorted \
                            .groupby(by='measurement_point_name') \
                            .mean() \
                            .sort_values(by='MAE_mean_validate_test')
    

print('DummyRegressor mean per measurement_point_name:', \
      file=log_file_handle)
MAE_mean_validate_test_mp.round(2).to_csv(log_file_handle,
                                          mode='a',
                                          index=True,
                                          header=True,
                                          sep=',',
                                          lineterminator='\n')
baseline_val = MAE_mean_validate_test_mp \
    .loc[:,['MAE_validate','MAE_test']].stack().mean()
print(f'Baseline pred1 pred2 mean = {baseline_val:.4f}',
      file=log_file_handle)

print('\n\n', file=log_file_handle)





# Baseline, plot
print('MAE_mean_validate_test:', file=log_file_handle)

mps = MAE_mean_validate_test_mp.index.values


fig, ax = plt.subplots(figsize=figseiz)

for idx_mp, mp_val in enumerate(mps):
    
    idxs = df_dummyregressor_sorted['measurement_point_name'] == mp_val
    
    holder1 = df_dummyregressor_sorted.loc[idxs, 'MAE_test']
    holder1.plot(ax=ax, style=':', color=colors_mp[idx_mp])
    
    holder2 = df_dummyregressor_sorted.loc[idxs, 'MAE_validate']
    holder2.plot(ax=ax, style='--', color=colors_mp[idx_mp])
        
    holder3 = df_dummyregressor_sorted.loc[idxs, 'MAE_train']
    holder3.plot(ax=ax, style='-', color=colors_mp[idx_mp])
        
    print(f'mp={mp_val}',
          f'test={holder1.mean():.2f}',
          f'validate={holder2.mean():.2f}',
          f'train={holder3.mean():.2f}',
          file=log_file_handle)

print('\n\n', file=log_file_handle)
    
ax.set_ylim(bottom=0.0, top=2.0)
ax.grid()
ax.set_xlabel('Cases using DummyRegressor')
ax.set_ylabel('MAE')
ax.legend(['pred2','pred1','train'], loc='lower right')

fname = os.path.join(output_folder,
                     'baseline.jpg')
fig.savefig(fname, dpi=dpi_val, bbox_inches='tight')
plt.close(fig)







## Scatter plot, all measurement_points with same color, not used
# R2 vs MAE, all points
fig, ax = plt.subplots(figsize=figseiz)
df.plot.scatter(x='MAE_mean_validate_test',
                y='R2_mean_validate_test',
                s=0.3,
                ax=ax)
ax.grid()
ax.set_xlabel('MAE')
ax.set_ylabel('$R^2$')
fname = os.path.join(output_folder,
                     'scatter_all_not_limited_R2_vs_MAE.jpg')
fig.savefig(fname, dpi=dpi_val, bbox_inches='tight')
plt.close(fig)




# R2 vs MAE, all points
fig, ax = plt.subplots(figsize=figseiz)
df.plot.scatter(x='MAE_mean_validate_test',
                y='R2_mean_validate_test',
                s=0.3,
                ax=ax)
ax.grid()
ax.set_xlim((-0.1, 3))
ax.set_xlabel('MAE')
ax.set_ylim((-0.1, 1.1))
ax.set_ylabel('$R^2$')
fname = os.path.join(output_folder,
                     'scatter_all_limited_axis_R2_vs_MAE.jpg')
fig.savefig(fname, dpi=dpi_val, bbox_inches='tight')
plt.close(fig)









## Scatter plots of single output variables

def plot_all_cases_by_index(df, output_folder):
    # Plot scatter plot of values against index
    
    print('plot_all_cases_by_index:', file=log_file_handle)
    
    to_plot = ['MAE', 'RMSE', 'R2', 'clock']
    
    for col in df.columns:
        
        if any(x in col for x in to_plot):
        
            fig, ax = plt.subplots(figsize=figseiz)
            
            for idx_mp, val_mp in enumerate(df['measurement_point_name'].unique()):
                
                idxs = df['measurement_point_name'] == val_mp
                print(f'{col:s} {val_mp:s} idxs.sum() = {idxs.sum():.0f}',
                      file=log_file_handle)
                
                df.loc[idxs, col].plot(ax=ax,
                                       style='.',
                                       ms=0.4,
                                       color=colors_mp[idx_mp])
            
            
            if 'R2' in col:
                ylim_top = 1.1
            else:
                ylim_top = 5.0
            ax.set_ylim(bottom=-0.1, top=ylim_top)
            ax.set_xlabel('Case index (all cases)')
            
            ylabel_str = col.replace('_',' ') \
                            .replace('validate','pred1') \
                            .replace('test', 'pred2')
            
            ax.set_ylabel(ylabel_str)
            
            ax.grid()
            
            col_str = col.replace('validate','pred1').replace('test', 'pred2')
            
            fname = os.path.join(output_folder,
                                 f'all_cases_by_index_{col_str}.jpg')
            fig.savefig(fname, dpi=dpi_val, bbox_inches='tight')
            plt.close(fig)
        
plot_all_cases_by_index(df, output_folder)








# Scatter plots of single output variables, but in sorted order

def plot_single_series_sorted(df, output_folder):
    
    for key1 in ['MAE', 'RMSE', 'R2']:
        
        # Select only a subset of data
        
        cols = [key1 + '_train',
                key1 + '_validate',
                key1 + '_test']
        
        df_full = df.loc[:, cols].copy()
        
        
        # Sort rows
        
        if key1 == 'R2':
            dummy_ylabel= 'R$^2$'
            ylim_top = 1.1
            is_ascending = False
        else:
            dummy_ylabel = key1
            ylim_top = 5.0
            is_ascending = True
        
        idx_new = df_full.loc[:, cols[1:]].mean(axis=1) \
                        .sort_values(ascending=is_ascending).index
        
        df_full = df_full.reindex(index=idx_new).reset_index(drop=True)
        
        
        
        # Plot, no averaging
        fig, ax = plt.subplots(figsize=figseiz)
        ax.plot(df_full,
                lw=0.4)
        ax.set_ylabel(dummy_ylabel)
        ax.set_xlabel('Cases, sorted')
        ax.set_ylim(bottom=-0.1, top=ylim_top)
        
        dummy_legend = [x.replace('_',', ') for x in df_full.columns]
        ax.legend(dummy_legend)
        fname = os.path.join(output_folder,
                             f'sorted_notgrouped_{key1}.jpg')
        fig.savefig(fname, dpi=dpi_val, bbox_inches='tight')
        plt.close(fig)
        
        
        
        # Plot, using median plus high and low quantiles
        q_list = ['0.05', '0.5', '0.95']
        col_list = ['C0', 'C1', 'C0']
        linestyle_list = ['--', '-', '--']
        
        n_splits = 200
        dfs_as_list = np.array_split(df_full, n_splits)
        
        
        for col in cols:
            
            col_str = col.replace('_',' ')
            print(col, col_str)
            
            
            fig, ax = plt.subplots(figsize=figseiz)
            
            y_vals = np.zeros((n_splits, 3))
            
            for idx_split, df_split in enumerate(dfs_as_list):
        
                y_vals[idx_split, 0] = df_split.loc[:,col].quantile(0.05)
                y_vals[idx_split, 1] = df_split.loc[:,col].quantile(0.5)
                y_vals[idx_split, 2] = df_split.loc[:,col].quantile(0.95)
            
            
            for idx_fmt in range(3):
                
                ax.plot(y_vals[:, idx_fmt],
                        label=f'{col_str}, {q_list[idx_fmt]}',
                        color=col_list[idx_fmt],
                        linestyle=linestyle_list[idx_fmt])
        
            ax.set_ylabel(dummy_ylabel)
            ax.set_xlabel('Sorted cases divided into groups')
            ax.set_ylim(bottom=-0.1, top=ylim_top)
            
            ax.legend()
            fname = os.path.join(output_folder,
                                 f'sorted_grouped_{col}.jpg')
            fig.savefig(fname, dpi=dpi_val, bbox_inches='tight')
            plt.close(fig)

    
    return(df_full)

df_full = plot_single_series_sorted(df, output_folder)








def plot_scatter_train_validate_test(df):
    
    fig, ax = plt.subplots(figsize=figseiz)
    
    idxs = (df['R2_validate'] >= R2_limit_min) \
            & (df['R2_test'] >= R2_limit_min)
    
    df.loc[idxs, :].plot.scatter(x='R2_validate',
                                 y='R2_test',
                                 s=0.4,
                                 ax=ax)
    
    ax.set_xlim(left=-0.1, right=1.5)
    ax.set_ylim(bottom=-0.1, top=1.5)
    
    ax.grid()
    
    fname = os.path.join(output_folder,
                         'correlation_MAE_test_vs_validate.jpg')
    fig.savefig(fname, dpi=dpi_val, bbox_inches='tight')
    plt.close(fig)

plot_scatter_train_validate_test(df)









## scatter plots colored be measurement_point


def plot_R2_vs_RMSE(df, output_folder):
    # Three scatter plots where the data is nicely in groups (lines)
    # for each measurement_point

    for key in ['train', 'validate', 'test']:
        
        fig, ax = plt.subplots(figsize=figseiz)
        
        for mp in df['measurement_point_name'].unique():
            
            idxs = (df['measurement_point_name'] == mp)
            
            df_group = df.loc[idxs, :]
            
            ax.scatter(x=f'RMSE_{key}', y=f'R2_{key}', \
                       s=0.5, marker=next(markers),
                       data=df_group, label=mp)
        ax.set_ylim((-0.1, 1.1))
        ax.set_xlim((-0.1, 2.5))
        ax.grid(True)
        ax.set_axisbelow(True)
        ax.legend()
        ax.set_xlabel(f'RMSE, {key}')
        ax.set_ylabel(f'R$^2$, {key}')
        fname = os.path.join(output_folder,
                             f'R2 vs RMSE {key}.jpg')
        fig.savefig(fname, dpi=dpi_val, bbox_inches='tight')
        plt.close(fig)

plot_R2_vs_RMSE(df, output_folder)





def plot_R2_vs_MAE(df, output_folder):
    # Basically the same as previous, but with a little bit more variation
    # R2 is calculated using squared errors
    # This is in manuscript

    for key in ['train', 'validate', 'test']:
        
        fig, ax = plt.subplots(figsize=figseiz)
        
        # The commented line produced 1-tuple error
        # for label, grp in df.groupby(['measurement_point_name']):
        for mp in df['measurement_point_name'].unique():
            
            idxs = (df['measurement_point_name'] == mp)
            df_group = df.loc[idxs, :]
            
            ax.scatter(x=f'MAE_{key}', y=f'R2_{key}', \
                        s=0.5, marker=next(markers),
                        data=df_group, label=mp)
        ax.set_ylim((-0.1, 1.1))
        ax.set_xlim((-0.1, 2.5))
        ax.grid(True)
        ax.set_axisbelow(True)
        ax.legend()
        
        xlabel_str = f'MAE, {key}' \
                        .replace('validate','pred1') \
                        .replace('test', 'pred2')
        
        ax.set_xlabel(xlabel_str)
        
        ylabel_str = f'R$^2$, {key}' \
                        .replace('validate','pred1') \
                        .replace('test', 'pred2')
        
        ax.set_ylabel(ylabel_str)
        
        key_str = key.replace('validate','pred1').replace('test','pred2')
        
        fname = os.path.join(output_folder,
                              f'R2 vs MAE {key_str}.jpg')
        fig.savefig(fname, dpi=dpi_val, bbox_inches='tight')
        plt.close(fig)

plot_R2_vs_MAE(df, output_folder)

# The R2-vs-RMSE and R2-vs-MAE show that different measurement_points form
# clusters. This means that the reported accuracy is data-dependent and proper
# evaluation of methods should be done using sufficiently large population of
# data sets and enough variation between them (to improve prediction accuracy).









def plot_boxplot_by_mp(df, output_folder):
    # Boxplots with five box-and-whiskers plots
    # The y-data is the main indicators
    
    print('plot_boxplot_by_mp:', file=log_file_handle)
    
    idxs = (df.loc[:, 'R2_validate'] > R2_limit_min) \
            & (df.loc[:,'R2_test'] > R2_limit_min)
    print('idxs.sum():', idxs.sum(),
          file=log_file_handle)
    
    df_dummy = df.loc[idxs, :].groupby('measurement_point_name').count().iloc[:,0]
    print('df_dummy:', 
          file=log_file_handle)
    print(df_dummy,
          file=log_file_handle)
    
    with pd.ExcelWriter(xlsx_output_file,
                        mode='a',
                        engine='openpyxl',
                        if_sheet_exists='replace') as writer:
        df_dummy.to_excel(writer,
                          sheet_name='boxp_by_mp')
    
    
    for y_target in ['R2_mean_validate_test',
                   'MAE_mean_validate_test',
                   'RMSE_mean_validate_test']:
        
        fig, ax = plt.subplots(figsize=figseiz)
        df.loc[idxs, :].boxplot(column=y_target,
                            by='measurement_point_name',
                            rot=45,
                            boxprops=boxprops,
                            medianprops=medianprops,
                            whiskerprops=whiskerprops,
                            capprops=capprops,
                            flierprops=flierprops,
                            ax=ax)
        ax.set_ylim(bottom=-0.1)
        
        if y_target == 'R2_mean_validate_test':
            ax.set_ylim(top=1.1)
        else:
            ax.set_ylim(top=1.4)
        
        ax.set_title('')
        fig.suptitle('')
        ax.set_xlabel('Measurement point name')
        
        y_target_str = y_target.replace('R2','R$^2$') \
                        .replace('_mean_validate_test', ', mean') \
                    + '\n R$^2_{validate}$ > 0 and R$^2_{test}$ > 0'
        ax.set_ylabel(y_target_str)
        
        fname = os.path.join(output_folder,
                             f'boxplot_mp_{y_target}.jpg')
        fig.savefig(fname, dpi=dpi_val, bbox_inches='tight')
        plt.close(fig)

print('\n\n', file=log_file_handle)        
plot_boxplot_by_mp(df, output_folder)










def plot_boxplot_by_mp_and_other(df, output_folder):
    # Boxplots with two controlled variables
    # measurement point and Xlag/ylag/opt/N_CV
    # This is in manuscript
    
    R2_limit_min_this = 0.0
    
    print('plot_boxplot_ny_mp_and_other',
          file=log_file_handle)
    
    idxs = (df.loc[:, 'R2_validate'] > R2_limit_min_this) \
            & (df.loc[:,'R2_test'] > R2_limit_min_this)
    print('idxs.sum():', idxs.sum(),
          file=log_file_handle)
    
    
    xlabelstr = {'X_lag': 'Xlag',
                 'y_lag': 'ylag',
                 'optimization_method': 'optimization method',
                 'N_CV': '$N_{CV}$',
                 'N_ITER': '$N_{ITER}$'}
    
    for y_target in ['R2_mean_validate_test',
                   'MAE_mean_validate_test',
                   'RMSE_mean_validate_test']:
        
        for y_key2 in ['X_lag', 'y_lag', 'optimization_method', 'N_CV', 'N_ITER']:
            
            df_dummy = df.loc[idxs, :] \
                      .groupby(by=['measurement_point_name', y_key2]) \
                      .count().iloc[:,0]
            print('\nGroup counts:',
                  file=log_file_handle)
            print(df_dummy,
                  file=log_file_handle)
            with pd.ExcelWriter(xlsx_output_file,
                                mode='a',
                                if_sheet_exists='replace',
                                engine='openpyxl') as writer:
                sh_name = 'count_{}_{}'.format(y_target.split('_')[0],
                                            y_key2)
                df_dummy.to_excel(writer,
                                  sheet_name=sh_name)
            
            
            fig, ax = plt.subplots(figsize=figseiz)
            df.loc[idxs, :].boxplot(column=y_target,
                                by=['measurement_point_name', y_key2],
                                rot=90,
                                boxprops=boxprops,
                                medianprops=medianprops,
                                whiskerprops=whiskerprops,
                                capprops=capprops,
                                flierprops=flierprops,
                                ax=ax)
            
            ax.set_ylim(bottom=-0.1)
            
            if y_target == 'R2_mean_validate_test':
                ax.set_ylim(top=1.1)
            else:
                ax.set_ylim(top=1.2)
            
            ax.set_title('')
            fig.suptitle('')
            ax.set_xlabel('({}, {})'.format('Measurement point name',
                                          xlabelstr[y_key2]))
            
            y_target_str = y_target.replace('R2','R$^2$') \
                            .replace('_mean_validate_test', ', mean') \
                        + '\n ' \
                        + '$R^2_{pred1}$' + f' > {R2_limit_min_this:.0f}' \
                        + ' and $R^2_{pred2}$' + f' > {R2_limit_min_this:.0f}'
            
            
            ax.set_ylabel(y_target_str)
            
            y_target_fname = y_target \
                            .replace('validate', 'pred1') \
                            .replace('test', 'pred2')
            
            fname = os.path.join(output_folder,
                                 f'boxplot2_mp_{y_key2}_{y_target_fname}.jpg')
            fig.savefig(fname, dpi=dpi_val, bbox_inches='tight')
            plt.close(fig)

print('\n\n', file=log_file_handle)
plot_boxplot_by_mp_and_other(df, output_folder)





# Calculate differences between otherwise same cases,
# but with different N_CV or N_ITER


def calc_differences(df):
    
    print('calc_differences:', file=log_file_handle)
    
    idxs = (df['R2_validate'] >= R2_limit_min) \
            & (df['R2_test'] >= R2_limit_min)
    df2 = df.loc[idxs, :].copy()
    
    # N_ITER
    for mp_val in df2['measurement_point_name'].unique():
        
        for mod_val in df2['model_name'].unique():
            
            for opt_val in df2['optimization_method'].unique():
                
                for Xlag_val in df2['X_lag'].unique():
                    
                    for ylag_val in df2['y_lag'].unique():
                        
                        for NCV_val in df2['N_CV'].unique():
                            
                            idxs_low = (df2['measurement_point_name']==mp_val) \
                                    & (df2['model_name']==mod_val) \
                                    & (df2['optimization_method']==opt_val) \
                                    & (df2['X_lag']==Xlag_val) \
                                    & (df2['y_lag']==ylag_val) \
                                    & (df2['N_CV']==NCV_val) \
                                    & (df2['N_ITER'] == 50)
                            
                            idxs_high = (df2['measurement_point_name']==mp_val) \
                                    & (df2['model_name']==mod_val) \
                                    & (df2['optimization_method']==opt_val) \
                                    & (df2['X_lag']==Xlag_val) \
                                    & (df2['y_lag']==ylag_val) \
                                    & (df2['N_CV']==NCV_val) \
                                    & (df2['N_ITER'] == 500)
                            
                            df_diff = df2.loc[idxs_high,
                                             'MAE_mean_validate_test'].min() \
                                        -df2.loc[idxs_low,
                                            'MAE_mean_validate_test'].min()
                            
                            if np.abs(df_diff) > 0.25:
                                print(mp_val, mod_val, opt_val, Xlag_val,
                                      ylag_val, NCV_val, df_diff,
                                      file=log_file_handle)
# calc_differences(df)
# print('\n\n', file=log_file_handle)







                            











def calculate_average_time_per_optimization_method(df):
    # List of wall_clock_time
    
    print('calculate_average_time_per_optimization_method:',
          file=log_file_handle)

    df_holder = df.groupby(by=['optimization_method']) \
                .mean(numeric_only=True) \
                 .loc[:,'wall_clock_time_minutes']
    
    print('Average time per optimization_method:',
          file=log_file_handle)
    df_dummy = df_holder.sort_values(ascending=True).round(1)
    print(df_dummy,
          file=log_file_handle)

    with pd.ExcelWriter(xlsx_output_file,
                        mode='a',
                        if_sheet_exists='replace',
                        engine='openpyxl') as writer:
    
        df_dummy.to_excel(writer,
                          sheet_name='avg_time')

print('\n\n', file=log_file_handle)
calculate_average_time_per_optimization_method(df)











def calculate_pivot_table_averages_per_MLO(df, include_all):
    # Analyse data using pivot tables
    # Four tables for different variables
    # If all data was included, then the badly off values made it more
    # difficult to analysed the data -> option to include or not to include.
    
    # Counts are in the manuscript
    # The table in the manuscript is created by hand in the spreadsheet program
    
    if include_all:
        idxs = df.index
        
        key_pivot = 'all'
    else:
        idxs = (df['R2_validate'] > R2_limit_min) \
             & (df['R2_test'] > R2_limit_min)
        
        key_pivot = 'pos'
    
    
    # Counts
    
    print(f'pivot table counts {key_pivot}:',
          file=log_file_handle, flush=True)
    print('\n', file=log_file_handle)
    
    df_holder = df.loc[idxs, ['model_name',
                               'optimization_method',
                               'MAE_mean_validate_test']]
    
    df_pivot = pd.pivot_table(data=df_holder,
                                index='model_name',
                                columns='optimization_method',
                                aggfunc='count')
    
    index_sorted = df_pivot.sum(axis=1) \
                    .sort_values(ascending=False).index
    df_pivot = df_pivot.loc[index_sorted, :]
    
    
    print(df_pivot.to_string(),
          file=log_file_handle)
    print('\n', file=log_file_handle)
    
    with pd.ExcelWriter(xlsx_output_file,
                        mode='a',
                        if_sheet_exists='replace',
                        engine='openpyxl') as writer:
        
        sh_name = 'pivot_' + key_pivot + '_count'
        df_pivot.to_excel(writer,
                          sheet_name=sh_name)
    
    
    
    
    # Mean of indicators
    
    for y_key in ['wall_clock_time_minutes',
                'RMSE_mean_validate_test',
                'MAE_mean_validate_test',
                'R2_mean_validate_test']:
        
        df_holder = df.loc[idxs, ['model_name',
                                   'optimization_method',
                                   y_key]]
        
        df_pivot = pd.pivot_table(data=df_holder,
                                    index='model_name',
                                    columns='optimization_method',
                                    aggfunc=np.mean)
        
        cols = [(y_key, 'pso'), 
                (y_key, 'randomizedsearchcv'), 
                (y_key, 'bayessearchcv')]
        df_pivot = df_pivot.loc[:, cols]
        
        df_pivot.sort_values(by=(y_key, 'pso'),
                             ascending=('R2' not in y_key),
                             inplace=True)
        
        print(f'pivot_table results{y_key}:',
              file=log_file_handle)

        print(df_pivot.round(2).to_string(),
              file=log_file_handle)
        
        
        with pd.ExcelWriter(xlsx_output_file,
                            mode='a',
                            if_sheet_exists='replace',
                            engine='openpyxl') as writer:
            
            sh_name = 'pivot_' + key_pivot + '_' + y_key.split('_')[0]
            df_pivot.round(2).to_excel(writer,
                                       sheet_name=sh_name)

print('\n\n', file=log_file_handle)
calculate_pivot_table_averages_per_MLO(df, True)
calculate_pivot_table_averages_per_MLO(df, False)














def plot_MAE_vs_wall_clock_time(df, output_folder, bool_limit_axis):
    
    # This is in the manuscript
    
    fig, ax = plt.subplots(figsize=figseiz)
    
    df.groupby(by=['model_name']).mean(numeric_only=True) \
        .plot.scatter(x='wall_clock_time_minutes',
                      y='MAE_mean_validate_test',
                      s=1.0,
                      ax=ax)
    
    if bool_limit_axis:
        ax.set_ylim(0,3.0) # (0, 1.5)
        ax.set_xlim(-1, 100)
        
        arg1 = 'limited'
    else:
        arg1 = 'notLimited'
    
    ax.set_xlabel('wall clock time, minutes')
    ax.set_ylabel('MAE mean pred1 pred2')
    
    fname = os.path.join(output_folder,
                         f'MAE_vs_wall_clock_time_{arg1:s}.jpg')
    fig.savefig(fname, dpi=dpi_val, bbox_inches='tight')
    plt.close(fig)

plot_MAE_vs_wall_clock_time(df, output_folder, False)
plot_MAE_vs_wall_clock_time(df, output_folder, True)











def plot_wall_clock_time_by_grouping(df, output_folder):
    
    print('plot_wall_clock_time_by_grouping:',
          file=log_file_handle)
    
    
    for key_grouper in ['model_name', 'optimization_method', 'N_ITER']:
    
        fig, ax = plt.subplots(figsize=figseiz)

        df_holder = df.loc[:, [key_grouper, 'wall_clock_time_minutes'] ] \
                     .groupby(by=[key_grouper]).mean(numeric_only=True) \
                     .loc[:,'wall_clock_time_minutes']
        
        df_dummy = df_holder.sort_values(ascending=True)
        df_dummy.sort_values(inplace=True)
        
        # print(df_dummy.round(1),
        #       file=log_file_handle)
        
        with pd.ExcelWriter(xlsx_output_file,
                            mode='a',
                            if_sheet_exists='replace',
                            engine='openpyxl') as writer:
            
            # WCT = Wall Clock Time, i.e. wall_clock_time_minutes
            sh_name = 'WCT_{}'.format(key_grouper)
            df_dummy.round(1).to_excel(writer,
                              sheet_name=sh_name)
        
        df_dummy.plot(rot=90,
                        style='-o',
                        ms=3,
                        ax=ax)
        
        fname = os.path.join(output_folder,
                             'wall_clock_time all {}.jpg'.format(key_grouper))
        fig.savefig(fname, dpi=dpi_val, bbox_inches='tight')
        plt.close(fig)

print('\n\n', file=log_file_handle)
plot_wall_clock_time_by_grouping(df, output_folder)
# svrpoly, nusvrpoly, kernelridgesigmoid were very slow
# pso 10 min, rand 21 min, bayes 31 min
# wall_clock_time / (N_iter/100) has a minimum at N_ITER = ~200-300










def plot_var_by_grouping_and_mp(df, output_folder, limit_to_R2_positive):
    # Other variables were added afterwards
    # Some of the cases had results way off, which distorts the results
    # overall. These are however kept here for documentation purposes.
    
    print('plot_var_by_grouping_and_mp:',
          file=log_file_handle)
    
    cols = ['wall_clock_time_minutes',
            'MAE_mean_validate_test',
            'RMSE_mean_validate_test',
            'R2_mean_validate_test']
    
    xticks_max = 0
    
    if limit_to_R2_positive:
        # Calculate ranks only for better-than-benchmark cases
        print(f'vars only for R2_validate > {R2_limit_min:.2f} ' \
              f'and R2_test > {R2_limit_min:.2f}:', flush=True,
              file=log_file_handle)
        idxs = (df['R2_validate'] > R2_limit_min) \
                & (df['R2_test'] > R2_limit_min)
        fname_key = 'pos'
    else:
        # Include all calculated cases in the ranking
        print('vars among all cases:', flush=True,
              file=log_file_handle)
        idxs = df.index == df.index
        fname_key = 'all'
    
    
    for key_grouper in ['model_name', 'optimization_method', 'N_ITER']:
        
        for col in cols:
            
            # Collect and then plot
            df_list = []
            
            for mp in df['measurement_point_name'].unique():
                
                # Get rows per measurement_point
                idxs_mp = idxs & (df['measurement_point_name'] == mp)
                
                df_holder = df.loc[idxs_mp, :].loc[:, [key_grouper, col] ] \
                              .groupby(by=[key_grouper]).mean() \
                              .loc[:, col].copy()
                
                df_list.append(df_holder)
                
            
            # sort
            df_full = pd.concat(df_list, axis=1)
            df_full = df_full.reindex(index \
                        =df_full.mean(axis=1).sort_values(ascending=True).index)
            
            # plot
            fig, ax = plt.subplots(figsize=figseiz)
                
            df_full.plot(rot=90,
                            style='-o',
                            ms=3,
                            ax=ax)
                
            ax.set_ylabel(col)
            if ('MAE' in col or 'RMSE' in col) and (fname_key == 'pos'):
                ax.set_ylim(bottom=-0.4, top=1.2)
            elif 'R2' in col:
                ax.set_ylim(bottom=-0.1, top=1.1)
            
            if key_grouper == 'model_name':
                xticks_max_new = df_holder.shape[0]
                xticks_max = np.max((xticks_max, xticks_max_new))
                    
            
            
            # Write to excel
            with pd.ExcelWriter(xlsx_output_file,
                                mode='a',
                                if_sheet_exists='replace',
                                engine='openpyxl') as writer:
                sh_name = 'gr_{}_{}_{}'.format(key_grouper[0:4],
                                               col.split('_')[0],
                                               fname_key)
                df_full.to_excel(writer,
                                   sheet_name=sh_name)
            
                        
            if key_grouper == 'model_name':
                ax.set_xticks(np.arange(0, xticks_max, 5))
                ax.set_xticklabels([])
                ax.set_xlabel('Cases, sorted')
            
            if 'R2' in col:
                ax.set_ylabel('R$^2$ mean')
            elif any(x in col for x in ['MAE', 'RMSE']):
                y_label_str = col.split('_')[0] + ' mean'
                ax.set_ylabel(y_label_str)
            
            fname = os.path.join(output_folder,
                                 'grouped {} {} {}.jpg'.format(fname_key,
                                                               key_grouper,
                                                               col))
            fig.savefig(fname, dpi=dpi_val, bbox_inches='tight')
            plt.close(fig)
        
print('\n\n', file=log_file_handle)
plot_var_by_grouping_and_mp(df, output_folder, True)
plot_var_by_grouping_and_mp(df, output_folder, False)







def impact_from_NCV(df):
    
    print('Impact of N_CV on the prediction accuracy:',
          file=log_file_handle)

    idxs = (df['R2_validate'] >= R2_limit_min) \
            & (df['R2_test'] >= R2_limit_min)
    df_pos = df.loc[idxs, :].copy()
    
    grouping = df_pos.groupby(by=['measurement_point_name',
                              'model_name',
                              'optimization_method',
                              'X_lag',
                              'y_lag',
                              'N_ITER'])
    
    dummy = []
    
    for group in grouping:
        
        if group[1].shape[0] >= 2:
            
            mae_mvt = group[1].sort_values(by='N_CV')['MAE_mean_validate_test'].values
            
            mae_mvt_range = mae_mvt[-1] - mae_mvt[0]
            
            if False:
                
                # This is a hack, set the condition to True or False to
                # get either all numbers or just some part of them
                
                print(group[0],
                      file=log_file_handle)
                
                dummy.append(mae_mvt_range)
            
            elif np.abs(mae_mvt_range) > 0.05:
                
                print(group[0],
                      file=log_file_handle)
            
                dummy.append(mae_mvt_range)
    
    
    print('\n\n', file=log_file_handle, flush=True)
    return(dummy)

dummy_impactFromNCV = impact_from_NCV(df)






def impact_from_NITER(df, output_folder):
    
    print('Impact of N_ITER on the prediction accuracy:',
          file=log_file_handle)

    idxs = (df['R2_validate'] >= R2_limit_min) \
            & (df['R2_test'] >= R2_limit_min)
    df_pos = df.loc[idxs, :].copy()
    
    grouping = df_pos.groupby(by=['measurement_point_name',
                              'model_name',
                              'optimization_method',
                              'X_lag',
                              'y_lag',
                              'N_CV'])

    dummy = []
    
    fig, ax = plt.subplots(figsize=figseiz)
    
    for group in grouping:
        
        if group[1].shape[0] >= 3:
            
            # Identify those where increasing N_ITER had impact on MAE
            
            mae_mvt = group[1].sort_values(by='N_ITER')['MAE_mean_validate_test'].values
            
            mae_mvt_range = mae_mvt[-1] - mae_mvt[0]
            
            # For plotting, comment out
            # if np.abs(mae_mvt[2] - mae_mvt[0]) > 0.15:
            if mae_mvt_range <= -0.05:
            
                group[1].sort_values(by='N_ITER') \
                    .plot.line(x='N_ITER', y='MAE_mean_validate_test',
                               ax=ax, legend=False, lw=0.4)
            
            
            # Gather those where it is worth the while to increase N_ITER
            if mae_mvt_range <= -0.05:
                
                print(group[0],
                      file=log_file_handle,
                      flush=True)
                
                dummy.append(group[0])
        
        
    ax.set_ylim(bottom=0.4, top=1.2)
    ax.set_xlim(left=0, right=550)
    
    fname = os.path.join(output_folder,
                         'MAE_vs_NITER.jpg')
    fig.savefig(fname, dpi=dpi_val, bbox_inches='tight')
    plt.close(fig)
    
    print('\n\n', file=log_file_handle, flush=True)
    return(dummy)

dummy = impact_from_NITER(df, output_folder)













def impact_from_opt(df, output_folder):
    
    print('Impact of optimization_method on the prediction accuracy:',
          file=log_file_handle)

    idxs = (df['R2_validate'] >= R2_limit_min) \
            & (df['R2_test'] >= R2_limit_min)
    df_pos = df.loc[idxs, :].copy()
    
    grouping = df_pos.groupby(by=['measurement_point_name',
                              'model_name',
                              'X_lag',
                              'y_lag',
                              'N_CV',
                              'N_ITER'])

    dummy = []
    
    fig, ax = plt.subplots(figsize=figseiz)
    
    for group in grouping:
        
        if group[1].shape[0] >= 3:
            
            # Identify those where optimization_method had impact on MAE
            
            mae_mvt = group[1].sort_values(by='optimization_method') \
                        ['MAE_mean_validate_test'].values
            
            mae_mvt_range = mae_mvt[-1] - mae_mvt[0]
            
            # For plotting, comment out
            # if np.abs(mae_mvt_range) > 0.15:
            if mae_mvt_range <= -0.05:
            
                group[1].sort_values(by='optimization_method') \
                    .plot(x='optimization_method', y='MAE_mean_validate_test',
                               ax=ax, legend=False, lw=0.4)
            
            
            # Gather those where it is worth the while to increase N_ITER
            if mae_mvt_range <= -0.05:
                
                print(group[0],
                      file=log_file_handle,
                      flush=True)
                
                dummy.append(group[0])
        
        
    ax.set_ylim(bottom=0.4, top=1.2)
    ax.set_xlim(left=0, right=550)
    
    fname = os.path.join(output_folder,
                         'MAE_vs_optimizationMethod.jpg')
    fig.savefig(fname, dpi=dpi_val, bbox_inches='tight')
    plt.close(fig)
    
    print('\n\n', file=log_file_handle, flush=True)
    return(dummy)

dummy = impact_from_opt(df, output_folder)












def impact_from_lag(df, arg1, output_folder):
    
    print(f'Impact of {arg1} on the prediction accuracy:',
          file=log_file_handle)

    idxs = (df['R2_validate'] >= R2_limit_min) \
            & (df['R2_test'] >= R2_limit_min)
    df_pos = df.loc[idxs, :].copy()
    
    
    if arg1 == 'y_lag':
        the_other = 'X_lag'
        cutpoint = 0.03
    elif arg1 == 'X_lag':
        the_other = 'y_lag'
        cutpoint = 0.06
    else:
        the_other = 'foobar'
    
    grouping = df_pos.groupby(by=['measurement_point_name',
                              'model_name',
                              'optimization_method',
                              'N_CV',
                              'N_ITER',
                              the_other])

    dummy_improve = []
    dummy_weaken = []
    dummy_nothing = []
    
    counter_improve = 0
    counter_weaken = 0
    counter_nothing = 0
    
    fig_improve, ax_improve = plt.subplots(figsize=figseiz)
    fig_weaken, ax_weaken = plt.subplots(figsize=figseiz)
    fig_nothing, ax_nothing = plt.subplots(figsize=figseiz)
    
    for group in grouping:
        
        if group[1].shape[0] >= 2:
            
            mae_mvt = group[1].sort_values(by=arg1) \
                        ['MAE_mean_validate_test'].values
            
            mae_mvt_range = mae_mvt[-1] - mae_mvt[0]
            
            
            
            if mae_mvt_range <= -cutpoint:
                counter_improve += 1
                dummy_improve.append(group[0])
                
                group[1].sort_values(by=arg1) \
                    .plot.line(x=arg1, y='MAE_mean_validate_test',
                               ax=ax_improve, legend=False, lw=0.4,
                               style='-o', ms=0.3)
                
            elif mae_mvt_range >= cutpoint:
                counter_weaken += 1
                dummy_weaken.append(group[0])
                
                group[1].sort_values(by=arg1) \
                    .plot.line(x=arg1, y='MAE_mean_validate_test',
                               ax=ax_weaken, legend=False, lw=0.4,
                               style='-o', ms=0.3)
                
            else:
                counter_nothing += 1
                dummy_nothing.append(group[0])
                
                group[1].sort_values(by=arg1) \
                    .plot.line(x=arg1, y='MAE_mean_validate_test',
                               ax=ax_nothing, legend=False, lw=0.4,
                               style='-o', ms=0.3)
        
    # improve
    ax_improve.set_ylim(bottom=0.4, top=1.2)
    ax_improve.set_ylabel('MAE mean validate test')
    fname = os.path.join(output_folder,
                         f'MAE_vs_{arg1} improve.jpg')
    fig_improve.savefig(fname, dpi=dpi_val, bbox_inches='tight')
    plt.close(fig_improve)
    
    # weaken
    ax_weaken.set_ylim(bottom=0.4, top=1.2)
    ax_weaken.set_ylabel('MAE mean validate test')
    fname = os.path.join(output_folder,
                         f'MAE_vs_{arg1} weaken.jpg')
    fig_weaken.savefig(fname, dpi=dpi_val, bbox_inches='tight')
    plt.close(fig_weaken)
    
    # nothing
    ax_nothing.set_ylim(bottom=0.4, top=1.2)
    ax_nothing.set_ylabel('MAE mean validate test')
    fname = os.path.join(output_folder,
                         f'MAE_vs_{arg1} nothing.jpg')
    fig_nothing.savefig(fname, dpi=dpi_val, bbox_inches='tight')
    plt.close(fig_nothing)
    
    
    
    print(f'counter_improve = {counter_improve}:',
          file=log_file_handle, flush=True)
    for val in dummy_improve:
        print(val, file=log_file_handle)
    print('\n', file=log_file_handle)


    print(f'counter_weaken = {counter_weaken}:',
          file=log_file_handle, flush=True)
    for val in dummy_weaken:
        print(val, file=log_file_handle)
    print('\n', file=log_file_handle)
    
    print(f'counter_nothing = {counter_nothing}:',
          file=log_file_handle, flush=True)
    for val in dummy_nothing:
        print(val, file=log_file_handle)
    print('\n', file=log_file_handle)
    
    
    print('\n\n', file=log_file_handle, flush=True)
    return(dummy_improve, dummy_weaken, dummy_nothing)


dummy_improve_X, dummy_weaken_X, dummy_nothing_X \
    = impact_from_lag(df, 'X_lag', output_folder)

dummy_improve_y, dummy_weaken_y, dummy_nothing_y \
    = impact_from_lag(df, 'y_lag', output_folder)











def calc_group_sizes(list_of_tup, arg1):
    
    print(f'calc_group_sizes: {arg1}',
          file=log_file_handle)
    
    # Fill dict
    res = {}
    
    for item in list_of_tup:
        
        if item[0] not in res:
            res[item[0]] = []
        
        else:
            res[item[0]].append(item[1])
    
    # Calculate group sizes
    for key in res:
        print(f'{key} {len(res[key])}',
              file=log_file_handle)
        
        for method_val in set(res[key]):
            
            print(f'  n {method_val} = {res[key].count(method_val)}',
                  file=log_file_handle)
    
    print('\n\n', file=log_file_handle, flush=True)
    return(res)

x = calc_group_sizes(dummy_improve_X, 'dummy_improve_X')
x = calc_group_sizes(dummy_weaken_X, 'dummy_weaken_X')
x = calc_group_sizes(dummy_nothing_X, 'dummy_nothing_X')

x = calc_group_sizes(dummy_improve_y, 'dummy_improve_y')
x = calc_group_sizes(dummy_weaken_y, 'dummy_weaken_y')
x = calc_group_sizes(dummy_nothing_y, 'dummy_nothing_y')




    











def calculate_ranking(df, output_folder, limit_to_R2_positive, arg1=''):
    
    # This is in the manuscript
    
    print('calculate ranking:',
          file=log_file_handle)
    
    print(f'df.shape = {df.shape}',
          file=log_file_handle)
    
    if limit_to_R2_positive:
        
        # Calculate ranks only for better-than-benchmark cases
        
        print(f'Rankings only for R2_validate > {R2_limit_min:.2f} ' \
              f'and R2_test > {R2_limit_min:.2f}:', flush=True,
              file=log_file_handle)
            
        idxs = (df['R2_validate'] > R2_limit_min) \
                & (df['R2_test'] > R2_limit_min)
                
        fname_key = 'pos' + arg1
        
    else:
        # Include all calculated cases in the ranking
        print('Rankings among all cases:',
              flush=True,
              file=log_file_handle)
        
        idxs = df.index
        
        fname_key = 'all' + arg1
    
    df_dummy = df.loc[idxs,:].copy()
    
    
    
    # Smaller MAE, RMSE and R2inv are better
    # df.loc[(meas_point, optimiz, model_name), ['MAE','RMSE','R2inv']]
    # RMSE is not used, because the information is contained in R2 already
    
    # NOTE! .mean(numeric_only=True) includes only finished cases
    # This means that e.g. in SVR methods, the ranking is about the finished
    # cases, and unfinished cases are left out
    
    df_holder = df_dummy \
                    .groupby(by=['measurement_point_name',
                                 'optimization_method',
                                 'model_name']) \
                    .mean(numeric_only=True) \
                    .loc[:, ['R2_mean_validate_test',
                             'MAE_mean_validate_test']] \
                    .sort_values(by='MAE_mean_validate_test', ascending=True) \
                    .copy()
    
    
    df_holder.sort_index(inplace=True)
    
    # Distance to perfect match: R2 + dR2 = 1 -> dR2 = 1 - R2
    df_holder.loc[:, 'dR2_mean_validate_test'] \
        = 1.0 - df_holder.loc[:,'R2_mean_validate_test']
    
    df_holder.drop(columns='R2_mean_validate_test', inplace=True)
    
    print(f'df_holder.shape = {df_holder.shape}',
          file=log_file_handle)
    
    
    
    # Calculate ranks and max rank sum
    # row-index has three levels (,,)
    print('calculate max rank sum...',
          file=log_file_handle)
    df_ranks = df_holder.copy()
    max_rank_sum = 0.0
    
    for mp in df_holder.index.levels[0]:
        
        print('mp:', mp, flush=True)
        
        for opt in df_holder.loc[(mp),:].index.levels[0]:
            
            print('  opt:', opt, flush=True)
            
            df_ranks.loc[(mp, opt), :] \
                = df_holder.loc[(mp,opt), :] \
                    .rank(axis=0, ascending=False).values
            
            
            helper_value = df_ranks.loc[(mp,opt),:].shape[0] \
                            * df_ranks.loc[(mp,opt),:].shape[1]
                            
            print('helper value:', helper_value,
                  file=log_file_handle)
            
            max_rank_sum += helper_value
            
            print('    max rank sum:', max_rank_sum, flush=True)
            
    
    # sum the ranks to a dict(model_name: sum(ranks))
    res_ranks_dict = {}
    
    for tup in df_ranks.index:
        
        if tup[2] not in res_ranks_dict:
            # tup = (measurement_point, optimization_method, ML_method)
            res_ranks_dict[tup[2]] = df_ranks.loc[tup, :].sum()
        
        else:
            res_ranks_dict[tup[2]] += df_ranks.loc[tup, :].sum()
    
    
    # convert dict to a DataFrame table and sort rows
    df_res_ranks = pd.DataFrame.from_dict(data=res_ranks_dict,
                                          orient='index',
                                          columns=['rank_sum'])
    
    df_res_ranks.sort_values(by='rank_sum',
                             ascending=False,
                             inplace=True)
    
    # Scale to 0...1
    df_res_ranks_relative = df_res_ranks / max_rank_sum
    
    
    
    # Add counts and median wall clock times
    df_res_ranks_relative['count'] \
        = np.zeros(df_res_ranks_relative.shape[0], dtype=np.int64)
        
    df_res_ranks_relative['wall_clock_time_minutes'] \
        = np.zeros(df_res_ranks_relative.shape[0])
    
    for mod in df_res_ranks_relative.index:
        
        idxs_mod = df_dummy['model_name'] == mod
        
        df_res_ranks_relative.loc[mod, 'count'] \
            = df_dummy.loc[idxs_mod, :].shape[0]
        
        df_res_ranks_relative.loc[mod, 'wall_clock_time_minutes'] \
            = df_dummy.loc[idxs_mod, 'wall_clock_time_minutes'].median()
    
    
    
    print(df_res_ranks_relative.round(2),
          file=log_file_handle)
    
    print(f'max_rank_sum = {max_rank_sum}',
          file=log_file_handle)
    
    with pd.ExcelWriter(xlsx_output_file,
                        mode='a',
                        if_sheet_exists='replace',
                        engine='openpyxl') as writer:
        
        sh_name = 'relrank_{}'.format(fname_key)
        df_res_ranks_relative.to_excel(writer,
                                       sheet_name=sh_name)
    
    print('\n\n', file=log_file_handle, flush=True)
    
    return(df_holder, df_ranks, df_res_ranks, df_res_ranks_relative)



# All cases and R2>=R2_limit min cases
df_holder_all, df_ranks_all, df_res_ranks_all, df_ranks_relative_all \
    = calculate_ranking(df, output_folder, False)

df_holder_pos, df_ranks_pos, df_res_ranks_pos, df_ranks_relative_pos \
    = calculate_ranking(df, output_folder, True)

# One measurement point at a time
for mp in df['measurement_point_name'].unique():
    
    print('mp', mp,
          file=log_file_handle)
    
    idxs = (df['measurement_point_name'] == mp)
    
    df_onemp = df.loc[idxs, :].copy()
    
    df_holder_pos_onemp, df_ranks_pos_onemp, \
    df_res_ranks_pos_onemp, df_res_ranks_relative_pos_onemp \
        = calculate_ranking(df_onemp, output_folder, True, mp)
    
    










# Plot correlation scatter plot of ranks
fig, ax = plt.subplots(figsize=figseiz)
df_ranks_pos.plot.scatter(x='MAE_mean_validate_test',
                          y='dR2_mean_validate_test',
                          xlabel='rank(MAE mean)',
                          ylabel='rank(1-R$^2$ mean)',
                          marker='.',
                          s=1.0,
                          grid=True,
                          ax=ax)
ax.set_axisbelow(True)
fname = os.path.join(output_folder,
                     'correlation ranks MAE vs ranks dR2.png')
fig.savefig(fname, dpi=dpi_val, bbox_inches='tight')
plt.close(fig)


# Correlation coefficient etc of ranks
print('\n\n', file=log_file_handle)
print(f'Correlation of ranks (all)), df_ranks_pos.shape = {df_ranks_pos.shape}:',
      file=log_file_handle)
print(df_ranks_pos.corr(),
      file=log_file_handle)
print('standard deviation:',
      file=log_file_handle)
print((df_ranks_pos.loc[:,'MAE_mean_validate_test'] \
       -df_ranks_pos.loc[:,'dR2_mean_validate_test']).std(),
      file=log_file_handle, flush=True)



for mp in df_ranks_pos.index.levels[0]:
    
    print('Correlation of ranks (mp)' \
          + 'df_ranks_pos.loc[].shape = ' \
          + f'{df_ranks_pos.loc[(mp, slice(None), slice(None)), :].shape}',
          file=log_file_handle)
    
    print(mp,
          file=log_file_handle)
    print(df_ranks_pos.loc[(mp, slice(None), slice(None)), :].corr(),
          file=log_file_handle)
    
    print((df_ranks_pos.loc[(mp, slice(None),slice(None)), 
                            'MAE_mean_validate_test']
           -df_ranks_pos.loc[(mp, slice(None),slice(None)), 
                                   'dR2_mean_validate_test']).std(),
          file=log_file_handle, flush=True)
        














def calculate_ranking_v2(df, output_folder):
    
    print('calculate_ranking_v2:',
          file=log_file_handle)
    
    mps = df['measurement_point_name'].unique()
    
    models = df['model_name'].unique()
    
    cols = ['optimization_method', 'X_lag', 'y_lag', 'N_CV', 'N_ITER']
    
    df_relrank = pd.DataFrame(data=np.zeros(( models.shape[0],
                                                len(mps) )),
                                index=models,
                                columns=mps)
    
    for mp in mps:
    # for mp in ['Tampere1']:
        
        # If we want to evaluate the capabilities of a model_name in multiple
        # situations, we should first split the data according to those
        # situations, then make the evaluations and the combine the results.
        
        # If the ranking is done with just one go, the best model will always
        # get a relative rank of 1.0, regardless of how well it was able to
        # keep its position in various situations.

        df_rank_sums = pd.DataFrame(data=np.zeros(( models.shape[0],
                                                    len(cols) )),
                                    index=models,
                                    columns=cols)
        
        counter = 0
        
        for col in cols:
            
            # print('col:', col)
            
            uniques = df[col].unique()
            
            for unique in uniques:
                
                # print('unique:', unique)
                
                idxs = (df['R2_train'] >= R2_limit_min) \
                        & (df['R2_validate'] >= R2_limit_min) \
                        & (df['R2_test'] >= R2_limit_min) \
                        & (df['measurement_point_name'] == mp) \
                        & (df[col] == unique)
                
                
                df_dummy = df.loc[idxs,:].copy()
                
                df_dummy['dR2_mean_validate_test'] = 1.0 - df_dummy['R2_mean_validate_test']
                
                # The correlation between MAE and dR2 rankings is not perfect, but good
                df_ranks = df_dummy.groupby(by='model_name') \
                                .mean(numeric_only=True) \
                                .loc[:, ['MAE_mean_validate_test',
                                         'dR2_mean_validate_test']] \
                                .rank(axis=0, ascending=False).mean(axis=1)
                
                df_rank_sums.loc[df_ranks.index, col] += df_ranks.values
                x = df_ranks.max()
                
                if not np.isnan(x):
                    counter += x
        
        df_relrank.loc[df_rank_sums.index, mp] \
            = df_rank_sums.sum(axis=1).values / counter
        

    df_relrank['mean_relrank'] = df_relrank.mean(axis=1)

    df_relrank.sort_values(by='mean_relrank',
                                ascending=False,
                                inplace=True)
    
    print(df_relrank,
          file=log_file_handle, flush=True)

    with pd.ExcelWriter(xlsx_output_file,
                        mode='a',
                        if_sheet_exists='replace',
                        engine='openpyxl') as writer:
        
        sh_name = 'relrankV2'
        df_relrank.to_excel(writer,
                            sheet_name=sh_name)
        
        # =B2>=LARGE(B$2:B$49;5)
        # TODO: Use openpyxl for conditional formatting the relranks
    
    return(df_relrank)


df_relrank = calculate_ranking_v2(df, output_folder)










def calc_wall_clock_time(df, output_folder, idxs_order):
    
    # This function is related to the previous function:
    # calculate_ranking_v2()
    
    print('calculate wall_clock_time V2:',
          file=log_file_handle)
    
    mps = df['measurement_point_name'].unique()
    
    models = df['model_name'].unique()
    
    shape_ = (models.shape[0], mps.shape[0])
    df_wct = pd.DataFrame(data=np.zeros(shape_),
                          index=models,
                          columns=mps)
    
    
    for mp in mps:
        
        idxs = (df['R2_train'] >= R2_limit_min) \
                & (df['R2_validate'] >= R2_limit_min) \
                & (df['R2_test'] >= R2_limit_min) \
                & (df['measurement_point_name'] == mp)
        
        df_dummy = df.loc[idxs, :].copy()
        
        df_wct_holder = df_dummy.groupby(by='model_name').mean(numeric_only=True)
        
        df_wct.loc[df_wct_holder.index, mp] \
            = df_wct_holder['wall_clock_time_minutes'].values
        
    df_wct['mean_wct'] = df_wct.mean(axis=1).values
    
    df_wct = df_wct.loc[idxs_order, :]

    print(df_wct,
          file=log_file_handle, flush=True)
    
    with pd.ExcelWriter(xlsx_output_file,
                        mode='a',
                        if_sheet_exists='overlay',
                        engine='openpyxl') as writer:
        
        sh_name = 'relrankV2'
        df_wct.to_excel(writer,
                        sheet_name=sh_name,
                        startcol=8)
    
    return(df_wct)
    
df_wct = calc_wall_clock_time(df, output_folder, df_relrank.index)


















## Find the RandomForestRegressor with best accuracy

cols_to_show = ['optimization_method','X_lag', 'y_lag',
                'wall_clock_time_minutes', 'N_CV', 'N_ITER',
                'MAE_mean_validate_test']


with pd.ExcelWriter(xlsx_output_file,
                    mode='a',
                    engine='openpyxl',
                    if_sheet_exists='replace') as writer:

    for mp in mps:
        
        # Find suitable rows
        
        idxs = (df['model_name'] == 'randomforest') \
                & (df['measurement_point_name'] == mp) \
                & (df['R2_train'] >= R2_limit_min) \
                & (df['R2_validate'] >= R2_limit_min) \
                & (df['R2_test'] >= R2_limit_min) \
                & (df['optimization_method'].isin(['pso','randomizedsearchcv'])) \
                & (df['wall_clock_time_minutes'] <= 120.0)
        
        df_helper = df.loc[idxs,:].copy()
        df_helper.sort_values(by=['optimization_method',
                                  'MAE_mean_validate_test'],
                              inplace=True,
                              ignore_index=True)
    
    
        # Write them into the output.xlsx file
        
        df_helper.to_excel(writer, sheet_name=f'RF_{mp}')
    




## Print wall clock times

print(df['wall_clock_time_minutes'].quantile(q=[0.0, 0.1, 0.5, 0.9, 1.0]))

print(df['wall_clock_time_minutes'].quantile(q=[0.0, 0.1, 0.5, 0.9, 1.0]),
      file=log_file_handle, flush=True)









print('myPostAnalysis.py, END', flush=True,
      file=log_file_handle)


log_file_handle.close()
