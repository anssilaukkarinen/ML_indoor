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

# Select best from multiple runs
# Select best from X_lag is 0 or 1; y_lag is 0 or 1



Narvi_folder = os.path.join(r'C:\Users\laukkara\OneDrive - TUNI.fi\PhD',
                            r'1_ML_indoor\ML_indoor_Narvi')

#github_folder = r'C:\Local\laukkara\Data\github\ML_indoor'

y_lag = 'y0' # 'y0', 'y1'



######################

# Read in the merger xlsx file from Narvi
fname_xlsx_input = os.path.join(Narvi_folder,
                                y_lag,
                                'merged_results',
                                'df_all.xlsx')

df = pd.read_excel(fname_xlsx_input, index_col=0)
df.sort_index(inplace=True)



# Make sure the output folder and output xlsx file exists
output_folder = os.path.join(Narvi_folder,
                             'myPostAnalysis',
                             y_lag)
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

xlsx_output_file = os.path.join(output_folder,
                                'output.xlsx')

if not os.path.exists(xlsx_output_file):
    with pd.ExcelWriter(xlsx_output_file,
                        mode='w',
                        engine='openpyxl') as writer:
        # This didn't work first. You need to open a desktop excel program
        # and save an empty file with this name to this locatio,
        # and then the current code starts to work.
        pd.DataFrame(data=[0]).to_excel(writer, sheet_name='start')


fname_logfile = os.path.join(output_folder,
                             '{}.log'.format(y_lag))
log_file_handle = open(fname_logfile, mode='w')




# Some additional input data

figseiz = (5.5, 3.5)
dpi_val = 200
markers = itertools.cycle(['o', '^', 's', 'x', 'd'])

col_names_numeric = ['MAE_train', 'MAE_validate', 'MAE_test',
                    'RMSE_train','RMSE_validate', 'RMSE_test',
                    'R2_train', 'R2_validate', 'R2_test',
                    'wall_clock_time_minutes']


cols_dup = ['measurement_point_name', 'model_name', 'optimization_method', \
            'X_lag', 'y_lag', 'N_CV', 'N_ITER', 'N_CPU']


########################################

# How to define worst cases?
# -> MAE > ?
# -> R2 < -3.0
# -> R2 < 0.0
# Worst/best from the duplicates


print(df.columns,
      file=log_file_handle)



# diff = max - min
key1 = 'MAE_mean_validate_test'
key1_max = 10

dy = df.groupby(by=cols_dup).max().loc[:, key1] \
        - df.groupby(by=cols_dup).min().loc[:, key1]
dy.sort_values(inplace=True, ascending=True)

print('dy.shape = {}'.format(dy.shape), 
      file=log_file_handle)

print('n for dy <= {} -> {}'.format(key1_max, ( dy <= key1_max).sum() ),
      file=log_file_handle)

print('n for dy > {} -> {}'.format(key1_max, (dy > key1_max).sum() ), 
      file=log_file_handle)




idxs = dy > key1_max

dy_bigdiff = dy.loc[idxs].reset_index(drop=False)

for col_name in dy_bigdiff.columns[:-1]:
    print('Next coming up:', col_name, 
          file=log_file_handle)
    
    print(dy_bigdiff.groupby(by=[col_name]).count().iloc[:,-1], 
          file=log_file_handle)
    
    print('\n',
          file=log_file_handle)



# Can I remove the worst cases easily?


# duplicated() is not good, because keeping just the first/last isn't useful
# idxs_duplicated = df.loc[:, cols].duplicated()
# print(df.shape)
# print(idxs_duplicated.shape)

# groupby()
key = 'MAE_validate'

df_mae = df.groupby(by=cols_dup).min().loc[:, 'MAE_mean_validate_test']
df_R2 = df.groupby(by=cols_dup).max().loc[:, 'R2_mean_validate_test']
fig, ax = plt.subplots(figsize=figseiz)
ax.plot(df_mae, df_R2, '.')
ax.set_xlim((-0.1, 3))
ax.set_ylim((-0.1, 1.1))
fname = os.path.join(output_folder,
                     'diff_MAE_vs_R2.png')
fig.savefig(fname, dpi=dpi_val, bbox_inches='tight')
plt.close(fig)










def plot_single_series(df, output_folder):
    # Plot scatter plot of values against index
    
    # R2 figures are just point clouds -> uninformative
    # to_plot = ['MAE', 'RMSE', 'R2', 'clock']
    to_plot = ['MAE', 'RMSE', 'clock']
    
    for col in df.columns:
        
        if any(x in col for x in to_plot):
        
            fig, ax = plt.subplots(figsize=figseiz)
            df.loc[:, col].plot(ax=ax, style='.', ms=0.5)
            ax.set_ylabel(col)
            ylim_top = np.quantile(df.loc[:,col].values, 0.98)
            ax.set_ylim(bottom=-0.1, top=ylim_top)
            fname = os.path.join(output_folder,
                                 f'single_{col}.png')
            fig.savefig(fname, dpi=dpi_val, bbox_inches='tight')
            plt.close(fig)
        
plot_single_series(df, output_folder)
# There is a limit at around MAE = 0.5 C on validation and testing accuracy






def plot_sorted(df, output_folder):
    # Three increasing lines for train, validation and test data
    # The x-axis is index sorted according to MAE, RMSE or R2
    
    # Sort values according to train, validate and test mean
    
    ylim_top = 0.0
    
    for key1 in ['MAE', 'RMSE', 'R2']:
        
        
        cols = [key1 + '_train',
                key1 + '_validate',
                key1 + '_test']
        
        df_full = df.loc[:, cols].copy()
        
        
        # Plot
        fig, ax = plt.subplots(figsize=figseiz)
        

        
        if key1 == 'R2':
            ax.set_ylabel('R$^2$')
            ylim_top = 1.05
            is_ascending=False
        else:
            ax.set_ylabel(key1)
            ylim_top = np.max((ylim_top,
                               np.quantile(df_full.values, q=0.98, axis=0).min() ))
            is_ascending=True
        
        idx_new = df_full.loc[:, cols[1:]].mean(axis=1) \
                        .sort_values(ascending=is_ascending).index
        
        df_full = df_full.reindex(index=idx_new) \
                .rolling(window=10, min_periods=10).mean() \
                .dropna(axis=0, how='any') \
                .reset_index(drop=True)
        
        ax.plot(df_full)
        ax.set_xlabel('Cases, sorted')
        
        ax.set_ylim(bottom=-0.1, top=ylim_top)
        ax.legend(df_full.columns)
        fname = os.path.join(output_folder,
                             f'sorted_{key1}.png')
        fig.savefig(fname, dpi=dpi_val, bbox_inches='tight')
        plt.close(fig)
    
    return(df_full)

df_full = plot_sorted(df, output_folder)
# In R2 the training, validation and testing lines are visually more apart,
# than the MAE and RMSE lines.
# There is a long plateau and small number of very bad and a little bit better cases.







def plot_sorted2(df, output_folder):
    # Similar to plot_sorted(), but the values to be plotted are group means
    # from all measurement_points
    
    # add ylab, xlabel
    
    cols = ['measurement_point_name', 'model_name', 'optimization_method',
            'X_lag', 'y_lag', 'N_CV', 'N_ITER', 'N_CPU']
    
    for key1 in ['MAE', 'RMSE', 'R2']:
        
        if key1 == 'R2':
            ylim_top = 1.1
            is_ascending = False
        
        else:
            # ylim_top = 3.0
            ylim_top = 30.0
            is_ascending = True
        
        cols_to_plot = [key1 + '_train',
                        key1 + '_validate',
                        key1 + '_test']
        
        # Plot
        fig, ax = plt.subplots()
        
        df_group_means = df.groupby(by=cols[1:]) \
                .mean(numeric_only=True).loc[:, cols_to_plot]
        
        idx_new = df_group_means.loc[:, cols_to_plot[1:]].mean(axis=1) \
                    .sort_values(ascending=is_ascending).index
        
        df_group_means_sorted = df_group_means.reindex(index=idx_new) \
                                    .rolling(window=10, min_periods=10).mean() \
                                    .dropna(axis=0, how='any') \
                                    .reset_index(drop=True)
        
        df_group_means_sorted.plot(ax=ax)
        
        ax.set_ylim((-0.1, ylim_top))
        ax.set_ylabel(key1 + ', mean of mp, rolling average')
        ax.set_xlabel('Cases, sorted')
        ax.grid(visible=True)
        fname = os.path.join(output_folder,
                             'plot_sorted2_{}.png'.format(key1))
        fig.savefig(fname, dpi=dpi_val, bbox_inches='tight')
        plt.close(fig)
        

plot_sorted2(df, output_folder)
        
        
        










def plot_R2_vs_RMSE(df, output_folder):
    # Three scatter plots where the data is nicely in groups (lines)
    # for each measurement_point

    for key in ['train', 'validate', 'test']:
        
        fig, ax = plt.subplots(figsize=figseiz)
        
        for mp in df['measurement_point_name'].unique():
            
            idxs = (df['measurement_point_name'] == mp)
            
            df_group = df.loc[idxs, :]
            
            ax.scatter(x=f'RMSE_{key}', y=f'R2_{key}', \
                       s=5.0, marker=next(markers),
                       data=df_group, label=mp)
        ax.set_ylim((-0.1, 1.1))
        ax.set_xlim((-0.1, 2.5))
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
    # Basically the same as previous, but with a little bit more variation
    # R2 is calculated using squared errors

    for key in ['train', 'validate', 'test']:
        
        fig, ax = plt.subplots(figsize=figseiz)
        
        # The commented line produced 1-tuple error
        # for label, grp in df.groupby(['measurement_point_name']):
        for mp in df['measurement_point_name'].unique():
            
            idxs = (df['measurement_point_name'] == mp)
            df_group = df.loc[idxs, :]
            
            ax.scatter(x=f'MAE_{key}', y=f'R2_{key}', \
                        s=5.0, marker=next(markers),
                        data=df_group, label=mp)
        ax.set_ylim((-0.1, 1.1))
        ax.set_xlim((-0.1, 2.5))
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

# The R2-vs-RMSE and R2-vs-MAE show that different measurement_points form
# clusters. This means that the reported accuracy is data-dependent and proper
# evaluation of methods should be done using sufficiently large population of
# data sets and enough variation between them (to improve prediction accuracy).








def plot_boxplot_by_mp(df, output_folder):
    # Boxplots with five box-and-whiskers plots
    # The y-data is the main indicators
    
    print('\n\nplot_boxplot_by_mp', file=log_file_handle)
    
    idxs = (df.loc[:, 'R2_validate'] > 0.0) & (df.loc[:,'R2_test'] > 0.0)
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
                            ax=ax)
        ax.set_ylim(bottom=-0.1)
        
        if y_target == 'R2_mean_validate_test':
            ax.set_ylim(top=1.1)
        else:
            ax.set_ylim(top=1.5)
        
        ax.set_title('')
        fig.suptitle('')
        ax.set_xlabel('Measurement point name')
        
        y_target_str = y_target.replace('R2','R$^2$') \
                        .replace('_mean_validate_test', ', mean') \
                    + '\n R$^2_{validate}$ > 0 and R$^2_{test}$ > 0'
        ax.set_ylabel(y_target_str)
        
        fname = os.path.join(output_folder,
                             f'boxplot_mp_{y_target}.png')
        fig.savefig(fname, dpi=dpi_val, bbox_inches='tight')
        plt.close(fig)
        
plot_boxplot_by_mp(df, output_folder)

# Boxplots show that there are differences betweeen measurement points.
# The number of samples is different per measurement point,
# i.e. some measurement points reacher easier the condition
# R2_validation > 0 & R2_test > 0.
# Boxplots with measurement_point categorization might not be needed,
# if the same information is available from othe boxplots.







def plot_boxplot_by_mp_and_other(df, output_folder):
    # Boxplots with two controlled variables
    # measurement point and Xlag/ylag/opt/N_CV
    
    print('\n\nplot_boxplot_ny_mp_and_other',
          file=log_file_handle)
    
    idxs = (df.loc[:, 'R2_validate'] > 0.0) & (df.loc[:,'R2_test'] > 0.0)
    print('idxs.sum():', idxs.sum(),
          file=log_file_handle)
    
    
    xlabelstr = {'X_lag': 'Xlag',
                 'y_lag': 'ylag',
                 'optimization_method': 'optimization method',
                 'N_CV': '$N_{CV}$'}
    
    for y_target in ['R2_mean_validate_test',
                   'MAE_mean_validate_test',
                   'RMSE_mean_validate_test']:
        
        for y_key2 in ['X_lag', 'y_lag', 'optimization_method', 'N_CV']:
            
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
                                flierprops={'markersize':1},
                                ax=ax)
            ax.set_ylim(bottom=-0.1)
            
            if y_target == 'R2_mean_validate_test':
                ax.set_ylim(top=1.1)
            else:
                ax.set_ylim(top=1.5)
            
            ax.set_title('')
            fig.suptitle('')
            ax.set_xlabel('{}, {}'.format('Measurement point name',
                                          xlabelstr[y_key2]))
            
            y_target_str = y_target.replace('R2','R$^2$') \
                            .replace('_mean_validate_test', ', mean') \
                        + '\n R$^2_{validate}$ > 0 and R$^2_{test}$ > 0'
            ax.set_ylabel(y_target_str)
            
            fname = os.path.join(output_folder,
                                 f'boxplot2_mp_{y_key2}_{y_target}.png')
            fig.savefig(fname, dpi=dpi_val, bbox_inches='tight')
            plt.close(fig)
            
plot_boxplot_by_mp_and_other(df, output_folder)

# There was very little difference between Xlag=0/1, ylag=0/1, 
# optimization_method=pso/rand/bayes or N_CV=3/4/5.
# In one case the variation between ML_O was smaller, when higher N_CV was used.
# Using ylag = 1 was (a bit surprisingly) not useful in the first round of
# calculations to improve prediction accuracy.






def calculate_average_time_per_optimization_method(df):
    # List of wall_clock_time
    
    print('\n\ncalculate_average_time_per_optimization_method',
          file=log_file_handle)

    df_holder = df.groupby(by=['optimization_method']) \
                .mean(numeric_only=True) \
                 .loc[:,'wall_clock_time_minutes']
    
    print('\nAverage time per optimization_method:',
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

calculate_average_time_per_optimization_method(df)
# pso 10 min, rand 20 min, bayes 30 min





def calculate_pivot_table_averages_per_MLO(df, include_all):
    # Analyse data using pivot tables
    # Four tables for different variables
    # If all data was included, then the badly off values made it more
    # difficult to analysed the data -> option to include or not to include.
    
    if include_all:
        idxs = df.index
    else:
        idxs = (df['R2_validate'] > 0.0) \
             & (df['R2_test'] > 0.0)
    
    
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
        
        print('\npivot_table results',
              file=log_file_handle)

        print(df_pivot.round(2).to_string(),
              file=log_file_handle)
        
        
        with pd.ExcelWriter(xlsx_output_file,
                            mode='a',
                            if_sheet_exists='replace',
                            engine='openpyxl') as writer:
            
            sh_name = y_key.split('_')[0]
            df_pivot.round(2).to_excel(writer,
                                       sheet_name=sh_name)

calculate_pivot_table_averages_per_MLO(df, False)
# 












def plot_MAE_vs_wall_clock_time(df, output_folder):
    
    fig, ax = plt.subplots(figsize=figseiz)
    
    df.groupby(by=['model_name']).mean(numeric_only=True) \
        .plot.scatter(x='wall_clock_time_minutes',
                      y='MAE_mean_validate_test',
                      ax=ax)
    ax.set_ylim(0,1.5)
    ax.set_xlim(-1, 100)
    fname = os.path.join(output_folder,
                         'MAE_vs_wall_clock_time.png')
    fig.savefig(fname, dpi=dpi_val, bbox_inches='tight')
    plt.close(fig)

plot_MAE_vs_wall_clock_time(df, output_folder)
# The variation is pretty big, not much can be said from the correlation 
# between MAE and wall_clock_time. A small improvement in MAE vs
# wall_clock_time does however seem to exist.








def plot_wall_clock_time_by_grouping(df, output_folder):
    
    print('\n\nplot_wall_clock_time_by_grouping',
          file=log_file_handle)
    
    
    for key_grouper in ['model_name', 'optimization_method', 'N_ITER']:
    
        fig, ax = plt.subplots(figsize=figseiz)

        df_holder = df.loc[:, [key_grouper, 'wall_clock_time_minutes'] ] \
                     .groupby(by=[key_grouper]).mean(numeric_only=True) \
                     .loc[:,'wall_clock_time_minutes']
        
        df_dummy = df_holder.sort_values(ascending=True)
        df_dummy.sort_values(inplace=True)
        
        print(df_dummy.round(1),
              file=log_file_handle)
        
        with pd.ExcelWriter(xlsx_output_file,
                            mode='a',
                            if_sheet_exists='replace',
                            engine='openpyxl') as writer:
        
            sh_name = 'wall_{}'.format(key_grouper)
            df_dummy.round(1).to_excel(writer,
                              sheet_name=sh_name)
        
        df_dummy.plot(rot=90,
                        style='-o',
                        ms=3,
                        ax=ax)
        
        fname = os.path.join(output_folder,
                             'wall_clock_time all {}.png'.format(key_grouper))
        fig.savefig(fname, dpi=dpi_val, bbox_inches='tight')
        plt.close(fig)

plot_wall_clock_time_by_grouping(df, output_folder)
# svrpoly, nusvrpoly, kernelridgesigmoid were very slow
# pso 10 min, rand 21 min, bayes 31 min
# wall_clock_time / (N_iter/100) has a minimum at N_ITER = ~200-300










def plot_var_by_grouping_and_mp(df, output_folder, limit_to_R2_positive):
    # Other variables were added afterwards
    # Some of the cases had results way off, which distorts the results
    # overall. These are however kept here for documentation purposes.
    
    print('\n\nplot_var_by_grouping_and_mp',
          file=log_file_handle)
    
    cols = ['wall_clock_time_minutes',
            'MAE_mean_validate_test',
            'RMSE_mean_validate_test',
            'R2_mean_validate_test']
    
    xticks_max = 0
    
    if limit_to_R2_positive:
        # Calculate ranks only for better-than-benchmark cases
        print('vars only for R2_validate > 0 and R2_test > 0', flush=True,
              file=log_file_handle)
        idxs = (df['R2_validate'] > 0) & (df['R2_test'] > 0)
        fname_key = 'pos'
    else:
        # Include all calculated cases in the ranking
        print('vars among all cases', flush=True,
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
            df_full = df_full.reindex(index=df_full.mean(axis=1).sort_values(ascending=True).index)
            
            # plot
            fig, ax = plt.subplots(figsize=figseiz)
                
            df_full.plot(rot=90,
                            style='-o',
                            ms=3,
                            ax=ax)
                
            ax.set_ylabel(col)
            if ('MAE' in col or 'RMSE' in col) and (fname_key == 'pos'):
                ax.set_ylim(bottom=-0.1, top=1.5)
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
                                 'grouped {} {} {}.png'.format(fname_key,
                                                               key_grouper,
                                                               col))
            fig.savefig(fname, dpi=dpi_val, bbox_inches='tight')
            plt.close(fig)
        

plot_var_by_grouping_and_mp(df, output_folder, True)
plot_var_by_grouping_and_mp(df, output_folder, False)
# wall_clock_time curves were pretty similar between different measurement_points








def calculate_ranking(df, output_folder, limit_to_R2_positive):
    
    print('\n\ncalculate_ranking',
          file=log_file_handle)
    
    if limit_to_R2_positive:
        # Calculate ranks only for better-than-benchmark cases
        print('Rankings only for R2_validate > 0 and R2_test > 0', flush=True,
              file=log_file_handle)
        idxs = (df['R2_validate'] > 0) & (df['R2_test'] > 0)
        fname_key = 'pos'
    else:
        # Include all calculated cases in the ranking
        print('Rankings among all cases', flush=True,
              file=log_file_handle)
        idxs = df.index
        fname_key = 'all'
    
    
    # The value are rounded before ranking, so that model_names that are
    # close to each other, would get similar ranks.
    df_holder = df.loc[idxs,:] \
                    .groupby(by=['measurement_point_name',
                                 'optimization_method',
                                 'model_name']) \
                    .mean(numeric_only=True) \
                    .loc[:, ['R2_mean_validate_test',
                             'MAE_mean_validate_test',
                             'RMSE_mean_validate_test']] \
                    .sort_values(by='MAE_mean_validate_test', ascending=True) \
                    .round(2).copy()
    
    df_holder.sort_index(inplace=True)
    
    # Distance to perfect match: R2val + dR2 = 1 -> dR2 = 1 - R2val
    df_holder.loc[:, 'R2inv_mean_validate_test'] \
        = 1.0 - df_holder.loc[:,'R2_mean_validate_test']
    df_holder.drop(columns='R2_mean_validate_test', inplace=True)
    
    
    # Calculate ranks and max rank sum
    df_ranks = df_holder.copy()
    max_rank_sum = 0.0
    
    for mp in df_holder.index.levels[0]:
        
        for opt in df_holder.loc[(mp),:].index.levels[0]:
                
                df_ranks.loc[(mp, opt), :] \
                    = df_holder.loc[(mp,opt), :] \
                        .rank(axis=0, ascending=False).values
                
                max_rank_sum += df_ranks.loc[(mp,opt),:].shape[0] \
                                * df_ranks.loc[(mp,opt),:].shape[1]
    
    
    
    
    # Sum and scale the ranks
    res_ranks_dict = {}
    
    for tup in df_ranks.index:
        
        if tup[2] in res_ranks_dict:
            res_ranks_dict[tup[2]] += df_ranks.loc[tup, :].sum()
        
        else:
            res_ranks_dict[tup[2]] = df_ranks.loc[tup, :].sum()
    
    df_res_ranks = pd.DataFrame.from_dict(data=res_ranks_dict,
                                          orient='index',
                                          columns=['rank_sum'])
    
    df_res_ranks.sort_values(by='rank_sum',
                             ascending=False,
                             inplace=True)
    
    df_res_ranks_relative = df_res_ranks / max_rank_sum
    
    print(df_res_ranks_relative.round(2),
          file=log_file_handle)
    
    with pd.ExcelWriter(xlsx_output_file,
                        mode='a',
                        if_sheet_exists='replace',
                        engine='openpyxl') as writer:
        
        sh_name = 'relrank_{}'.format(fname_key)
        df_res_ranks_relative.to_excel(writer,
                                       sheet_name=sh_name)
    
            

calculate_ranking(df, output_folder, True)

calculate_ranking(df, output_folder, False)











print('myPostAnalysis.py, END', flush=True,
      file=log_file_handle)


log_file_handle.close()

