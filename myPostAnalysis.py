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
import pandas as pd
import matplotlib.pyplot as plt

import myPostAnalysis_helper


path_repo_root = r'C:\Storage\github\ML_indoor'
run_combiner_bool = True

# Paths for folders that contain combined.csv:
# LenovoP51
# root_folder = r'C:\Local\laukkara\Data\github\ML_indoor\output_2022-09-17-09-36-39_säästä'

# Lenovo L340
# root_folder = 'output_2022-09-21-00-56-00_säästä'
# root_folder = 'output_2022-09-23-00-10-57_säästä'
# root_folder = 'output_2022-09-23-17-08-12_säästä'
# root_folder = 'output_2022-09-24-13-41-33_säästä'

## New results.txt formatting
# root_folder = r'C:\Storage\github\ML_indoor\output_2022-09-25-21-38-36'

# folders = ['output_2022-09-24-13-41-33_säästä',
#            'output_2022-09-27-01-38-00_save']

folders = [f for f in os.listdir(path_repo_root) if 'output_' in f]


folder_res = os.path.join(path_repo_root,
                                 'merged_results')
if not os.path.exists(folder_res):
    os.makedirs(folder_res)


########################################


# This makes sure that there is the combined.csv files are available
# and the '0output' folder ready for plotting.
if run_combiner_bool:
    # Create combine.csv
    myPostAnalysis_helper.run_combine_results_files(path_repo_root)


# Do some basic plotting for each of the output folders
for folder in folders:
    print(folder)

    # Basic plot
    fname = os.path.join(path_repo_root, folder, 'combined.csv')
    df_single = pd.read_csv(fname)
    
    output_folder = os.path.join(path_repo_root,
                                 folder,
                                 '0output')
    myPostAnalysis_helper.plot_single_series(df_single, output_folder)
    myPostAnalysis_helper.plot_R2_vs_RMSE(df_single, output_folder)
    myPostAnalysis_helper.plot_R2_vs_MAE(df_single, output_folder)



# Read all combined.csv files to a single pandas DataFrame:
df_list = []

#for folder in folders:
for folder in folders:
    
    fname = os.path.join(path_repo_root, folder, 'combined.csv')
    df_single = pd.read_csv(fname)
    df_list.append(df_single)

df_all = pd.concat(df_list, ignore_index=True)

df_all['R2_mean_validate_test'] \
    = df_all.loc[:, ['R2_validate','R2_test']].mean(axis=1)

df_all['RMSE_mean_validate_test'] \
    = df_all.loc[:, ['RMSE_validate','RMSE_test']].mean(axis=1)

df_all['MAE_mean_validate_test'] \
    = df_all.loc[:, ['MAE_validate','MAE_test']].mean(axis=1)



# markers = itertools.cycle(['o', '^', 's', 'x', 'd'])
# figseiz = (5.5, 3.5)
# dpi_val = 200


##########################

# 
print(df_all.columns)

# 
idxs = df_all['model_name']=='dummyregressor'
df_holder = df_all.loc[idxs, :].groupby('measurement_point_name').mean()
print(df_holder.round(2).T)

# 
fname = os.path.join(folder_res, 'sorted_all.xlsx')
df_all.sort_values(by='R2_mean_validate_test', ascending=False).round(2).to_excel(fname)


# 



myPostAnalysis_helper.plot_R2_vs_RMSE(df_all, folder_res)







