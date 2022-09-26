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
import pandas as pd
import matplotlib.pyplot as plt
import itertools
from main import combine_results_files_old
from main import combine_results_files


## Old results.txt formatting
# LenovoP51
# root_folder = r'C:\Local\laukkara\Data\github\ML_indoor\output_2022-09-17-09-36-39_säästä'

# Lenovo L340
# root_folder = r'C:\Storage\github\ML_indoor\output_2022-09-21-00-56-00_säästä'
# root_folder = r'C:\Storage\github\ML_indoor\output_2022-09-23-00-10-57_säästä'
# root_folder = r'C:\Storage\github\ML_indoor\output_2022-09-23-17-08-12_säästä'
# root_folder = r'C:\Storage\github\ML_indoor\output_2022-09-24-13-41-33_säästä'

## New results.txt formatting
root_folder = r'C:\Storage\github\ML_indoor\output_2022-09-25-21-38-36'


if 'säästä' in root_folder:
    combine_results_files_old(root_folder)
else:
    combine_results_files(root_folder)


fname = os.path.join(root_folder, 'combined.csv')
df = pd.read_csv(fname)

markers = itertools.cycle(['o', '^', 's', 'x', 'd'])


output_folder = os.path.join(root_folder,
                           '0output')
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

figseiz = (5.5, 3.5)
dpi_val = 200


##########################

idxs = df['model_name']=='dummyregressor'
df_holder = df.loc[idxs, :].groupby('measurement_point_name').mean()
print(df_holder.round(2).T)


# wall_clock_time
fig, ax = plt.subplots(figsize=figseiz)
df.loc[:, 'wall_clock_time_minutes'].plot()
fname = os.path.join(output_folder,
                     'wall_clock_time.png')
fig.savefig(fname, dpi=dpi_val, bbox_inches='tight')
plt.close(fig)



## R2 vs RMSE

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




