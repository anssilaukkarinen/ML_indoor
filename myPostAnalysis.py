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


root_folder = r'C:\Local\laukkara\Data\github\ML_indoor\output_2022-09-17-09-36-39_säästä'

fname = os.path.join(root_folder, 'combined.csv')
df = pd.read_csv(fname)

markers = itertools.cycle(['o', '^', 's', 'x', 'd'])


output_fold = os.path.join(root_folder,
                           '0output')
if not os.path.exists(output_fold):
    os.makedirs(output_fold)

dpi_val = 200


##########################

idxs = df['model_name']=='dummyregressor'
df_holder = df.loc[idxs, :].groupby('measurement_point_name').mean()
print(df_holder.round(2).T)


## R2 vs RMSE

for key in ['train', 'validate', 'test']:
    # key = 'train'
    fig, ax = plt.subplots()
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
    fname = os.path.join(output_fold,
                         f'{key} R2 vs RMSE.png')
    fig.savefig(fname, dpi=dpi_val, bbox_inches='tight')




