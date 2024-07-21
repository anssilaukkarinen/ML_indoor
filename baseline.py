# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 19:59:45 2024

@author: laukkara


Calculate the baseline prediction accuracy using the mean of
the training data mean.

This code was written after the main calculations. In the original calculation
run, the baseline was calculated including the cross-validation splits. Here
the baseline is calculated without the splits and with cleaner plot.

"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.dummy import DummyRegressor


root_folder = '.'

input_folder = os.path.join(root_folder,
                            'input')

output_folder = r'C:\Users\laukkara\Data\ML_indoor_Narvi'


idx_train = 8760
idx_pred2 = -1000

figseiz = (5.5, 3.5)
dpi_val = 300


### Read in data


data = {}

for file in os.listdir(input_folder):
    
    if file.endswith('.csv'):
        
        fname = os.path.join(input_folder,
                             file)
        
        data[file[:-4]] = pd.read_csv(fname)




### Split and calculate baselines

MAE_baseline = {}


for key in data.keys():
    
    # Divide all data into X and y
    cols_X = data[key].columns != 'Ti'
    
    X_meas = data[key].loc[:, cols_X]
    
    y_meas = data[key].loc[:, ['Ti']]
    
    # Split data to train, pred1 and pred2
    X_meas_train = X_meas.iloc[:idx_train, :]
    y_meas_train = y_meas.iloc[:idx_train, :].values
    
    X_meas_pred1 = X_meas.iloc[idx_train:idx_pred2, :]
    y_meas_pred1 = y_meas.iloc[idx_train:idx_pred2, :].values
    
    X_meas_pred2 = X_meas.iloc[idx_pred2:, :]
    y_meas_pred2 = y_meas.iloc[idx_pred2:, :].values
    
    # 
    # Centering and scaling is not done, because there is no need for it
    
    # Calculate predictions
    model = DummyRegressor(strategy='mean')
    model.fit(X_meas_train, y_meas_train)
    
    y_pred_train = model.predict(X_meas_train)
    y_pred_pred1 = model.predict(X_meas_pred1)
    y_pred_pred2 = model.predict(X_meas_pred2)
    
    # Evaluate prediction accuracy
    # y_meas = y_pred + dy -> dy = y_meas - y_pred
    MAE_train = np.mean( np.abs( y_meas_train - y_pred_train ) )
    MAE_pred1 = np.mean( np.abs( y_meas_pred1 - y_pred_pred1 ) )
    MAE_pred2 = np.mean( np.abs( y_meas_pred2 - y_pred_pred2 ) )
    
    MAE_baseline[key] = {'train': MAE_train,
                         'pred1': MAE_pred1,
                         'pred2': MAE_pred2}


df_MAE = pd.DataFrame(data=MAE_baseline).T



### Plot

major_ticks = np.arange(start=0, stop=1.81, step=0.2)
minor_ticks = np.arange(start=0, stop=1.81, step=0.1)


fig, ax = plt.subplots(figsize=figseiz)
df_MAE.plot(kind='bar',
            rot=45,
            ax=ax)
ax.set_ylabel('Mean Absolute Error MAE, $^\circ$C')

ax.set_yticks(major_ticks)
ax.set_yticks(minor_ticks, minor=True)
ax.yaxis.grid(which='both')
ax.yaxis.grid(which='major', alpha=0.8)
ax.yaxis.grid(which='minor', alpha=0.2)

ax.set_ylim(bottom=0.0, top=1.8)
ax.set_axisbelow(True)

fname = os.path.join(output_folder,
                     'baseline_barchart.png')
fig.savefig(fname, dpi=dpi_val, bbox_inches='tight')



### Print numerical values

fname = os.path.join(output_folder,
                     'baseline_logfile.txt')

with open(fname, 'w') as f:
    
    print('Baseline MAE:', file=f)
    print(df_MAE.round(2), file=f)
    
    print('min, mean and max:', file=f)
    print(df_MAE.values.min().round(4), 
          df_MAE.values.mean().round(4),
          df_MAE.values.max().round(4), 
          file=f)



    
    