# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 10:35:57 2020

@author: laukkara

python3 main.py 0 5 10 11 0 3 3 4 1
python3 main.py 0 1 10 11 0 1 3 4 1

"""

import sys
sys.path.append(".")

import os

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

from multiprocessing import Pool

import myModels
import myResults





def load_and_split(input_folder, measurement_point_name):
    
    print('main: load_and_split...', flush=True)
    # Read in data
    fname = os.path.join(input_folder,
                         measurement_point_name + '.csv')
    vals = pd.read_csv(fname)
    y = vals.iloc[:, [0]].values
    X = vals.iloc[:, 1:].values
    y_names = vals.columns[0:1].to_list()
    X_names = vals.columns[1:].to_list()
    print('Variable names:', y_names, X_names, flush=True)
    
    print('Data loaded for', measurement_point_name, flush=True)
    
    
    # Split
    X_train = X[0:8760, :]
    X_validate = X[8760:-1000, :]
    X_test = X[-1000:, :]
    
    y_train = y[0:8760, :]
    y_validate = y[8760:-1000, :]
    y_test = y[-1000:, :]
    
    return(X_train, X_validate, X_test, y_train, y_validate, y_test)


def create_features(X, y, n_lags_X, n_lags_y):
    
    # Indicator variables, e.g. val > 21; val1 > 1 & val2 > 100, val3 == 'C'
    # Interaction features, e.g. sum, difference, product, quotient
    # Feature representation, e.g. day_of_week, years_in_school, 'other' class
    # External data, e.g. time series, geospatial, output from external software
    
    # Afterwards: Error analysis, i.e. start from large errors, classification etc
    
    
    
    # Custom features
    print('Create_features...', flush=True)
    
    Te_mean = pd.DataFrame(X[:,0]).rolling(window=12, min_periods=1).mean().values.reshape(-1,1)
    dTe = pd.DataFrame(X[:,0]).diff(1).values.reshape(-1,1)
    
    Rglob = np.sum(X[:,1:3], axis=1).reshape(-1,1)
    dRglob_pos = pd.DataFrame(Rglob).diff(1).values.reshape(-1,1)
    dRglob_pos[0,:] = dRglob_pos[1,:]
    dRglob_pos[dRglob_pos < 0] = 0
    dRglob_neg = pd.DataFrame(Rglob).diff(1).values.reshape(-1,1)
    dRglob_neg[0,:] = dRglob_neg[1,:]
    dRglob_neg[dRglob_neg > 0] = 0    
    
    TeRglob = X[:,0].reshape(-1,1) * Rglob.reshape(-1,1)
    
    dws = pd.DataFrame(X[:,4]).diff(1).values.reshape(-1,1)
    
    # X = np.hstack((X, Te_mean, dTe, Rglob, dRglob_pos, dRglob_neg, TeRglob, dws))
    X = np.hstack((X, Te_mean, dRglob_pos, dRglob_neg))
    X[0,:] = X[1,:]
    
    # Lagged X
    X_copy = X.copy() # Take from X, put to X_copy
    for i in range(1, n_lags_X+1):
        X_lagged = pd.DataFrame(X).shift(i).values
        X_copy = np.concatenate((X_copy, X_lagged), axis=1)
    
    
    # Lagged y
    for i in range(1, 1+n_lags_y):
        y_lagged = pd.DataFrame(y).shift(i).values
        X_copy = np.concatenate((X_copy, y_lagged), axis=1)
    
    n_lags_max = np.max((n_lags_X, n_lags_y))
    X = X_copy[n_lags_max:, :]
    y = y[n_lags_max:, :]

    print('Features created!', flush=True)
    

    return(X, y)
    
    

    
def scale_train(X_train, y_train):
    
    print('Scaling features...', flush=True)
    scaler_X = StandardScaler()
    scaler_X.fit(X_train)
    X_train_scaled = scaler_X.transform(X_train)
    
    scaler_y = StandardScaler()
    scaler_y.fit(y_train)
    y_train_scaled = scaler_y.transform(y_train)
    
    print('Features scaled!', flush=True)
    
    return(X_train_scaled, y_train_scaled,
           scaler_X, scaler_y)



def predict(y0, X0, n_lags_X, n_lags_y, scaler_X, scaler_y, model):
    print('We are at main_predict', flush=True)
    y_pred = 22.0 * np.ones(y0.shape)
    X_pred, y_pred = create_features(X0, y_pred, n_lags_X, n_lags_y)
    X_pred_scaled = scaler_X.transform(X_pred)
    y_pred_scaled = scaler_y.transform(y_pred)
    y_pred_scaled = myModels.predict(model, 
                                     X_pred_scaled,
                                     y_pred_scaled,
                                     n_lags_y)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    return(y_pred, X_pred_scaled)




def main(input_folder,
         measurement_point_name,
         model_name,
         optimization_method,
         n_lags_X,
         n_lags_y,
         N_CV, N_ITER, N_CPU):
    

    
    time_start = time.time()
    print('Start time:', \
          time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time_start)), flush=True)
    
    # Create and fit model
    print('Load data, create features and fit model...', flush=True)
    X_train0, X_validate0, X_test0, y_train0, y_validate0, y_test0 \
        = load_and_split(input_folder, measurement_point_name)
    
    X_train, y_train = create_features(X_train0, y_train0, n_lags_X, n_lags_y)
    X_train_scaled, y_train_scaled, scaler_X, scaler_y \
        = scale_train(X_train, y_train)
    print('scale_train ok!', flush=True)
    xopt, fopt, model = myModels.fit_model(X_train_scaled, y_train_scaled,
                                           model_name, optimization_method,
                                           N_CV, N_ITER, N_CPU)
    
    

    
    # Create features for validation and test
    print('Features for validation and test sets...', flush=True)
    X_validate, y_validate = create_features(X_validate0, y_validate0, n_lags_X, n_lags_y)
    X_test, y_test = create_features(X_test0, y_test0, n_lags_X, n_lags_y)
    
    
    
    # Predict, train
    print('Predict train, validate and test...', flush=True)
    print('Starting prediction at:', \
        time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), flush=True)
    
    if 'kernelridge' in model_name or 'histgradientboosting' in model_name:
        y_train_pred, X_train_pred_scaled = predict(y_train0, X_train0, n_lags_X, n_lags_y, scaler_X, scaler_y, model)
        y_validate_pred, X_validate_pred_scaled = predict(y_validate0, X_validate0, n_lags_X, n_lags_y, scaler_X, scaler_y, model)
        y_test_pred, X_test_pred_scaled = predict(y_test0, X_test0, n_lags_X, n_lags_y, scaler_X, scaler_y, model)
        
        
    else:
        list_args = [(y_train0, X_train0, n_lags_X, n_lags_y, scaler_X, scaler_y, model),
                     (y_validate0, X_validate0, n_lags_X, n_lags_y, scaler_X, scaler_y, model),
                     (y_test0, X_test0, n_lags_X, n_lags_y, scaler_X, scaler_y, model)]
        with Pool(3) as p:
            list_results = p.starmap(predict, list_args)
        y_train_pred, X_train_pred_scaled = list_results[0]
        y_validate_pred, X_validate_pred_scaled = list_results[1]
        y_test_pred, X_test_pred_scaled = list_results[2]
    
    
    time_end = time.time()
    print('End time:', \
          time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time_end)), flush=True)
        
    wall_clock_time = time_end - time_start
    print('Simulation wall clock duration:', wall_clock_time/60, 'min\n', flush=True)
    
    return(xopt, fopt, model, 
           y_train, y_train_pred,
           y_validate, y_validate_pred,
           y_test, y_test_pred,
           X_train_pred_scaled, X_validate_pred_scaled, X_test_pred_scaled,
           wall_clock_time)
    
    

if __name__ == '__main__':
    """
    # Run multiple times from command line
    python main.py Tampere1 $idx_start $idx_end
    python main.py 3 4 6 7 1 2
    
    python main.py mp_names models optimization_methods N_CV N_ITER N_CPU
    python main.py 3 4 6 7 1 2 4 10 1
    
    python3 main.py 0 5 0 48 0 3 3 2 1
    """
    
    
    # Input and output folder
    input_folder = os.path.join(os.getcwd(),
                                'input')
    
    time_str = time.strftime("%Y-%m-%d-%H-%M-%S",
                                time.localtime())
    output_folder = os.path.join(os.getcwd(),
                                 'output_{}'.format(time_str))
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    
    # Redirect sys.stdout
    # sys_stdout = sys.stdout
    # fname = os.path.join(output_folder,
    #                      'log_all.txt')
    # with open(fname, 'w') as sys.stdout:
        
    # Run all the code        
    print(sys.argv, flush=True)
    
    # measurement points
    measurement_point_names = ['Espoo1', 'Espoo2', 'Tampere1', 'Tampere2', 'Valkeakoski']
    measurement_point_names = measurement_point_names[int(sys.argv[1]):int(sys.argv[2])]
    print(measurement_point_names, flush=True)
    
    
    
    # ML methods, 0 48
    model_names = ['dummyregressor',
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
                    'kernelridge_cosine',
                    'kernelridge_linear',
                    'kernelridge_polynomial',
                    'kernelridge_sigmoid',
                    'kernelridge_rbf',
                    'kernelridge_laplacian',
                    'linearsvr',
                    'nusvr_linear',
                    'nusvr_poly',
                    'nusvr_rbf',
                    'nusvr_sigmoid',
                    'svr_linear',
                    'svr_poly',
                    'svr_rbf',
                    'svr_sigmoid',
                    'kneighborsregressor_uniform',
                    'kneighborsregressor_distance',
                    'decisiontreeregressor_best',
                    'decisiontreeregressor_random',
                    'extratreeregressor_best',
                    'extratreeregressor_random',
                    'adaboost_decisiontree',
                    'adaboost_extratree',
                    'bagging_decisiontree',
                    'bagging_extratree',
                    'extratreesregressor_bootstrapfalse',
                    'extratreesregressor_bootstraptrue',
                    'gradientboostingregressor_lad',
                    'histgradientboostingregressor_lad',
                    'randomforest',
                    'lgb_gbdt',
                    'lgb_goss',
                    'lgb_dart',
                    'lgb_rf',
                    'xgb_gbtree',
                    'xgb_dart']
    model_names = model_names[int(sys.argv[3]):int(sys.argv[4])]
    print(model_names, flush=True)
    
    # svr_poly is very slow
    # kernelridge_sigmoid gives dual problem/least squares warnings, and LinAlgError errors
    # nusvr_poly: convergence warning terminated early (max_iter=1000000)


    # theilsen, ransac
    
    # check randomforest, histgradientboosting
    
    
    # Optimization methods
    optimization_methods = ['pso', 'randomizedsearchcv', 'bayessearchcv']
    optimization_methods = optimization_methods[int(sys.argv[5]):int(sys.argv[6])]
    print(optimization_methods, flush=True)
    
    
    # number of runs and jobs
    N_CV = int(sys.argv[7])
    N_ITER = int(sys.argv[8])
    N_CPU = int(sys.argv[9])
    print('N_CV:', N_CV, 'N_ITER:', N_ITER, 'N_CPU:', N_CPU, flush=True)
    
    
    
    # Other parameters
    n_lags_X = 0
    n_lags_y_max = 1
    
    
    
    # Loop through different situations
    
    for model_name in model_names:
        
        for optimization_method in optimization_methods:
            
            for measurement_point_name in measurement_point_names:            
                
                results = []
                
                for idx in range(n_lags_y_max):
                    
                    # Fit model and predict
                    print('\n\nylag:', idx, flush=True)
                    
                    xopt, fopt, model, \
                    y_train, y_train_pred, \
                    y_validate, y_validate_pred, \
                    y_test, y_test_pred, \
                    X_train_pred_scaled, X_validate_pred_scaled, X_test_pred_scaled, \
                    wall_clock_time \
                        = main(input_folder,
                               measurement_point_name,
                               model_name,
                               optimization_method,
                               n_lags_X,
                               idx,
                               N_CV, N_ITER, N_CPU)
                        
                        
                        
                    
                    results.append({'xopt':xopt, 'fopt':fopt, 'model':model,
                                    'y_train':y_train, 'y_train_pred':y_train_pred,
                                    'y_validate':y_validate, 'y_validate_pred':y_validate_pred,
                                    'y_test':y_test, 'y_test_pred':y_test_pred,
                                    'wall_clock_time':wall_clock_time})
                    
                
    
                # Plot and save results
                print('Export results...', flush=True)
                myResults.main(output_folder,
                               measurement_point_name,
                               model_name,
                               optimization_method,
                               n_lags_X,
                               n_lags_y_max,
                               results)
    
    
    # combine all results files to single 
    
    print('End', flush=True)
    




