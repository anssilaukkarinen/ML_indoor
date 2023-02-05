# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 10:35:57 2020

@author: laukkara


Workflow, local laptop:
python3 main.py <args>




Workflow, Narvi:
1) Create folder structure to Narvi
    - Create base folder to Narvi by hand
    - Create "/.../base/slurm_logs folder to Narvi by hand
2) Create sbatch files and shell script for running them all
    - Run: sbatch ML_indoor_create_sbatch_files.sh
3) Submit multiple sbatch files to slurm using the bash script that was created
    - Run: ./ML_indoor_shell_script_for_sbatch_files.sh
    - Remember to give execution rights to script
4) Merge results
    - Run: sbatch ML_indoor_merge.sh
5) Copy-paste "df_all.xlsx" to local computer and run: "myPostAnalysis.py"
6) Report results.
    
python3 path/to/main.py 
    idx_case_start
    idx_case_end
    idx_model_start
    idx_model_end
    idx_optimization_start
    idx_optimization_end
    N_CV
    N_ITER
    N_CPU
    n_lags_X
    n_lags_y




"""

import sys
sys.path.append(".")

if 'lin' in sys.platform:
    sys.path.append(r'/home/laukkara/github/ML_indoor')

import os
print('Current dir:', os.getcwd())
print('args:', sys.argv)



import time
import numpy as np
import pandas as pd

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
    
    # For the current time stamp, Rglob has increased from the previous time stamp
    Rglob = np.sum(X[:,1:3], axis=1).reshape(-1,1)
    dRglob_pos = pd.DataFrame(Rglob).diff(1).values.reshape(-1,1)
    dRglob_pos[0,:] = dRglob_pos[1,:] # First value is NaN, which is approximated
    dRglob_pos[dRglob_pos < 0] = 0
    
    # For the current time stamp, Rglob has decreased from the previous time stamp
    dRglob_neg = pd.DataFrame(Rglob).diff(1).values.reshape(-1,1)
    dRglob_neg[0,:] = dRglob_neg[1,:]
    dRglob_neg[dRglob_neg > 0] = 0
    
    # Concatenate all columns horizontally (side-by-side)
    X = np.hstack((X, Te_mean, dRglob_pos, dRglob_neg))
    X[0,:] = X[1,:] # First row is the same as the second row, because dGlob
    
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
    

    # Record wall clock date and time
    time_start = time.time()
    print('Start time:', \
          time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time_start)),
          flush=True)
    
    # Load data and split into train, validate and test sets
    print('Load and split data...', flush=True)
    X_train0, X_validate0, X_test0, y_train0, y_validate0, y_test0 \
        = load_and_split(input_folder, measurement_point_name)
    
    # Decide on the features of the data matrix
    print('Create features...', flush=True)
    X_train, y_train = create_features(X_train0, y_train0, n_lags_X, n_lags_y)
    
    # Scale data
    print('Create scalers based on training data and scale...', flush=True)
    X_train_scaled, y_train_scaled, scaler_X, scaler_y \
        = scale_train(X_train, y_train)
    print('X.shape:', X_train_scaled.shape, \
          'y.shape:', y_train_scaled.shape, flush=True)
    
    # Fit model
    print('Fit model...', flush=True)
    xopt, fopt, model = myModels.fit_model(X_train_scaled, y_train_scaled,
                                           model_name, optimization_method,
                                           N_CV, N_ITER, N_CPU)
    
    
    # Create features for validation and test sets
    print('Features for validation and test sets...', flush=True)
    X_validate, y_validate = create_features(X_validate0, y_validate0, n_lags_X, n_lags_y)
    X_test, y_test = create_features(X_test0, y_test0, n_lags_X, n_lags_y)
    
    # Create predictions for train, validate and test sets
    print('Predict train, validate and test...', flush=True)
    print('Starting prediction at:', \
        time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
        flush=True)
    
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
          time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time_end)),
          flush=True)
        
    wall_clock_time = time_end - time_start
    print('Simulation wall clock duration:', wall_clock_time/60, 'min\n',
          flush=True)
    
    res_main = {'xopt':xopt, 'fopt':fopt, 'model':model,
               'y_train':y_train, 'y_train_pred':y_train_pred,
               'y_validate':y_validate, 'y_validate_pred':y_validate_pred,
               'y_test':y_test, 'y_test_pred':y_test_pred,
               'X_train_pred_scaled':X_train_pred_scaled,
               'X_validate_pred_scaled':X_validate_pred_scaled,
               'X_test_pred_scaled':X_test_pred_scaled,
               'wall_clock_time':wall_clock_time}
    
    # return(xopt, fopt, model, 
    #        y_train, y_train_pred,
    #        y_validate, y_validate_pred,
    #        y_test, y_test_pred,
    #        X_train_pred_scaled, X_validate_pred_scaled, X_test_pred_scaled,
    #        wall_clock_time)
    return(res_main)
    
    

if __name__ == '__main__':
    
    # Input and output folder
    if 'win' in sys.platform:
        input_folder = r'C:\Users\laukkara\github\ML_indoor\input'
        output_folder_base = r'C:\Users\laukkara\Data\ML_indoor_Narvi'
    
    elif 'lin' in sys.platform:
        input_folder = '/home/laukkara/github/ML_indoor/input'
        output_folder_base = '/lustre/scratch/laukkara/ML_indoor'
    else:
        print('Unknown platform!', flush=True)
    
    
    # output_folder_custom_str = 'output_2022-09-16-21-53-41'
    output_folder_custom_str = ''
    
    if output_folder_custom_str != '':
        output_folder = os.path.join(output_folder_base,
                                     output_folder_custom_str)
    else:
        time_str = time.strftime("%Y-%m-%d-%H-%M-00",
                                    time.localtime())
        output_folder = os.path.join(output_folder_base,
                                     'output_{}'.format(time_str))
    
    if not os.path.exists(output_folder):
        try:
            os.makedirs(output_folder)
        except:
            print(f'The folder {output_folder} already existed')
    
    # Redirect stdout and stderr
    sys_stdout_original = sys.stdout
    sys_stderr_original = sys.stderr
    
    fname = os.path.join(output_folder,
                         'std_out_and_err_file.txt')
    
    if os.path.isfile(fname):
        f_out = open(fname, 'a')
        print('\n\n\n\n', file=f_out, flush=True)
        print('#####################################', file=f_out, flush=True)
        print('File reopened for appending!', file=f_out, flush=True)
    else:
        f_out = open(fname, 'w')
    
    sys.stdout = f_out
    sys.stderr = f_out

    
        
    # Run all the code
    # The output goes to the new custom-made stdout file
    print(sys.argv, flush=True)
    
    # measurement points
    measurement_point_names = ['Espoo1', 'Espoo2', 'Tampere1', 'Tampere2',
                               'Valkeakoski', 'Klaukkala']
    measurement_point_names = measurement_point_names[int(sys.argv[1]):int(sys.argv[2])]
    print(measurement_point_names, flush=True)
    
    
    # LassoLarsICAIC
    # LassoLarsICBIC
    # QuantileRegressor
    # TweedieRegressor (Poisson, Gamma, Inverse Gaussian)
    
    
    model_names = ['kernelridgesigmoid',
                   'nusvrpoly',
                   'nusvrrbf',
                   'svrlinear',
                   'svrpoly',
                   'kneighborsregressoruniform',
                   'kneighborsregressordistance',
                    'dummyregressor',
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
                    'kernelridgelinear',
                    'kernelridgerbf',
                    'kernelridgelaplacian',
                    'kernelridgecosine',
                    'kernelridgepolynomial',
                    'linearsvr',
                    'nusvrlinear',
                    'nusvrsigmoid',
                    'svrrbf',
                    'svrsigmoid',
                    'decisiontreeregressorbest',
                    'decisiontreeregressorrandom',
                    'extratreeregressorbest',
                    'extratreeregressorrandom',
                    'adaboostdecisiontree',
                    'adaboostextratree',
                    'baggingdecisiontree',
                    'baggingextratree',
                    'extratreesregressorbootstrapfalse',
                    'extratreesregressorbootstraptrue',
                    'gradientboostingregressor',
                    'histgradientboostingregressor',
                    'randomforest',
                    'lgbgbdt',
                    'lgbgoss',
                    'lgbdart',
                    'lgbrf',
                    'xgbgbtree',
                    'xgbdart']
    model_names = model_names[int(sys.argv[3]):int(sys.argv[4])]
    print(model_names, flush=True)
    
    # UserWarning: The objective has been evaluated at this point before.
    # -> ridge bayes; lasso bayes; elasticnet bayes; lars bayes; lassolars bayes;
    # -> theilsen bayes
    
    # calculations stopped (swap memory full?) at kernelridgecosine pso (counter=1)
    # kernelridge cosine bayessearch Tampere1 Tampere2: the objective has been evaluated at this point before
    
    
    # kernelridgepoly is a bit slow compared to earlier methods,
    # but stil ran through in a reasonable time
    # kernelridgepolynomial, pso, Espoo2 frozen due to Swap memory full in Linux
    
    # kernelridge_sigmoid gives dual problem/least squares warnings, and LinAlgError errors
    # The program crashed here!!! -> Espoo1 ok, Espoo2 (+1 crash), Valkeakoski (crash+3)
    # kernelridgesigmoid Tampere2 randomizedsearchcv -> 12 cpus in use for NCPU=10, is that ok?
    # kernelridgesigmoid Tampere2 bayessearchcv SVD didn't converge linear least squares
    # kernelridgesigmoid very slow in some cases (large alpha?)
    
    # nusvr_poly: convergence warning terminated early (max_iter=1000000)
    
    # svrpoly is very slow
    
    
    
    # Optimization methods
    optimization_methods = ['pso', 'randomizedsearchcv', 'bayessearchcv']
    optimization_methods = optimization_methods[int(sys.argv[5]):int(sys.argv[6])]
    print(optimization_methods, flush=True)
    
    
    # number of runs and jobs
    N_CV = int(sys.argv[7])
    N_ITER = int(sys.argv[8])
    N_CPU = int(sys.argv[9])
    n_lags_X = int(sys.argv[10])
    n_lags_y = int(sys.argv[11])
    print('N_CV:', N_CV, 'N_ITER:', N_ITER, 'N_CPU:', N_CPU, \
          'n_lags_X:', n_lags_X, 'n_lags_y:', n_lags_y, flush=True)    
    
    
    # Loop through different situations
    
    for model_name in model_names:
        print('model_name:', model_name, flush=True)
        
        for optimization_method in optimization_methods:
            print('optimization_method:', optimization_method, flush=True)
            
            for measurement_point_name in measurement_point_names:  
                print('measurement_point_name:', measurement_point_name, flush=True)
                
                # Prepare input data, fit model, make predictions
                results = main(input_folder,
                               measurement_point_name,
                               model_name,
                               optimization_method,
                               n_lags_X,
                               n_lags_y,
                               N_CV, N_ITER, N_CPU)

                # Plot and save results
                print('Export results...', flush=True)
                myResults.main(output_folder,
                               model_name,
                               optimization_method,
                               n_lags_X,
                               n_lags_y,
                               N_CV, N_ITER, N_CPU,
                               measurement_point_name,
                               results)
    
    # combine all results files to single file
    # Here the combining is done for a single "output_..." folder.
    # combine_results is removed from here, because it is run in any case
    # after all the calculations have finished.
    # myResults.combine_results_files(output_folder)
    
    
    
    # End
    print('End', flush=True)
    sys.stdout = sys_stdout_original
    sys.stderr = sys_stderr_original
    f_out.close()
    




