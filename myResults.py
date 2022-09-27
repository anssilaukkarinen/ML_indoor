# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 00:18:13 2020

@author: laukkara
"""

import os
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score



def main(output_folder,
         measurement_point_name,
         model_name,
         optimization_method,
         n_lags_X,
         n_lags_y_max,
         N_CV, N_ITER, N_CPU,
         results):
    
    dummy = '{}_{}_Xlag{}_ylagmax{}_NCV{}_NITER{}_NCPU{}_{}'.format(
                                                model_name,
                                                optimization_method,
                                                str(n_lags_X),
                                                str(n_lags_y_max),
                                                str(N_CV), str(N_ITER), str(N_CPU),
                                                measurement_point_name)
    
    results_folder = os.path.join(output_folder,
                                  dummy)
    
    if not os.path.exists(results_folder):
        # The folder didn't exist, so we created a new one
        os.makedirs(results_folder)
    else:
        # The folder already existed, so we append the name with datetime
        # and make a new folder
        time_str = time.strftime("%Y-%m-%d-%H-%M-%S",
                                    time.localtime())
        results_folder = os.path.join(output_folder,
                                      dummy +'__' + time_str)
        os.makedirs(results_folder)
    
    
    print('saveplots', flush=True)
    savePlots(results, results_folder)
    
    print('save numeric', flush=True)
    saveNumeric(measurement_point_name,
                model_name,
                optimization_method,
                n_lags_X,
                n_lags_y_max,
                results,
                results_folder)
    
    print('save model', flush=True)
    saveModel(results, results_folder)
    
    print('saving results done!', flush=True)






def savePlots(results, results_folder):
    
    figseiz = (5.0, 3.0)
    dpi_val = 200

    ## Training
    for idx in range(len(results)):
        
        # Scatter plot
        fig, ax = plt.subplots(figsize=figseiz)
        ax.plot(results[idx]['y_train'], results[idx]['y_train_pred'],'.')
        ax.set_xlabel('gt')
        ax.set_ylabel('pred')
        ax.plot((20,30),(20,30))
        fname = results_folder + '/train_scatter_' + str(idx) + '.png'
        fig.savefig(fname, dpi=dpi_val, bbox_inches='tight')
        plt.close(fig)

        # Line plot
        fig, ax = plt.subplots(figsize=figseiz)
        ax.plot(results[idx]['y_train'], label='gt')
        ax.plot(results[idx]['y_train_pred'], label='pred')
        ax.legend()
        fname = results_folder + '/train_line_' + str(idx) + '.png'
        plt.savefig(fname, dpi=dpi_val, bbox_inches='tight')
        plt.close()
    
    
    ## Validation
    for idx in range(len(results)):
        
        # Scatter plot
        fig, ax = plt.subplots(figsize=figseiz)
        ax.plot(results[idx]['y_validate'], results[idx]['y_validate_pred'],'.')
        ax.plot((20,30),(20,30))
        ax.set_xlabel('gt')
        ax.set_ylabel('pred')
        fname = results_folder + '/validate_scatter_' + str(idx) + '.png'
        fig.savefig(fname, dpi=dpi_val, bbox_inches='tight')
        plt.close(fig)
    
        # Line plot
        fig, ax = plt.subplots(figsize=figseiz)
        ax.plot(results[idx]['y_validate'], label='gt')
        ax.plot(results[idx]['y_validate_pred'], label='pred')
        ax.legend()
        fname = results_folder + '/validate_line_' + str(idx) + '.png'
        fig.savefig(fname, dpi=dpi_val, bbox_inches='tight')
        plt.close(fig)
    
    
    ## Test
    for idx in range(len(results)):
        
        # Scatter plot
        fig, ax = plt.subplots(figsize=figseiz)
        ax.plot(results[idx]['y_test'], results[idx]['y_test_pred'],'.')
        ax.plot((20,30),(20,30))
        ax.set_xlabel('gt')
        ax.set_ylabel('pred')
        fname = results_folder + '/test_scatter_' + str(idx) + '.png'
        fig.savefig(fname, dpi=dpi_val, bbox_inches='tight')
        plt.close(fig)
    
        # Line plot
        fig, ax = plt.subplots(figsize=figseiz)
        ax.plot(results[idx]['y_test'], label='gt')
        ax.plot(results[idx]['y_test_pred'], label='pred')
        ax.legend()
        fname = results_folder + '/test_line_' + str(idx) + '.png'
        fig.savefig(fname, dpi=dpi_val, bbox_inches='tight')
        plt.close(fig)
    




def saveNumeric(measurement_point_name,
                model_name,
                optimization_method,
                n_lags_X,
                n_lags_y_max,
                results,
                results_folder):
    
    fname_res = results_folder + '/' + 'log.txt'
    
    fname_com = os.path.join(results_folder,
                             'results.txt')
    
    
    # with open(fname_res, 'w') as f, open(fname_com, 'a') as g:
    with open(fname_res, 'w') as f:
           
        # Basic info
        s = measurement_point_name
        f.write(s + '\n')
        
        s = model_name
        f.write(s + '\n')
        
        s = optimization_method
        f.write(s + '\n')
        
        s = 'X_lag:' + str(n_lags_X)
        f.write(s + '\n')
        
        
        for idx in range(len(results)):
            
            s = 'y_lag: ' + str(idx)
            f.write('\n' + s + '\n')
            
            # MAE
            mae_train = mean_absolute_error(results[idx]['y_train'], 
                                            results[idx]['y_train_pred'])
            mae_validate = mean_absolute_error(results[idx]['y_validate'], 
                                               results[idx]['y_validate_pred'])
            mae_test = mean_absolute_error(results[idx]['y_test'], 
                                           results[idx]['y_test_pred'])
            s = 'MAE_y train {:.4f} validate {:.4f} test {:.4f}' \
                .format(mae_train, mae_validate, mae_test)
            f.write(s + '\n')
            
            # RMSE
            rmse_train = mean_squared_error(results[idx]['y_train'],
                                            results[idx]['y_train_pred'],
                                            squared=False)
            rmse_validate = mean_squared_error(results[idx]['y_validate'],
                                               results[idx]['y_validate_pred'],
                                               squared=False)
            rmse_test = mean_squared_error(results[idx]['y_test'],
                                           results[idx]['y_test_pred'],
                                           squared=False)
            s = 'RMSE_y train {:.4f} validate {:.4f} test {:.4f}' \
                .format(rmse_train, rmse_validate, rmse_test)
            f.write(s + '\n')
            
            
            # R2
            R2_train = r2_score(results[idx]['y_train'], 
                                results[idx]['y_train_pred'])
            R2_validate = r2_score(results[idx]['y_validate'], 
                                   results[idx]['y_validate_pred'])
            R2_test = r2_score(results[idx]['y_test'], 
                               results[idx]['y_test_pred'])
            s = 'R2 train {:.4f} validate {:.4f} test {:.4f}' \
                .format(R2_train, R2_validate, R2_test)
            f.write(s + '\n')
            
            
            # Parameters
            s = 'Params:'
            f.write('\n' + s + '\n')
            
            # Get names and parameters
            if 'best_estimator_' in dir(results[idx]['model']):
                # RandomizedSearchCV or BayesSearchCV
                dict_to_write = results[idx]['model'].best_estimator_.get_params()
            
            else:
                # sklearn model
                dict_to_write = results[idx]['model'].get_params()
                
                for key in dict_to_write:
                    # This is needed for Ridge
                    if type(dict_to_write[key]) is np.ndarray:
                        dict_to_write[key] = dict_to_write[key].tolist()
                    
            # base_estimator in AdaBoost is not json-serializable
            if 'adaboost' in model_name or 'bagging' in model_name:
                dict_to_write.pop('base_estimator')
            
            f.write(json.dumps(dict_to_write, sort_keys=True, indent=4))
            f.write('\n')
            
            # Some additional printing
            if model_name == 'dummyregressor':
                c = results[idx]['model'].constant_
                s = 'model.constant_:\n' + str(c)
                f.write(s + '\n')
            
            elif model_name == 'linearregression':
                c = results[idx]['model'].coef_
                s = 'model.coef_:\n' + str(c)
                f.write(s + '\n')
                i = results[idx]['model'].intercept_
                s = 'model.intercept_:\n' + str(i)
                f.write(s + '\n')
            
            elif model_name in ['ridge', 'lasso', 'elasticnet',
                                'lars', 'lassolars', 'huberregressor',
                                'theilsenregressor']:
                if optimization_method in ['randomizedsearchcv', 'bayessearchcv']:
                    c = results[idx]['model'].best_estimator_.coef_
                    i = results[idx]['model'].best_estimator_.intercept_
                else:
                    c = results[idx]['model'].coef_
                    i = results[idx]['model'].intercept_
                s = 'model.coef_:\n' + str(c)
                f.write(s + '\n')
                s = 'model.intercept_:\n' + str(i)
                f.write(s + '\n')
            
            elif model_name in ['kernelridge_linear',
                                'kernelridge_cosine',
                                'kernelridge_polynomial']:
                # len(dual_coef_ == 8760)
                if optimization_method in ['randomizedsearchcv', 'bayessearchcv']:
                    c = results[idx]['model'].best_estimator_.dual_coef_
                else:
                    c = results[idx]['model'].dual_coef_
                s = 'model.dual_coef_:\n' + str(c)
                f.write(s + '\n')
            
            
            # Wall clock time the fitting and prediction
            s = 'wall_clock_time, minutes {:.5f}'.format(results[idx]['wall_clock_time']/60)
            f.write(s + '\n')
    
    
            
            ## the other file, the short results.txt
            data_dummy = {'measurement_point_name': measurement_point_name,
                            'model_name': model_name,
                            'optimization_method': optimization_method,
                            'X_lag': float(n_lags_X),
                            'y_lag': float(idx),
                            'MAE_train': round(float(mae_train), 4),
                            'MAE_validate': round(float(mae_validate), 4),
                            'MAE_test': round(float(mae_test), 4),
                            'RMSE_train': round(float(rmse_train), 4),
                            'RMSE_validate': round(float(rmse_validate), 4),
                            'RMSE_test': round(float(rmse_test), 4),
                            'R2_train': round(float(R2_train), 4),
                            'R2_validate': round(float(R2_validate), 4),
                            'R2_test': round(float(R2_test), 4),
                            'wall_clock_time_minutes': round(float(results[idx]['wall_clock_time'])/60, 4)}
            
            df_dummy = pd.DataFrame(data=data_dummy, index=[0])

            fname = os.path.join(results_folder,
                                 'results.txt')
            df_dummy.to_csv(path_or_buf=fname, index=False)
            
            print(data_dummy, flush=True)
    
    
    
def saveModel(results, results_folder):
    
    for idx_res, res in enumerate(results):
        
        fname = os.path.join(results_folder, \
                             'savemodel_ylag{}.pickle'.format(idx_res))
        
        with open(fname, 'wb') as f:
            pickle.dump(res, f)
        
            
            
            
            
            
            
            
            
            
            