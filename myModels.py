# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 17:19:05 2020

@author: laukkara

this is from the linux machine
"""
import numpy as np


from sklearn.model_selection import cross_val_score
from sklearn.model_selection import TimeSeriesSplit

# Open interval: (0,1) = {x | 0 < x < 1}
# Closed interval: [0,1] = {x | 0 \leq x \leq 1}
from scipy.stats import randint as sp_randint # half-open, integers -> [low, high)
from scipy.stats import uniform as sp_uniform # half-open, real numbers -> [loc, loc+scale]
from scipy.stats import loguniform as sp_loguniform # half-open, real numbers -> [a, b)
# BayesSearchCV Integer: [low, high]
# BayesSearhcCV Real: [low, high]


# Optimization methods
# https://pythonhosted.org/pyswarm/
# This can be installed directly from github using conda
from pyswarm import pso

from sklearn.model_selection import RandomizedSearchCV

# https://scikit-optimize.github.io/stable/modules/generated/skopt.Optimizer.html
# https://scikit-optimize.github.io/stable/modules/generated/skopt.BayesSearchCV.html#skopt.BayesSearchCV
# http://blairhudson.com/blog/posts/optimising-hyper-parameters-efficiently-with-scikit-optimize/
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical # Integer limits are inclusive



# Machine learning algorithms
from sklearn.dummy import DummyRegressor

import myClasses
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Lars
from sklearn.linear_model import LassoLars
from sklearn.linear_model import HuberRegressor
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import TheilSenRegressor

from sklearn.kernel_ridge import KernelRidge

from sklearn.svm import LinearSVR
from sklearn.svm import NuSVR
from sklearn.svm import SVR

from sklearn.neighbors import KNeighborsRegressor

from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import ExtraTreeRegressor

from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
# experimentals need to be enabled first:
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor

# conda install -c conda-forge xgboost lightgbm
import xgboost as xgb
import lightgbm as lgb

N_CV = 5
N_CPU = 4







# 
N_SWARMSIZE_MINIMAL_3 = 20
N_MAXITER_MINIMAL_3 = 50

N_ITER_MINIMAL_3 = 1000

N_BAYESITER_MINIMAL_3 = 250


# 
N_SWARMSIZE_MINIMAL_2 = 10
N_MAXITER_MINIMAL_2 = 40

N_ITER_MINIMAL_2 = 400

N_BAYESITER_MINIMAL_2 = 200



# 
N_SWARMSIZE_MINIMAL_1 = 10
N_MAXITER_MINIMAL_1 = 10

N_ITER_MINIMAL_1 = 100

N_BAYESITER_MINIMAL_1 = 50



# swarmsize 3, maxiter 2, iter 6, bayesiter 3
N_SWARMSIZE_MINIMAL_0 = 5
N_MAXITER_MINIMAL_0 = 8

N_ITER_MINIMAL_0 = 40

N_BAYESITER_MINIMAL_0 = 20



def fit_model(X_train_scaled, y_train_scaled, model_name, optimization_method):

    print(model_name, optimization_method, flush=True)

    
    
    
    if model_name == 'dummyregressor':
        print('ML: dummyregressor')
        print('Hyperparameter tuning: None')
        kwargs = {'X_train_scaled': X_train_scaled,
                  'y_train_scaled': y_train_scaled,
                  'return_model': True}
        xopt = np.nan
        fopt, model = dummyregressor(kwargs)
        
    
    elif model_name == 'expfunc':
        print('ML: expfunc')
        print('Hyperparameter tuning: None')
        kwargs = {'X_train_scaled': X_train_scaled, 
                  'y_train_scaled': y_train_scaled}
        xopt, fopt, model = expfunc(kwargs)
        
    elif model_name == 'piecewisefunc':
        print('ML: piecewisefunc')
        print('Hyperparameter tuning: None')
        kwargs = {'X_train_scaled': X_train_scaled, 
                  'y_train_scaled': y_train_scaled}
        xopt, fopt, model = piecewise(kwargs)
    
    
    elif model_name == 'linearregression':
        
        print('ML: linearregression')
        print('Hyperparameter tuning: None')
        
        kwargs = {'X_train_scaled': X_train_scaled,
                  'y_train_scaled': y_train_scaled,
                  'return_model': True}
        xopt = np.nan
        fopt, model = linearregression(kwargs)
        
    
    elif model_name == 'ridge':
        print('ML: ridge')
        # Alpha must be a positive float in Ridge
        # Larger values mean stronger regularization
        
        if optimization_method == 'pso':
            print('hyperparameter tuning: pso')
            kwargs = {'X_train_scaled': X_train_scaled,
                      'y_train_scaled': y_train_scaled,
                      'return_model': False}
            lb = [10.0]
            ub = [10000.0]
            xopt, fopt = pso(ridge, lb, ub,
                            kwargs=kwargs,
                            swarmsize=N_SWARMSIZE_MINIMAL_3,
                            omega=0.5, phip=0.5, phig=0.5, 
                            maxiter=N_MAXITER_MINIMAL_3, minstep=1e-8,
                            minfunc=1e-5, debug=True,
                            processes=1)
            print('pso ready!')
            # Fit model with best hyperparameters:
            kwargs['return_model'] = True
            fopt, model = ridge(xopt, **kwargs)
            print('Model ready!')
        
        elif optimization_method == 'randomizedsearchcv':
            print('Hyperparameter tuning: RandomizedSearchCV')
            reg_model = Ridge()
            distributions = {'alpha': sp_loguniform(a=10.0, b=10000.0)}
            tss = TimeSeriesSplit(n_splits=N_CV)
            model = RandomizedSearchCV(estimator=reg_model, 
                                       param_distributions=distributions,
                                       n_iter=N_ITER_MINIMAL_3,
                                       scoring='neg_mean_absolute_error',
                                       n_jobs=1,
                                       cv=tss,
                                       verbose=1)
            model.fit(X_train_scaled, y_train_scaled)
            xopt = model.best_params_
            fopt = -model.best_score_
            print('Model ready!')
    
        elif optimization_method == 'bayessearchcv':
            print('Hyperparameter tuning: BayesSearchCV')
            reg_model = Ridge()
            distributions = {'alpha': Real(10.0, 10000.0, prior='log-uniform')}
            tss = TimeSeriesSplit(n_splits=N_CV)
            model = BayesSearchCV(estimator=reg_model, 
                                    search_spaces=distributions,
                                    n_iter=N_BAYESITER_MINIMAL_3,
                                    scoring='neg_mean_absolute_error',
                                    n_jobs=1,
                                    n_points=1,
                                    cv=tss,
                                    verbose=1)
            model.fit(X_train_scaled, y_train_scaled.ravel())
            xopt = model.best_params_
            fopt = -model.best_score_
            print('Model ready!')
        



    elif model_name == 'lasso':
        print('ML: lasso')
        
        if optimization_method == 'pso':
            print('hyperparameter tuning: pso')
            kwargs = {'X_train_scaled': X_train_scaled,
                      'y_train_scaled': y_train_scaled,
                      'return_model': False}
            lb = [1e-1]
            ub = [10]
            xopt, fopt = pso(lasso, lb, ub,
                            kwargs=kwargs,
                            swarmsize=N_SWARMSIZE_MINIMAL_3,
                            omega=0.5, phip=0.5, phig=0.5, 
                            maxiter=N_MAXITER_MINIMAL_3, minstep=1e-8,
                            minfunc=1e-5, debug=True,
                            processes=1)
            print('pso ready!')
            kwargs['return_model'] = True
            fopt, model = lasso(xopt, **kwargs)
            print('Model ready!')
        
        elif optimization_method == 'randomizedsearchcv':
            print('Hyperparameter tuning: RandomizedSearchCV')
            reg_model = Lasso()
            distributions = {'alpha': sp_loguniform(a=1e-1, b=10.0)}
            tss = TimeSeriesSplit(n_splits=N_CV)
            model = RandomizedSearchCV(estimator=reg_model, 
                                       param_distributions=distributions,
                                       n_iter=N_ITER_MINIMAL_3,
                                       scoring='neg_mean_absolute_error',
                                       n_jobs=1,
                                       cv=tss,
                                       verbose=1)
            model.fit(X_train_scaled, y_train_scaled)
            xopt = model.best_params_
            fopt = -model.best_score_
            print('Model ready!')
    
        elif optimization_method == 'bayessearchcv':
            print('Hyperparameter tuning: BayesSearchCV')
            reg_model = Lasso()
            distributions = {'alpha': Real(1e-1, 10.0, prior='log-uniform')}
            tss = TimeSeriesSplit(n_splits=N_CV)
            model = BayesSearchCV(estimator=reg_model, 
                                    search_spaces=distributions,
                                    n_iter=N_BAYESITER_MINIMAL_3,
                                    scoring='neg_mean_absolute_error',
                                    n_jobs=1,
                                    n_points=1,
                                    cv=tss,
                                    verbose=1)
            model.fit(X_train_scaled, y_train_scaled.ravel())
            xopt = model.best_params_
            fopt = -model.best_score_
            print('Model ready!')
    
    
    elif model_name == 'elasticnet':
        print('ML: elasticnet')
        
        if optimization_method == 'pso':
            print('hyperparameter tuning: pso')
            kwargs = {'X_train_scaled': X_train_scaled,
                      'y_train_scaled': y_train_scaled,
                      'return_model': False}
            lb = [1e-5,  0.01]
            ub = [100.0, 0.99]
            xopt, fopt = pso(elasticnet, lb, ub,
                            kwargs=kwargs,
                            swarmsize=N_SWARMSIZE_MINIMAL_3,
                            omega=0.5, phip=0.5, phig=0.5, 
                            maxiter=N_MAXITER_MINIMAL_3, minstep=1e-8,
                            minfunc=1e-5, debug=True,
                            processes=1)
            print('pso ready!')
            kwargs['return_model'] = True
            fopt, model = elasticnet(xopt, **kwargs)
            print('Model ready!')
        
        elif optimization_method == 'randomizedsearchcv':
            print('Hyperparameter tuning: RandomizedSearchCV')
            reg_model = ElasticNet()
            distributions = {'alpha': sp_loguniform(a=1e-5, b=100.0),
                             'l1_ratio': sp_uniform(0.01, 0.98)}
            tss = TimeSeriesSplit(n_splits=N_CV)
            model = RandomizedSearchCV(estimator=reg_model, 
                                       param_distributions=distributions,
                                       n_iter=N_ITER_MINIMAL_3,
                                       scoring='neg_mean_absolute_error',
                                       n_jobs=1,
                                       cv=tss,
                                       verbose=1)
            model.fit(X_train_scaled, y_train_scaled)
            xopt = model.best_params_
            fopt = -model.best_score_
            print('Model ready!')
    
        elif optimization_method == 'bayessearchcv':
            print('Hyperparameter tuning: BayesSearchCV')
            reg_model = ElasticNet()
            distributions = {'alpha': Real(1e-5, 100.0, prior='log-uniform'),
                             'l1_ratio': Real(0.01, 0.99, prior='uniform')}
            tss = TimeSeriesSplit(n_splits=N_CV)
            model = BayesSearchCV(estimator=reg_model, 
                                    search_spaces=distributions,
                                    n_iter=N_BAYESITER_MINIMAL_3,
                                    scoring='neg_mean_absolute_error',
                                    n_jobs=1,
                                    n_points=1,
                                    cv=tss,
                                    verbose=1)
            model.fit(X_train_scaled, y_train_scaled.ravel())
            xopt = model.best_params_
            fopt = -model.best_score_
            print('Model ready!')
    
    
    elif model_name == 'lars':
        print('ML: lars')
        
        if optimization_method == 'pso':
            print('hyperparameter tuning: pso')
            kwargs = {'X_train_scaled': X_train_scaled,
                      'y_train_scaled': y_train_scaled,
                      'return_model': False}
            lb = [1]
            ub = [50]
            xopt, fopt = pso(lars, lb, ub,
                            kwargs=kwargs,
                            swarmsize=N_SWARMSIZE_MINIMAL_3,
                            omega=0.5, phip=0.5, phig=0.5, 
                            maxiter=N_MAXITER_MINIMAL_3, minstep=1e-8,
                            minfunc=1e-5, debug=True,
                            processes=1)
            print('pso ready!')
            kwargs['return_model'] = True
            fopt, model = lars(xopt, **kwargs)
            print('Model ready!')
        
        elif optimization_method == 'randomizedsearchcv':
            print('Hyperparameter tuning: RandomizedSearchCV')
            reg_model = Lars()
            distributions = {'n_nonzero_coefs': sp_randint(1, 50)}
            tss = TimeSeriesSplit(n_splits=N_CV)
            model = RandomizedSearchCV(estimator=reg_model, 
                                       param_distributions=distributions,
                                       n_iter= N_ITER_MINIMAL_3,
                                       scoring='neg_mean_absolute_error',
                                       n_jobs=1,
                                       cv=tss,
                                       verbose=1)
            model.fit(X_train_scaled, y_train_scaled)
            xopt = model.best_params_
            fopt = -model.best_score_
            print('Model ready!')
    
        elif optimization_method == 'bayessearchcv':
            print('Hyperparameter tuning: BayesSearchCV')
            reg_model = Lars()
            distributions = {'n_nonzero_coefs': Integer(1, 50, prior='uniform')}
            tss = TimeSeriesSplit(n_splits=N_CV)
            model = BayesSearchCV(estimator=reg_model, 
                                    search_spaces=distributions,
                                    n_iter=N_BAYESITER_MINIMAL_3,
                                    scoring='neg_mean_absolute_error',
                                    n_jobs=1,
                                    n_points=1,
                                    cv=tss,
                                    verbose=1)
            model.fit(X_train_scaled, y_train_scaled.ravel())
            xopt = model.best_params_
            fopt = -model.best_score_
            print('Model ready!')


    elif model_name == 'lassolars':
        print('ML: lassolars')
        
        if optimization_method == 'pso':
            print('hyperparameter tuning: pso')
            kwargs = {'X_train_scaled': X_train_scaled,
                      'y_train_scaled': y_train_scaled,
                      'return_model': False}
            lb = [1e-5]
            ub = [50]
            xopt, fopt = pso(lassolars, lb, ub,
                            kwargs=kwargs,
                            swarmsize=N_SWARMSIZE_MINIMAL_3,
                            omega=0.5, phip=0.5, phig=0.5, 
                            maxiter=N_MAXITER_MINIMAL_3, minstep=1e-8,
                            minfunc=1e-5, debug=True,
                            processes=1)
            print('pso ready!')
            kwargs['return_model'] = True
            fopt, model = lassolars(xopt, **kwargs)
            print('Model ready!')
        
        elif optimization_method == 'randomizedsearchcv':
            print('Hyperparameter tuning: RandomizedSearchCV')
            reg_model = LassoLars()
            distributions = {'alpha': sp_loguniform(a=1e-5, b=100)}
            tss = TimeSeriesSplit(n_splits=N_CV)
            model = RandomizedSearchCV(estimator=reg_model, 
                                       param_distributions=distributions,
                                       n_iter=N_ITER_MINIMAL_3,
                                       scoring='neg_mean_absolute_error',
                                       n_jobs=1,
                                       cv=tss,
                                       verbose=1)
            model.fit(X_train_scaled, y_train_scaled)
            xopt = model.best_params_
            fopt = -model.best_score_
            print('Model ready!')
    
        elif optimization_method == 'bayessearchcv':
            print('Hyperparameter tuning: BayesSearchCV')
            reg_model = LassoLars()
            distributions = {'alpha': Real(1e-5, 100, prior='log-uniform')}
            tss = TimeSeriesSplit(n_splits=N_CV)
            model = BayesSearchCV(estimator=reg_model, 
                                    search_spaces=distributions,
                                    n_iter=N_BAYESITER_MINIMAL_3,
                                    scoring='neg_mean_absolute_error',
                                    n_jobs=1,
                                    n_points=1,
                                    cv=tss,
                                    verbose=1)
            model.fit(X_train_scaled, y_train_scaled.ravel())
            xopt = model.best_params_
            fopt = -model.best_score_
            print('Model ready!')


    elif model_name == 'huberregressor':
        print('ML: huberregressor')
        # epsilon: min=1.0, default=1.35
        # as epsilon increases, Huber approaches ridge
        # epsilon is the width of the standardized tube
        
        if optimization_method == 'pso':
            print('hyperparameter tuning: pso')
            kwargs = {'X_train_scaled': X_train_scaled,
                      'y_train_scaled': y_train_scaled,
                      'max_iter': 10000,
                      'return_model': False}
            lb = [1.01, 1e-5]
            ub = [2.0,    1e5]
            xopt, fopt = pso(huberregressor, lb, ub,
                            kwargs=kwargs,
                            swarmsize=N_SWARMSIZE_MINIMAL_3,
                            omega=0.5, phip=0.5, phig=0.5, 
                            maxiter=N_MAXITER_MINIMAL_3, minstep=1e-8,
                            minfunc=1e-5, debug=True,
                            processes=1)
            print('pso ready!')
            kwargs['return_model'] = True
            fopt, model = huberregressor(xopt, **kwargs)
            print('Model ready!')
        
        elif optimization_method == 'randomizedsearchcv':
            print('Hyperparameter tuning: RandomizedSearchCV')
            reg_model = HuberRegressor()
            distributions = {'epsilon':sp_loguniform(a=1.01,b=2),
                             'max_iter': [10000],
                             'alpha':sp_loguniform(a=1e-5, b=1e5)}
            tss = TimeSeriesSplit(n_splits=N_CV)
            model = RandomizedSearchCV(estimator=reg_model, 
                                       param_distributions=distributions,
                                       n_iter=N_ITER_MINIMAL_3,
                                       scoring='neg_mean_absolute_error',
                                       n_jobs=1,
                                       cv=tss,
                                       verbose=1)
            model.fit(X_train_scaled, y_train_scaled)
            xopt = model.best_params_
            fopt = -model.best_score_
            print('Model ready!')
    
        elif optimization_method == 'bayessearchcv':
            print('Hyperparameter tuning: BayesSearchCV')
            reg_model = HuberRegressor()
            distributions = {'epsilon': Real(1.01, 2, prior='log-uniform'),
                             'max_iter': Categorical([10000]),
                             'alpha': Real(1e-5, 1e5, prior='log-uniform')}
            tss = TimeSeriesSplit(n_splits=N_CV)
            model = BayesSearchCV(estimator=reg_model, 
                                    search_spaces=distributions,
                                    n_iter=N_BAYESITER_MINIMAL_3,
                                    scoring='neg_mean_absolute_error',
                                    n_jobs=1,
                                    n_points=1,
                                    cv=tss,
                                    verbose=1)
            model.fit(X_train_scaled, y_train_scaled.ravel()) # Adding .values here doesn't help
            xopt = model.best_params_
            fopt = -model.best_score_
            print('Model ready!')


    elif model_name == 'ransacregressor':
        # RANSACRegressor has some parameters that were not optimized
        # data points with residual smaller than residual_threshold are inliers
        print('ML: ransacregressor')
        
        if optimization_method == 'pso':
            print('hyperparameter tuning: pso')
            kwargs = {'X_train_scaled': X_train_scaled,
                      'y_train_scaled': y_train_scaled,
                      'return_model': False}
            # min_samples, residual_threshold, max_trials, stop_probability
            lb = [0.05, 1e-3,  50.0,  0.5]
            ub = [0.95, 100.0, 300.0, 0.95]
            xopt, fopt = pso(ransacregressor, lb, ub,
                            kwargs=kwargs,
                            swarmsize=N_SWARMSIZE_MINIMAL_3,
                            omega=0.5, phip=0.5, phig=0.5, 
                            maxiter=N_MAXITER_MINIMAL_3, minstep=1e-8,
                            minfunc=1e-5, debug=True,
                            processes=1)
            print('pso ready!')
            kwargs['return_model'] = True
            fopt, model = ransacregressor(xopt, **kwargs)
            print('Model ready!')
        
        elif optimization_method == 'randomizedsearchcv':
            print('Hyperparameter tuning: RandomizedSearchCV')
            reg_model = RANSACRegressor()
            distributions = {'min_samples': sp_uniform(0.05, 0.9),
                             'residual_threshold':sp_loguniform(a=1e-3, b=2.0),
                             'max_trials': sp_randint(50,300),
                             'stop_probability':sp_uniform(0.5, 0.95)}
            tss = TimeSeriesSplit(n_splits=N_CV)
            model = RandomizedSearchCV(estimator=reg_model, 
                                       param_distributions=distributions,
                                       n_iter=N_ITER_MINIMAL_3,
                                       scoring='neg_mean_absolute_error',
                                       n_jobs=1,
                                       cv=tss,
                                       verbose=1)
            model.fit(X_train_scaled, y_train_scaled)
            xopt = model.best_params_
            fopt = -model.best_score_
            print('Model ready!')
    
        elif optimization_method == 'bayessearchcv':
            print('Hyperparameter tuning: BayesSearchCV')
            reg_model = RANSACRegressor()
            distributions = {'min_samples': Real(0.05, 0.9, prior='uniform'),
                             'residual_threshold': Real(1e-3, 2, prior='log-uniform'),
                             'max_trials': Integer(50, 300, prior='uniform'),
                             'stop_probability': Real(0.5, 0.95, prior='uniform')}
            tss = TimeSeriesSplit(n_splits=N_CV)
            model = BayesSearchCV(estimator=reg_model, 
                                    search_spaces=distributions,
                                    n_iter=N_BAYESITER_MINIMAL_3,
                                    scoring='neg_mean_absolute_error',
                                    n_jobs=1,
                                    n_points=1,
                                    cv=tss,
                                    verbose=1)
            model.fit(X_train_scaled, y_train_scaled.ravel())
            xopt = model.best_params_
            fopt = -model.best_score_
            print('Model ready!')
        


    elif model_name == 'theilsenregressor':
        print('ML: theilsenregressor')
        
        if optimization_method == 'pso':
            print('hyperparameter tuning: pso')
            kwargs = {'X_train_scaled': X_train_scaled,
                      'y_train_scaled': y_train_scaled.ravel(),
                      'n_jobs': 1,
                      'return_model': False}
            lb = [1000,  200]
            ub = [10000, 1000]
            xopt, fopt = pso(theilsenregressor, lb, ub,
                            kwargs=kwargs,
                            swarmsize=N_SWARMSIZE_MINIMAL_3,
                            omega=0.5, phip=0.5, phig=0.5, 
                            maxiter=N_MAXITER_MINIMAL_3, minstep=1e-8,
                            minfunc=1e-5, debug=True,
                            processes=1)
            print('pso ready!')
            kwargs['return_model'] = True
            fopt, model = theilsenregressor(xopt, **kwargs)
            print('Model ready!')
        
        elif optimization_method == 'randomizedsearchcv':
            print('Hyperparameter tuning: RandomizedSearchCV')
            reg_model = TheilSenRegressor()
            distributions = {'max_subpopulation': sp_randint(1000, 10000),
                             'n_jobs': [1],
                             'n_subsamples': sp_randint(200, 1000)}
            tss = TimeSeriesSplit(n_splits=N_CV)
            model = RandomizedSearchCV(estimator=reg_model, 
                                       param_distributions=distributions,
                                       n_iter=N_ITER_MINIMAL_3,
                                       scoring='neg_mean_absolute_error',
                                       n_jobs=1,
                                       cv=tss,
                                       verbose=1)
            model.fit(X_train_scaled, y_train_scaled.ravel())
            xopt = model.best_params_
            fopt = -model.best_score_
            print('Model ready!')
    
        elif optimization_method == 'bayessearchcv':
            print('Hyperparameter tuning: BayesSearchCV')
            reg_model = TheilSenRegressor()
            distributions = {'max_subpopulation': Integer(1000, 10000, prior='uniform'),
                             'n_jobs': [1],
                             'n_subsamples': Integer(200, 1000, prior='uniform')}
            tss = TimeSeriesSplit(n_splits=N_CV)
            model = BayesSearchCV(estimator=reg_model, 
                                    search_spaces=distributions,
                                    n_iter=N_BAYESITER_MINIMAL_3,
                                    scoring='neg_mean_absolute_error',
                                    n_jobs=1, # This was slow with 1; Might have been wrong and this should be 1
                                    n_points=1, # This was slow with 1
                                    cv=tss,
                                    verbose=1)
            model.fit(X_train_scaled, y_train_scaled.ravel())
            xopt = model.best_params_
            fopt = -model.best_score_
            print('Model ready!')
            

    
    elif model_name == 'kernelridge_cosine':
        print('ML: kernelridge_cosine')
        
        if optimization_method == 'pso':
            print('hyperparameter tuning: pso')
            kwargs = {'X_train_scaled': X_train_scaled,
                      'y_train_scaled': y_train_scaled.ravel(),
                      'kernel': 'cosine',
                      'return_model': False}
            # # alpha
            lb = [1.0]
            ub = [1000.0]
            xopt, fopt = pso(kernelridge, lb, ub,
                            kwargs=kwargs,
                            swarmsize=N_SWARMSIZE_MINIMAL_3,
                            omega=0.5, phip=0.5, phig=0.5, 
                            maxiter=N_MAXITER_MINIMAL_3, minstep=1e-8,
                            minfunc=1e-5, debug=True,
                            processes=1)
            print('pso ready!')
            kwargs['return_model'] = True
            fopt, model = kernelridge(xopt, **kwargs)
            print('Model ready!')
        
        elif optimization_method == 'randomizedsearchcv':
            print('Hyperparameter tuning: RandomizedSearchCV')
            reg_model = KernelRidge()
            distributions = {'kernel':['cosine'],
                             'alpha': sp_uniform(1.0, 999.0)}
            tss = TimeSeriesSplit(n_splits=N_CV)
            model = RandomizedSearchCV(estimator=reg_model, 
                                       param_distributions=distributions,
                                       n_iter=N_ITER_MINIMAL_3,
                                       scoring='neg_mean_absolute_error',
                                       n_jobs=1,
                                       cv=tss,
                                       verbose=1)
            model.fit(X_train_scaled, y_train_scaled.ravel())
            xopt = model.best_params_
            fopt = -model.best_score_
            print('Model ready!')
    
        elif optimization_method == 'bayessearchcv':
            print('Hyperparameter tuning: BayesSearchCV')
            reg_model = KernelRidge()
            distributions = {'kernel':['cosine'],
                             'alpha': Real(1, 1000, prior='uniform')}
            tss = TimeSeriesSplit(n_splits=N_CV)
            model = BayesSearchCV(estimator=reg_model, 
                                    search_spaces=distributions,
                                    n_iter=N_BAYESITER_MINIMAL_3,
                                    scoring='neg_mean_absolute_error',
                                    n_jobs=1,
                                    n_points=1,
                                    cv=tss,
                                    verbose=1)
            model.fit(X_train_scaled, y_train_scaled.ravel())
            xopt = model.best_params_
            fopt = -model.best_score_
            print('Model ready!') 
            
    
    
    elif model_name == 'kernelridge_linear':
        print('ML: kernelridge_linear')
        
        if optimization_method == 'pso':
            print('hyperparameter tuning: pso')
            kwargs = {'X_train_scaled': X_train_scaled,
                      'y_train_scaled': y_train_scaled.ravel(),
                      'kernel': 'linear',
                      'return_model': False}
            # alpha, gamma, degree, coef0
            lb = [1,    1e-3,   1,  0]
            ub = [1000, 1.0,    3,  3]
            xopt, fopt = pso(kernelridge, lb, ub,
                            kwargs=kwargs,
                            swarmsize=N_SWARMSIZE_MINIMAL_3,
                            omega=0.5, phip=0.5, phig=0.5, 
                            maxiter=N_MAXITER_MINIMAL_3, minstep=1e-8,
                            minfunc=1e-5, debug=True,
                            processes=1)
            print('pso ready!')
            kwargs['return_model'] = True
            fopt, model = kernelridge(xopt, **kwargs)
            print('Model ready!')
        
        elif optimization_method == 'randomizedsearchcv':
            print('Hyperparameter tuning: RandomizedSearchCV')
            reg_model = KernelRidge()
            distributions = {'kernel':['linear'],
                             'alpha': sp_uniform(1.0, 999.0)}
            tss = TimeSeriesSplit(n_splits=N_CV)
            model = RandomizedSearchCV(estimator=reg_model, 
                                       param_distributions=distributions,
                                       n_iter=N_ITER_MINIMAL_3,
                                       scoring='neg_mean_absolute_error',
                                       n_jobs=1,
                                       cv=tss,
                                       verbose=1)
            model.fit(X_train_scaled, y_train_scaled.ravel())
            xopt = model.best_params_
            fopt = -model.best_score_
            print('Model ready!')
    
        elif optimization_method == 'bayessearchcv':
            print('Hyperparameter tuning: BayesSearchCV')
            reg_model = KernelRidge()
            distributions = {'kernel':['linear'],
                             'alpha': Real(1, 1000, prior='uniform')}
            tss = TimeSeriesSplit(n_splits=N_CV)
            model = BayesSearchCV(estimator=reg_model, 
                                    search_spaces=distributions,
                                    n_iter=N_BAYESITER_MINIMAL_3,
                                    scoring='neg_mean_absolute_error',
                                    n_jobs=1,
                                    n_points=1,
                                    cv=tss,
                                    verbose=1)
            model.fit(X_train_scaled, y_train_scaled.ravel())
            xopt = model.best_params_
            fopt = -model.best_score_
            print('Model ready!') 
            
            

    elif model_name == 'kernelridge_polynomial':
        print('ML: kernelridge_polynomial')
        
        if optimization_method == 'pso':
            print('hyperparameter tuning: pso')
            kwargs = {'X_train_scaled': X_train_scaled,
                      'y_train_scaled': y_train_scaled.ravel(),
                      'kernel': 'polynomial',
                      'return_model': False}
            # alpha, gamma, degree, coef0
            lb = [1,    1e-3,   1,  0]
            ub = [1000, 1.0,    2,  2]
            xopt, fopt = pso(kernelridge, lb, ub,
                            kwargs=kwargs,
                            swarmsize=N_SWARMSIZE_MINIMAL_3,
                            omega=0.5, phip=0.5, phig=0.5, 
                            maxiter=N_MAXITER_MINIMAL_3, minstep=1e-8,
                            minfunc=1e-5, debug=True,
                            processes=1)
            print('pso ready!')
            kwargs['return_model'] = True
            fopt, model = kernelridge(xopt, **kwargs)
            print('Model ready!')
        
        elif optimization_method == 'randomizedsearchcv':
            print('Hyperparameter tuning: RandomizedSearchCV')
            reg_model = KernelRidge()
            distributions = {'kernel':['polynomial'],
                             'alpha': sp_loguniform(a=1.0, b=999.0),
                             'gamma': sp_loguniform(a=1e-3, b=1e0),
                             'degree': [1, 2],
                             'coef0': [0, 1, 2]}
            tss = TimeSeriesSplit(n_splits=N_CV)
            model = RandomizedSearchCV(estimator=reg_model, 
                                       param_distributions=distributions,
                                       n_iter=N_ITER_MINIMAL_3,
                                       scoring='neg_mean_absolute_error',
                                       n_jobs=1,
                                       cv=tss,
                                       verbose=1)
            model.fit(X_train_scaled, y_train_scaled.ravel())
            xopt = model.best_params_
            fopt = -model.best_score_
            print('Model ready!')
    
        elif optimization_method == 'bayessearchcv':
            print('Hyperparameter tuning: BayesSearchCV')
            reg_model = KernelRidge()
            distributions = {'kernel':['polynomial'],
                             'alpha': Real(1, 1000, prior='log-uniform'),
                             'gamma': Real(1e-3, 1, prior='log-uniform'),
                             'degree': Integer(1, 2, prior='uniform'),
                             'coef0': Integer(0, 2, prior='uniform')}
            tss = TimeSeriesSplit(n_splits=N_CV)
            model = BayesSearchCV(estimator=reg_model, 
                                    search_spaces=distributions,
                                    n_iter=N_BAYESITER_MINIMAL_3,
                                    scoring='neg_mean_absolute_error',
                                    n_jobs=1,
                                    n_points=1,
                                    cv=tss,
                                    verbose=1)
            model.fit(X_train_scaled, y_train_scaled.ravel())
            xopt = model.best_params_
            fopt = -model.best_score_
            print('Model ready!')


    elif model_name == 'kernelridge_sigmoid':
        print('ML: kernelridge_sigmoid')
        
        if optimization_method == 'pso':
            print('hyperparameter tuning: pso')
            kwargs = {'X_train_scaled': X_train_scaled,
                      'y_train_scaled': y_train_scaled.ravel(),
                      'kernel': 'sigmoid',
                      'return_model': False}
            # alpha, gamma, degree, coef0
            lb = [0.1,  1e-3,   1,  0]
            ub = [1000, 1.0,    2,  2]
            xopt, fopt = pso(kernelridge, lb, ub,
                            kwargs=kwargs,
                            swarmsize=N_SWARMSIZE_MINIMAL_1,
                            omega=0.5, phip=0.5, phig=0.5, 
                            maxiter=N_MAXITER_MINIMAL_1, minstep=1e-8,
                            minfunc=1e-5, debug=True,
                            processes=1)
            print('pso ready!')
            kwargs['return_model'] = True
            fopt, model = kernelridge(xopt, **kwargs)
            print('Model ready!')
        
        elif optimization_method == 'randomizedsearchcv':
            print('Hyperparameter tuning: RandomizedSearchCV')
            reg_model = KernelRidge()
            distributions = {'kernel':['sigmoid'],
                             'alpha': sp_loguniform(a=1.0, b=999.0),
                             'gamma': sp_loguniform(a=1e-3, b=1.0),
                             'degree': [1, 2],
                             'coef0': [0, 1, 2]}
            tss = TimeSeriesSplit(n_splits=N_CV)
            model = RandomizedSearchCV(estimator=reg_model, 
                                       param_distributions=distributions,
                                       n_iter=N_ITER_MINIMAL_1,
                                       scoring='neg_mean_absolute_error',
                                       n_jobs=1,
                                       cv=tss,
                                       verbose=1)
            model.fit(X_train_scaled, y_train_scaled.ravel())
            xopt = model.best_params_
            fopt = -model.best_score_
            print('Model ready!')
    
        elif optimization_method == 'bayessearchcv':
            print('Hyperparameter tuning: BayesSearchCV')
            reg_model = KernelRidge()
            distributions = {'kernel':['sigmoid'],
                             'alpha': Real(1, 1000, prior='log-uniform'),
                             'gamma': Real(1e-3, 1, prior='log-uniform'),
                             'degree': Integer(1, 2, prior='uniform'),
                             'coef0': Integer(0, 2, prior='uniform')}
            tss = TimeSeriesSplit(n_splits=N_CV)
            model = BayesSearchCV(estimator=reg_model, 
                                    search_spaces=distributions,
                                    n_iter=N_BAYESITER_MINIMAL_1,
                                    scoring='neg_mean_absolute_error',
                                    n_jobs=1,
                                    n_points=1,
                                    cv=tss,
                                    verbose=1)
            model.fit(X_train_scaled, y_train_scaled.ravel())
            xopt = model.best_params_
            fopt = -model.best_score_
            print('Model ready!') 



    elif model_name == 'kernelridge_rbf':
        print('ML: kernelridge_rbf')
        
        if optimization_method == 'pso':
            print('hyperparameter tuning: pso')
            kwargs = {'X_train_scaled': X_train_scaled,
                      'y_train_scaled': y_train_scaled.ravel(),
                      'kernel': 'rbf',
                      'return_model': False}
            # alpha, gamma, degree, coef0
            lb = [0.01, 1e-4,   1,  0]
            ub = [100,  10.0,    2,  2]
            xopt, fopt = pso(kernelridge, lb, ub,
                            kwargs=kwargs,
                            swarmsize=N_SWARMSIZE_MINIMAL_2,
                            omega=0.5, phip=0.5, phig=0.5, 
                            maxiter=N_MAXITER_MINIMAL_2, minstep=1e-8,
                            minfunc=1e-5, debug=True,
                            processes=1)
            print('pso ready!')
            kwargs['return_model'] = True
            fopt, model = kernelridge(xopt, **kwargs)
            print('Model ready!')
        
        elif optimization_method == 'randomizedsearchcv':
            print('Hyperparameter tuning: RandomizedSearchCV')
            reg_model = KernelRidge()
            distributions = {'kernel':['rbf'],
                             'alpha': sp_loguniform(a=0.01, b=100.0),
                             'gamma': sp_loguniform(a=1e-4, b=10.0),
                             'degree': [1, 2],
                             'coef0': [0, 1, 2]}
            tss = TimeSeriesSplit(n_splits=N_CV)
            model = RandomizedSearchCV(estimator=reg_model, 
                                       param_distributions=distributions,
                                       n_iter=N_ITER_MINIMAL_2,
                                       scoring='neg_mean_absolute_error',
                                       n_jobs=1,
                                       cv=tss,
                                       verbose=1)
            model.fit(X_train_scaled, y_train_scaled.ravel())
            xopt = model.best_params_
            fopt = -model.best_score_
            print('Model ready!')
    
        elif optimization_method == 'bayessearchcv':
            print('Hyperparameter tuning: BayesSearchCV')
            reg_model = KernelRidge()
            distributions = {'kernel':['rbf'],
                             'alpha': Real(0.01, 100, prior='log-uniform'),
                             'gamma': Real(1e-4, 10, prior='log-uniform'),
                             'degree': Integer(1, 2, prior='uniform'),
                             'coef0': Integer(0, 2, prior='uniform')}
            tss = TimeSeriesSplit(n_splits=N_CV)
            model = BayesSearchCV(estimator=reg_model, 
                                    search_spaces=distributions,
                                    n_iter=N_BAYESITER_MINIMAL_2,
                                    scoring='neg_mean_absolute_error',
                                    n_jobs=1,
                                    n_points=1,
                                    cv=tss,
                                    verbose=1)
            model.fit(X_train_scaled, y_train_scaled.ravel())
            xopt = model.best_params_
            fopt = -model.best_score_
            print('Model ready!')



    elif model_name == 'kernelridge_laplacian':
        print('ML: kernelridge_laplacian')
        
        if optimization_method == 'pso':
            print('hyperparameter tuning: pso')
            kwargs = {'X_train_scaled': X_train_scaled,
                      'y_train_scaled': y_train_scaled.ravel(),
                      'kernel': 'laplacian',
                      'return_model': False}
            # alpha, gamma, degree, coef0
            lb = [0.01, 1e-4,   1,  0]
            ub = [100,  10.0,    2,  2]
            xopt, fopt = pso(kernelridge, lb, ub,
                            kwargs=kwargs,
                            swarmsize=N_SWARMSIZE_MINIMAL_2,
                            omega=0.5, phip=0.5, phig=0.5, 
                            maxiter=N_MAXITER_MINIMAL_2, minstep=1e-8,
                            minfunc=1e-5, debug=True,
                            processes=1)
            print('pso ready!')
            kwargs['return_model'] = True
            fopt, model = kernelridge(xopt, **kwargs)
            print('Model ready!')
        
        elif optimization_method == 'randomizedsearchcv':
            print('Hyperparameter tuning: RandomizedSearchCV')
            reg_model = KernelRidge()
            distributions = {'kernel':['laplacian'],
                             'alpha': sp_uniform(0.01, 100.0),
                             'gamma': sp_loguniform(a=1e-4, b=10.0),
                             'degree': [1, 2],
                             'coef0': [0, 1, 2]}
            tss = TimeSeriesSplit(n_splits=N_CV)
            model = RandomizedSearchCV(estimator=reg_model, 
                                       param_distributions=distributions,
                                       n_iter=N_ITER_MINIMAL_2,
                                       scoring='neg_mean_absolute_error',
                                       n_jobs=1,
                                       cv=tss,
                                       verbose=1)
            model.fit(X_train_scaled, y_train_scaled.ravel())
            xopt = model.best_params_
            fopt = -model.best_score_
            print('Model ready!')
    
        elif optimization_method == 'bayessearchcv':
            print('Hyperparameter tuning: BayesSearchCV')
            reg_model = KernelRidge()
            distributions = {'kernel':['laplacian'],
                             'alpha': Real(0.01, 100, prior='uniform'),
                             'gamma': Real(1e-4, 10, prior='log-uniform'),
                             'degree': Integer(1, 2, prior='uniform'),
                             'coef0': Integer(0, 2, prior='uniform')}
            tss = TimeSeriesSplit(n_splits=N_CV)
            model = BayesSearchCV(estimator=reg_model, 
                                    search_spaces=distributions,
                                    n_iter=N_BAYESITER_MINIMAL_2,
                                    scoring='neg_mean_absolute_error',
                                    n_jobs=1,
                                    n_points=1,
                                    cv=tss,
                                    verbose=1)
            model.fit(X_train_scaled, y_train_scaled.ravel())
            xopt = model.best_params_
            fopt = -model.best_score_
            print('Model ready!')



    elif model_name == 'linearsvr':
        print('ML: linearsvr')
        
        if optimization_method == 'pso':
            print('hyperparameter tuning: pso')
            kwargs = {'X_train_scaled': X_train_scaled,
                      'y_train_scaled': y_train_scaled.ravel(),
                      'loss': 'squared_epsilon_insensitive',
                      'dual': False,
                      'max_iter': 2000,
                      'return_model': False}
            # epsilon, C
            lb = [0.01, 0.1]
            ub = [1.0, 100.0]
            xopt, fopt = pso(linearsvr, lb, ub,
                            kwargs=kwargs,
                            swarmsize=N_SWARMSIZE_MINIMAL_3,
                            omega=0.5, phip=0.5, phig=0.5, 
                            maxiter=N_MAXITER_MINIMAL_3, minstep=1e-8,
                            minfunc=1e-5, debug=True,
                            processes=1)
            print('pso ready!')
            kwargs['return_model'] = True
            fopt, model = linearsvr(xopt, **kwargs)
            print('Model ready!')
        
        elif optimization_method == 'randomizedsearchcv':
            # Set dual = True if number of features > number of examples and vice versa.
            # If dual = True, then loss='squared_epsilon_insensitive'
            print('Hyperparameter tuning: RandomizedSearchCV')
            reg_model = LinearSVR()
            distributions = {'epsilon': sp_uniform(0.01, 1.0),
                             'C': sp_uniform(0.1, 100.0),
                             'loss': ['squared_epsilon_insensitive'],
                             'dual': [False],
                             'max_iter': [2000]}
            tss = TimeSeriesSplit(n_splits=N_CV)
            model = RandomizedSearchCV(estimator=reg_model, 
                                       param_distributions=distributions,
                                       n_iter=N_ITER_MINIMAL_3,
                                       scoring='neg_mean_absolute_error',
                                       n_jobs=1,
                                       cv=tss,
                                       verbose=1)
            model.fit(X_train_scaled, y_train_scaled.ravel())
            xopt = model.best_params_
            fopt = -model.best_score_
            print('Model ready!')
    
        elif optimization_method == 'bayessearchcv':
            print('Hyperparameter tuning: BayesSearchCV')
            reg_model = LinearSVR()
            distributions = {'epsilon': Real(0.01, 1.0, prior='uniform'),
                             'C': Real(0.1, 100, prior='uniform'),
                             'loss': Categorical(['squared_epsilon_insensitive']),
                             'dual': Categorical([False]),
                             'max_iter': [2000]}
            tss = TimeSeriesSplit(n_splits=N_CV)
            model = BayesSearchCV(estimator=reg_model, 
                                    search_spaces=distributions,
                                    n_iter=N_BAYESITER_MINIMAL_3,
                                    scoring='neg_mean_absolute_error',
                                    n_jobs=1,
                                    n_points=1,
                                    cv=tss,
                                    verbose=1)
            model.fit(X_train_scaled, y_train_scaled.ravel())
            xopt = model.best_params_
            fopt = -model.best_score_
            print('Model ready!')




    elif model_name == 'nusvr_linear':
        print('ML', model_name)
        
        if optimization_method == 'pso':
            print('hyperparameter tuning: pso')
            kwargs = {'X_train_scaled': X_train_scaled,
                      'y_train_scaled': y_train_scaled.ravel(),
                      'kernel': 'linear',
                      'shrinking':  True,
                      'cache_size': 1000,
                      'max_iter': -1,
                      'return_model': False}
            # nu, C, degree, gamma, coef0
            lb = [0.05, -5, 2, 0.01, 0]
            ub = [0.5,  0, 4,  1.0,  2]
            xopt, fopt = pso(nusvr, lb, ub,
                            kwargs=kwargs,
                            swarmsize=N_SWARMSIZE_MINIMAL_3,
                            omega=0.5, phip=0.5, phig=0.5, 
                            maxiter=N_MAXITER_MINIMAL_3, minstep=1e-8,
                            minfunc=1e-5, debug=True,
                            processes=N_CPU)
            print('pso ready!')
            kwargs['return_model'] = True
            fopt, model = nusvr(xopt, **kwargs)
            print('Model ready!')
        
        elif optimization_method == 'randomizedsearchcv':
            # Set dual = True if number of features > number of examples and vice versa.
            # If dual = True, then loss='squared_epsilon_insensitive'
            print('Hyperparameter tuning: RandomizedSearchCV')
            reg_model = NuSVR()
            distributions = {'nu': sp_uniform(0.05, 0.45),
                             'C': sp_loguniform(a=1e-5, b=2.0),
                             'degree': sp_randint(2, 5),
                             'gamma': sp_uniform(0.01, 0.99),
                             'coef0': sp_uniform(0, 2),
                             'kernel': ['linear'],
                             'shrinking': [True],
                             'cache_size': [1000],
                             'max_iter': [-1]}
            tss = TimeSeriesSplit(n_splits=N_CV)
            model = RandomizedSearchCV(estimator=reg_model, 
                                       param_distributions=distributions,
                                       n_iter=N_ITER_MINIMAL_3,
                                       scoring='neg_mean_absolute_error',
                                       n_jobs=N_CPU,
                                       cv=tss,
                                       verbose=1)
            model.fit(X_train_scaled, y_train_scaled.ravel())
            xopt = model.best_params_
            fopt = -model.best_score_
            print('Model ready!')
    
        elif optimization_method == 'bayessearchcv':
            print('Hyperparameter tuning: BayesSearchCV')
            reg_model = NuSVR()
            distributions = {'nu': Real(0.05, 0.5, prior='uniform'),
                             'C': Real(1e-5, 2.0, prior='log-uniform'),
                             'degree': Integer(2, 4, prior='uniform'),
                             'gamma': Real(0.01, 1.0, prior='uniform'),
                             'coef0': Real(0, 2, prior='uniform'),
                             'kernel': Categorical(['linear']),
                             'shrinking': Categorical([True]),
                             'cache_size': Categorical([1000]),
                             'max_iter': Categorical([-1])}
            tss = TimeSeriesSplit(n_splits=N_CV)
            model = BayesSearchCV(estimator=reg_model, 
                                    search_spaces=distributions,
                                    n_iter=N_BAYESITER_MINIMAL_3,
                                    scoring='neg_mean_absolute_error',
                                    n_jobs=N_CPU,
                                    n_points=N_CPU // N_CV,
                                    cv=tss,
                                    verbose=1)
            model.fit(X_train_scaled, y_train_scaled.ravel())
            xopt = model.best_params_
            fopt = -model.best_score_
            print('Model ready!')




    elif model_name == 'nusvr_poly':
        print('ML:', model_name)
        # Jos y_lag == 2, niin nusvr_poly -karkaa numeerisesti prediction
        # -vaiheessa inf/nan -arvoihin
        
        if optimization_method == 'pso':
            print('hyperparameter tuning: pso')
            kwargs = {'X_train_scaled': X_train_scaled,
                      'y_train_scaled': y_train_scaled.ravel(),
                      'kernel': 'poly',
                      'shrinking':  True,
                      'cache_size': 1000,
                      'max_iter': 1e6, # This is changed to
                      'return_model': False}
            # nu, C, degree, gamma, coef0
            lb = [0.2, -5, 2,  0.01, 0]
            ub = [0.5, 0,  3,  0.5,  2]
            xopt, fopt = pso(nusvr, lb, ub,
                            kwargs=kwargs,
                            swarmsize=N_SWARMSIZE_MINIMAL_1,
                            omega=0.5, phip=0.5, phig=0.5, 
                            maxiter=N_MAXITER_MINIMAL_1, minstep=1e-8,
                            minfunc=1e-5, debug=True,
                            processes=1)
            print('pso ready!')
            kwargs['return_model'] = True
            fopt, model = nusvr(xopt, **kwargs)
            print('Model ready!')
        
        elif optimization_method == 'randomizedsearchcv':
            # Set dual = True if number of features > number of examples and vice versa.
            # If dual = True, then loss='squared_epsilon_insensitive'
            print('Hyperparameter tuning: RandomizedSearchCV')
            reg_model = NuSVR()
            distributions = {'nu': sp_uniform(0.2, 0.3),
                             'C': sp_loguniform(a=1e-5, b=1e5),
                             'degree': sp_randint(2, 3),
                             'gamma': sp_uniform(0.01, 0.49),
                             'coef0': sp_uniform(0, 2),
                             'kernel': ['poly'],
                             'shrinking': [True],
                             'cache_size': [1000],
                             'max_iter': [1e6]} # This is changed to
            tss = TimeSeriesSplit(n_splits=N_CV)
            model = RandomizedSearchCV(estimator=reg_model, 
                                       param_distributions=distributions,
                                       n_iter=N_ITER_MINIMAL_1,
                                       scoring='neg_mean_absolute_error',
                                       n_jobs=1,
                                       cv=tss,
                                       verbose=1)
            model.fit(X_train_scaled, y_train_scaled.ravel())
            xopt = model.best_params_
            fopt = -model.best_score_
            print('Model ready!')
    
        elif optimization_method == 'bayessearchcv':
            print('Hyperparameter tuning: BayesSearchCV')
            reg_model = NuSVR()
            distributions = {'nu': Real(0.2, 0.5, prior='uniform'),
                             'C': Real(1e-5, 1e5, prior='log-uniform'),
                             'degree': Integer(2, 3, prior='uniform'),
                             'gamma': Real(0.01, 0.5, prior='uniform'),
                             'coef0': Real(0, 2, prior='uniform'),
                             'kernel': Categorical(['poly']),
                             'shrinking': Categorical([True]),
                             'cache_size': Categorical([1000]),
                             'max_iter': Categorical([1e6])} # This is changed to
            tss = TimeSeriesSplit(n_splits=N_CV)
            model = BayesSearchCV(estimator=reg_model, 
                                    search_spaces=distributions,
                                    n_iter=N_BAYESITER_MINIMAL_1,
                                    scoring='neg_mean_absolute_error',
                                    n_jobs=1,
                                    n_points=1,
                                    cv=tss,
                                    verbose=1)
            model.fit(X_train_scaled, y_train_scaled.ravel())
            xopt = model.best_params_
            fopt = -model.best_score_
            print('Model ready!')




    elif model_name == 'nusvr_rbf':
        print('ML:', model_name)
        # Jos y_lag == 2, niin nusvr_poly -karkaa numeerisesti prediction
        # -vaiheessa inf/nan -arvoihin
        
        if optimization_method == 'pso':
            print('hyperparameter tuning: pso')
            kwargs = {'X_train_scaled': X_train_scaled,
                      'y_train_scaled': y_train_scaled.ravel(),
                      'kernel': 'rbf',
                      'shrinking':  True,
                      'cache_size': 1000,
                      'max_iter': -1,
                      'return_model': False}
            # nu, C, degree, gamma, coef0
            lb = [0.2, -5, 2,  0.01, 0]
            ub = [0.5,  0, 4,  0.5,  2]
            xopt, fopt = pso(nusvr, lb, ub,
                            kwargs=kwargs,
                            swarmsize=N_SWARMSIZE_MINIMAL_3,
                            omega=0.5, phip=0.5, phig=0.5, 
                            maxiter=N_MAXITER_MINIMAL_3, minstep=1e-8,
                            minfunc=1e-5, debug=True,
                            processes=N_CPU)
            print('pso ready!')
            kwargs['return_model'] = True
            fopt, model = nusvr(xopt, **kwargs)
            print('Model ready!')
        
        elif optimization_method == 'randomizedsearchcv':
            # Set dual = True if number of features > number of examples and vice versa.
            # If dual = True, then loss='squared_epsilon_insensitive'
            print('Hyperparameter tuning: RandomizedSearchCV')
            reg_model = NuSVR()
            distributions = {'nu': sp_uniform(0.2, 0.3),
                             'C': sp_loguniform(a=0.01, b=100),
                             'degree': sp_randint(2, 4),
                             'gamma': sp_uniform(0.01, 0.49),
                             'coef0': sp_uniform(0, 2),
                             'kernel': ['rbf'],
                             'shrinking': [True],
                             'cache_size': [1000],
                             'max_iter': [-1]}
            tss = TimeSeriesSplit(n_splits=N_CV)
            model = RandomizedSearchCV(estimator=reg_model, 
                                       param_distributions=distributions,
                                       n_iter=N_ITER_MINIMAL_3,
                                       scoring='neg_mean_absolute_error',
                                       n_jobs=N_CPU,
                                       cv=tss,
                                       verbose=1)
            model.fit(X_train_scaled, y_train_scaled.ravel())
            xopt = model.best_params_
            fopt = -model.best_score_
            print('Model ready!')
    
        elif optimization_method == 'bayessearchcv':
            print('Hyperparameter tuning: BayesSearchCV')
            reg_model = NuSVR()
            distributions = {'nu': Real(0.2, 0.5, prior='uniform'),
                             'C': Real(1e-6, 1e-4, prior='log-uniform'),
                             'degree': Integer(2, 4, prior='uniform'),
                             'gamma': Real(0.01, 0.5, prior='uniform'),
                             'coef0': Real(0, 2, prior='uniform'),
                             'kernel': Categorical(['rbf']),
                             'shrinking': Categorical([True]),
                             'cache_size': Categorical([1000]),
                             'max_iter': Categorical([-1])}
            tss = TimeSeriesSplit(n_splits=N_CV)
            model = BayesSearchCV(estimator=reg_model, 
                                    search_spaces=distributions,
                                    n_iter=N_BAYESITER_MINIMAL_3,
                                    scoring='neg_mean_absolute_error',
                                    n_jobs=N_CPU,
                                    n_points=N_CPU // N_CV,
                                    cv=tss,
                                    verbose=1)
            model.fit(X_train_scaled, y_train_scaled.ravel())
            xopt = model.best_params_
            fopt = -model.best_score_
            print('Model ready!')




    elif model_name == 'nusvr_sigmoid':
        print('ML:', model_name)
        
        if optimization_method == 'pso':
            print('hyperparameter tuning: pso')
            kwargs = {'X_train_scaled': X_train_scaled,
                      'y_train_scaled': y_train_scaled.ravel(),
                      'kernel': 'sigmoid',
                      'shrinking':  True,
                      'cache_size': 1000,
                      'max_iter': -1,
                      'return_model': False}
            # nu, C, degree, gamma, coef0
            lb = [0.2, -5, 2,  0.01, 0]
            ub = [0.5,  0, 4,  0.5,  2]
            xopt, fopt = pso(nusvr, lb, ub,
                            kwargs=kwargs,
                            swarmsize=N_SWARMSIZE_MINIMAL_3,
                            omega=0.5, phip=0.5, phig=0.5, 
                            maxiter=N_MAXITER_MINIMAL_3, minstep=1e-8,
                            minfunc=1e-5, debug=True,
                            processes=N_CPU)
            print('pso ready!')
            kwargs['return_model'] = True
            fopt, model = nusvr(xopt, **kwargs)
            print('Model ready!')
        
        elif optimization_method == 'randomizedsearchcv':
            # Set dual = True if number of features > number of examples and vice versa.
            # If dual = True, then loss='squared_epsilon_insensitive'
            print('Hyperparameter tuning: RandomizedSearchCV')
            reg_model = NuSVR()
            distributions = {'nu': sp_uniform(0.2, 0.3),
                             'C': sp_loguniform(a=0.01, b=100),
                             'degree': sp_randint(2, 4),
                             'gamma': sp_uniform(0.01, 0.49),
                             'coef0': sp_uniform(0, 2),
                             'kernel': ['sigmoid'],
                             'shrinking': [True],
                             'cache_size': [1000],
                             'max_iter': [-1]}
            tss = TimeSeriesSplit(n_splits=N_CV)
            model = RandomizedSearchCV(estimator=reg_model, 
                                       param_distributions=distributions,
                                       n_iter=N_ITER_MINIMAL_3,
                                       scoring='neg_mean_absolute_error',
                                       n_jobs=N_CPU,
                                       cv=tss,
                                       verbose=1)
            model.fit(X_train_scaled, y_train_scaled.ravel())
            xopt = model.best_params_
            fopt = -model.best_score_
            print('Model ready!')
    
        elif optimization_method == 'bayessearchcv':
            print('Hyperparameter tuning: BayesSearchCV')
            reg_model = NuSVR()
            distributions = {'nu': Real(0.2, 0.5, prior='uniform'),
                             'C': Real(0.01, 100, prior='log-uniform'),
                             'degree': Integer(2, 4, prior='uniform'),
                             'gamma': Real(0.01, 0.5, prior='uniform'),
                             'coef0': Real(0, 2, prior='uniform'),
                             'kernel': Categorical(['sigmoid']),
                             'shrinking': Categorical([True]),
                             'cache_size': Categorical([1000]),
                             'max_iter': Categorical([-1])}
            tss = TimeSeriesSplit(n_splits=N_CV)
            model = BayesSearchCV(estimator=reg_model, 
                                    search_spaces=distributions,
                                    n_iter=N_BAYESITER_MINIMAL_3,
                                    scoring='neg_mean_absolute_error',
                                    n_jobs=N_CPU,
                                    n_points=N_CPU // N_CV,
                                    cv=tss,
                                    verbose=1)
            model.fit(X_train_scaled, y_train_scaled.ravel())
            xopt = model.best_params_
            fopt = -model.best_score_
            print('Model ready!')





    elif model_name == 'svr_linear':
        print('ML:', model_name)
        
        if optimization_method == 'pso':
            print('hyperparameter tuning: pso')
            kwargs = {'X_train_scaled': X_train_scaled,
                      'y_train_scaled': y_train_scaled.ravel(),
                      'kernel': 'linear',
                      'shrinking':  True,
                      'cache_size': 1000,
                      'max_iter': -1,
                      'return_model': False}
            # epsilon, C, degree, gamma, coef0
            lb = [0.01, -3, 2, 0.01, 0]
            ub = [0.5,  2,  4, 0.5,  2]
            xopt, fopt = pso(svr, lb, ub,
                            kwargs=kwargs,
                            swarmsize=N_SWARMSIZE_MINIMAL_2,
                            omega=0.5, phip=0.5, phig=0.5, 
                            maxiter=N_MAXITER_MINIMAL_2, minstep=1e-8,
                            minfunc=1e-5, debug=True,
                            processes=1)
            print('pso ready!')
            kwargs['return_model'] = True
            fopt, model = svr(xopt, **kwargs)
            print('Model ready!')
        
        elif optimization_method == 'randomizedsearchcv':
            # Set dual = True if number of features > number of examples and vice versa.
            # If dual = True, then loss='squared_epsilon_insensitive'
            print('Hyperparameter tuning: RandomizedSearchCV')
            reg_model = SVR()
            distributions = {'epsilon': sp_uniform(0.01, 0.49),
                             'C': sp_loguniform(a=0.001, b=10),
                             'degree': sp_randint(2, 4),
                             'gamma': sp_uniform(0.01, 0.49),
                             'coef0': sp_uniform(0, 2),
                             'kernel': ['linear'],
                             'shrinking': [True],
                             'cache_size': [1000],
                             'max_iter': [-1]}
            tss = TimeSeriesSplit(n_splits=N_CV)
            model = RandomizedSearchCV(estimator=reg_model, 
                                       param_distributions=distributions,
                                       n_iter=N_ITER_MINIMAL_2,
                                       scoring='neg_mean_absolute_error',
                                       n_jobs=N_CPU,
                                       cv=tss,
                                       verbose=1)
            model.fit(X_train_scaled, y_train_scaled.ravel())
            xopt = model.best_params_
            fopt = -model.best_score_
            print('Model ready!')
    
        elif optimization_method == 'bayessearchcv':
            print('Hyperparameter tuning: BayesSearchCV')
            reg_model = SVR()
            distributions = {'epsilon': Real(0.01, 1.0, prior='uniform'),
                             'C': Real(0.001, 10, prior='log-uniform'),
                             'degree': Integer(2, 4, prior='uniform'),
                             'gamma': Real(0.01, 0.5, prior='uniform'),
                             'coef0': Real(0, 2, prior='uniform'),
                             'kernel': Categorical(['linear']),
                             'shrinking': Categorical([True]),
                             'cache_size': Categorical([1000]),
                             'max_iter': Categorical([-1])}
            tss = TimeSeriesSplit(n_splits=N_CV)
            model = BayesSearchCV(estimator=reg_model, 
                                    search_spaces=distributions,
                                    n_iter=N_BAYESITER_MINIMAL_2,
                                    scoring='neg_mean_absolute_error',
                                    n_jobs=N_CPU,
                                    n_points=N_CPU // N_CV,
                                    cv=tss,
                                    verbose=1)
            model.fit(X_train_scaled, y_train_scaled.ravel())
            xopt = model.best_params_
            fopt = -model.best_score_
            print('Model ready!')




    elif model_name == 'svr_poly':
        print('ML:', model_name, flush=True)
        
        if optimization_method == 'pso':
            print('hyperparameter tuning: pso', flush=True)
            kwargs = {'X_train_scaled': X_train_scaled,
                      'y_train_scaled': y_train_scaled.ravel(),
                      'kernel': 'poly',
                      'shrinking': True,
                      'cache_size': 10000,
                      'max_iter': 1e6,
                      'return_model': False}
            # epsilon, C, degree, gamma, coef0; C is given as log10(C)
            lb = [0.01, -3, 2, 0.01, 0]
            ub = [0.4,  0,  4, 0.5,  2]
            xopt, fopt = pso(svr, lb, ub,
                            kwargs=kwargs,
                            swarmsize=N_SWARMSIZE_MINIMAL_0,
                            omega=0.5, phip=0.5, phig=0.5, 
                            maxiter=N_MAXITER_MINIMAL_0, minstep=1e-8,
                            minfunc=1e-5, debug=True,
                            processes=1) # This was exceptionally slow with N_CPU
            print('pso ready!', flush=True)
            kwargs['return_model'] = True
            fopt, model = svr(xopt, **kwargs)
            print('Model ready!')
        
        elif optimization_method == 'randomizedsearchcv':
            # Set dual = True if number of features > number of examples and vice versa.
            # If dual = True, then loss='squared_epsilon_insensitive'
            print('Hyperparameter tuning: RandomizedSearchCV')
            reg_model = SVR()
            distributions = {'epsilon': sp_uniform(0.01, 0.39),
                             'C': sp_loguniform(a=0.001, b=1),
                             'degree': sp_randint(2, 4),
                             'gamma': sp_uniform(0.01, 0.49),
                             'coef0': sp_uniform(0, 2),
                             'kernel': ['poly'],
                             'shrinking': [True],
                             'cache_size': [10000],
                             'max_iter': [1e6]}
            tss = TimeSeriesSplit(n_splits=N_CV)
            model = RandomizedSearchCV(estimator=reg_model, 
                                       param_distributions=distributions,
                                       n_iter=N_ITER_MINIMAL_0,
                                       scoring='neg_mean_absolute_error',
                                       n_jobs=1, # This is changed to N_CPU
                                       cv=tss,
                                       verbose=1)
            model.fit(X_train_scaled, y_train_scaled.ravel())
            xopt = model.best_params_
            fopt = -model.best_score_
            print('Model ready!')
    
        elif optimization_method == 'bayessearchcv':
            print('Hyperparameter tuning: BayesSearchCV')
            reg_model = SVR()
            distributions = {'epsilon': Real(0.01, 0.4, prior='uniform'),
                             'C': Real(0.001, 1, prior='log-uniform'),
                             'degree': Integer(2, 4, prior='uniform'),
                             'gamma': Real(0.01, 0.5, prior='uniform'),
                             'coef0': Real(0, 2, prior='uniform'),
                             'kernel': Categorical(['poly']),
                             'shrinking': Categorical([True]),
                             'cache_size': Categorical([10000]),
                             'max_iter': Categorical([1e6])}
            tss = TimeSeriesSplit(n_splits=N_CV)
            model = BayesSearchCV(estimator=reg_model, 
                                    search_spaces=distributions,
                                    n_iter=N_BAYESITER_MINIMAL_0,
                                    scoring='neg_mean_absolute_error',
                                    n_jobs=1, # This is changed to 
                                    n_points=1, # This is changed to 
                                    cv=tss,
                                    verbose=1)
            model.fit(X_train_scaled, y_train_scaled.ravel())
            xopt = model.best_params_
            fopt = -model.best_score_
            print('Model ready!')
        else:
            print('Unknown optimization method!', flush=True)




    elif model_name == 'svr_rbf':
        print('ML: svr_rbf')
        
        if optimization_method == 'pso':
            print('hyperparameter tuning: pso')
            kwargs = {'X_train_scaled': X_train_scaled,
                      'y_train_scaled': y_train_scaled.ravel(),
                      'kernel': 'rbf',
                      'shrinking':  True,
                      'cache_size': 1000,
                      'max_iter': -1,
                      'return_model': False}
            # epsilon, C, degree, gamma, coef0
            lb = [0.01, -3, 2, 0.01, 0]
            ub = [0.5,  2,  4, 0.5,  2]
            xopt, fopt = pso(svr, lb, ub,
                            kwargs=kwargs,
                            swarmsize=N_SWARMSIZE_MINIMAL_3,
                            omega=0.5, phip=0.5, phig=0.5, 
                            maxiter=N_MAXITER_MINIMAL_3, minstep=1e-8,
                            minfunc=1e-5, debug=True,
                            processes=N_CPU)
            print('pso ready!')
            kwargs['return_model'] = True
            fopt, model = svr(xopt, **kwargs)
            print('Model ready!')
        
        elif optimization_method == 'randomizedsearchcv':
            # Set dual = True if number of features > number of examples and vice versa.
            # If dual = True, then loss='squared_epsilon_insensitive'
            print('Hyperparameter tuning: RandomizedSearchCV')
            reg_model = SVR()
            distributions = {'epsilon': sp_uniform(0.01, 0.49),
                             'C': sp_loguniform(a=0.001, b=10),
                             'degree': sp_randint(2, 4),
                             'gamma': sp_uniform(0.01, 0.49),
                             'coef0': sp_uniform(0, 2),
                             'kernel': ['rbf'],
                             'shrinking': [True],
                             'cache_size': [1000],
                             'max_iter': [-1]}
            tss = TimeSeriesSplit(n_splits=N_CV)
            model = RandomizedSearchCV(estimator=reg_model, 
                                       param_distributions=distributions,
                                       n_iter=N_ITER_MINIMAL_3,
                                       scoring='neg_mean_absolute_error',
                                       n_jobs=N_CPU,
                                       cv=tss,
                                       verbose=1)
            model.fit(X_train_scaled, y_train_scaled.ravel())
            xopt = model.best_params_
            fopt = -model.best_score_
            print('Model ready!')
    
        elif optimization_method == 'bayessearchcv':
            print('Hyperparameter tuning: BayesSearchCV')
            reg_model = SVR()
            distributions = {'epsilon': Real(0.01, 0.5, prior='uniform'),
                             'C': Real(0.001, 10, prior='log-uniform'),
                             'degree': Integer(2, 4, prior='uniform'),
                             'gamma': Real(0.01, 0.5, prior='uniform'),
                             'coef0': Real(0, 2, prior='uniform'),
                             'kernel': Categorical(['rbf']),
                             'shrinking': Categorical([True]),
                             'cache_size': Categorical([1000]),
                             'max_iter': Categorical([-1])}
            tss = TimeSeriesSplit(n_splits=N_CV)
            model = BayesSearchCV(estimator=reg_model, 
                                    search_spaces=distributions,
                                    n_iter=N_BAYESITER_MINIMAL_3,
                                    scoring='neg_mean_absolute_error',
                                    n_jobs=N_CPU,
                                    n_points=N_CPU // N_CV,
                                    cv=tss,
                                    verbose=1)
            model.fit(X_train_scaled, y_train_scaled.ravel())
            xopt = model.best_params_
            fopt = -model.best_score_
            print('Model ready!')





    elif model_name == 'svr_sigmoid':
        print('ML:', model_name)
        
        if optimization_method == 'pso':
            print('hyperparameter tuning: pso')
            kwargs = {'X_train_scaled': X_train_scaled,
                      'y_train_scaled': y_train_scaled.ravel(),
                      'kernel': 'sigmoid',
                      'shrinking':  True,
                      'cache_size': 1000,
                      'max_iter': -1,
                      'return_model': False}
            # epsilon, C, degree, gamma, coef0
            lb = [0.01, -3, 2, 0.01, 0]
            ub = [0.5,  2,  4, 0.5,  2]
            xopt, fopt = pso(svr, lb, ub,
                            kwargs=kwargs,
                            swarmsize=N_SWARMSIZE_MINIMAL_3,
                            omega=0.5, phip=0.5, phig=0.5, 
                            maxiter=N_MAXITER_MINIMAL_3, minstep=1e-8,
                            minfunc=1e-5, debug=True,
                            processes=N_CPU)
            print('pso ready!')
            kwargs['return_model'] = True
            fopt, model = svr(xopt, **kwargs)
            print('Model ready!')
        
        elif optimization_method == 'randomizedsearchcv':
            # Set dual = True if number of features > number of examples and vice versa.
            # If dual = True, then loss='squared_epsilon_insensitive'
            print('Hyperparameter tuning: RandomizedSearchCV')
            reg_model = SVR()
            distributions = {'epsilon': sp_uniform(0.01, 0.49),
                             'C': sp_loguniform(a=0.001, b=10),
                             'degree': sp_randint(2, 4),
                             'gamma': sp_uniform(0.01, 0.49),
                             'coef0': sp_uniform(0, 2),
                             'kernel': ['sigmoid'],
                             'shrinking': [True],
                             'cache_size': [1000],
                             'max_iter': [-1]}
            tss = TimeSeriesSplit(n_splits=N_CV)
            model = RandomizedSearchCV(estimator=reg_model, 
                                       param_distributions=distributions,
                                       n_iter=N_ITER_MINIMAL_3,
                                       scoring='neg_mean_absolute_error',
                                       n_jobs=N_CPU,
                                       cv=tss,
                                       verbose=1)
            model.fit(X_train_scaled, y_train_scaled.ravel())
            xopt = model.best_params_
            fopt = -model.best_score_
            print('Model ready!')
    
        elif optimization_method == 'bayessearchcv':
            print('Hyperparameter tuning: BayesSearchCV')
            reg_model = SVR()
            distributions = {'epsilon': Real(0.01, 0.5, prior='uniform'),
                             'C': Real(0.001, 10, prior='log-uniform'),
                             'degree': Integer(2, 4, prior='uniform'),
                             'gamma': Real(0.01, 0.5, prior='uniform'),
                             'coef0': Real(0, 2, prior='uniform'),
                             'kernel': Categorical(['sigmoid']),
                             'shrinking': Categorical([True]),
                             'cache_size': Categorical([1000]),
                             'max_iter': Categorical([-1])}
            tss = TimeSeriesSplit(n_splits=N_CV)
            model = BayesSearchCV(estimator=reg_model, 
                                    search_spaces=distributions,
                                    n_iter=N_BAYESITER_MINIMAL_3,
                                    scoring='neg_mean_absolute_error',
                                    n_jobs=N_CPU,
                                    n_points=N_CPU // N_CV,
                                    cv=tss,
                                    verbose=1)
            model.fit(X_train_scaled, y_train_scaled.ravel())
            xopt = model.best_params_
            fopt = -model.best_score_
            print('Model ready!')





    elif model_name == 'kneighborsregressor_uniform':
        print('ML:', model_name)
        
        if optimization_method == 'pso':
            print('hyperparameter tuning: pso')
            kwargs = {'X_train_scaled': X_train_scaled,
                      'y_train_scaled': y_train_scaled.ravel(),
                      'weights': 'uniform',
                      'n_jobs': 1,
                      'return_model': False}
            # n_neighbors, leaf_size, p
            lb = [3,  2,  2]
            ub = [30, 20, 4]
            xopt, fopt = pso(kneighborsregressor, lb, ub,
                            kwargs=kwargs,
                            swarmsize=N_SWARMSIZE_MINIMAL_3,
                            omega=0.5, phip=0.5, phig=0.5, 
                            maxiter=N_MAXITER_MINIMAL_3, minstep=1e-8,
                            minfunc=1e-5, debug=True,
                            processes=N_CPU) # Changed this to 1
            print('pso ready!')
            kwargs['return_model'] = True
            fopt, model = kneighborsregressor(xopt, **kwargs)
            print('Model ready!')
        
        elif optimization_method == 'randomizedsearchcv':
            print('Hyperparameter tuning: RandomizedSearchCV')
            reg_model = KNeighborsRegressor()
            distributions = {'n_neighbors': sp_randint(3, 31),
                             'weights': ['uniform'],
                             'leaf_size': sp_randint(2, 21),
                             'p': sp_randint(2, 5),
                             'n_jobs': [1]}
            tss = TimeSeriesSplit(n_splits=N_CV)
            model = RandomizedSearchCV(estimator=reg_model, 
                                       param_distributions=distributions,
                                       n_iter=N_ITER_MINIMAL_3,
                                       scoring='neg_mean_absolute_error',
                                       n_jobs=N_CPU,
                                       cv=tss,
                                       verbose=1)
            model.fit(X_train_scaled, y_train_scaled.ravel())
            xopt = model.best_params_
            fopt = -model.best_score_
            print('Model ready!')
    
        elif optimization_method == 'bayessearchcv':
            print('Hyperparameter tuning: BayesSearchCV')
            reg_model = KNeighborsRegressor()
            distributions = {'n_neighbors': Integer(3, 30, prior='uniform'),
                             'weights': Categorical(['uniform']),
                             'leaf_size': Integer(2, 20, prior='uniform'),
                             'p': Integer(2, 4, prior='uniform'),
                             'n_jobs': [1]}
            tss = TimeSeriesSplit(n_splits=N_CV)
            model = BayesSearchCV(estimator=reg_model, 
                                    search_spaces=distributions,
                                    n_iter=N_BAYESITER_MINIMAL_3,
                                    scoring='neg_mean_absolute_error',
                                    n_jobs=N_CPU, # Changed this to 1
                                    n_points=N_CPU // N_CV, # Changed this to 1
                                    cv=tss,
                                    verbose=1)
            model.fit(X_train_scaled, y_train_scaled.ravel())
            xopt = model.best_params_
            fopt = -model.best_score_
            print('Model ready!')




    elif model_name == 'kneighborsregressor_distance':
        print('ML:', model_name)
        
        if optimization_method == 'pso':
            print('hyperparameter tuning: pso')
            kwargs = {'X_train_scaled': X_train_scaled,
                      'y_train_scaled': y_train_scaled.ravel(),
                      'weights': 'distance',
                      'n_jobs': 1,
                      'return_model': False}
            # n_neighbors, leaf_size, p
            lb = [3,  2,  2]
            ub = [30, 20, 4]
            xopt, fopt = pso(kneighborsregressor, lb, ub,
                            kwargs=kwargs,
                            swarmsize=N_SWARMSIZE_MINIMAL_3,
                            omega=0.5, phip=0.5, phig=0.5, 
                            maxiter=N_MAXITER_MINIMAL_3, minstep=1e-8,
                            minfunc=1e-5, debug=True,
                            processes=N_CPU)
            print('pso ready!')
            kwargs['return_model'] = True
            fopt, model = kneighborsregressor(xopt, **kwargs)
            print('Model ready!')
        
        elif optimization_method == 'randomizedsearchcv':
            print('Hyperparameter tuning: RandomizedSearchCV')
            reg_model = KNeighborsRegressor()
            distributions = {'n_neighbors': sp_randint(3, 31),
                             'weights': ['distance'],
                             'leaf_size': sp_randint(2, 21),
                             'p': sp_randint(2, 5),
                             'n_jobs': [1]}
            tss = TimeSeriesSplit(n_splits=N_CV)
            model = RandomizedSearchCV(estimator=reg_model, 
                                       param_distributions=distributions,
                                       n_iter=N_ITER_MINIMAL_3,
                                       scoring='neg_mean_absolute_error',
                                       n_jobs=N_CPU,
                                       cv=tss,
                                       verbose=1)
            model.fit(X_train_scaled, y_train_scaled.ravel())
            xopt = model.best_params_
            fopt = -model.best_score_
            print('Model ready!')
    
        elif optimization_method == 'bayessearchcv':
            print('Hyperparameter tuning: BayesSearchCV')
            reg_model = KNeighborsRegressor()
            distributions = {'n_neighbors': Integer(3, 30, prior='uniform'),
                             'weights': Categorical(['distance']),
                             'leaf_size': Integer(2, 20, prior='uniform'),
                             'p': Integer(2, 4, prior='uniform'),
                             'n_jobs': [1]}
            tss = TimeSeriesSplit(n_splits=N_CV)
            model = BayesSearchCV(estimator=reg_model, 
                                    search_spaces=distributions,
                                    n_iter=N_BAYESITER_MINIMAL_3,
                                    scoring='neg_mean_absolute_error',
                                    n_jobs=N_CPU,
                                    n_points=N_CPU // N_CV,
                                    cv=tss,
                                    verbose=1)
            model.fit(X_train_scaled, y_train_scaled.ravel())
            xopt = model.best_params_
            fopt = -model.best_score_
            print('Model ready!')
    
    
    
    

    elif model_name == 'decisiontreeregressor_best':
        print('ML:', model_name)
        
        
        if optimization_method == 'pso':
            print('hyperparameter tuning: pso')
            kwargs = {'X_train_scaled': X_train_scaled,
                      'y_train_scaled': y_train_scaled.ravel(),
                      'splitter': 'best', # best, random
                      'criterion': 'mae', # mse, friedman_mse, mae
                      'max_features': 'auto', # auto, sqrt, log2
                      'max_depth': None,
                      'return_model': False}
            # 'min_samples_leaf', 'min_samples_split', 'min_weight_fraction_leaf'
            lb = [1,  3,  1e-5]
            ub = [30, 30, 0.1]
            xopt, fopt = pso(decisiontreeregressor, lb, ub,
                            kwargs=kwargs,
                            swarmsize=N_SWARMSIZE_MINIMAL_3,
                            omega=0.5, phip=0.5, phig=0.5, 
                            maxiter=N_MAXITER_MINIMAL_3, minstep=1e-8,
                            minfunc=1e-5, debug=True,
                            processes=N_CPU)
            print('pso ready!')
            kwargs['return_model'] = True
            fopt, model = decisiontreeregressor(xopt, **kwargs)
            print('Model ready!')
        
        elif optimization_method == 'randomizedsearchcv':
            print('Hyperparameter tuning: RandomizedSearchCV')
            reg_model = DecisionTreeRegressor()
            distributions = {'min_samples_leaf': sp_randint(1, 31),
                             'min_samples_split': sp_randint(3,31),
                             'min_weight_fraction_leaf': sp_loguniform(a=1e-5, b=0.1),
                             'splitter':['best'],
                             'criterion':['mae'],
                             'max_features':['auto'],
                             'max_depth': [None]}
            tss = TimeSeriesSplit(n_splits=N_CV)
            model = RandomizedSearchCV(estimator=reg_model, 
                                       param_distributions=distributions,
                                       n_iter=N_ITER_MINIMAL_3,
                                       scoring='neg_mean_absolute_error',
                                       n_jobs=N_CPU,
                                       cv=tss,
                                       verbose=1)
            model.fit(X_train_scaled, y_train_scaled.ravel())
            xopt = model.best_params_
            fopt = -model.best_score_
            print('Model ready!')
    
        elif optimization_method == 'bayessearchcv':
            print('Hyperparameter tuning: BayesSearchCV')
            reg_model = DecisionTreeRegressor()
            distributions = {'min_samples_leaf': Integer(1, 30, prior='uniform'),
                             'min_samples_split': Integer(3, 31, prior='uniform'),
                             'min_weight_fraction_leaf': Real(1e-5, 0.1, prior='log-uniform'),
                             'splitter': Categorical(['best']),
                             'criterion': Categorical(['mae']),
                             'max_features': Categorical(['auto']),
                             'max_depth': Categorical([None])}
            tss = TimeSeriesSplit(n_splits=N_CV)
            model = BayesSearchCV(estimator=reg_model, 
                                    search_spaces=distributions,
                                    n_iter=N_BAYESITER_MINIMAL_3,
                                    scoring='neg_mean_absolute_error',
                                    n_jobs=N_CPU,
                                    n_points=N_CPU // N_CV,
                                    cv=tss,
                                    verbose=1)
            model.fit(X_train_scaled, y_train_scaled.ravel())
            xopt = model.best_params_
            fopt = -model.best_score_
            print('Model ready!')
    
    


    elif model_name == 'decisiontreeregressor_random':
        print('ML:', model_name)
        
        
        if optimization_method == 'pso':
            print('hyperparameter tuning: pso')
            kwargs = {'X_train_scaled': X_train_scaled,
                      'y_train_scaled': y_train_scaled.ravel(),
                      'splitter': 'random', # best, random
                      'criterion': 'mae', # mse, friedman_mse, mae
                      'max_features': 'auto', # auto, sqrt, log2
                      'max_depth': None,
                      'return_model': False}
            # randint 'min_samples_leaf', randint 'min_samples_split', loguniform 'min_weight_fraction_leaf'
            lb = [1,  3,  1e-5]
            ub = [30, 30, 0.1]
            xopt, fopt = pso(decisiontreeregressor, lb, ub,
                            kwargs=kwargs,
                            swarmsize=N_SWARMSIZE_MINIMAL_3,
                            omega=0.5, phip=0.5, phig=0.5, 
                            maxiter=N_MAXITER_MINIMAL_3, minstep=1e-8,
                            minfunc=1e-5, debug=True,
                            processes=N_CPU)
            print('pso ready!')
            kwargs['return_model'] = True
            fopt, model = decisiontreeregressor(xopt, **kwargs)
            print('Model ready!')
        
        elif optimization_method == 'randomizedsearchcv':
            print('Hyperparameter tuning: RandomizedSearchCV')
            reg_model = DecisionTreeRegressor()
            distributions = {'min_samples_leaf': sp_randint(1, 31),
                             'min_samples_split': sp_randint(3,31),
                             'min_weight_fraction_leaf': sp_loguniform(a=1e-5, b=0.1),
                             'splitter':['random'],
                             'criterion':['mae'],
                             'max_features':['auto'],
                             'max_depth': [None]}
            tss = TimeSeriesSplit(n_splits=N_CV)
            model = RandomizedSearchCV(estimator=reg_model, 
                                       param_distributions=distributions,
                                       n_iter=N_ITER_MINIMAL_3,
                                       scoring='neg_mean_absolute_error',
                                       n_jobs=N_CPU,
                                       cv=tss,
                                       verbose=1)
            model.fit(X_train_scaled, y_train_scaled.ravel())
            xopt = model.best_params_
            fopt = -model.best_score_
            print('Model ready!')
    
        elif optimization_method == 'bayessearchcv':
            print('Hyperparameter tuning: BayesSearchCV')
            reg_model = DecisionTreeRegressor()
            distributions = {'min_samples_leaf': Integer(1, 30, prior='uniform'),
                             'min_samples_split': Integer(3, 31, prior='uniform'),
                             'min_weight_fraction_leaf': Real(1e-5, 0.1, prior='log-uniform'),
                             'splitter': Categorical(['random']),
                             'criterion': Categorical(['mae']),
                             'max_features': Categorical(['auto']),
                             'max_depth': Categorical([None])}
            tss = TimeSeriesSplit(n_splits=N_CV)
            model = BayesSearchCV(estimator=reg_model, 
                                    search_spaces=distributions,
                                    n_iter=N_BAYESITER_MINIMAL_3,
                                    scoring='neg_mean_absolute_error',
                                    n_jobs=N_CPU,
                                    n_points=N_CPU // N_CV,
                                    cv=tss,
                                    verbose=1)
            model.fit(X_train_scaled, y_train_scaled.ravel())
            xopt = model.best_params_
            fopt = -model.best_score_
            print('Model ready!')






    elif model_name == 'extratreeregressor_best':
        print('ML:', model_name)
        
        if optimization_method == 'pso':
            print('hyperparameter tuning: pso')
            kwargs = {'X_train_scaled': X_train_scaled,
                      'y_train_scaled': y_train_scaled.ravel(),
                      'criterion': 'mae',
                      'splitter': 'best',
                      'max_depth': None,
                      'max_features': 'auto',
                      'return_model': False}
            # int min_samples_leaf, int min_samples_split, loguniform min_weight_fraction_leaf
            lb = [1,  3,  1e-5]
            ub = [30, 30, 0.1]
            xopt, fopt = pso(extratreeregressor, lb, ub,
                            kwargs=kwargs,
                            swarmsize=N_SWARMSIZE_MINIMAL_3,
                            omega=0.5, phip=0.5, phig=0.5, 
                            maxiter=N_MAXITER_MINIMAL_3, minstep=1e-8,
                            minfunc=1e-5, debug=True,
                            processes=N_CPU)
            print('pso ready!')
            kwargs['return_model'] = True
            fopt, model = extratreeregressor(xopt, **kwargs)
            print('Model ready!')
        
        elif optimization_method == 'randomizedsearchcv':
            print('Hyperparameter tuning: RandomizedSearchCV')
            reg_model = ExtraTreeRegressor()
            distributions = {'min_samples_leaf': sp_randint(1, 30),
                             'min_samples_split': sp_randint(3,30),
                             'min_weight_fraction_leaf': sp_loguniform(a=1e-5, b=0.1),
                             'splitter':['best'],
                             'criterion':['mae'],
                             'max_features':['auto'],
                             'max_depth': [None]}
            tss = TimeSeriesSplit(n_splits=N_CV)
            model = RandomizedSearchCV(estimator=reg_model, 
                                       param_distributions=distributions,
                                       n_iter=N_ITER_MINIMAL_3,
                                       scoring='neg_mean_absolute_error',
                                       n_jobs=N_CPU,
                                       cv=tss,
                                       verbose=1)
            model.fit(X_train_scaled, y_train_scaled.ravel())
            xopt = model.best_params_
            fopt = -model.best_score_
            print('Model ready!')
    
        elif optimization_method == 'bayessearchcv':
            print('Hyperparameter tuning: BayesSearchCV')
            reg_model = ExtraTreeRegressor()
            distributions = {'min_samples_leaf': Integer(1, 30, prior='uniform'),
                             'min_samples_split': Integer(3, 30, prior='uniform'),
                             'min_weight_fraction_leaf': Real(1e-5, 0.1, prior='log-uniform'),
                             'splitter': Categorical(['best']),
                             'criterion': Categorical(['mae']),
                             'max_features': Categorical(['auto']),
                             'max_depth': Categorical([None])}
            tss = TimeSeriesSplit(n_splits=N_CV)
            model = BayesSearchCV(estimator=reg_model, 
                                    search_spaces=distributions,
                                    n_iter=N_BAYESITER_MINIMAL_3,
                                    scoring='neg_mean_absolute_error',
                                    n_jobs=N_CPU,
                                    n_points=N_CPU // N_CV,
                                    cv=tss,
                                    verbose=1)
            model.fit(X_train_scaled, y_train_scaled.ravel())
            xopt = model.best_params_
            fopt = -model.best_score_
            print('Model ready!')
            





    elif model_name == 'extratreeregressor_random':
        print('ML:', model_name)
        
        if optimization_method == 'pso':
            print('hyperparameter tuning: pso')
            kwargs = {'X_train_scaled': X_train_scaled,
                      'y_train_scaled': y_train_scaled.ravel(),
                      'criterion': 'mae',
                      'splitter': 'random',
                      'max_depth': None,
                      'max_features': 'auto',
                      'return_model': False}
            # int min_samples_leaf, int min_samples_split, loguniform min_weight_fraction_leaf
            lb = [1,  3,  1e-5]
            ub = [30, 30, 0.1]
            xopt, fopt = pso(extratreeregressor, lb, ub,
                            kwargs=kwargs,
                            swarmsize=N_SWARMSIZE_MINIMAL_3,
                            omega=0.5, phip=0.5, phig=0.5, 
                            maxiter=N_MAXITER_MINIMAL_3, minstep=1e-8,
                            minfunc=1e-5, debug=True,
                            processes=N_CPU)
            print('pso ready!')
            kwargs['return_model'] = True
            fopt, model = extratreeregressor(xopt, **kwargs)
            print('Model ready!')
        
        elif optimization_method == 'randomizedsearchcv':
            print('Hyperparameter tuning: RandomizedSearchCV')
            reg_model = ExtraTreeRegressor()
            distributions = {'min_samples_leaf': sp_randint(1, 30),
                             'min_samples_split': sp_randint(3,30),
                             'min_weight_fraction_leaf': sp_loguniform(a=1e-5, b=0.1),
                             'splitter':['random'],
                             'criterion':['mae'],
                             'max_features':['auto'],
                             'max_depth': [None]}
            tss = TimeSeriesSplit(n_splits=N_CV)
            model = RandomizedSearchCV(estimator=reg_model, 
                                       param_distributions=distributions,
                                       n_iter=N_ITER_MINIMAL_3,
                                       scoring='neg_mean_absolute_error',
                                       n_jobs=N_CPU,
                                       cv=tss,
                                       verbose=1)
            model.fit(X_train_scaled, y_train_scaled.ravel())
            xopt = model.best_params_
            fopt = -model.best_score_
            print('Model ready!')
    
        elif optimization_method == 'bayessearchcv':
            print('Hyperparameter tuning: BayesSearchCV')
            reg_model = ExtraTreeRegressor()
            distributions = {'min_samples_leaf': Integer(1, 30, prior='uniform'),
                             'min_samples_split': Integer(3, 30, prior='uniform'),
                             'min_weight_fraction_leaf': Real(1e-5, 0.1, prior='log-uniform'),
                             'splitter': Categorical(['random']),
                             'criterion': Categorical(['mae']),
                             'max_features': Categorical(['auto']),
                             'max_depth': Categorical([None])}
            tss = TimeSeriesSplit(n_splits=N_CV)
            model = BayesSearchCV(estimator=reg_model, 
                                    search_spaces=distributions,
                                    n_iter=N_BAYESITER_MINIMAL_3,
                                    scoring='neg_mean_absolute_error',
                                    n_jobs=N_CPU,
                                    n_points=N_CPU // N_CV,
                                    cv=tss,
                                    verbose=1)
            model.fit(X_train_scaled, y_train_scaled.ravel())
            xopt = model.best_params_
            fopt = -model.best_score_
            print('Model ready!')




    elif model_name == 'adaboost_decisiontree':
        print('ML:', model_name)
        
        if optimization_method == 'pso':
            print('hyperparameter tuning: pso')
            kwargs = {'X_train_scaled': X_train_scaled,
                      'y_train_scaled': y_train_scaled.ravel(),
                      'return_model': False}
            # n_estimators, learning_rate, base_estimator__max_depth
            lb = [50,  0.1, 1]
            ub = [300, 1.0, 5]
            xopt, fopt = pso(adaboost_decisiontree, lb, ub,
                            kwargs=kwargs,
                            swarmsize=N_SWARMSIZE_MINIMAL_3,
                            omega=0.5, phip=0.5, phig=0.5, 
                            maxiter=N_MAXITER_MINIMAL_3, minstep=1e-8,
                            minfunc=1e-5, debug=True,
                            processes=N_CPU)
            print('pso ready!')
            kwargs['return_model'] = True
            fopt, model = adaboost_decisiontree(xopt, **kwargs)
            print('Model ready!')
        
        elif optimization_method == 'randomizedsearchcv':
            print('Hyperparameter tuning: RandomizedSearchCV')
            reg_model = AdaBoostRegressor()
            distributions = {'n_estimators': sp_randint(50, 300),
                             'learning_rate': sp_uniform(0.1, 0.9),
                             'base_estimator': [DecisionTreeRegressor()],
                             'base_estimator__max_depth': sp_randint(1, 6)}
            tss = TimeSeriesSplit(n_splits=N_CV)
            model = RandomizedSearchCV(estimator=reg_model, 
                                       param_distributions=distributions,
                                       n_iter=N_ITER_MINIMAL_3,
                                       scoring='neg_mean_absolute_error',
                                       n_jobs=N_CPU,
                                       cv=tss,
                                       verbose=1)
            model.fit(X_train_scaled, y_train_scaled.ravel())
            xopt = model.best_params_
            fopt = -model.best_score_
            print('Model ready!')
    
        elif optimization_method == 'bayessearchcv':
            print('Hyperparameter tuning: BayesSearchCV')
            reg_model = AdaBoostRegressor()
            distributions = {'n_estimators': Integer(50, 300, prior='uniform'),
                             'learning_rate': Real(0.1, 1.0, prior='uniform'),
                             'base_estimator': [DecisionTreeRegressor()],
                             'base_estimator__max_depth': Integer(1, 5, prior='uniform')}
            tss = TimeSeriesSplit(n_splits=N_CV)
            model = BayesSearchCV(estimator=reg_model, 
                                    search_spaces=distributions,
                                    n_iter=N_BAYESITER_MINIMAL_3,
                                    scoring='neg_mean_absolute_error',
                                    n_jobs=N_CPU,
                                    n_points=N_CPU // N_CV,
                                    cv=tss,
                                    verbose=1)
            model.fit(X_train_scaled, y_train_scaled.ravel())
            xopt = model.best_params_
            fopt = -model.best_score_
            print('Model ready!')





    elif model_name == 'adaboost_extratree':
        print('ML:', model_name)
        
        if optimization_method == 'pso':
            print('hyperparameter tuning: pso')
            kwargs = {'X_train_scaled': X_train_scaled,
                      'y_train_scaled': y_train_scaled.ravel(),
                      'return_model': False}
            # n_estimators, learning_rate, base_estimator__max_depth
            lb = [50,  0.1, 1]
            ub = [300, 1.0, 5]
            xopt, fopt = pso(adaboost_extratree, lb, ub,
                            kwargs=kwargs,
                            swarmsize=N_SWARMSIZE_MINIMAL_3,
                            omega=0.5, phip=0.5, phig=0.5, 
                            maxiter=N_MAXITER_MINIMAL_3, minstep=1e-8,
                            minfunc=1e-5, debug=True,
                            processes=N_CPU)
            print('pso ready!')
            kwargs['return_model'] = True
            fopt, model = adaboost_extratree(xopt, **kwargs)
            print('Model ready!')
        
        elif optimization_method == 'randomizedsearchcv':
            print('Hyperparameter tuning: RandomizedSearchCV')
            reg_model = AdaBoostRegressor()
            distributions = {'n_estimators': sp_randint(50, 300),
                             'learning_rate': sp_uniform(0.1, 0.9),
                             'base_estimator': [ExtraTreeRegressor()],
                             'base_estimator__max_depth': sp_randint(1, 6)}
            tss = TimeSeriesSplit(n_splits=N_CV)
            model = RandomizedSearchCV(estimator=reg_model, 
                                       param_distributions=distributions,
                                       n_iter=N_ITER_MINIMAL_3,
                                       scoring='neg_mean_absolute_error',
                                       n_jobs=N_CPU,
                                       cv=tss,
                                       verbose=1)
            model.fit(X_train_scaled, y_train_scaled.ravel())
            xopt = model.best_params_
            fopt = -model.best_score_
            print('Model ready!')
    
        elif optimization_method == 'bayessearchcv':
            print('Hyperparameter tuning: BayesSearchCV')
            reg_model = AdaBoostRegressor()
            distributions = {'n_estimators': Integer(50, 300, prior='uniform'),
                             'learning_rate': Real(0.1, 1.0, prior='uniform'),
                             'base_estimator': [ExtraTreeRegressor()],
                             'base_estimator__max_depth': Integer(1, 5, prior='uniform')}
            tss = TimeSeriesSplit(n_splits=N_CV)
            model = BayesSearchCV(estimator=reg_model, 
                                    search_spaces=distributions,
                                    n_iter=N_BAYESITER_MINIMAL_3,
                                    scoring='neg_mean_absolute_error',
                                    n_jobs=N_CPU,
                                    n_points=N_CPU // N_CV,
                                    cv=tss,
                                    verbose=1)
            model.fit(X_train_scaled, y_train_scaled.ravel())
            xopt = model.best_params_
            fopt = -model.best_score_
            print('Model ready!')




    elif model_name == 'bagging_decisiontree':
        print('ML:', model_name)
        
        if optimization_method == 'pso':
            print('hyperparameter tuning: pso')
            kwargs = {'X_train_scaled': X_train_scaled,
                      'y_train_scaled': y_train_scaled.ravel(),
                      'n_jobs': 1,
                      'return_model': False}
            # n_estimators, base_estimator__max_depth
            lb = [50,  1]
            ub = [300, 5]
            xopt, fopt = pso(bagging_decisiontree, lb, ub,
                            kwargs=kwargs,
                            swarmsize=N_SWARMSIZE_MINIMAL_3,
                            omega=0.5, phip=0.5, phig=0.5, 
                            maxiter=N_MAXITER_MINIMAL_3, minstep=1e-8,
                            minfunc=1e-5, debug=True,
                            processes=N_CPU)
            print('pso ready!')
            kwargs['return_model'] = True
            fopt, model = bagging_decisiontree(xopt, **kwargs)
            print('Model ready!')
        
        elif optimization_method == 'randomizedsearchcv':
            print('Hyperparameter tuning: RandomizedSearchCV')
            reg_model = BaggingRegressor()
            distributions = {'n_estimators': sp_randint(50, 300),
                             'n_jobs': [1],
                             'base_estimator': [DecisionTreeRegressor()],
                             'base_estimator__max_depth': sp_randint(1, 6)}
            tss = TimeSeriesSplit(n_splits=N_CV)
            model = RandomizedSearchCV(estimator=reg_model, 
                                       param_distributions=distributions,
                                       n_iter=N_ITER_MINIMAL_3,
                                       scoring='neg_mean_absolute_error',
                                       n_jobs=N_CPU,
                                       cv=tss,
                                       verbose=1)
            model.fit(X_train_scaled, y_train_scaled.ravel())
            xopt = model.best_params_
            fopt = -model.best_score_
            print('Model ready!')
    
        elif optimization_method == 'bayessearchcv':
            print('Hyperparameter tuning: BayesSearchCV')
            reg_model = BaggingRegressor()
            distributions = {'n_estimators': Integer(50, 300, prior='uniform'),
                             'n_jobs': [1],
                             'base_estimator': [DecisionTreeRegressor()],
                             'base_estimator__max_depth': Integer(1, 5, prior='uniform')}
            tss = TimeSeriesSplit(n_splits=N_CV)
            model = BayesSearchCV(estimator=reg_model, 
                                    search_spaces=distributions,
                                    n_iter=N_BAYESITER_MINIMAL_3,
                                    scoring='neg_mean_absolute_error',
                                    n_jobs=N_CPU,
                                    n_points=N_CPU // N_CV,
                                    cv=tss,
                                    verbose=1)
            model.fit(X_train_scaled, y_train_scaled.ravel())
            xopt = model.best_params_
            fopt = -model.best_score_
            print('Model ready!')



    elif model_name == 'bagging_extratree':
        print('ML:', model_name)
        
        if optimization_method == 'pso':
            print('hyperparameter tuning: pso')
            kwargs = {'X_train_scaled': X_train_scaled,
                      'y_train_scaled': y_train_scaled.ravel(),
                      'n_jobs': 1,
                      'return_model': False}
            # n_estimators, base_estimator__max_depth
            lb = [50,  1]
            ub = [300, 5]
            xopt, fopt = pso(bagging_extratree, lb, ub,
                            kwargs=kwargs,
                            swarmsize=N_SWARMSIZE_MINIMAL_3,
                            omega=0.5, phip=0.5, phig=0.5, 
                            maxiter=N_MAXITER_MINIMAL_3, minstep=1e-8,
                            minfunc=1e-5, debug=True,
                            processes=N_CPU)
            print('pso ready!')
            kwargs['return_model'] = True
            fopt, model = bagging_extratree(xopt, **kwargs)
            print('Model ready!')
        
        elif optimization_method == 'randomizedsearchcv':
            print('Hyperparameter tuning: RandomizedSearchCV')
            reg_model = BaggingRegressor()
            distributions = {'n_estimators': sp_randint(50, 300),
                             'n_jobs': [1],
                             'base_estimator': [ExtraTreeRegressor()],
                             'base_estimator__max_depth': sp_randint(1, 6)}
            tss = TimeSeriesSplit(n_splits=N_CV)
            model = RandomizedSearchCV(estimator=reg_model, 
                                       param_distributions=distributions,
                                       n_iter=N_ITER_MINIMAL_3,
                                       scoring='neg_mean_absolute_error',
                                       n_jobs=N_CPU,
                                       cv=tss,
                                       verbose=1)
            model.fit(X_train_scaled, y_train_scaled.ravel())
            xopt = model.best_params_
            fopt = -model.best_score_
            print('Model ready!')
    
        elif optimization_method == 'bayessearchcv':
            print('Hyperparameter tuning: BayesSearchCV')
            reg_model = BaggingRegressor()
            distributions = {'n_estimators': Integer(50, 300, prior='uniform'),
                             'n_jobs': [1],
                             'base_estimator': [ExtraTreeRegressor()],
                             'base_estimator__max_depth': Integer(1, 5, prior='uniform')}
            tss = TimeSeriesSplit(n_splits=N_CV)
            model = BayesSearchCV(estimator=reg_model, 
                                    search_spaces=distributions,
                                    n_iter=N_BAYESITER_MINIMAL_3,
                                    scoring='neg_mean_absolute_error',
                                    n_jobs=N_CPU,
                                    n_points=N_CPU // N_CV,
                                    cv=tss,
                                    verbose=1)
            model.fit(X_train_scaled, y_train_scaled.ravel())
            xopt = model.best_params_
            fopt = -model.best_score_
            print('Model ready!')






    elif model_name == 'extratreesregressor_bootstrapfalse':
        print('ML:', model_name)
        
        if optimization_method == 'pso':
            print('hyperparameter tuning: pso')
            kwargs = {'X_train_scaled': X_train_scaled,
                      'y_train_scaled': y_train_scaled.ravel(),
                      'criterion': 'mae',
                      'max_features': 'auto',
                      'bootstrap': False,
                      'n_jobs': 1,
                      'return_model': False}
            # int n_estimators, int max_depth,
            # int min_samples_split, int min_samples_leaf, float min_weight_fraction_leaf
            lb = [20,  3,  3,  3,  1e-6]
            ub = [150, 7, 30, 30, 1e-2]
            xopt, fopt = pso(extratreesregressor, lb, ub,
                            kwargs=kwargs,
                            swarmsize=N_SWARMSIZE_MINIMAL_0,
                            omega=0.5, phip=0.5, phig=0.5, 
                            maxiter=N_MAXITER_MINIMAL_0, minstep=1e-8,
                            minfunc=1e-5, debug=True,
                            processes=1)
            print('pso ready!')
            kwargs['return_model'] = True
            fopt, model = extratreesregressor(xopt, **kwargs)
            print('Model ready!')
        
        elif optimization_method == 'randomizedsearchcv':
            print('Hyperparameter tuning: RandomizedSearchCV')
            reg_model = ExtraTreesRegressor()
            distributions = {'n_estimators': sp_randint(20, 151),
                             'max_depth': sp_randint(3, 7),
                             'min_samples_split': sp_randint(3, 31),
                             'min_samples_leaf': sp_randint(3, 31),
                             'min_weight_fraction_leaf': sp_loguniform(a=1e-6, b=1e-2),
                             'criterion': ['mae'],
                             'max_features': ['auto'],
                             'n_jobs': [1],
                             'bootstrap': [False]}
            tss = TimeSeriesSplit(n_splits=N_CV)
            model = RandomizedSearchCV(estimator=reg_model, 
                                       param_distributions=distributions,
                                       n_iter=N_ITER_MINIMAL_0,
                                       scoring='neg_mean_absolute_error',
                                       n_jobs=1,
                                       cv=tss,
                                       verbose=1)
            model.fit(X_train_scaled, y_train_scaled.ravel())
            xopt = model.best_params_
            fopt = -model.best_score_
            print('Model ready!')
    
        elif optimization_method == 'bayessearchcv':
            print('Hyperparameter tuning: BayesSearchCV')
            reg_model = ExtraTreesRegressor()
            distributions = {'n_estimators': Integer(20, 150, prior='uniform'),
                             'max_depth': Integer(3, 7, prior='uniform'),
                             'min_samples_split': Integer(3, 30, prior='uniform'),
                             'min_samples_leaf': Integer(3, 30, prior='uniform'),
                             'min_weight_fraction_leaf': Real(1e-6, 1e-2, prior='log-uniform'),
                             'criterion': Categorical(['mae']),
                             'max_features': Categorical(['auto']),
                             'n_jobs': [1],
                             'bootstrap': Categorical([False])}
            tss = TimeSeriesSplit(n_splits=N_CV)
            model = BayesSearchCV(estimator=reg_model, 
                                    search_spaces=distributions,
                                    n_iter=N_BAYESITER_MINIMAL_0,
                                    scoring='neg_mean_absolute_error',
                                    n_jobs=1,
                                    n_points=1,
                                    cv=tss,
                                    verbose=1)
            model.fit(X_train_scaled, y_train_scaled.ravel())
            xopt = model.best_params_
            fopt = -model.best_score_
            print('Model ready!')




    elif model_name == 'extratreesregressor_bootstraptrue':
        print('ML:', model_name)
        
        if optimization_method == 'pso':
            print('hyperparameter tuning: pso')
            kwargs = {'X_train_scaled': X_train_scaled,
                      'y_train_scaled': y_train_scaled.ravel(),
                      'criterion': 'mae',
                      'max_features': 'auto',
                      'bootstrap': True,
                      'n_jobs': 1,
                      'return_model': False}
            # int n_estimators, int max_depth,
            # int min_samples_split, int min_samples_leaf, float min_weight_fraction_leaf
            lb = [20,  3, 3,  3,  1e-6]
            ub = [150, 7, 30, 30, 1e-2]
            xopt, fopt = pso(extratreesregressor, lb, ub,
                            kwargs=kwargs,
                            swarmsize=N_SWARMSIZE_MINIMAL_1,
                            omega=0.5, phip=0.5, phig=0.5, 
                            maxiter=N_MAXITER_MINIMAL_1, minstep=1e-8,
                            minfunc=1e-5, debug=True,
                            processes=1)
            print('pso ready!')
            kwargs['return_model'] = True
            fopt, model = extratreesregressor(xopt, **kwargs)
            print('Model ready!')
        
        elif optimization_method == 'randomizedsearchcv':
            print('Hyperparameter tuning: RandomizedSearchCV')
            reg_model = ExtraTreesRegressor()
            distributions = {'n_estimators': sp_randint(20, 151),
                             'max_depth': sp_randint(3, 7),
                             'min_samples_split': sp_randint(3, 31),
                             'min_samples_leaf': sp_randint(3, 31),
                             'min_weight_fraction_leaf': sp_loguniform(a=1e-6, b=1e-2),
                             'criterion': ['mae'],
                             'max_features': ['auto'],
                             'n_jobs': [1],
                             'bootstrap': [True]}
            tss = TimeSeriesSplit(n_splits=N_CV)
            model = RandomizedSearchCV(estimator=reg_model, 
                                       param_distributions=distributions,
                                       n_iter=N_ITER_MINIMAL_1,
                                       scoring='neg_mean_absolute_error',
                                       n_jobs=1,
                                       cv=tss,
                                       verbose=1)
            model.fit(X_train_scaled, y_train_scaled.ravel())
            xopt = model.best_params_
            fopt = -model.best_score_
            print('Model ready!')
    
        elif optimization_method == 'bayessearchcv':
            print('Hyperparameter tuning: BayesSearchCV')
            reg_model = ExtraTreesRegressor()
            distributions = {'n_estimators': Integer(20, 150, prior='uniform'),
                             'max_depth': Integer(3, 7, prior='uniform'),
                             'min_samples_split': Integer(3, 30, prior='uniform'),
                             'min_samples_leaf': Integer(3, 30, prior='uniform'),
                             'min_weight_fraction_leaf': Real(1e-6, 1e-2, prior='log-uniform'),
                             'criterion': Categorical(['mae']),
                             'max_features': Categorical(['auto']),
                             'n_jobs': [1],
                             'bootstrap': Categorical([True])}
            tss = TimeSeriesSplit(n_splits=N_CV)
            model = BayesSearchCV(estimator=reg_model, 
                                    search_spaces=distributions,
                                    n_iter=N_BAYESITER_MINIMAL_1,
                                    scoring='neg_mean_absolute_error',
                                    n_jobs=1,
                                    n_points=1,
                                    cv=tss,
                                    verbose=1)
            model.fit(X_train_scaled, y_train_scaled.ravel())
            xopt = model.best_params_
            fopt = -model.best_score_
            print('Model ready!')
            
            
            
            
    elif model_name == 'gradientboostingregressor_lad':
        print('ML:', model_name)
        
        if optimization_method == 'pso':
            print('hyperparameter tuning: pso')
            kwargs = {'X_train_scaled': X_train_scaled,
                      'y_train_scaled': y_train_scaled.ravel(),
                      'loss': 'lad',
                      'criterion': 'mae',
                      'max_features': 'auto',
                      'return_model': False}
            # learning_rate, n_estimators, subsample, min_samples_split,
            # min_samples_leaf, min_weight_fraction_leaf, max_depth
            lb = [0.01, 5,  0.5, 2,  2, 1e-6,  5]
            ub = [0.9,  20, 1.0, 31, 31, 1e-2, 10]
            xopt, fopt = pso(gradientboostingregressor, lb, ub,
                            kwargs=kwargs,
                            swarmsize=N_SWARMSIZE_MINIMAL_2, # SMALL was too much
                            omega=0.5, phip=0.5, phig=0.5, 
                            maxiter=N_MAXITER_MINIMAL_2, minstep=1e-8,
                            minfunc=1e-5, debug=True,
                            processes=N_CPU)
            print('pso ready!')
            kwargs['return_model'] = True
            fopt, model = gradientboostingregressor(xopt, **kwargs)
            print('Model ready!')
        
        elif optimization_method == 'randomizedsearchcv':
            print('Hyperparameter tuning: RandomizedSearchCV')
            reg_model = GradientBoostingRegressor()
            distributions = {'learning_rate': sp_uniform(0.01, 0.89),
                             'n_estimators': sp_randint(5, 20),
                             'subsample': sp_uniform(0.5, 0.5),
                             'min_samples_split': sp_randint(2, 31),
                             'min_samples_leaf': sp_randint(2, 31),
                             'min_weight_fraction_leaf': sp_loguniform(a=1e-6, b=1e-2),
                             'max_depth': sp_randint(5, 10),
                             'loss': ['lad'],
                             'criterion': ['mae'],
                             'max_features': ['auto']}
            tss = TimeSeriesSplit(n_splits=N_CV)
            model = RandomizedSearchCV(estimator=reg_model, 
                                       param_distributions=distributions,
                                       n_iter=N_ITER_MINIMAL_2,
                                       scoring='neg_mean_absolute_error',
                                       n_jobs=N_CPU,
                                       cv=tss,
                                       verbose=1)
            model.fit(X_train_scaled, y_train_scaled.ravel())
            xopt = model.best_params_
            fopt = -model.best_score_
            print('Model ready!')
    
        elif optimization_method == 'bayessearchcv':
            print('Hyperparameter tuning: BayesSearchCV')
            reg_model = GradientBoostingRegressor()
            distributions = {'learning_rate': Real(0.01, 0.9, prior='uniform'),
                             'n_estimators': Integer(5, 20, prior='uniform'),
                             'subsample': Real(0.5, 1.0, prior='uniform'),
                             'min_samples_split': Integer(2, 30, prior='uniform'),
                             'min_samples_leaf': Integer(2, 30, prior='uniform'),
                             'min_weight_fraction_leaf': Real(1e-6, 1e-2, prior='log-uniform'),
                             'max_depth': Integer(5, 10, prior='uniform'),
                             'loss': Categorical(['lad']),
                             'criterion': Categorical(['mae']),
                             'max_features': Categorical(['auto'])}
            tss = TimeSeriesSplit(n_splits=N_CV)
            model = BayesSearchCV(estimator=reg_model, 
                                    search_spaces=distributions,
                                    n_iter=N_BAYESITER_MINIMAL_2,
                                    scoring='neg_mean_absolute_error',
                                    n_jobs=N_CPU,
                                    n_points=N_CPU // N_CV,
                                    cv=tss,
                                    verbose=1)
            model.fit(X_train_scaled, y_train_scaled.ravel())
            xopt = model.best_params_
            fopt = -model.best_score_
            print('Model ready!')
            



    elif model_name == 'histgradientboostingregressor_lad':
        print('ML:', model_name)
        
        if optimization_method == 'pso':
            print('hyperparameter tuning: pso')
            kwargs = {'X_train_scaled': X_train_scaled,
                      'y_train_scaled': y_train_scaled.ravel(),
                      'loss': 'least_absolute_deviation',
                      'early_stopping': False,
                      'return_model': False}
            # learning_rate, max_iter, max_leaf_nodes, max_depth,
            # min_samples_leaf, l2_regularization, max_bins, 
            lb = [0.01, 10, 100, 2, 2, 1e-6, 100]
            ub = [0.9, 20, 300, 6, 30, 1e-2, 255]
            xopt, fopt = pso(histgradientboostingregressor, lb, ub,
                            kwargs=kwargs,
                            swarmsize=N_SWARMSIZE_MINIMAL_3,
                            omega=0.5, phip=0.5, phig=0.5, 
                            maxiter=N_MAXITER_MINIMAL_3, minstep=1e-8,
                            minfunc=1e-5, debug=True,
                            processes=1)
            print('pso ready!')
            kwargs['return_model'] = True
            fopt, model = histgradientboostingregressor(xopt, **kwargs)
            print('Model ready!')
        
        elif optimization_method == 'randomizedsearchcv':
            print('Hyperparameter tuning: RandomizedSearchCV')
            reg_model = HistGradientBoostingRegressor()
            distributions = {'learning_rate': sp_uniform(0.01, 0.89),
                             'max_iter': sp_randint(10, 20),
                             'max_leaf_nodes': sp_randint(100, 300),
                             'max_depth': sp_randint(5, 10),
                             'min_samples_leaf': sp_randint(2, 31),
                             'l2_regularization': sp_loguniform(a=1e-6, b=1e-2),
                             'max_bins': sp_randint(100, 255),
                             'loss': ['least_absolute_deviation'],
                             'early_stopping': [False]}
            tss = TimeSeriesSplit(n_splits=N_CV)
            model = RandomizedSearchCV(estimator=reg_model, 
                                       param_distributions=distributions,
                                       n_iter=N_ITER_MINIMAL_3,
                                       scoring='neg_mean_absolute_error',
                                       n_jobs=1,
                                       cv=tss,
                                       verbose=1)
            model.fit(X_train_scaled, y_train_scaled.ravel())
            xopt = model.best_params_
            fopt = -model.best_score_
            print('Model ready!')
    
        elif optimization_method == 'bayessearchcv':
            print('Hyperparameter tuning: BayesSearchCV')
            reg_model = HistGradientBoostingRegressor()
            distributions = {'learning_rate': Real(0.01, 0.9, prior='uniform'),
                             'max_iter': Integer(10, 20, prior='uniform'),
                             'max_leaf_nodes': Integer(100, 300, prior='uniform'),
                             'max_depth': Integer(5, 10, prior='uniform'),
                             'min_samples_leaf': Integer(2, 30, prior='uniform'),
                             'l2_regularization': Real(1e-6, 1e-2, prior='log-uniform'),
                             'max_bins': Integer(100, 255, prior='uniform'),
                             'loss': Categorical(['least_absolute_deviation']),
                             'early_stopping': Categorical([False])}
            tss = TimeSeriesSplit(n_splits=N_CV)
            model = BayesSearchCV(estimator=reg_model, 
                                    search_spaces=distributions,
                                    n_iter=N_BAYESITER_MINIMAL_3,
                                    scoring='neg_mean_absolute_error',
                                    n_jobs=1,
                                    n_points=1,
                                    cv=tss,
                                    verbose=1)
            model.fit(X_train_scaled, y_train_scaled.ravel())
            xopt = model.best_params_
            fopt = -model.best_score_
            print('Model ready!')
            









    elif model_name == 'randomforest':
        print('ML:', model_name)
        # max_features = (0, 1]
        # max_samples should be in the interval (0, 1).
        if optimization_method == 'pso':
            print('Hyperparameter tuning: PSO')
            kwargs = {'X_train_scaled': X_train_scaled,
                      'y_train_scaled': y_train_scaled.ravel(),
                      'n_jobs': 1,
                      'return_model': False}
            lb = [30,  2,  1,  1,  0.2, 0.2]
            ub = [300, 10, 20, 20, 1.0, 0.99]
            xopt, fopt = pso(randomforest, lb, ub,
                            kwargs=kwargs,
                            swarmsize=N_SWARMSIZE_MINIMAL_3,
                            omega=0.5, phip=0.5, phig=0.5, 
                            maxiter=N_MAXITER_MINIMAL_3, minstep=1e-8,
                            minfunc=1e-5, debug=True,
                            processes=N_CPU)
            print('pso ready!')
            kwargs['return_model'] = True
            fopt, model = randomforest(xopt, **kwargs)
            print('Model ready!')
        
        
        elif optimization_method == 'randomizedsearchcv':
            print('Hyperparameter tuning: RandomizedSearchCV')
            reg_model = RandomForestRegressor()
            distributions = {'n_estimators': sp_randint(30, 300),
                             'max_depth': sp_randint(2, 10),
                             'min_samples_split': sp_randint(1, 21),
                             'min_samples_leaf': sp_randint(1, 21),
                             'max_features': sp_uniform(loc=0.2, scale=0.8),
                             'max_leaf_nodes': [None],
                             'min_impurity_decrease': [0.0],
                             'n_jobs': [1],
                             'max_samples': sp_uniform(loc=0.2, scale=0.79)}
            tss = TimeSeriesSplit(n_splits=N_CV)
            model = RandomizedSearchCV(estimator=reg_model, 
                                       param_distributions=distributions,
                                       n_iter=N_ITER_MINIMAL_3,
                                       scoring='neg_mean_absolute_error',
                                       n_jobs=N_CPU,
                                       cv=tss,
                                       verbose=1)
            model.fit(X_train_scaled, y_train_scaled.ravel())
            xopt = model.best_params_
            fopt = -model.best_score_
            print('Model ready!')
            
    
        elif optimization_method == 'bayessearchcv':
            print('Hyperparameter tuning: BayesSearchCV')
            reg_model = RandomForestRegressor()
            distributions = {'n_estimators': Integer(30, 300, prior='uniform'),
                             'max_depth': Integer(2, 10, prior='uniform'),
                             'min_samples_split': Integer(1, 20, prior='uniform'),
                             'min_samples_leaf': Integer(1, 20, prior='uniform'),
                             'max_features': Real(0.2, 1.0, prior='uniform'),
                             'max_leaf_nodes': [None],
                             'min_impurity_decrease': [0.0],
                             'n_jobs': [1],
                             'max_samples': Real(0.2, 0.99, prior='uniform')}
            tss = TimeSeriesSplit(n_splits=N_CV)
            model = BayesSearchCV(estimator=reg_model, 
                                    search_spaces=distributions,
                                    n_iter=N_BAYESITER_MINIMAL_3,
                                    scoring='neg_mean_absolute_error',
                                    n_jobs=N_CPU,
                                    n_points=N_CPU // N_CV,
                                    cv=tss,
                                    verbose=1)
            model.fit(X_train_scaled, y_train_scaled.ravel())
            xopt = model.best_params_
            fopt = -model.best_score_
            print('Model ready!')
            



    elif model_name == 'lgb_gbdt':
        print('ML:', model_name)    
        if optimization_method == 'pso':
            print('Hyperparameter tuning: PSO')
            kwargs = {'X_train_scaled': X_train_scaled, 
                      'y_train_scaled': y_train_scaled.ravel(),
                      'boosting_type': 'gbdt',
                      'n_jobs': 1,
                      'return_model':False}
            # num_leaves, max_depth, min_child_samples, learning_rate, n_estimators,
            # reg_alpha, reg_lambda
            lb = [2,  4,  10, 0.01, 20,  1e-6, 1e-6]
            ub = [40, 10, 30, 0.9,  300, 1e-1, 1e1]
            xopt, fopt = pso(lgb_regressor, lb, ub,
                            kwargs=kwargs,
                            swarmsize=N_SWARMSIZE_MINIMAL_3,
                            omega=0.5, phip=0.5, phig=0.5, 
                            maxiter=N_MAXITER_MINIMAL_3, minstep=1e-8,
                            minfunc=1e-5, debug=False,
                            processes=N_CPU)
            print(model_name, 'ready!')
            kwargs['return_model'] = True
            fopt, model = lgb_regressor(xopt, **kwargs)
        
        elif optimization_method == 'randomizedsearchcv':
            print('Hyperparameter tuning: RandomizedSearchCV')
            reg_model = lgb.LGBMRegressor()
            distributions = {'num_leaves': sp_randint(2, 40),
                             'max_depth': sp_randint(2, 10),
                             'min_child_samples': sp_randint(10, 30),
                             'learning_rate': sp_uniform(loc=0.01, scale=0.99),
                             'n_estimators': sp_randint(20, 300),
                             'reg_alpha': sp_loguniform(a=1e-6, b=1e-1),
                             'reg_lambda': sp_loguniform(a=1e-6,b=1e1),
                             'boosting_type': ['gbdt'],
                             'n_jobs': [1]}
            tss = TimeSeriesSplit(n_splits=N_CV)
            model = RandomizedSearchCV(estimator=reg_model, 
                                       param_distributions=distributions,
                                       n_iter=N_ITER_MINIMAL_3,
                                       scoring='neg_mean_absolute_error',
                                       n_jobs=N_CPU,
                                       cv=tss,
                                       verbose=1)
            model.fit(X_train_scaled, y_train_scaled.ravel())
            xopt = model.best_params_
            fopt = -model.best_score_
    
        elif optimization_method == 'bayessearchcv':
            print('Hyperparameter tuning: BayesSearchCV')
            reg_model = lgb.LGBMRegressor()
            distributions = {'num_leaves': Integer(2, 40, prior='uniform'),
                             'max_depth': Integer(2, 10, prior='uniform'),
                             'min_child_samples': Integer(10, 30, prior='uniform'),
                             'learning_rate': Real(0.01, 0.99, prior='uniform'),
                             'n_estimators': Integer(20, 300, prior='uniform'),
                             'reg_alpha': Real(1e-6, 1e-1, prior='log-uniform'),
                             'reg_lambda': Real(1e-6,1e1, prior='log-uniform'),
                             'boosting_type': ['gbdt'],
                             'n_jobs': [1]}
            tss = TimeSeriesSplit(n_splits=N_CV)
            model = BayesSearchCV(estimator=reg_model, 
                                    search_spaces=distributions,
                                    n_iter=N_BAYESITER_MINIMAL_3,
                                    scoring='neg_mean_absolute_error',
                                    n_jobs=N_CPU,
                                    n_points=N_CPU // N_CV,
                                    cv=tss,
                                    verbose=1)
            model.fit(X_train_scaled, y_train_scaled.ravel())
            xopt = model.best_params_
            fopt = -model.best_score_






    elif model_name == 'lgb_goss':
        print('ML:', model_name)
        if optimization_method == 'pso':
            print('Hyperparameter tuning: PSO')
            kwargs = {'X_train_scaled': X_train_scaled, 
                      'y_train_scaled': y_train_scaled.ravel(),
                      'boosting_type': 'goss',
                      'n_jobs': 1,
                      'return_model':False}
            # num_leaves, max_depth, min_child_samples, learning_rate, n_estimators,
            # reg_alpha, reg_lambda
            lb = [2,  4,  10, 0.01, 20,  1e-6, 1e-6]
            ub = [40, 10, 30, 0.9,  300, 1e-1, 1e1]
            xopt, fopt = pso(lgb_regressor, lb, ub,
                            kwargs=kwargs,
                            swarmsize=N_SWARMSIZE_MINIMAL_3,
                            omega=0.5, phip=0.5, phig=0.5, 
                            maxiter=N_MAXITER_MINIMAL_3, minstep=1e-8,
                            minfunc=1e-5, debug=False,
                            processes=N_CPU)
            print(model_name, 'ready!')
            kwargs['return_model'] = True
            fopt, model = lgb_regressor(xopt, **kwargs)
        
        elif optimization_method == 'randomizedsearchcv':
            print('Hyperparameter tuning: RandomizedSearchCV')
            reg_model = lgb.LGBMRegressor()
            distributions = {'num_leaves': sp_randint(2, 40),
                             'max_depth': sp_randint(2, 10),
                             'min_child_samples': sp_randint(10, 30),
                             'learning_rate': sp_uniform(loc=0.01, scale=0.99),
                             'n_estimators': sp_randint(20, 300),
                             'reg_alpha': sp_loguniform(a=1e-6, b=1e-1),
                             'reg_lambda': sp_loguniform(a=1e-6,b=1e1),
                             'boosting_type': ['goss'],
                             'n_jobs': [1]}
            tss = TimeSeriesSplit(n_splits=N_CV)
            model = RandomizedSearchCV(estimator=reg_model, 
                                       param_distributions=distributions,
                                       n_iter=N_ITER_MINIMAL_3,
                                       scoring='neg_mean_absolute_error',
                                       n_jobs=N_CPU,
                                       cv=tss,
                                       verbose=1)
            model.fit(X_train_scaled, y_train_scaled.ravel())
            xopt = model.best_params_
            fopt = -model.best_score_
    
        elif optimization_method == 'bayessearchcv':
            print('Hyperparameter tuning: BayesSearchCV')
            reg_model = lgb.LGBMRegressor()
            distributions = {'num_leaves': Integer(2, 40, prior='uniform'),
                             'max_depth': Integer(2, 10, prior='uniform'),
                             'min_child_samples': Integer(10, 30, prior='uniform'),
                             'learning_rate': Real(0.01, 0.99, prior='uniform'),
                             'n_estimators': Integer(20, 300, prior='uniform'),
                             'reg_alpha': Real(1e-6, 1e-1, prior='log-uniform'),
                             'reg_lambda': Real(1e-6,1e1, prior='log-uniform'),
                             'boosting_type': ['goss'],
                             'n_jobs': [1]}
            tss = TimeSeriesSplit(n_splits=N_CV)
            model = BayesSearchCV(estimator=reg_model, 
                                    search_spaces=distributions,
                                    n_iter=N_BAYESITER_MINIMAL_3,
                                    scoring='neg_mean_absolute_error',
                                    n_jobs=N_CPU,
                                    n_points=N_CPU // N_CV,
                                    cv=tss,
                                    verbose=1)
            model.fit(X_train_scaled, y_train_scaled.ravel())
            xopt = model.best_params_
            fopt = -model.best_score_



    elif model_name == 'lgb_dart':
        print('ML:', model_name)    
        if optimization_method == 'pso':
            print('Hyperparameter tuning: PSO')
            kwargs = {'X_train_scaled': X_train_scaled, 
                      'y_train_scaled': y_train_scaled.ravel(),
                      'boosting_type': 'dart',
                      'n_jobs': 1,
                      'return_model':False}
            # num_leaves, max_depth, min_child_samples, learning_rate, n_estimators,
            # reg_alpha, reg_lambda
            lb = [2,  4,  10, 0.01, 20,  1e-6, 1e-6]
            ub = [40, 10, 30, 0.9,  300, 1e-1, 1e1]
            xopt, fopt = pso(lgb_regressor, lb, ub,
                            kwargs=kwargs,
                            swarmsize=N_SWARMSIZE_MINIMAL_3,
                            omega=0.5, phip=0.5, phig=0.5, 
                            maxiter=N_MAXITER_MINIMAL_3, minstep=1e-8,
                            minfunc=1e-5, debug=False,
                            processes=N_CPU)
            print(model_name, 'ready!')
            kwargs['return_model'] = True
            fopt, model = lgb_regressor(xopt, **kwargs)
        
        elif optimization_method == 'randomizedsearchcv':
            print('Hyperparameter tuning: RandomizedSearchCV')
            reg_model = lgb.LGBMRegressor()
            distributions = {'num_leaves': sp_randint(2, 40),
                             'max_depth': sp_randint(2, 10),
                             'min_child_samples': sp_randint(10, 30),
                             'learning_rate': sp_uniform(loc=0.01, scale=0.99),
                             'n_estimators': sp_randint(20, 300),
                             'reg_alpha': sp_loguniform(a=1e-6, b=1e-1),
                             'reg_lambda': sp_loguniform(a=1e-6,b=1e1),
                             'boosting_type': ['dart'],
                             'n_jobs': [1]}
            tss = TimeSeriesSplit(n_splits=N_CV)
            model = RandomizedSearchCV(estimator=reg_model, 
                                       param_distributions=distributions,
                                       n_iter=N_ITER_MINIMAL_3,
                                       scoring='neg_mean_absolute_error',
                                       n_jobs=N_CPU,
                                       cv=tss,
                                       verbose=1)
            model.fit(X_train_scaled, y_train_scaled.ravel())
            xopt = model.best_params_
            fopt = -model.best_score_
    
        elif optimization_method == 'bayessearchcv':
            print('Hyperparameter tuning: BayesSearchCV')
            reg_model = lgb.LGBMRegressor()
            distributions = {'num_leaves': Integer(2, 40, prior='uniform'),
                             'max_depth': Integer(2, 10, prior='uniform'),
                             'min_child_samples': Integer(10, 30, prior='uniform'),
                             'learning_rate': Real(0.01, 0.99, prior='uniform'),
                             'n_estimators': Integer(30, 200, prior='uniform'),
                             'reg_alpha': Real(1e-6, 1e-1, prior='log-uniform'),
                             'reg_lambda': Real(1e-6,1e1, prior='log-uniform'),
                             'boosting_type': ['dart'],
                             'n_jobs': [1]}
            tss = TimeSeriesSplit(n_splits=N_CV)
            model = BayesSearchCV(estimator=reg_model, 
                                    search_spaces=distributions,
                                    n_iter=N_BAYESITER_MINIMAL_3,
                                    scoring='neg_mean_absolute_error',
                                    n_jobs=N_CPU,
                                    n_points=N_CPU // N_CV,
                                    cv=tss,
                                    verbose=1)
            model.fit(X_train_scaled, y_train_scaled.ravel())
            xopt = model.best_params_
            fopt = -model.best_score_



    elif model_name == 'lgb_rf':
        print('ML:', model_name)
        if optimization_method == 'pso':
            print('Hyperparameter tuning: PSO')
            kwargs = {'X_train_scaled': X_train_scaled, 
                      'y_train_scaled': y_train_scaled.ravel(),
                      'boosting_type': 'rf',
                      'n_jobs': 1,
                      'return_model':False}
            # num_leaves, max_depth, min_child_samples, learning_rate, n_estimators,
            # subsample, subsample_freq, reg_alpha, reg_lambda
            lb = [2,  2, 10, 0.01, 20,  0.5, 1, 1e-6, 1e-6]
            ub = [31, 6, 30, 0.9,  300, 0.9, 3, 1e-1, 1e1]
            xopt, fopt = pso(lgb_regressor, lb, ub,
                            kwargs=kwargs,
                            swarmsize=N_SWARMSIZE_MINIMAL_3,
                            omega=0.5, phip=0.5, phig=0.5, 
                            maxiter=N_MAXITER_MINIMAL_3, minstep=1e-8,
                            minfunc=1e-5, debug=False,
                            processes=N_CPU)
            print(model_name, 'ready!')
            kwargs['return_model'] = True
            fopt, model = lgb_regressor(xopt, **kwargs)
        
        elif optimization_method == 'randomizedsearchcv':
            print('Hyperparameter tuning: RandomizedSearchCV')
            reg_model = lgb.LGBMRegressor()
            distributions = {'num_leaves': sp_randint(2, 31),
                             'max_depth': sp_randint(2, 6),
                             'min_child_samples': sp_randint(10, 30),
                             'learning_rate': sp_uniform(loc=0.01, scale=0.99),
                             'n_estimators': sp_randint(20, 300),
                             'subsample': sp_uniform(loc=0.5, scale=0.4),
                             'subsample_freq': sp_randint(1, 4),
                             'reg_alpha': sp_loguniform(a=1e-6, b=1e-1),
                             'reg_lambda': sp_loguniform(a=1e-6,b=1e1),
                             'boosting_type': ['rf'],
                             'n_jobs': [1]}
            tss = TimeSeriesSplit(n_splits=N_CV)
            model = RandomizedSearchCV(estimator=reg_model, 
                                       param_distributions=distributions,
                                       n_iter=N_ITER_MINIMAL_3,
                                       scoring='neg_mean_absolute_error',
                                       n_jobs=N_CPU,
                                       cv=tss,
                                       verbose=1)
            model.fit(X_train_scaled, y_train_scaled.ravel())
            xopt = model.best_params_
            fopt = -model.best_score_
    
        elif optimization_method == 'bayessearchcv':
            print('Hyperparameter tuning: BayesSearchCV')
            reg_model = lgb.LGBMRegressor()
            distributions = {'num_leaves': Integer(2, 31, prior='uniform'),
                             'max_depth': Integer(2, 6, prior='uniform'),
                             'min_child_samples': Integer(10, 30, prior='uniform'),
                             'learning_rate': Real(0.01, 0.99, prior='uniform'),
                             'n_estimators': Integer(20, 300, prior='uniform'),
                             'subsample': Real(0.5, 0.9, prior='uniform'),
                             'subsample_freq': Integer(1, 3, prior='uniform'),
                             'reg_alpha': Real(1e-6, 1e-1, prior='log-uniform'),
                             'reg_lambda': Real(1e-6,1e1, prior='log-uniform'),
                             'boosting_type': ['rf'],
                             'n_jobs': [1]}
            tss = TimeSeriesSplit(n_splits=N_CV)
            model = BayesSearchCV(estimator=reg_model, 
                                    search_spaces=distributions,
                                    n_iter=N_BAYESITER_MINIMAL_3,
                                    scoring='neg_mean_absolute_error',
                                    n_jobs=N_CPU,
                                    n_points=N_CPU // N_CV,
                                    cv=tss,
                                    verbose=1)
            model.fit(X_train_scaled, y_train_scaled.ravel())
            xopt = model.best_params_
            fopt = -model.best_score_





    elif model_name == 'xgb_gbtree':
        print('ML:', model_name)
        
        try:
            print(xgb.__version__, flush=True)
        except:
            print("Could not load xgb!", flush=True)
        
        if optimization_method == 'pso':
            print('Hyperparameter tuning: PSO')
            kwargs = {'X_train_scaled': X_train_scaled, 
                      'y_train_scaled': y_train_scaled,
                      'booster': 'gbtree',
                      'objective': 'reg:squarederror',
                      'n_jobs': 1,
                      'return_model':False}
            #n_estimators, max_depth, learning_rate, gamma, subsample,
            # reg_alpha, reg_lambda
            lb = [30,  2,  0.1, 1e-6, 0.2, 1e-6, 1e-6]
            ub = [100, 5, 1.0,  1e1,  1.0, 1e-1,  1e1]
            xopt, fopt = pso(xgb_gbtree, lb, ub,
                             kwargs=kwargs,
                             swarmsize=N_SWARMSIZE_MINIMAL_3,
                             omega=0.5, phip=0.5, phig=0.5, 
                             maxiter=N_MAXITER_MINIMAL_3, minstep=1e-8,
                             minfunc=1e-5, debug=False,
                             processes=N_CPU)
            kwargs['return_model'] = True
            fopt, model = xgb_gbtree(xopt, **kwargs)
        
        elif optimization_method == 'randomizedsearchcv':
            print('Hyperparameter tuning: RandomizedSearchCV')
            reg_model = xgb.XGBRegressor()
            distributions = {'booster':['gbtree'],
                             'n_jobs': [1],
                             'n_estimators': sp_randint(30, 100),
                             'max_depth': sp_randint(2, 10),
                             'learning_rate': sp_uniform(loc=0.01, scale=0.99),
                             'gamma': sp_loguniform(a=1e-6, b=1e1),
                             'subsample': sp_uniform(loc=0.2, scale=0.8),
                             'reg_alpha': sp_loguniform(a=1e-6, b=1e-1),
                             'reg_lambda': sp_loguniform(a=1e-6,b=1e1)}
            tss = TimeSeriesSplit(n_splits=N_CV)
            model = RandomizedSearchCV(estimator=reg_model, 
                                       param_distributions=distributions,
                                       n_iter=N_ITER_MINIMAL_3,
                                       scoring='neg_mean_absolute_error',
                                       n_jobs=N_CPU,
                                       cv=tss,
                                       verbose=1)
            model.fit(X_train_scaled, y_train_scaled)
            xopt = model.best_params_
            fopt = -model.best_score_
    
        elif optimization_method == 'bayessearchcv':
            print('Hyperparameter tuning: BayesSearchCV')
            reg_model = xgb.XGBRegressor()
            distributions = {'booster':['gbtree'],
                             'n_jobs': [1],
                             'n_estimators': Integer(30, 100, prior='uniform'),
                             'max_depth': Integer(2, 10, prior='uniform'),
                             'learning_rate': Real(0.01, 0.99, prior='uniform'),
                             'gamma': Real(1e-6, 1e1, prior='log-uniform'),
                             'subsample': Real(0.2, 1.0, prior='uniform'),
                             'reg_alpha': Real(1e-6, 1e-1, prior='log-uniform'),
                             'reg_lambda': Real(1e-6,1e1, prior='log-uniform')}
            tss = TimeSeriesSplit(n_splits=N_CV)
            model = BayesSearchCV(estimator=reg_model, 
                                  search_spaces=distributions,
                                  n_iter=N_BAYESITER_MINIMAL_3,
                                  scoring='neg_mean_absolute_error',
                                  n_jobs=N_CPU,
                                  n_points=N_CPU // N_CV,
                                  cv=tss,
                                  verbose=1)
            model.fit(X_train_scaled, y_train_scaled)
            xopt = model.best_params_
            fopt = -model.best_score_





    
    elif model_name == 'xgb_dart':
        print('ML:', model_name)
        if optimization_method == 'pso':
            print('Hyperparameter tuning: PSO')
            kwargs = {'X_train_scaled': X_train_scaled, 
                      'y_train_scaled': y_train_scaled,
                      'booster': 'dart',
                      'objective': 'reg:squarederror',
                      'n_jobs': 1,
                      'return_model':False}
            #n_estimators, max_depth, learning_rate, gamma, subsample,
            # reg_alpha, reg_lambda
            lb = [30,  2,  0.01, 1e-6, 0.2, 1e-6, 1e-6]
            ub = [100, 10, 1.0,  1e1,  1.0, 1e-1,  1e1]
            xopt, fopt = pso(xgb_gbtree, lb, ub,
                            kwargs=kwargs,
                            swarmsize=N_SWARMSIZE_MINIMAL_3,
                            omega=0.5, phip=0.5, phig=0.5, 
                            maxiter=N_MAXITER_MINIMAL_3, minstep=1e-8,
                            minfunc=1e-5, debug=False,
                            processes=N_CPU)
            kwargs['return_model'] = True
            fopt, model = xgb_gbtree(xopt, **kwargs)
        
        elif optimization_method == 'randomizedsearchcv':
            print('Hyperparameter tuning: RandomizedSearchCV')
            reg_model = xgb.XGBRegressor()
            distributions = {'booster':['dart'],
                             'n_jobs': [1],
                             'n_estimators': sp_randint(30, 100),
                             'max_depth': sp_randint(2, 10),
                             'learning_rate': sp_uniform(loc=0.01, scale=0.99),
                             'gamma': sp_loguniform(a=1e-6, b=1e1),
                             'subsample': sp_uniform(loc=0.2, scale=0.8),
                             'reg_alpha': sp_loguniform(a=1e-6, b=1e-1),
                             'reg_lambda': sp_loguniform(a=1e-6, b=1e1)}
            tss = TimeSeriesSplit(n_splits=N_CV)
            model = RandomizedSearchCV(estimator=reg_model, 
                                       param_distributions=distributions,
                                       n_iter=N_ITER_MINIMAL_3,
                                       scoring='neg_mean_absolute_error',
                                       n_jobs=N_CPU,
                                       cv=tss,
                                       verbose=1)
            model.fit(X_train_scaled, y_train_scaled)
            xopt = model.best_params_
            fopt = -model.best_score_
    
        elif optimization_method == 'bayessearchcv':
            print('Hyperparameter tuning: BayesSearchCV')
            reg_model = xgb.XGBRegressor()
            distributions = {'booster':['dart'],
                             'n_jobs': [1],
                             'n_estimators': Integer(30, 100, prior='uniform'),
                             'max_depth': Integer(2, 10, prior='uniform'),
                             'learning_rate': Real(0.01, 0.99, prior='uniform'),
                             'gamma': Real(1e-6, 1e1, prior='log-uniform'),
                             'subsample': Real(0.2, 1.0, prior='uniform'),
                             'reg_alpha': Real(1e-6, 1e-1, prior='log-uniform'),
                             'reg_lambda': Real(1e-6, 1e1, prior='log-uniform')}
            tss = TimeSeriesSplit(n_splits=N_CV)
            model = BayesSearchCV(estimator=reg_model, 
                                    search_spaces=distributions,
                                    n_iter=N_BAYESITER_MINIMAL_3,
                                    scoring='neg_mean_absolute_error',
                                    n_jobs=N_CPU,
                                    n_points=N_CPU // N_CV,
                                    cv=tss,
                                    verbose=1)
            model.fit(X_train_scaled, y_train_scaled)
            xopt = model.best_params_
            fopt = -model.best_score_
            



    
    
    else:
        print('Unknown model_name in fit_model!')
        
    # Return
    return(xopt, fopt, model)


        
# Everything here has n_jobs = 1
def dummyregressor(kwargs):
    # This is not optimized, because only default hyperparameters are used
    X_train_scaled = kwargs['X_train_scaled']
    y_train_scaled = kwargs['y_train_scaled']
    model = DummyRegressor()
    tss = TimeSeriesSplit(n_splits=N_CV)
    scores = cross_val_score(model,
                             X_train_scaled,
                             y_train_scaled,
                             cv=tss,
                             scoring='neg_mean_absolute_error',
                             n_jobs=1)
    score = -scores.mean()
    if not kwargs['return_model']:
        return(score)
    else:
        model.fit(X_train_scaled, y_train_scaled)
        return(score, model)


def expfunc(kwargs):
    X_train_scaled = kwargs['X_train_scaled']
    y_train_scaled = kwargs['y_train_scaled']
    model = myClasses.ExpFuncRegressor()
    tss = TimeSeriesSplit(n_splits=N_CV)
    scores = cross_val_score(model, 
                             X_train_scaled,
                             y_train_scaled,
                             cv=tss,
                             scoring='neg_mean_absolute_error',
                             n_jobs=1)
    score = -scores.mean()
    model.fit(X_train_scaled, y_train_scaled)
    xopt = model.get_params()
    return(xopt, score, model)


def piecewise(kwargs):
    X_train_scaled = kwargs['X_train_scaled']
    y_train_scaled = kwargs['y_train_scaled']
    model = myClasses.PiecewiseRegressor()
    tss = TimeSeriesSplit(n_splits=N_CV)
    scores = cross_val_score(model, 
                             X_train_scaled,
                             y_train_scaled,
                             cv=tss,
                             scoring='neg_mean_absolute_error',
                             n_jobs=1)
    score = -scores.mean()
    model.fit(X_train_scaled, y_train_scaled)
    xopt = model.get_params()
    return(xopt, score, model)



def linearregression(kwargs):
    X_train_scaled = kwargs['X_train_scaled']
    y_train_scaled = kwargs['y_train_scaled']
    model = LinearRegression(n_jobs=1)
    
    tss = TimeSeriesSplit(n_splits=N_CV)
    scores = cross_val_score(model,
                             X_train_scaled,
                             y_train_scaled,
                             cv=tss,
                             scoring='neg_mean_absolute_error',
                             n_jobs=1)
    score = -scores.mean()
    if not kwargs['return_model']:
        return(score)
    else:
        model.fit(X_train_scaled, y_train_scaled)
        return(score, model)


def ridge(x, **kwargs):
    # x is linear in PSO, but log-linear in randomizedsearchcv
    # and bayessearchcv
    X_train_scaled = kwargs['X_train_scaled']
    y_train_scaled = kwargs['y_train_scaled']
    model = Ridge(alpha=x)
    tss = TimeSeriesSplit(n_splits=N_CV)
    scores = cross_val_score(model,
                             X_train_scaled,
                             y_train_scaled,
                             cv=tss,
                             scoring='neg_mean_absolute_error',
                             n_jobs=1)
    score = -scores.mean()
    if not kwargs['return_model']:
        return(score)
    else:
        model.fit(X_train_scaled, y_train_scaled)
        return(score, model)


def lasso(x, **kwargs):
    # x is linear in PSO, but log-linear in randomizedsearchcv
    # and bayessearchcv
    X_train_scaled = kwargs['X_train_scaled']
    y_train_scaled = kwargs['y_train_scaled']
    model = Lasso(alpha=x)
    tss = TimeSeriesSplit(n_splits=N_CV)
    scores = cross_val_score(model,
                             X_train_scaled,
                             y_train_scaled,
                             cv=tss,
                             scoring='neg_mean_absolute_error',
                             n_jobs=1)
    score = -scores.mean()
    if not kwargs['return_model']:
        return(score)
    else:
        model.fit(X_train_scaled, y_train_scaled)
        return(score, model)


def elasticnet(x, **kwargs):
    X_train_scaled = kwargs['X_train_scaled']
    y_train_scaled = kwargs['y_train_scaled']
    model = ElasticNet(alpha=x[0], l1_ratio=x[1])
    tss = TimeSeriesSplit(n_splits=N_CV)
    scores = cross_val_score(model,
                             X_train_scaled,
                             y_train_scaled,
                             cv=tss,
                             scoring='neg_mean_absolute_error',
                             n_jobs=1)
    score = -scores.mean()
    if not kwargs['return_model']:
        return(score)
    else:
        model.fit(X_train_scaled, y_train_scaled)
        return(score, model)


def lars(x, **kwargs):
    X_train_scaled = kwargs['X_train_scaled']
    y_train_scaled = kwargs['y_train_scaled']
    model = Lars(n_nonzero_coefs=int(x))
    tss = TimeSeriesSplit(n_splits=N_CV)
    scores = cross_val_score(model,
                             X_train_scaled,
                             y_train_scaled,
                             cv=tss,
                             scoring='neg_mean_absolute_error',
                             n_jobs=1)
    score = -scores.mean()
    if not kwargs['return_model']:
        return(score)
    else:
        model.fit(X_train_scaled, y_train_scaled)
        return(score, model)

def lassolars(x, **kwargs):
    # x is linear in PSO, but log-linear in randomizedsearchcv
    # and bayessearchcv
    X_train_scaled = kwargs['X_train_scaled']
    y_train_scaled = kwargs['y_train_scaled']
    model = LassoLars(alpha=x)
    tss = TimeSeriesSplit(n_splits=N_CV)
    scores = cross_val_score(model,
                             X_train_scaled,
                             y_train_scaled,
                             cv=tss,
                             scoring='neg_mean_absolute_error',
                             n_jobs=1)
    score = -scores.mean()
    if not kwargs['return_model']:
        return(score)
    else:
        model.fit(X_train_scaled, y_train_scaled)
        return(score, model)


def huberregressor(x, **kwargs):
    X_train_scaled = kwargs['X_train_scaled']
    y_train_scaled = kwargs['y_train_scaled']
    max_iter = kwargs['max_iter']
    epsilon = x[0]
    alpha = x[1]
    model = HuberRegressor(epsilon=epsilon,
                           max_iter=max_iter,
                           alpha=alpha)
    tss = TimeSeriesSplit(n_splits=N_CV)
    scores = cross_val_score(model,
                             X_train_scaled,
                             y_train_scaled,
                             cv=tss,
                             scoring='neg_mean_absolute_error',
                             n_jobs=1)
    score = -scores.mean()
    if not kwargs['return_model']:
        return(score)
    else:
        model.fit(X_train_scaled, y_train_scaled)
        return(score, model)
    

def ransacregressor(x, **kwargs):
    min_samples = x[0]
    residual_threshold = x[1]
    max_trials = int(np.round(x[2]))
    stop_probability = x[3]
    
    X_train_scaled = kwargs['X_train_scaled']
    y_train_scaled = kwargs['y_train_scaled']   
    model = RANSACRegressor(min_samples=min_samples,
                            residual_threshold=residual_threshold,
                            max_trials=max_trials,
                            stop_probability=stop_probability)
    tss = TimeSeriesSplit(n_splits=N_CV)
    scores = cross_val_score(model,
                             X_train_scaled,
                             y_train_scaled,
                             cv=tss,
                             scoring='neg_mean_absolute_error',
                             n_jobs=1)
    score = -scores.mean()
    if not kwargs['return_model']:
        return(score)
    else:
        model.fit(X_train_scaled, y_train_scaled)
        return(score, model)
    

def theilsenregressor(x, **kwargs):
    max_subpopulation = int(np.round(x[0]))
    n_subsamples = int(np.round(x[1]))
    n_jobs = kwargs['n_jobs']
    X_train_scaled = kwargs['X_train_scaled']
    y_train_scaled = kwargs['y_train_scaled']
    
    model = TheilSenRegressor(max_subpopulation=max_subpopulation,
                              n_subsamples=n_subsamples,
                              n_jobs=n_jobs)
    tss = TimeSeriesSplit(n_splits=N_CV)
    scores = cross_val_score(model,
                             X_train_scaled,
                             y_train_scaled,
                             cv=tss,
                             scoring='neg_mean_absolute_error',
                             n_jobs=1)
    score = -scores.mean()
    if not kwargs['return_model']:
        return(score)
    else:
        model.fit(X_train_scaled, y_train_scaled)
        return(score, model)


def kernelridge(x, **kwargs):
    print('x:', x)
    
    if kwargs['kernel'] == 'cosine':
        alpha = x[0]
        X_train_scaled = kwargs['X_train_scaled']
        y_train_scaled = kwargs['y_train_scaled']
        kernel = kwargs['kernel']
        model = KernelRidge(alpha=alpha)
    
    else:
        alpha = x[0]
        gamma = x[1]
        degree = int(np.round(x[2]))
        coef0 = int(np.round(x[3]))
        X_train_scaled = kwargs['X_train_scaled']
        y_train_scaled = kwargs['y_train_scaled']
        kernel = kwargs['kernel']
        model = KernelRidge(alpha=alpha, kernel=kernel, gamma=gamma,
                            degree=degree, coef0=coef0)
    
    tss = TimeSeriesSplit(n_splits=N_CV)
    scores = cross_val_score(model,
                             X_train_scaled,
                             y_train_scaled,
                             cv=tss,
                             scoring='neg_mean_absolute_error',
                             n_jobs=1)
    score = -scores.mean()
    if not kwargs['return_model']:
        return(score)
    else:
        model.fit(X_train_scaled, y_train_scaled)
        return(score, model)

    


def linearsvr(x, **kwargs):
    print('x:', x)
    epsilon = x[0]
    C = x[1]
    X_train_scaled = kwargs['X_train_scaled']
    y_train_scaled = kwargs['y_train_scaled']
    loss = kwargs['loss']
    dual = kwargs['dual']
    max_iter = kwargs['max_iter']
    
    model = LinearSVR(epsilon=epsilon,
                        C=C, loss=loss, dual=dual,
                        max_iter=max_iter)
    tss = TimeSeriesSplit(n_splits=N_CV)
    scores = cross_val_score(model,
                             X_train_scaled,
                             y_train_scaled,
                             cv=tss,
                             scoring='neg_mean_absolute_error',
                             n_jobs=1)
    score = -scores.mean()
    if not kwargs['return_model']:
        return(score)
    else:
        model.fit(X_train_scaled, y_train_scaled)
        return(score, model)



def nusvr(x, **kwargs):
    print('x:', x)
    nu = x[0]
    C = 10**(x[1])
    degree = int(np.round(x[2]))
    gamma = x[3]
    coef0 = x[4]
    X_train_scaled = kwargs['X_train_scaled']
    y_train_scaled = kwargs['y_train_scaled']
    kernel = kwargs['kernel']
    shrinking = kwargs['shrinking']
    cache_size = kwargs['cache_size']
    max_iter = kwargs['max_iter']
    model = NuSVR(nu=nu, C=C, degree=degree, gamma=gamma, coef0=coef0,
                  kernel=kernel, shrinking=shrinking,
                  cache_size=cache_size,
                  max_iter=max_iter)
    tss = TimeSeriesSplit(n_splits=N_CV)
    scores = cross_val_score(model,
                             X_train_scaled,
                             y_train_scaled,
                             cv=tss,
                             scoring='neg_mean_absolute_error',
                             n_jobs=1)
    score = -scores.mean()
    if not kwargs['return_model']:
        return(score)
    else:
        model.fit(X_train_scaled, y_train_scaled)
        return(score, model)


def svr(x, **kwargs):
    print('x:', x, flush=True)
    epsilon = x[0]
    C = 10**(x[1])
    degree = int(np.round(x[2]))
    gamma = x[3]
    coef0 = x[4]
    X_train_scaled = kwargs['X_train_scaled']
    y_train_scaled = kwargs['y_train_scaled']
    kernel = kwargs['kernel']
    shrinking = kwargs['shrinking']
    cache_size = kwargs['cache_size']
    model = SVR(kernel=kernel, degree=degree, gamma=gamma, coef0=coef0,
                C=C, epsilon=epsilon, shrinking=shrinking,
                cache_size=cache_size)
    tss = TimeSeriesSplit(n_splits=N_CV)
    scores = cross_val_score(model,
                             X_train_scaled,
                             y_train_scaled,
                             cv=tss,
                             scoring='neg_mean_absolute_error',
                             n_jobs=1)
    score = -scores.mean()
    if not kwargs['return_model']:
        return(score)
    else:
        model.fit(X_train_scaled, y_train_scaled)
        return(score, model)


def kneighborsregressor(x, **kwargs):
    print('x:', x)
    n_neighbors = int(np.round(x[0]))
    leaf_size = int(np.round(x[1]))
    p = int(np.round(x[2]))
    X_train_scaled = kwargs['X_train_scaled']
    y_train_scaled = kwargs['y_train_scaled']
    weights = kwargs['weights']
    n_jobs = kwargs['n_jobs']
    model = KNeighborsRegressor(n_neighbors=n_neighbors,
                                leaf_size=leaf_size, p=p,
                                weights=weights,
                                n_jobs=n_jobs)
    tss = TimeSeriesSplit(n_splits=N_CV)
    scores = cross_val_score(model,
                             X_train_scaled,
                             y_train_scaled,
                             cv=tss,
                             scoring='neg_mean_absolute_error',
                             n_jobs=1)
    score = -scores.mean()
    if not kwargs['return_model']:
        return(score)
    else:
        model.fit(X_train_scaled, y_train_scaled)
        return(score, model)





def decisiontreeregressor(x, **kwargs):
    print('x:', x)
    min_samples_leaf = int(np.round(x[0]))
    min_samples_split = int(np.round(x[1]))
    min_weight_fraction_leaf = x[2]
    X_train_scaled = kwargs['X_train_scaled']
    y_train_scaled = kwargs['y_train_scaled']
    
    model = DecisionTreeRegressor(criterion=kwargs['criterion'],
                                  splitter=kwargs['splitter'],
                                  max_depth=kwargs['max_depth'],
                                  min_samples_split=min_samples_split,
                                  min_samples_leaf=min_samples_leaf,
                                  min_weight_fraction_leaf=min_weight_fraction_leaf,
                                  max_features=kwargs['max_features'])
    tss = TimeSeriesSplit(n_splits=N_CV)
    scores = cross_val_score(model,
                             X_train_scaled,
                             y_train_scaled,
                             cv=tss,
                             n_jobs=1,
                             scoring='neg_mean_absolute_error')
    score = -scores.mean()
    if not kwargs['return_model']:
        return(score)
    else:
        model.fit(X_train_scaled, y_train_scaled)
        return(score, model)




def extratreeregressor(x, **kwargs):
    print('x:', x)
    min_samples_leaf = int(np.round(x[0]))
    min_samples_split = int(np.round(x[1]))
    min_weight_fraction_leaf = x[2]
    X_train_scaled = kwargs['X_train_scaled']
    y_train_scaled = kwargs['y_train_scaled']
    
    model = ExtraTreeRegressor(criterion=kwargs['criterion'],
                                  splitter=kwargs['splitter'],
                                  max_depth=kwargs['max_depth'],
                                  min_samples_split=min_samples_split,
                                  min_samples_leaf=min_samples_leaf,
                                  min_weight_fraction_leaf=min_weight_fraction_leaf,
                                  max_features=kwargs['max_features'])
    tss = TimeSeriesSplit(n_splits=N_CV)
    scores = cross_val_score(model,
                             X_train_scaled,
                             y_train_scaled,
                             cv=tss,
                             n_jobs=1,
                             scoring='neg_mean_absolute_error')
    score = -scores.mean()
    if not kwargs['return_model']:
        return(score)
    else:
        model.fit(X_train_scaled, y_train_scaled)
        return(score, model)




def adaboost_decisiontree(x, **kwargs):
    print('x:', x)
    n_estimators = int(np.round(x[0]))
    learning_rate = x[1]
    base_estimator_max_depth=int(np.round(x[2]))    
    X_train_scaled = kwargs['X_train_scaled']
    y_train_scaled = kwargs['y_train_scaled']
    
    base_estimator = DecisionTreeRegressor(max_depth=base_estimator_max_depth)
    model = AdaBoostRegressor(base_estimator=base_estimator,
                              n_estimators=n_estimators,
                              learning_rate=learning_rate)
    tss = TimeSeriesSplit(n_splits=N_CV)
    scores = cross_val_score(model,
                             X_train_scaled,
                             y_train_scaled,
                             cv=tss,
                             n_jobs=1,
                             scoring='neg_mean_absolute_error')
    score = -scores.mean()
    if not kwargs['return_model']:
        return(score)
    else:
        model.fit(X_train_scaled, y_train_scaled)
        return(score, model)




def adaboost_extratree(x, **kwargs):
    print('x:', x)
    n_estimators = int(np.round(x[0]))
    learning_rate = x[1]
    base_estimator_max_depth=int(np.round(x[2]))
    X_train_scaled = kwargs['X_train_scaled']
    y_train_scaled = kwargs['y_train_scaled']
    
    base_estimator = ExtraTreeRegressor(max_depth=base_estimator_max_depth)
    model = AdaBoostRegressor(base_estimator=base_estimator,
                              n_estimators=n_estimators,
                              learning_rate=learning_rate)
    tss = TimeSeriesSplit(n_splits=N_CV)
    scores = cross_val_score(model,
                             X_train_scaled,
                             y_train_scaled,
                             cv=tss,
                             n_jobs=1,
                             scoring='neg_mean_absolute_error')
    score = -scores.mean()
    if not kwargs['return_model']:
        return(score)
    else:
        model.fit(X_train_scaled, y_train_scaled)
        return(score, model)



def bagging_decisiontree(x, **kwargs):
    print('x:', x)
    n_estimators = int(np.round(x[0]))
    base_estimator_max_depth=int(np.round(x[1]))
    X_train_scaled = kwargs['X_train_scaled']
    y_train_scaled = kwargs['y_train_scaled']
    n_jobs = kwargs['n_jobs']
    base_estimator = DecisionTreeRegressor(max_depth=base_estimator_max_depth)
    model = BaggingRegressor(base_estimator=base_estimator,
                             n_estimators=n_estimators,
                             n_jobs=n_jobs)
    tss = TimeSeriesSplit(n_splits=N_CV)
    scores = cross_val_score(model,
                             X_train_scaled,
                             y_train_scaled,
                             cv=tss,
                             n_jobs=1,
                             scoring='neg_mean_absolute_error')
    score = -scores.mean()
    if not kwargs['return_model']:
        return(score)
    else:
        model.fit(X_train_scaled, y_train_scaled)
        return(score, model)


def bagging_extratree(x, **kwargs):
    print('x:', x)
    n_estimators = int(np.round(x[0]))
    base_estimator_max_depth=int(np.round(x[1]))
    X_train_scaled = kwargs['X_train_scaled']
    y_train_scaled = kwargs['y_train_scaled']
    n_jobs = kwargs['n_jobs']
    base_estimator = ExtraTreeRegressor(max_depth=base_estimator_max_depth)
    model = BaggingRegressor(base_estimator=base_estimator,
                             n_estimators=n_estimators,
                             n_jobs=n_jobs)
    tss = TimeSeriesSplit(n_splits=N_CV)
    scores = cross_val_score(model,
                             X_train_scaled,
                             y_train_scaled,
                             cv=tss,
                             n_jobs=1,
                             scoring='neg_mean_absolute_error')
    score = -scores.mean()
    if not kwargs['return_model']:
        return(score)
    else:
        model.fit(X_train_scaled, y_train_scaled)
        return(score, model)







def extratreesregressor(x, **kwargs):
    print('x:', x)
    n_estimators = int(np.round(x[0]))
    max_depth = int(np.round(x[1]))
    min_samples_split = int(np.round(x[2]))
    min_samples_leaf = int(np.round(x[3]))
    min_weight_fraction_leaf = x[4]
    X_train_scaled = kwargs['X_train_scaled']
    y_train_scaled = kwargs['y_train_scaled']
    n_jobs = kwargs['n_jobs']
    model = ExtraTreesRegressor(criterion=kwargs['criterion'],
                                max_features=kwargs['max_features'],
                                bootstrap=kwargs['bootstrap'],
                                n_estimators=n_estimators,
                                max_depth=max_depth,
                                min_samples_split=min_samples_split,
                                min_samples_leaf=min_samples_leaf,
                                min_weight_fraction_leaf=min_weight_fraction_leaf,
                                n_jobs=n_jobs,
                                verbose=2)
    tss = TimeSeriesSplit(n_splits=N_CV)
    scores = cross_val_score(model,
                             X_train_scaled,
                             y_train_scaled,
                             cv=tss,
                             n_jobs=n_jobs,
                             scoring='neg_mean_absolute_error',
                             verbose=2)
    score = -scores.mean()
    if not kwargs['return_model']:
        return(score)
    else:
        model.fit(X_train_scaled, y_train_scaled)
        return(score, model)
    
    

def gradientboostingregressor(x, **kwargs):
    print('x:', x)
    
    learning_rate = x[0]
    n_estimators = int(np.round(x[1]))
    subsample = x[2]
    min_samples_split = int(np.round(x[3]))
    min_samples_leaf = int(np.round(x[4]))
    min_weight_fraction_leaf = x[5]
    max_depth = int(np.round(x[6]))
    X_train_scaled = kwargs['X_train_scaled']
    y_train_scaled = kwargs['y_train_scaled']
    model = GradientBoostingRegressor(loss=kwargs['loss'],
                                      learning_rate=learning_rate,
                                      n_estimators=n_estimators,
                                      subsample=subsample,
                                      criterion=kwargs['criterion'],
                                      min_samples_split=min_samples_split,
                                      min_samples_leaf=min_samples_leaf,
                                      min_weight_fraction_leaf=min_weight_fraction_leaf,
                                      max_depth=max_depth)
    tss = TimeSeriesSplit(n_splits=N_CV)
    scores = cross_val_score(model,
                             X_train_scaled,
                             y_train_scaled,
                             cv=tss,
                             n_jobs=1,
                             scoring='neg_mean_absolute_error')
    score = -scores.mean()
    if not kwargs['return_model']:
        return(score)
    else:
        model.fit(X_train_scaled, y_train_scaled)
        return(score, model)


def histgradientboostingregressor(x, **kwargs):
    print('x:', x)
    learning_rate = x[0]
    max_iter = int(np.round(x[1]))
    max_leaf_nodes = int(np.round(x[2]))
    max_depth = int(np.round(x[3]))
    min_samples_leaf = int(np.round(x[4]))
    l2_regularization = x[5]
    max_bins = int(np.round(x[6]))
    X_train_scaled = kwargs['X_train_scaled']
    y_train_scaled = kwargs['y_train_scaled']
    model = HistGradientBoostingRegressor(learning_rate=learning_rate,
                                          max_iter=max_iter,
                                          max_leaf_nodes=max_leaf_nodes,
                                          max_depth=max_depth,
                                          min_samples_leaf=min_samples_leaf,
                                          l2_regularization=l2_regularization,
                                          max_bins=max_bins,
                                          loss=kwargs['loss'],
                                          early_stopping=kwargs['early_stopping'])
    tss = TimeSeriesSplit(n_splits=N_CV)
    scores = cross_val_score(model,
                             X_train_scaled,
                             y_train_scaled,
                             cv=tss,
                             n_jobs=1,
                             scoring='neg_mean_absolute_error')
    score = -scores.mean()
    if not kwargs['return_model']:
        return(score)
    else:
        model.fit(X_train_scaled, y_train_scaled)
        return(score, model)





def randomforest(x, **kwargs):
    n_estimators = int(np.round(x[0]))
    max_depth = int(np.round(x[1]))
    min_samples_split = int(np.round(x[2]))
    min_samples_leaf = int(np.round(x[3]))
    max_features = x[4]
    max_samples = x[5]
    
    X_train_scaled = kwargs['X_train_scaled']
    y_train_scaled = kwargs['y_train_scaled']
    n_jobs = kwargs['n_jobs']
    model = RandomForestRegressor(n_estimators=n_estimators,
                                  max_depth=max_depth,
                                  min_samples_split=min_samples_split,
                                  min_samples_leaf=min_samples_leaf,
                                  max_features=max_features,
                                  max_leaf_nodes=None,
                                  min_impurity_decrease=0.0,
                                  max_samples=max_samples,
                                  n_jobs=n_jobs)
    tss = TimeSeriesSplit(n_splits=N_CV)
    scores = cross_val_score(model,
                             X_train_scaled,
                             y_train_scaled,
                             cv=tss,
                             n_jobs=1,
                             scoring='neg_mean_absolute_error')
    score = -scores.mean()
    if not kwargs['return_model']:
        return(score)
    else:
        model.fit(X_train_scaled, y_train_scaled)
        return(score, model)
    



def lgb_regressor(x, **kwargs):
    
    X_train_scaled = kwargs['X_train_scaled']
    y_train_scaled = kwargs['y_train_scaled']
    boosting_type = kwargs['boosting_type']
    n_jobs = kwargs['n_jobs']
    
    if kwargs['boosting_type'] == 'rf':
        num_leaves = int(np.round(x[0]))
        max_depth = int(np.round(x[1]))
        min_child_samples = int(np.round(x[2]))
        learning_rate = x[3]
        n_estimators = int(np.round(x[4]))
        subsample = x[5]
        subsample_freq = int(np.round(x[6]))
        reg_alpha = x[7]
        reg_lambda = x[8]
        
        model = lgb.LGBMRegressor(boosting_type=boosting_type,
                                  num_leaves=num_leaves,
                                  max_depth=max_depth,
                                  min_child_samples=min_child_samples,
                                  learning_rate=learning_rate,
                                  n_estimators=n_estimators,
                                  subsample=subsample,
                                  subsample_freq=subsample_freq,
                                  reg_alpha=reg_alpha,
                                  reg_lambda=reg_lambda,
                                  n_jobs=n_jobs)
    
    else:
        num_leaves = int(np.round(x[0]))
        max_depth = int(np.round(x[1]))
        min_child_samples = int(np.round(x[2]))
        learning_rate = x[3]
        n_estimators = int(np.round(x[4]))
        reg_alpha = x[5]
        reg_lambda = x[6]
        
        model = lgb.LGBMRegressor(boosting_type=boosting_type,
                                  num_leaves=num_leaves,
                                  max_depth=max_depth,
                                  min_child_samples=min_child_samples,
                                  learning_rate=learning_rate,
                                  n_estimators=n_estimators,
                                  reg_alpha=reg_alpha,
                                  reg_lambda=reg_lambda,
                                  n_jobs=n_jobs)
    
    tss = TimeSeriesSplit(n_splits=N_CV)
    scores = cross_val_score(model, 
                             X_train_scaled, 
                             y_train_scaled, 
                             cv=tss, 
                             scoring='neg_mean_absolute_error',
                             n_jobs=1)
    score = -scores.mean()
    if not kwargs['return_model']:
        return(score)
    else:
        model.fit(X_train_scaled, y_train_scaled)
        return(score, model)





def xgb_gbtree(x, **kwargs):
    # This function is used only for PSO
    n_estimators = int(np.round(x[0]))
    max_depth = int(np.round(x[1]))
    learning_rate = x[2]
    gamma = x[3]
    subsample = x[4]
    reg_alpha = x[5]
    reg_lambda = x[6]
    
    X_train_scaled = kwargs['X_train_scaled']
    y_train_scaled = kwargs['y_train_scaled']
    booster=kwargs['booster']
    objective=kwargs['objective']
    n_jobs = kwargs['n_jobs']
    model = xgb.XGBRegressor(n_estimators=n_estimators,
                             max_depth=max_depth, learning_rate=learning_rate,
                             gamma=gamma, subsample=subsample,
                             reg_alpha=reg_alpha, reg_lambda=reg_lambda,
                             objective=objective,
                             booster=booster,
                             n_jobs=n_jobs)
    tss = TimeSeriesSplit(n_splits=N_CV)
    scores = cross_val_score(model, 
                             X_train_scaled, 
                             y_train_scaled, 
                             cv=tss, 
                             scoring='neg_mean_absolute_error',
                             n_jobs=1)
    score = -scores.mean()
    if not kwargs['return_model']:
        return(score)
    else:
        model.fit(X_train_scaled, y_train_scaled)
        return(score, model)
    
    
def predict(model, X, y, n_lags_y):
    # Incoming X and y are 2D arrays
    
    n_y = y.shape[0]
    
    print('Predict idxs:', n_lags_y, n_y, flush=True)
    
    for t in range(n_lags_y, n_y):
        
        # Fill data matrix with past values
        for lag in range(1, 1+n_lags_y):
            X[t, -lag] = y[t-lag]
        
        # Calculate new value
        y[t] = model.predict( X[t,:].reshape((1,-1)) )
        
        # Print result
        if t % 730 == 0:
            print('We are at:', t, y[t], flush=True)
    
    return(y)
    
    
    
    
    
    