# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 12:20:08 2023

@author: laukkara
"""

import os
import sys
import time


if 'win' in sys.platform:
    folder_github = r'C:\Users\laukkara\github\ML_indoor'
    output_folder_base = os.path.join(r'C:\Users\laukkara\Data\ML_indoor_Narvi')

elif 'lin' in sys.platform:
    folder_github = r'/home/laukkara/github/ML_indoor'
    output_folder_base = r'/lustre/scratch/laukkara/ML_indoor'
    





counter_for_parameter_sets = 1

create_or_check = 'create'





#######################################

print('We are at: myCreate_slurm_sbatch_files', flush=True)




#######################################
if counter_for_parameter_sets == 1:
    
    print('We are using parameter set:', counter_for_parameter_sets, flush=True)
    
    ## Here is one set of cases that should be completed
    measurement_point_names = ['Espoo1', 'Espoo2', \
                               'Tampere1', 'Tampere2', \
                               'Valkeakoski', 'Klaukkala']
    
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
    
    # models_per_array_job = ','.join([str(x) for x in range(0,7)])
    # models_per_array_job = ','.join([str(x) for x in range(7,48)])
    models_per_array_job = ','.join([str(x) for x in range(0,48)])
    
    
    optimization_methods = ['pso', 'randomizedsearchcv', 'bayessearchcv']
    
    # First set
    # X_lags = [0, 1, 12, 24]
    # y_lags = [0, 1, 12, 24]
    
    # Second set
    X_lags = [2, 3]
    y_lags = [2, 3]
    
    
    N_CVs = [3, 5]
    N_ITERs = [50, 250, 500]
    N_CPUs = [1]
    



############################

n_tot = len(measurement_point_names) \
        * len(model_names) \
        * len(optimization_methods) \
        * len(X_lags) \
        * len(y_lags) \
        * len(N_CVs) \
        * len(N_ITERs) \
        * len(N_CPUs)

print(f'n_tot = {n_tot}', flush=True)



#################################


if create_or_check == 'create':
    # Create SLURM sbatch files and a single shell script for running
    # various sensitivity study cases
    
    print('Create new sbatch files', flush=True)
    
    path_to_main = os.path.join(folder_github,
                                'main.py')
    
    
    # Make sure output folder exists
    time_str = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    folder_sbatch = os.path.join(output_folder_base,
                                 'sbatch_files_{}'.format(time_str))
    if not os.path.exists(folder_sbatch):
        os.makedirs(folder_sbatch)


    # Create command line prompts
    
    command_line_prompts_list = []
    
    # for idx_mp, mp in enumerate(measurement_point_names):
        
    # for idx_model_name, model_name in enumerate(model_names):
        
    # for idx_opt_method, opt_method in enumerate(optimization_methods):
        
    for N_CV in N_CVs:
        
        for N_ITER in N_ITERs:
            
            for N_CPU in N_CPUs:

                for X_lag in X_lags:
                    
                    for y_lag in y_lags:
                        
                        # str_command_line = 'python3 ' \
                        #     + path_to_main + ' ' \
                        #     + f'{idx_mp} {idx_mp+1} ' \
                        #     + f'{idx_model_name} {idx_model_name+1} ' \
                        #     + f'{idx_opt_method} {idx_opt_method+1} ' \
                        #     + f'{N_CV} ' \
                        #     + f'{N_ITER} ' \
                        #     + f'{N_CPU} ' \
                        #     + f'{X_lag} ' \
                        #     + f'{y_lag}'
                        
                        str_command_line = 'python3 ' \
                            + path_to_main + ' ' \
                            + '0 3 ' \
                            + '$SLURM_ARRAY_TASK_ID $(( $SLURM_ARRAY_TASK_ID + 1 )) ' \
                            + '0 3 ' \
                            + f'{N_CV} ' \
                            + f'{N_ITER} ' \
                            + f'{N_CPU} ' \
                            + f'{X_lag} ' \
                            + f'{y_lag}'
                        
                        command_line_prompts_list.append(str_command_line)
                        
                        
                        str_command_line = 'python3 ' \
                            + path_to_main + ' ' \
                            + '3 6 ' \
                            + '$SLURM_ARRAY_TASK_ID $(( $SLURM_ARRAY_TASK_ID + 1 )) ' \
                            + '0 3 ' \
                            + f'{N_CV} ' \
                            + f'{N_ITER} ' \
                            + f'{N_CPU} ' \
                            + f'{X_lag} ' \
                            + f'{y_lag}'
                        
                        command_line_prompts_list.append(str_command_line)
                        
                        
    
    
    # Write prompts to a text file
    fname = os.path.join(folder_sbatch,
                         'command_line_prompts.txt')
    
    with open(fname, 'w') as f:
        for line in command_line_prompts_list:
            f.write(f'{line}\n')
    

    # Read in the base case shell script
    sbatch_template = []
    
    fname_ML_indoor_template = os.path.join(folder_github,
                                            'ML_indoor_template.sh')
    
    with open(fname_ML_indoor_template, 'r') as f:
        for line in f:
            sbatch_template.append(line.rstrip())



    # write new sbatch files
    
    filenames_list = []
    
    for idx_prompt, prompt in enumerate(command_line_prompts_list):
        
        idx_str_helper = prompt.find('_ID + 1 ))')
        
        str_helper = prompt[(idx_str_helper+11):].replace(' ', '_')
        
        sbatch_single_file = f'No{idx_prompt}_MLindoor_{str_helper}.sh'
        
        fname = os.path.join(folder_sbatch,
                             sbatch_single_file)
        
        filenames_list.append(sbatch_single_file)
        
        with open(fname, 'w') as f:
            
            for line_template in sbatch_template:
                
                if '{JOB_NAME}' in line_template:
                    s = line_template
                    s = s.replace('{JOB_NAME}',
                                  'ML'+str_helper)
                    f.write(s + '\n')
                
                elif '{ARRAY}' in line_template:
                    s = line_template
                    s = s.replace('{ARRAY}',
                                  models_per_array_job)
                    f.write(s + '\n')
                    
                elif '{FUNCTION_CALL}' in line_template:
                    s = line_template
                    s = s.replace('{FUNCTION_CALL}',
                                  prompt)
                    f.write(s + '\n')
                
                else:
                    f.write(line_template + '\n')
                
    
    
    # Write the single sbatch calls to a shell script file
    fname = os.path.join(folder_sbatch,
                         'ML_indoor_shell_script_for_sbatch_files.sh')
    with open(fname, 'w') as f:
        for line in filenames_list:
            f.write('sbatch ' + line + '\n')

    
    print('command line prompt file, sbatch files and shell script created!',
          flush=True)
    
    




####################################



if create_or_check == 'check':
    
    pass














