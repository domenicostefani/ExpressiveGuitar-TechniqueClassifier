#!/usr/bin/env python3

import os
import time
import subprocess
from glob import glob
import shutil

VERBOSE = True

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'output')

START_FROM_RUN_NUMBER = 1 # Change this to the run number you want to start from !!! WARNING !!! 1-based indexing
NUM_PARALLEL_RUNS = 6 # Change this to the number of parallel runs you want to run


start_from_run_index = START_FROM_RUN_NUMBER-1
# print('Converting notebook to python script...')
# os.system('/home/user/Develop/Domenico/timbre-classifier/convert_to_script.py')
# print('Done.')


runs_done = []
runs_done_dict = {}

infofiles = glob(os.path.join(OUTPUT_DIR,'*','info.txt'))
for iff in infofiles:
    if not os.path.exists(os.path.join(os.path.dirname(iff),'finalModel')) or\
        not os.path.exists(os.path.join(os.path.dirname(iff),'backup_expressive-technique-classifier-phase3.py')):
        print('Deleting '+os.path.dirname(iff))
        shutil.rmtree(os.path.dirname(iff))
    else:
        commandline = ''
        with open(iff) as oif:
            oif.readline()
            commandline = oif.readline()
        assert 'expressive-technique-classifier-phase3.py -f ' in commandline
        commandline = commandline.replace('expressive-technique-classifier-phase3.py -f ','').replace(' -d ',' ').replace(' -w ',' ').replace(' -dr ',' ').replace(' -lr ',' ').replace(' -bs ',' ').replace(' -e ',' ').replace(' -k ',' ')
        runparams = commandline.strip().split(' ')
        runs_done.append(runparams)
        runs_done_dict[os.path.basename(os.path.dirname(iff))] = runparams

print('Runs already in the output folder: '+str(len(runs_done)))

print('#----------------------------#')
print('# Parameter value ranges:    #')
print('#----------------------------#')

features_params = list(range(10,50,15)) + list(range(50,450,80))
print('Feature parameters:',features_params)

net_depth_params = [1,2,3]
print('Net depth parameters:',net_depth_params)

net_width_params = [20,40,50,100,200]
print('Net width parameters:',net_width_params)

dropout_rate_params = [0.15,0.3,0.5,0.8]
print('Dropout rate parameters:',dropout_rate_params)

learning_rate_params = [0.0001,0.001]
print('Learning rate parameters:',learning_rate_params)

batchsize_params = [1024]
print('Batch size parameters:',batchsize_params)

train_epochs_params = [10,50,100,200]
print('Train epochs parameters:',train_epochs_params)

k_fold_parameters = [5]
print('K-fold parameters:',k_fold_parameters)

parameter_lists = [features_params,net_depth_params,net_width_params,dropout_rate_params,learning_rate_params,batchsize_params,train_epochs_params,k_fold_parameters]
expected_length = len(features_params)*len(net_depth_params)*len(net_width_params)*len(dropout_rate_params)*len(learning_rate_params)*len(batchsize_params)*len(train_epochs_params)*len(k_fold_parameters)

# Compute the product of all the parameter values
import itertools
product = list(itertools.product(*parameter_lists))
assert len(product) == expected_length



# for strparameters in strproduct:
#     if strparameters in runs_done:
#         print('Run already done: '+str(strparameters) + '. Skipping...')


strproduct = [[str(inel) for inel in el] for el in product]
for rd in runs_done:
    if not rd in strproduct:
        curname = [k for k,v in runs_done_dict.items() if [str(e) for e in v] == rd][0]
        if VERBOSE:
            print('Warning! Run "'+curname+'" with params '+str(rd)+' is in the output folder but not in the list of runs to do.')


print('#--------------------------------------------------------------------#')
print('# Number of combined runs:',len(product))
print('#')
print('# Starting from run No.',start_from_run_index+1)
print('# Which leaves '+str(len(product)-start_from_run_index)+' training runs to execute.')
print('#--------------------------------------------------------------------#')

product = product[start_from_run_index:]

# Run the training sessions
currently_running = []

for i,parameters in enumerate(product):

    # Busy wait until there is a free slot
    while len(currently_running) >= NUM_PARALLEL_RUNS:
        for process in currently_running:
            if process.poll() is not None:
                currently_running.remove(process)
        time.sleep(1)

    strparameters = [str(p) for p in parameters]
    if strparameters in runs_done:
        print('Run already done: '+str(strparameters) + '. Skipping...')
    else:
        
        assert len(currently_running) < NUM_PARALLEL_RUNS

        # Run the training session
        print('Run',i+1+start_from_run_index,'of',len(product)+start_from_run_index)
        print('Parameters:',parameters)
        features,net_depth,net_width,dropout_rate,learning_rate,batchsize,train_epochs,k_fold = parameters
        command = 'python3 expressive-technique-classifier-phase3.py -f {} -d {} -w {} -dr {} -lr {} -bs {} -e {} -k {}'.format(features,net_depth,net_width,dropout_rate,learning_rate,batchsize,train_epochs,k_fold)

        print(command)

        with open(os.devnull, 'w') as outfile:
            process = subprocess.Popen(command.split(' '),stdout=outfile,stderr=outfile)
            currently_running.append(process)
        time.sleep(1)
        
    
