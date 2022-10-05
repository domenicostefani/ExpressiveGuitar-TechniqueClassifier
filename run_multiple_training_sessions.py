#!/usr/bin/env python3

import os


print('Converting notebook to python script...')
os.system('/home/user/Develop/Domenico/timbre-classifier/convert_to_script.py')
print('Done.')

print('#----------------------------#')
print('# Parameter value ranges:    #')
print('#----------------------------#')

features_params = list(range(50,150,20)) + list(range(150,210,30)) + list(range(210,381,50))
print('Feature parameters:',features_params)

net_depth_params = [3,5,8]
print('Net depth parameters:',net_depth_params)

net_width_params = [80,100,200,400,800]
print('Net width parameters:',net_width_params)

dropout_rate_params = [0.15,0.3,0.5]
print('Dropout rate parameters:',dropout_rate_params)

learning_rate_params = [0.0001]
print('Learning rate parameters:',learning_rate_params)

batchsize_params = list(reversed([256,512,1024]))
print('Batch size parameters:',batchsize_params)

train_epochs_params = [1000]
print('Train epochs parameters:',train_epochs_params)

k_fold_parameters = [5]
print('K-fold parameters:',k_fold_parameters)

parameter_lists = [features_params,net_depth_params,net_width_params,dropout_rate_params,learning_rate_params,batchsize_params,train_epochs_params,k_fold_parameters]
expected_length = len(features_params)*len(net_depth_params)*len(net_width_params)*len(dropout_rate_params)*len(learning_rate_params)*len(batchsize_params)*len(train_epochs_params)*len(k_fold_parameters)

# Compute the product of all the parameter values
import itertools
product = list(itertools.product(*parameter_lists))
assert len(product) == expected_length


print('#----------------------------------#')
print('# Number of combined runs:',len(product))
print('#----------------------------------#')

# Run the training sessions
for i,parameters in enumerate(product):
    print('Run',i+1,'of',len(product))
    print('Parameters:',parameters)
    features,net_depth,net_width,dropout_rate,learning_rate,batchsize,train_epochs,k_fold = parameters
    print('python3 expressive-technique-classifier-phase3.py -f {} -d {} -w {} -dr {} -lr {} -bs {} -e {} -k {}'.format(features,net_depth,net_width,dropout_rate,learning_rate,batchsize,train_epochs,k_fold))