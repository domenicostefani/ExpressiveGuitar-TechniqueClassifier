#!/usr/bin/env python3
DEBUG_MODE = False
# print big title that says 'RUN MULTIPLE'
print('''

██████╗ ██╗   ██╗███╗   ██╗    ███╗   ███╗██╗   ██╗██╗  ████████╗██╗██████╗ ██╗     ███████╗
██╔══██╗██║   ██║████╗  ██║    ████╗ ████║██║   ██║██║  ╚══██╔══╝██║██╔══██╗██║     ██╔════╝
██████╔╝██║   ██║██╔██╗ ██║    ██╔████╔██║██║   ██║██║     ██║   ██║██████╔╝██║     █████╗  
██╔══██╗██║   ██║██║╚██╗██║    ██║╚██╔╝██║██║   ██║██║     ██║   ██║██╔═══╝ ██║     ██╔══╝  
██║  ██║╚██████╔╝██║ ╚████║    ██║ ╚═╝ ██║╚██████╔╝███████╗██║   ██║██║     ███████╗███████╗
╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═══╝    ╚═╝     ╚═╝ ╚═════╝ ╚══════╝╚═╝   ╚═╝╚═╝     ╚══════╝╚══════╝ v2.0
''')


# MODES TO VERIFY
problem = 'classification_task = ClassificationTask.FULL_8_CLASS_PROBLEM'
fsize =   'FEATURE_WINDOW_SIZE = FeatureWindowSize._704windowed'
wm =      'WINDOWED_INPUT_MODE = WindowedInputMode._2D'

notebook = './expressive-technique-classifier-phase3.ipynb'
script = './expressive-technique-classifier-phase3.py'
backup_script = 'backup_'+script

convert_script_path = './convert_to_script.py'

VERBOSE = True








import os
import time
import subprocess
from glob import glob
import shutil
import numpy as np
import itertools
import re

this_folder = os.path.dirname(os.path.realpath(__file__))

# Check that notebook and script exist
assert os.path.exists(os.path.relpath(os.path.join(this_folder,notebook))), 'Notebook '+notebook+' not found'
assert os.path.exists(os.path.relpath(os.path.join(this_folder,script))), 'Script '+script+' not found'

# Check that notebook is not newer than script
notebook_time = os.path.getmtime(notebook)
script_time = os.path.getmtime(script)
if (notebook_time > script_time):
    print('Training Notebook is newer than script. Converting notebook to script...')
    os.system(convert_script_path)
    print('Done.')
    print('Restart this script manually')
    exit()
print('Training script is up to date. Continuing...')

found_classproblem_enum = False
# Check that script is configured as expected, i.e. that it is configured to run the 8-class problem, with 704 window size and 2D windowed input mode
found = {problem:False, fsize:False, wm:False}
with open(script) as f:
    lines = f.readlines()
    for l in lines:
        if problem == l.strip():
            found[problem] = True
        elif fsize == l.strip():
            found[fsize] = True
        elif wm == l.strip():
            found[wm] = True

        # Also check that this run_multiple script is not out of date
        if "FULL_8_CLASS_PROBLEM,BINARY_PERCUSSIVE_PITCHED,PERCUSSIVE_4_ONLY,PITCHED_4_ONLY,PERCUSSIVE_PLUS_PITCHED_CLASS,ONE_GUITARIST_FULL = ((1,'full'), (2,'binary'), (3,'perc'), (4,'pitch'), (5,'perc+pitch'), (6,'one-guit-full'))" in l:
            found_classproblem_enum = True

for line in found:
    assert found[line], '!The notebook is not configured as expected (Line "'+line+'" not found in script '+script+')'
assert found_classproblem_enum, 'The enum used in this very script is not up to date with the training notebook or script. Please update it IN THIS RUN_MULTIPLE SCRIPT.'



base_output_folder = os.path.join(this_folder,'output')

assert len(re.findall(r'FEATURE_WINDOW_SIZE[ ]*=[ ]*FeatureWindowSize\.', fsize)) == 1, '"fsize" does not match pattern'
window_folder = re.findall(r'FEATURE_WINDOW_SIZE[ ]*=[ ]*FeatureWindowSize\.(.*)', fsize)[0] # take text that comes after the regex r'FEATURE_WINDOW_SIZE[ ]*=[ ]*FeatureWindowSize\.'

#problem = 'classification_task = ClassificationTask.FULL_8_CLASS_PROBLEM'
assert len(re.findall(r'classification_task[ ]*=[ ]*ClassificationTask\.', problem)) == 1, '"problem" does not match pattern'
problem_enum = re.findall(r'classification_task[ ]*=[ ]*ClassificationTask\.(.*)', problem)[0]

converss = {'FULL_8_CLASS_PROBLEM' : 'full', 'BINARY_PERCUSSIVE_PITCHED' : 'binary', 'PERCUSSIVE_4_ONLY' : 'perc', 'PITCHED_4_ONLY' : 'pitch', 'PERCUSSIVE_PLUS_PITCHED_CLASS' : 'perc+pitch', 'ONE_GUITARIST_FULL' : 'one-guit-full'}
assert problem_enum in converss, 'problem_enum not found in converss'
problem_folder = converss[problem_enum]

full_run_folder = os.path.join(base_output_folder, window_folder, problem_folder)
# print('full_run_folder: '+full_run_folder)

if not os.path.exists(full_run_folder):
    print('WARNING: output folder '+full_run_folder+' does not exist. Are you sure that this is OK?')
    print('Press ENTER to continue or CTRL+C to exit')
    input()



OUTPUT_DIR = full_run_folder

START_FROM_RUN_NUMBER = 1 # Change this to the run number you want to start from !!! WARNING !!! 1-based indexing
NUM_PARALLEL_RUNS = 6 # Change this to the number of parallel runs you want to run


start_from_run_index = START_FROM_RUN_NUMBER-1
# print('Converting notebook to python script...')
# os.system('/home/base-user/Develop/Domenico/timbre-classifier/convert_to_script.py')
# print('Done.')


runs_done = []
runs_done_dict = {}

print('Check training sessions already done')

chars_to_delete = 0


parameter_names = {}
parameter_names['features'] = '-f'
parameter_names['net-depth'] = '-d'
parameter_names['net-width'] = '-w'
parameter_names['dropout'] = '-dr'
parameter_names['learning-rate'] = '-lr'
parameter_names['batchsize'] = '-bs'
parameter_names['epochs'] = '-e'
parameter_names['k-folds'] = '-k'
parameter_names['oversampling-aggressiveness'] = '-osagg'
parameter_names['conv'] = '-c1d'
parameter_names['conv-kernels'] = '-ck'
parameter_names['conv-strides'] = '-cs'
parameter_names['conv-filters'] = '-cf'
parameter_names['conv-activations'] = '-c1dact'
parameter_names['conv-padding'] = '-cp'
parameter_names['pool-layers'] = '-pl'


RUNS_DONE_CACHEFILE = os.path.join(OUTPUT_DIR, 'runs_done.txt') # Files with the parameters of the runs already done
CACHE_UP_TO_DATE = False
# if os.path.exists(RUNS_DONE_CACHEFILE):
#     with open(RUNS_DONE_CACHEFILE, 'r') as f:
#         lines = f.readlines()
#         # If the number of lines corresponds to the number of files in the output directory, then we can assume that the cache file is up to date
#         if len(lines) == len(glob(os.path.join(OUTPUT_DIR, 'c_acc*'))):
#             print('Cache file is up to date. Reading from cache file...')
#             CACHE_UP_TO_DATE = True

#             for line in lines:
#                 line = line.strip()
#                 splt = line.split(':')
#                 splt[0]

if not CACHE_UP_TO_DATE:
    infofiles = glob(os.path.join(OUTPUT_DIR,'*','info.txt'))
    for fidx,iff in enumerate(infofiles):
        print('\b'*chars_to_delete,end='')
        strpr = '['+str(fidx+1)+'/'+str(len(infofiles))+']'
        print(strpr,end='',flush = True)
        chars_to_delete = len(strpr)
                
        # if not os.path.exists(os.path.join(os.path.dirname(iff),'finalModel')) or\
        #     not os.path.exists(os.path.join(os.path.dirname(iff), backup_script)):
        #     # print('Moving '+os.path.dirname(iff) + ' to trash')
        #     # if not os.path.exists(os.path.join(OUTPUT_DIR,'trash')):
        #     #     os.mkdir(os.path.join(OUTPUT_DIR,'trash'))
        #     # shutil.move(os.path.dirname(iff),os.path.join(OUTPUT_DIR,'trash'))
        #     pass
        # else:


        commandline = ''
        with open(iff) as oif:
            oif.readline()
            commandline = oif.readline().strip()

        if commandline == '':
            # print('WARNING: commandline empty in file '+iff)
            continue
        # assert script+' -f ' in commandline
        

        # get script name from commandline
        scriptstr = re.findall(r'([^ ]*\.py)',commandline)[0]
        assert scriptstr == script, 'script name in commandline does not match script name in scriptstr'

        #remove everythin up to script name end in commandline
        commandline = commandline[commandline.find(scriptstr)+len(scriptstr):]

        def extract_argument_value(commandline, argument):
            if argument in commandline:
                cstart = commandline.find(string_to_find)+len(string_to_find)
                cend = commandline.find(' ',cstart)
                cend = len(commandline) if cend == -1 else cend
                return commandline[cstart:cend].strip()
            else:
                assert False, 'argument not found in commandline'
                return None

        thisruns_parameters = {}
        for p in parameter_names:
            string_to_find = ' --'+p+' '
            if string_to_find in commandline:
                argument_value = extract_argument_value(commandline, string_to_find)
                commandline = commandline.replace(re.findall(string_to_find+'[ ]*'+argument_value,commandline)[0],'')
                thisruns_parameters[p] = argument_value
        
        for longp, shortp in parameter_names.items():
            string_to_find = ' '+shortp+' '
            if string_to_find in commandline:
                argument_value = extract_argument_value(commandline, string_to_find)
                commandline = commandline.replace(re.findall(string_to_find+'[ ]*'+argument_value,commandline)[0],'')
                thisruns_parameters[longp] = argument_value

        if commandline.strip() != '':
            print('Warning, a run contains additional parameters that are not deal with in this script: "'+commandline+'"')
            print('Do you wish to proceed anyway? Press enter to continue, or ctrl+c to exit')
            input()


        runs_done.append(thisruns_parameters)
        runs_done_dict[os.path.basename(os.path.dirname(iff))] = thisruns_parameters

        # print(runs_done_dict)

            #TODO: save to cache
print(' Done.') # Cause code before does not print endline

# print('runs_done_dict:'+str(runs_done_dict))




print('Runs already in the output folder: '+str(len(runs_done)))

print('#----------------------------#')
print('# Parameter value ranges:    #')
print('#----------------------------#')

parameter_values = {
    'features'                      : [200,100,'all'],                      
    'net-depth'                     : [0,1,2,4],
    'net-width'                     : [16,32,100.200],
    'dropout'                       : [0.5,0.2],
    'learning-rate'                 : [0.0001,0.00005,0.00001],
    'batchsize'                     : [32,64,128],
    'epochs'                        : [100,200,600],
    'k-folds'                       : [5],
    'oversampling-aggressiveness'   : [0.5,1.0],
    'conv'                          : [1],
    'conv-kernels'                  : ['3','5'],
    'conv-strides'                  : ['2','1'],
    'conv-filters'                  : ['2','8','16','32','128'],
    'conv-activations'              : ['relu'],
    'conv-padding'                  : ['same'],
    'pool-layers'                   : ['M','A'],
}

for p in parameter_names:
    assert p in parameter_values, 'Value ranges are not specified for parameter '+p+'.'





print(''.join([str(p)+': '+str(parameter_values[p])+'\n' for p in parameter_values]))

parameter_lists = [parameter_values[p] for p in parameter_values]

# Compute the product of all the parameter values
product = list(itertools.product(*parameter_lists))
expected_length = np.prod([len(parameter_values[p]) for p in parameter_values])
assert len(product) == expected_length, 'Expected length: '+str(expected_length)+', actual length: '+str(len(product))


# for strparameters in strproduct:
#     if strparameters in runs_done:
#         print('Run already done: '+str(strparameters) + '. Skipping...')


# strproduct = [[str(inel) for inel in el] for el in product]
# for rd in runs_done:
#     if not rd in strproduct:
#         curname = [k for k,v in runs_done_dict.items() if [str(e) for e in v] == rd][0]
        # if VERBOSE:
        #     print('Warning! Run "'+curname+'" with params '+str(rd)+' is in the output folder but not in the list of runs to do.')

print('#--------------------------------------------------------------------#')
print('# Number of combined runs:',len(product))
print('#')






origlen = len(product)
product = product[start_from_run_index:]

already_done = 0

todl = []

for i,pd in enumerate(product):
    assert len(pd) == len(parameter_values), 'Expected length: '+str(len(parameter_values))+', actual length: '+str(len(pd))
    assert len(list(parameter_values.keys())) == len(pd)
    cur_params = dict(zip(parameter_values.keys(),pd))

    cur_params_str = {k:str(v) for k,v in cur_params.items()}

    if cur_params_str in runs_done:
        print('Run already done: '+str(cur_params) + '. Skipping...')
        already_done += 1
    else:
        # print('Run not done: '+str(cur_params) + '. Adding to list of runs to do.')
        # TODO: fix this big mess
        todl.append(cur_params)


print('# Starting from run %d / %d'%(start_from_run_index+1,origlen))
print('# Runs already done: '+str(already_done))
print('# Which leaves '+str(len(todl))+' training runs to execute.')
print('#--------------------------------------------------------------------#')




# Run the training sessions
currently_running = []

for i,pd in enumerate(product):
    assert len(pd) == len(parameter_values), 'Expected length: '+str(len(parameter_values))+', actual length: '+str(len(pd))
    assert len(list(parameter_values.keys())) == len(pd)
    cur_params = dict(zip(parameter_values.keys(),pd))

    # Busy wait until there is a free slot
    while len(currently_running) >= NUM_PARALLEL_RUNS:
        for process in currently_running:
            if process.poll() is not None:
                currently_running.remove(process)
        time.sleep(1)

    strparameters = [str(p) for p in parameter_names]

    print('\nRun',i+1+start_from_run_index,'of',len(product)+start_from_run_index)
    print('-> parameters "'+str([cur_params[e] for e in cur_params])+'"\n--> not done already, doing now...')

    
    assert len(currently_running) < NUM_PARALLEL_RUNS

    # Run the training session
    # print('Parameters:',parameter_names)

    command = 'python3 '+script+' '
    for p in parameter_names:
        assert p in cur_params, 'Parameter "'+p+'" not in cur_params: '+str(cur_params)
        command += '--'+p+' '+str(cur_params[p])+' '

    if DEBUG_MODE:
        print(command + '\n\n')
        time.sleep(5)
    else:
        with open(os.devnull, 'w') as outfile:
            process = subprocess.Popen(command.split(' '),stdout=outfile,stderr=outfile)
            currently_running.append(process)
        print(command + '\n\n')
        time.sleep(1)
        
    
# Example command:
# python3 expressive-technique-classifier-phase3.py -f 150 -d 4 -w 200 -dr 0.7 -lr 0.001 -bs 1024 -e 70 -k 5 -os -osagg 0   
