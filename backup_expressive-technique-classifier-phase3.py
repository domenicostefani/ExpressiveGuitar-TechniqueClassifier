#!/usr/bin/env python
# coding: utf-8

# <a href='https://colab.research.google.com/github/domenicostefani/timbre-classifier/blob/main/expressive-technique-classifier-phase3.ipynb' target='_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Expressive Guitar Technique classifier
# Ph.D. research project of [Domenico Stefani](work.domenicostefani.com)  
# This notebook loads a dataset of feature vectors extracted from **pitched** and **percussive** sounds recorded with many acoustic guitars.
# The techniques/classes recorded are:  
# 0.    **Kick**      (Palm on lower body)
# 1.    **Snare 1**   (All fingers on lower side)
# 2.    **Tom**       (Thumb on higher body)
# 3.    **Snare 2**   (Fingers on the muted strings, over the end
# of the fingerboard)
# ___
# 4.    **Natural Harmonics** (Stop strings from playing the dominant frequency, letting harmonics ring)
# 5.    **Palm Mute** (Muting partially the strings with the palm
# of the pick hand)
# 6.    **Pick Near Bridge** (Playing toward the bridge/saddle)
# 7.    **Pick Over the Soundhole** (Playing over the sound hole)
# 

# ## Import modules and mount drive folder

# In[170]:


# Install module for the ReliefF feature selection
# !pip install skrebate
# !pip install tensorboard
# !pip3 install pickle5
# !pip3 install --quiet tensorflow==2.4.1
# !pip3 install tensorboard


# In[171]:




# In[172]:


REQUIRE_GPU = True
DO_SAVE_TENSORBOARD_LOGS = False 
DO_SAVE_FOLD_MODELS = False 

# Load the TensorBoard notebook extension
#<redacted ipython line>get_ipython().run_line_magic('load_ext', 'tensorboard')
None
import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import datetime
import pandas as pd
import numpy as np
import tensorflow as tf
print("Tensorflow version: " + tf.version.VERSION)
import time
from tensorflow import keras
import pickle
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
import bz2 # To open compressed data
import re
import shutil
from sklearn.metrics import f1_score
from sklearn import metrics

global_random_state = 42
np.random.seed(global_random_state)
tf.random.set_seed(global_random_state)

COLAB = False
if COLAB:
    print('Running on CoLab')
    #Connect and mount the drive folder that contains the train dataset and the output folder
    from google.colab import drive
    drive.mount('/content/gdrive', force_remount=False)

    HOMEBASE = os.path.join('/content','gdrive','MyDrive','dottorato','Publications','02-IEEE-RTEmbeddedTimbreClassification(submitted)','Classifier')
    THISDIR = "/content/"
else:
    print('Not running on CoLab')
    HOMEBASE = "."
    THISDIR = "./"
DATAFOLDER = HOMEBASE + "/data"
MODELFOLDER = HOMEBASE + "/output"

RELIEF_CACHE_FILEPATH = DATAFOLDER + '/relief_cache.pickle'


# In[173]:


def is_notebook() -> bool:
    try:
#<redacted ipython line>        shell = get_ipython().__class__.__name__
        None
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
#<redacted ipython line>            return False  # Terminal running IPython
            None
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter


# ## Parse Command Line arguments
# 
# *_Important_*: If you are running this from a jupyter Notebook, change the run parameters at the end of the next cell

# In[174]:


args = None
if not is_notebook():
    import argparse
    parser = argparse.ArgumentParser(description='Train the expressive guitar technique classifier.')

    def featnum_type(x):
        (MIN,MAX) = (1,495) 
        x = int(x)
        if x < MIN or x > MAX:
            raise argparse.ArgumentTypeError("Feature number must be between {} and {}".format(MIN, MAX))
        return x
    def netdepth_type(x):
        (MIN,MAX) = (1,20) 
        x = int(x)
        if x < MIN or x > MAX:
            raise argparse.ArgumentTypeError("Network depth must be between {} and {}".format(MIN, MAX))
        return x
    def netwidth_type(x):
        (MIN,MAX) = (1,2000) 
        x = int(x)
        if x < MIN or x > MAX:
            raise argparse.ArgumentTypeError("Network width must be between {} and {}".format(MIN, MAX))
        return x
    def dropout_type(x):
        (MIN,MAX) = (0,1) 
        x = float(x)
        if x < MIN or x > MAX:
            raise argparse.ArgumentTypeError("Network depth must be between {} and {}".format(MIN, MAX))
        return x
    def lr_type(x):
        (MIN,MAX) = (0,1) 
        x = float(x)
        if x < MIN or x > MAX:
            raise argparse.ArgumentTypeError("Learning rate must be between {} and {}".format(MIN, MAX))
        return x
    def batchsize_type(x):
        (MIN,MAX) = (1,4096) 
        x = int(x)
        if x < MIN or x > MAX:
            raise argparse.ArgumentTypeError("Batchsize must be between {} and {}".format(MIN, MAX))
        return x
    def epochs_type(x):
        (MIN,MAX) = (1,10000) 
        x = int(x)
        if x < MIN or x > MAX:
            raise argparse.ArgumentTypeError("Batchsize must be between {} and {}".format(MIN, MAX))
        return x
    def kfold_type(x):
        (MIN,MAX) = (1,20) 
        x = int(x)
        if x < MIN or x > MAX:
            raise argparse.ArgumentTypeError("Batchsize must be between {} and {}".format(MIN, MAX))
        return x
    parser.add_argument('-f',  '--features',      default=80,     type=featnum_type,   help='Number of features to use for training [1-495] (default: 80)')
    parser.add_argument('-d',  '--net-depth',     default=3,      type=netdepth_type,  help='Number of layers in the FFNN [1-20] (default: 3)')
    parser.add_argument('-w',  '--net-width',     default=100,    type=netwidth_type,  help='Number of layers in the FFNN [1-2000] (default: 100)')
    parser.add_argument('-dr', '--dropout',       default=0.15,   type=dropout_type,   help='Dropout amount [0-1] (default: 0.15)')
    parser.add_argument('-lr', '--learning-rate', default=0.0001, type=lr_type,        help='Learning rate [0-1] (default: 0.0001)')
    parser.add_argument('-bs', '--batchsize',     default=256,    type=batchsize_type, help='Learning rate [1-4096] (default: 256)')
    parser.add_argument('-e',  '--epochs',        default=1000,   type=epochs_type,    help='Learning rate [1-10000] (default: 1000)')
    parser.add_argument('-k',  '--k-folds',       default=5,      type=kfold_type,     help='K of K-folds [1-20] (default: 5)')
    parser.add_argument('-os', '--oversampling',  action='store_true', help='Perform oversampling')
    parser.add_argument('-v', '--verbose',        action='store_true', help='increase output verbosity')
    args = parser.parse_args()
    args = vars(args)
else:
    
    #-------------------------------------------------------------------------------------------------------------------------------------------------------------------#
    """                               +-----------------------------------------------------------------------------------------------+                                 #
    #                                 |    CHANGE THE VALUES HERE IF RUNNING THE TRAINING FROM A JUPYTER NOTEBOOK (e.g., on Colab)    |                                 #
    #                                 + ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ +                                 #
    """ #↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓#
    args = {'features':      80, 
            'net_depth':     3, 
            'net_width':     100, 
            'dropout':       0.15,
            'learning_rate': 0.0001,
            'batchsize':     256,
            'epochs':        2,
            'k_folds':       5,
            'oversampling':  False,
            'verbose':       False}
    #↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑#
    """                               + ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ +                                 #
    #                                 |    CHANGE THE VALUES HERE IF RUNNING THE TRAINING FROM A JUPYTER NOTEBOOK (e.g., on Colab)    |                                 #
    #                                 +-----------------------------------------------------------------------------------------------+                                 #
    """#----------------------------------------------------------------------------------------------------------------------------------------------------------------#


# ## Enforce GPU usage

# In[175]:


# sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))
physical_devices = tf.config.list_physical_devices('GPU') 

for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

print(physical_devices)
if REQUIRE_GPU:
  assert len(tf.config.experimental.list_physical_devices('GPU')) >= 1


# ## Check Real avaliable GRAM

# In[176]:


import subprocess
import sys

def pip_install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


# In[177]:


CHECK_GRAM = False

if CHECK_GRAM:
    # memory footprint support libraries/code
    os.symlink('/opt/bin/nvidia-smi','/usr/bin/nvidia-smi')
    pip_install('gputil')
    pip_install('psutil')
    pip_install('humanize')
    import psutil
    import humanize
    import os
    import GPUtil as GPU
    GPUs = GPU.getGPUs()
    # XXX: only one GPU on Colab and isn’t guaranteed
    gpu = GPUs[0]
    def printm():
        process = psutil.Process(os.getpid())
        print("Gen RAM Free: " + humanize.naturalsize( psutil.virtual_memory().available ), " | Proc size: " + humanize.naturalsize( process.memory_info().rss))
        print("GPU RAM Free: {0:.0f}MB | Used: {1:.0f}MB | Util {2:3.0f}% | Total {3:.0f}MB".format(gpu.memoryFree, gpu.memoryUsed, gpu.memoryUtil*100, gpu.memoryTotal))
    printm()


# # Import Dataset

# In[178]:


LOAD_DATA_FROM = 'pickle'
# LOAD_DATA_FROM = 'compressedpickle'

if LOAD_DATA_FROM == 'compressedpickle':
    print("Reading dataset from compressed pickle...")
    DATASET_PATH = os.path.join(DATAFOLDER,'phase3','20220929_115154_onlycorrectdetections.bz2')
    startime = time.time()
    ifile = bz2.BZ2File(DATASET_PATH,'rb')
    featuredataset = pickle.load(ifile)
    ifile.close()
    print('Successfully Loaded!\nIt took %.1fs to load from compressed pickle' % (time.time()-startime))
elif LOAD_DATA_FROM == 'pickle':
    print("Reading dataset from pickle...")
    DATASET_PATH = os.path.join(DATAFOLDER,'phase3','20220929_115154_onlycorrectdetections.pickle')
    startime = time.time()
    with open(DATASET_PATH,'rb') as pf:
        featuredataset = pickle.load(pf)
    print('Successfully Loaded!\nIt took %.1fs to load from regular pickle' % (time.time()-startime))

# display(featuredataset)


# ### Drop features that we have found to be problematic with feature selection and training

# In[179]:


if 'attackTime_peaksamp' in featuredataset.columns.to_list() or 'attackTime_attackStartIdx' in featuredataset.columns.to_list() or 'peakSample_index' in featuredataset.columns.to_list():
    featuredataset.drop(columns=['attackTime_peaksamp',                                'attackTime_attackStartIdx',                                'peakSample_index'], inplace=True)


# In[180]:


# Extract separate DFs
metadata = featuredataset.filter(regex='^meta_',axis=1)
labels = featuredataset.meta_expressive_technique_id
# TODO: if the prev does not work, replace with:
# featuredataset.loc[:,['meta_expressive_technique_id']]
features = featuredataset.loc[:,[col for col in featuredataset.columns if col not in metadata.columns]]
# Convert to numeric formats where possible (somehow convert_dtypes doesn't work [https://stackoverflow.com/questions/65915048/pandas-convert-dtypes-not-working-on-numbers-marked-as-objects])
metadata = metadata.apply(pd.to_numeric, errors='ignore')
labels = labels.apply(pd.to_numeric, errors='ignore')
features = features.apply(pd.to_numeric, errors='ignore')


# In[181]:


assert metadata.shape[1] == 9
assert features.shape[1] == 495


# In[182]:


original_dataset_features = features.copy()
dataset_labels = labels.copy()

#TODO: in the future replace with whole dataset
CLASSES_DESC = {0:"Kick",
                1:"Snare 1",
                2:"Tom",
                3:"Snare 2",
                4:"Natural Harmonics",
                5:"Palm Mute",
                6:"Pick Near Bridge",
                7:"Pick Over the Soundhole"}
CLASSES = list(CLASSES_DESC.keys())

assert np.equal(np.sort(CLASSES),np.sort(pd.unique(dataset_labels))).all()


print("Dataset read")
print("Dataset entries: "+str(original_dataset_features.shape[0]))
print("Dataset features: "+str(original_dataset_features.shape[1]))

original_feature_number = original_dataset_features.shape[1]
(relief_data_X,relief_data_y) = (original_dataset_features.values,dataset_labels.values.ravel())


# In[183]:


# Compute has of the dataset files.
# This are used to cache precomputed feature selection with ReliefF (Which is rather slow)
import hashlib
 
dataset_sha256_hash = hashlib.sha256()
with open(DATASET_PATH,"rb") as fy:
    for byte_block in iter(lambda: fy.read(4096),b""):    # Read and update hash string value in blocks of 4K
        dataset_sha256_hash.update(byte_block)
dataset_sha256_hash = dataset_sha256_hash.hexdigest()

print(dataset_sha256_hash)


# ## Subset features

# In[184]:


def get_manual_selected_features(data):
    print ("Subsetting features...")
    columns_to_keep = []
    # if USE_ATTACKTIME_PEAKSAMP:
    #     columns_to_keep.append("attackTime_peaksamp")
    # if USE_ATTACKTIME_ATTACKSTARTIDX:
    #     columns_to_keep.append("attackTime_attackStartIdx")
    if USE_ATTACKTIME_VALUE:
        columns_to_keep.append("attackTime_value")
    if USE_BARKSPECBRIGHTNESS:
        columns_to_keep.append("barkSpecBrightness")
    if USE_PEAKSAMPLE_VALUE:
        columns_to_keep.append("peakSample_value")
    # if USE_PEAKSAMPLE_INDEX:
    #     columns_to_keep.append("peakSample_index")
    if USE_ZEROCROSSING:
        columns_to_keep.append("zeroCrossing")

    assert USE_BARKSPEC <= 50 and USE_BARKSPEC >= 0 and USE_BFCC <= 49 and USE_BFCC >= 0 and USE_CEPSTRUM <= 353 and USE_CEPSTRUM >= 0 and USE_MFCC <= 37 and USE_MFCC >= 0

    if USE_BARKSPEC > 0:
        columns_to_keep += ['barkSpec_'+str(i+1) for i in range(USE_BARKSPEC)]
    if USE_BFCC > 0:
        columns_to_keep += ['bfcc_'+str(i+2) for i in range(USE_BFCC)]  # +2 is correct here since we want to skip the first normalized coefficient
    if USE_CEPSTRUM > 0:
        columns_to_keep += ['cepstrum_'+str(i+1) for i in range(USE_CEPSTRUM)]
    if USE_MFCC > 0:
        columns_to_keep += ['mfcc_'+str(i+2) for i in range(USE_MFCC)]  # +2 is correct here since we want to skip the first normalized coefficient

    return columns_to_keep


# In[185]:


## To Compeltely reset RelieFF cache
# with open(RELIEF_CACHE_FILEPATH, 'wb') as rcf:
#     pickle.dump(set(), rcf)


# In[186]:



        
# how_many_examples_per_class =10
# subselection = list(range(0,how_many_examples_per_class))+\
#                list(range(600,600+how_many_examples_per_class))+\
#                list(range(1100,1100+how_many_examples_per_class))+\
#                list(range(1400,1400+how_many_examples_per_class))+\
#                list(range(1900,1900+how_many_examples_per_class))+\
#                list(range(3000,3000+how_many_examples_per_class))+\
#                list(range(9000,9000+how_many_examples_per_class))+\
#                list(range(14000,14000+how_many_examples_per_class))

# testprova_dataset_features = original_dataset_features.iloc[subselection]
# testprova_dataset_labels = dataset_labels.iloc[subselection]
import os, platform, subprocess, re

def get_processor_name():
    if platform.system() == "Windows":
        return platform.processor()
    elif platform.system() == "Darwin":
        os.environ['PATH'] = os.environ['PATH'] + os.pathsep + '/usr/sbin'
        command ="sysctl -n machdep.cpu.brand_string"
        return subprocess.check_output(command).strip()
    elif platform.system() == "Linux":
        command = "cat /proc/cpuinfo"
        all_info = subprocess.check_output(command, shell=True).decode().strip()
        for line in all_info.split("\n"):
            if "model name" in line:
                return re.sub( ".*model name.*:", "", line,1)
    return ""

class ReliefCacheElem(dict):

    PRINT_HASH = False

    def __init__(self,dataset_sha256,n_neighbors,relieff_top_features,relieff_feature_importances,time_of_computation):
        self.dataset_sha256 = dataset_sha256
        self.n_neighbors = n_neighbors
        self.relieff_top_features = relieff_top_features
        self.relieff_feature_importances = relieff_feature_importances
        self.date = time.strftime("%Y/%m/%d-%H:%M:%S")

        self.cpu_info = get_processor_name()
        self.time_of_computation = time_of_computation

    def __key(self):
        return tuple([self.dataset_sha256,
                     self.n_neighbors,
                     str(self.relieff_top_features),
                     str(self.relieff_feature_importances)])

    def __hash__(self):
        return hash(self.__key())

    def __str__(self):
        res = '{date: '+self.date+', n_neighbors:'+str(self.n_neighbors)
        
        if self.PRINT_HASH:
            res += 'dataset_sha256:'+str(self.dataset_sha256)+','

        res += 'cpu_info:'+str(self.cpu_info)+','
        res += 'time_of_computation:'+str(self.time_of_computation)+','
        res += '}'
        return res


def relieff_selection(X:list,y:list,n_features,n_neighbors,relief_cache_filepath,verbose_ = True):
    relief_data_X = X
    relief_data_y = y
    relief_top_features_ = None
    relief_feature_importances_ = None
    # First check if result is already cached
    ## Load Cache
    relief_cache = None

    ##----------------------------------------------##
    if not os.path.exists(relief_cache_filepath):
        raise Exception("RELIEF CACHE NOT FOUND at '"+relief_cache_filepath+"'! Comment exception to create empty cache")
        with open(relief_cache_filepath, 'wb') as rcf:
            pickle.dump(set(), rcf)
    ##----------------------------------------------##

    with open(relief_cache_filepath,'rb') as rcf:
        relief_cache = pickle.load(rcf)
        if verbose_: 
            print('Loaded Relief cache ('+str(len(relief_cache))+' solutions)')
    # Check if present
    for cache_elem in relief_cache:
        if cache_elem.dataset_sha256 == dataset_sha256_hash and           cache_elem.n_neighbors == n_neighbors:
            if verbose_:
                print("Result found in cache!")
            return cache_elem.relieff_top_features[:n_features]
    
    # If not present, compute
    if verbose_:
        print("Result NOT found in cache, computing now... (might take a long while)")
    
    from skrebate import ReliefF
    r = ReliefF(n_neighbors=n_neighbors,verbose=verbose_)
    
    start_fit = time.time()
    r.fit(X=relief_data_X,y=relief_data_y)
    top_features = r.top_features_
    feature_importances = r.feature_importances_
    stop_fit = time.time()

    if verbose_:
        print("Done. Now storing in cache...")

    savedata = ReliefCacheElem(
        dataset_sha256 = dataset_sha256_hash,
        n_neighbors = n_neighbors,
        relieff_top_features = top_features,
        relieff_feature_importances = feature_importances,
        time_of_computation = stop_fit - start_fit)
    relief_cache.add(savedata)
    with open(relief_cache_filepath, 'wb') as rcf:
        pickle.dump(relief_cache, rcf)

    if verbose_:
        print("Done.")


    return top_features[:n_features]


# In[187]:


with open(RELIEF_CACHE_FILEPATH,'rb') as rcf:
    relief_cache = pickle.load(rcf)
    
    print(len(relief_cache),'cached relief runs:')

    if len(relief_cache) != 0:
        samedataset = [e for e in relief_cache if e.dataset_sha256 == dataset_sha256_hash]
        print('('+str(len(samedataset))+'/'+str(len(relief_cache)), 'are from the same dataset)')
        if len(samedataset) != len(relief_cache):
            raise Exception('Some of the cached results are from a different dataset!')

        for i,e in enumerate(relief_cache):
            print(i,':',e)


# In[188]:


from enum import Enum
class FeatureSelection(Enum):
    MANUAL_VARIABLES = 1
    MANUAL_LIST = 2
    AUTO_ANOVA = 3      # https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.f_classif.html
    AUTO_RELIEF = 4


# In[189]:


# ------------------------------------------------------------------------------------------------------------------------------- #
#
# FEATURE_SELECTION = FeatureSelection.MANUAL_VARIABLES
# FEATURE_SELECTION = FeatureSelection.MANUAL_LIST
# FEATURE_SELECTION = FeatureSelection.AUTO_ANOVA
FEATURE_SELECTION = FeatureSelection.AUTO_RELIEF
AUTO_FEATURE_NUMBER = args['features']    # If FEATURE_SELECTION is AUTO_ANOVA or AUTO_RELIEF, select this number of features automatically
#
# ------------------------------------------------------------------------------------------------------------------------------- #

if FEATURE_SELECTION == FeatureSelection.MANUAL_VARIABLES:
    ''' Features '''
    USE_ATTACKTIME_VALUE = True
    USE_BARKSPECBRIGHTNESS = True
    USE_PEAKSAMPLE_VALUE = True
    USE_ZEROCROSSING = False

    USE_BARKSPEC = 40 # Number in range [0-50]
    USE_BFCC = 40     # Number in range [0-50]
    USE_CEPSTRUM = 60 # Number in range [0-353]
    USE_MFCC = 30     # Number in range [0-38]

    selected_features = get_manual_selected_features(original_dataset_features)
elif FEATURE_SELECTION == FeatureSelection.MANUAL_LIST:
    selected_features = ['attackTime_value', 'barkSpecBrightness', 'barkSpec_1', 'barkSpec_2', 'barkSpec_3', 'barkSpec_4', 'barkSpec_5', 'barkSpec_6', 'barkSpec_7', 'barkSpec_8', 'barkSpec_9', 'barkSpec_10', 'barkSpec_11', 'barkSpec_12', 'barkSpec_13', 'barkSpec_14', 'barkSpec_15', 'barkSpec_16', 'barkSpec_17', 'barkSpec_18', 'barkSpec_19', 'barkSpec_20', 'barkSpec_21', 'barkSpec_22', 'barkSpec_23', 'barkSpec_24', 'barkSpec_25', 'barkSpec_26', 'barkSpec_27', 'barkSpec_28', 'barkSpec_29', 'barkSpec_30', 'barkSpec_31', 'barkSpec_32', 'barkSpec_33', 'barkSpec_34', 'barkSpec_35', 'barkSpec_36', 'barkSpec_37', 'barkSpec_38', 'barkSpec_39', 'barkSpec_40', 'barkSpec_41', 'barkSpec_42', 'barkSpec_43', 'barkSpec_44', 'barkSpec_45', 'barkSpec_46', 'barkSpec_47', 'barkSpec_48', 'barkSpec_49', 'barkSpec_50', 'bfcc_2', 'bfcc_3', 'bfcc_4', 'bfcc_5', 'bfcc_6', 'bfcc_7', 'bfcc_8', 'bfcc_9', 'bfcc_10', 'bfcc_11', 'bfcc_12', 'bfcc_13', 'bfcc_15', 'bfcc_16', 'bfcc_17', 'bfcc_18', 'bfcc_19', 'bfcc_20', 'bfcc_21', 'bfcc_25', 'bfcc_26', 'bfcc_27', 'bfcc_28', 'bfcc_29', 'bfcc_30', 'bfcc_31', 'bfcc_35', 'bfcc_36', 'bfcc_37', 'bfcc_39', 'bfcc_40', 'bfcc_42', 'bfcc_43', 'bfcc_44', 'bfcc_45', 'bfcc_46', 'bfcc_48', 'cepstrum_1', 'cepstrum_2', 'cepstrum_3', 'cepstrum_4', 'cepstrum_5', 'cepstrum_6', 'cepstrum_7', 'cepstrum_8', 'cepstrum_9', 'cepstrum_10', 'cepstrum_11', 'cepstrum_12', 'cepstrum_13', 'cepstrum_14', 'cepstrum_15', 'cepstrum_16', 'cepstrum_17', 'cepstrum_18', 'cepstrum_19', 'cepstrum_20', 'cepstrum_21', 'cepstrum_22', 'cepstrum_23', 'cepstrum_24', 'cepstrum_25', 'cepstrum_26', 'cepstrum_27', 'cepstrum_28', 'cepstrum_29', 'cepstrum_30', 'cepstrum_31', 'cepstrum_32', 'cepstrum_33', 'cepstrum_34', 'cepstrum_35', 'cepstrum_36', 'cepstrum_37', 'cepstrum_41', 'cepstrum_42', 'cepstrum_43', 'cepstrum_44', 'cepstrum_45', 'cepstrum_46', 'cepstrum_47', 'cepstrum_48', 'cepstrum_49', 'cepstrum_54', 'cepstrum_56', 'cepstrum_59', 'cepstrum_60', 'cepstrum_67', 'cepstrum_72', 'cepstrum_86', 'cepstrum_87', 'cepstrum_108', 'cepstrum_164', 'cepstrum_205', 'cepstrum_206', 'mfcc_1', 'mfcc_2', 'mfcc_3', 'mfcc_4', 'mfcc_5', 'mfcc_6', 'mfcc_7', 'mfcc_8', 'mfcc_9', 'mfcc_10', 'mfcc_11', 'mfcc_12', 'mfcc_13', 'mfcc_14', 'mfcc_15', 'mfcc_16', 'mfcc_17', 'mfcc_18', 'mfcc_19', 'mfcc_20', 'mfcc_21', 'mfcc_22', 'mfcc_23', 'mfcc_24', 'mfcc_25', 'mfcc_26', 'mfcc_32', 'mfcc_33', 'mfcc_34', 'mfcc_35', 'mfcc_36', 'peakSample_value', 'zeroCrossing']
elif FEATURE_SELECTION == FeatureSelection.AUTO_ANOVA:
    if original_dataset_features.shape[1] != original_feature_number:
        raise ValueError("ERROR: please import dataset again since you are trying to subset an already processed feature set ("+str(dataset_features.shape[1])+"<"+str(original_feature_number)+")")

    # ANOVA feature selection for numeric input and categorical output (https://machinelearningmastery.com/feature-selection-with-real-and-categorical-data/#:~:text=Feature%20selection%20is%20the%20process,the%20performance%20of%20the%20model)
    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import f_classif
    
    fs = SelectKBest(score_func=f_classif, k=AUTO_FEATURE_NUMBER) # Define feature selection
    X_selected = fs.fit_transform(original_dataset_features.to_numpy(), dataset_labels.to_numpy().ravel())                         # Apply feature selection
    support = fs.get_support(indices=True)                      # Extract selected indexes
    selected_features = original_dataset_features.columns[support].tolist()
    print(str(AUTO_FEATURE_NUMBER)+" best features:" + str(selected_features))
elif FEATURE_SELECTION == FeatureSelection.AUTO_RELIEF:
    
    support = relieff_selection(relief_data_X,relief_data_y,AUTO_FEATURE_NUMBER,n_neighbors=5,relief_cache_filepath=RELIEF_CACHE_FILEPATH,verbose_= True)
    selected_features = original_dataset_features.columns[support].tolist()
    print(str(AUTO_FEATURE_NUMBER)+" best features:" + str(selected_features))
    

else:
    raise Exception("ERROR! This type of feature selection is not supported")

dataset_features = original_dataset_features.copy().loc[:,selected_features]
print("Features reduced "+('manually' if (FEATURE_SELECTION == FeatureSelection.MANUAL_LIST or FEATURE_SELECTION == FeatureSelection.MANUAL_VARIABLES) else 'automatically')+" ("+str(FEATURE_SELECTION)+") from "+str(original_feature_number)+" to : "+str(dataset_features.shape[1]))


# ## Evaluate class support
# (What percentage of dataset entries represent each class)

# In[190]:


DO_PRINT_SUPPORT = False
def printSupport (labels_ds):
    binc = np.bincount(np.reshape(labels_ds,labels_ds.size))
    for i in range(binc.size):
        print("Class " + str(i) + " support: " + str("{:.2f}".format(binc[i]/sum(binc) * 100)) + "%")
        
if DO_PRINT_SUPPORT:
    printSupport(dataset_labels.to_numpy())


# # Define model architecture

# In[191]:


def define_model_architecture(_verbose = False):
    tf.keras.backend.set_floatx('float32')

    net_width = args['net_width']

    dropout_rate = args['dropout']

    # sequential_structure = [tf.keras.Input(shape=(args['features'],))]
    sequential_structure = []

    for i in range(0,args['net_depth']):
        sequential_structure += [tf.keras.layers.Dense(net_width,activation='relu',
                                                       kernel_initializer='he_uniform'), #   X   |           |         |
                                 tf.keras.layers.BatchNormalization(),                   #       |     X     |         |
                                 tf.keras.layers.Dropout(dropout_rate)                   #       |           |    X    |
                                ]

    sequential_structure += [tf.keras.layers.Dense(net_width,activation='relu',
                                                   kernel_initializer='he_uniform'),     #   X   |           |         |
                                                   tf.keras.layers.Dense(8)]               

    model = tf.keras.models.Sequential(sequential_structure)

    timestr = time.strftime("%Y%m%d-%H%M%S")
    model._name = "guitar_timbre_classifier_" + timestr
    if _verbose:
        print("Created model: '" + model.name + "'")

    return model

def get_loss():
    return tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)


# 
# ### Define optimizer and compile model

# In[192]:


def compile_model(model,optimizer,loss_fn,_verbose = False):
    opt = None
    if optimizer["method"] == "sgd":
        opt = keras.optimizers.SGD(learning_rate = optimizer["learning_rate"], momentum=optimizer["momentum"])
    elif optimizer["method"] == "adam":
        opt = keras.optimizers.Adam(learning_rate = optimizer["learning_rate"])
    else:
        raise Exception("Optimizer method not supported")

    model.compile(optimizer=opt,
                  loss=loss_fn,
                  metrics=['accuracy'])
    
    if _verbose:
        print("Model compiled")


# # Save Models and Info functions

# In[193]:


def save_model_info(model,optimizer,final_cross_validation_results,folds,metrics,outpath, fold_zerobased = None):
    info_filename = '/info.txt' if fold_zerobased is None else '/info_fold_'+str(fold_zerobased+1)+'.txt'
    assert not (final_cross_validation_results and (fold_zerobased is not None))

    with open(outpath + info_filename, "w") as f:
        if not is_notebook():
            f.write('Execution command:\n')
            f.write(" ".join(sys.argv[:])+'\n')
        else:
            f.write('Trained with the jupyter notebook (not the script version)\n')
        f.write("\n\n")

        if fold_zerobased is not None:
            f.write("FOLD ["+str(fold_zerobased+1)+"/"+str(folds)+"]\n\n")
        f.write("Summary:\n")
        model.summary(print_fn=lambda x: f.write(x + '\n'))
        f.write("\n\n")
        f.write("+--| Features: \n")
        f.write('Number of features selected: '+str(len(selected_features))+'\n')
        f.write('Selected features: '+str(selected_features)+'\n')
        f.write('Feature Selection method: '+str(FEATURE_SELECTION)+'\n')
        f.write("\n\n")
        f.write('Run arguments: '+str(args)+'\n')
        f.write("\n\n")
        f.write("Optimizer: " + optimizer["method"])
        if optimizer["method"] == "sgd":
            f.write(" lr: " + str(optimizer["learning_rate"]) + " momentum: " + str(optimizer["momentum"]))
        elif optimizer["method"] == "adam":
            f.write(" lr: " + str(optimizer["learning_rate"]))
        else:
            assert(False) # If triggered check new optimizer and add case
        f.write("\n\n")
        if final_cross_validation_results:
            f.write("Trained for " + str(args['epochs']) + " and with_batch size '" + str(args['batchsize']) + "'" + " epochs for each fold ("+str(folds)+"-foldCrossValidation)\n")
            f.write("Single results in the folds directories\n")
            f.write('\n\n-------- Average results --------\n\n')
        else:
            f.write("Trained for " + str(args['epochs']) + " and with_batch size '" + str(args['batchsize']) + "'" + " epochs\n")

            if fold_zerobased is not None:
                f.write('(K-Fold cross validation run (fold '+str(fold_zerobased+1)+'/' +str(folds)+ '))\n')
            else:
                f.write('(Single run, NO k-fold cross validation)\n')

        for metric in metrics.keys():
            value = metrics[metric] if fold_zerobased is None else metrics[metric][fold_zerobased]
            f.write(str(metric) + ":\n" + str(value) + "\n\n")
        f.close()

    # Copy Tensorboard Logs
    if fold_zerobased == None and DO_SAVE_TENSORBOARD_LOGS:
        LOGPATH=outpath+"/tensorboardlogs"
        shutil.copytree('./logs', LOGPATH)

    if not COLAB and fold_zerobased == None:
        # Copy script or notebook depending on the execution environment
        script_path = None
        if is_notebook():
            script_path = 'expressive-technique-classifier-phase3.ipynb'
            pass #TODO: make this work
        else:
            script_path = os.path.realpath(__file__)
        shutil.copyfile(script_path, os.path.join(outpath, 'backup_'+os.path.basename(script_path)))


# # Prepare Logs

# In[194]:


if os.path.exists('./logs'):
    shutil.rmtree('./logs', ignore_errors=True) #Clear logs if necessary


# In[195]:


def start_tensorboard(tb_dir,logname):
    log_dir = tb_dir
    if logname is not None: 
        log_dir += logname
    return tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
tb_dir = "logs/fit/"
#<redacted ipython line>get_ipython().run_line_magic('tensorboard', '--logdir $tb_dir')
None


# In[196]:


import matplotlib.pyplot as plt

def plot_history(train_metric, validation_metric, title, xlabel, ylabel, filename=None, show = False):
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.plot(train_metric)
    ax.plot(validation_metric)
    ax.legend(['Training','Validation'])
    if show:
        fig.show()
    if filename is not None:
        plt.savefig(filename+".pdf",bbox_inches='tight')


# F1-Score on Test dataset

# In[197]:


def macroweighted_f1(y_true,y_pred):
    f1scores = []
    numSamples = []
    for selclass in CLASSES:
        classSelection = (y_true == (np.ones(np.shape(y_true)[0])*selclass))
        numSamples.append(sum(classSelection))
        classPrediction = (y_pred == (np.ones(np.shape(y_true)[0])*selclass))
        true_positives = np.sum(np.logical_and(classSelection,(y_true == y_pred)))

        precision = 1.0 * true_positives / np.sum(classPrediction)
        recall = 1.0 * true_positives / np.sum(classSelection)
        f1score = 2 /((1/precision)+(1/recall))
        f1scores.append(f1score)
    macroWeightedF1 = sum(np.array(f1scores) * np.array(numSamples)) / sum(numSamples)
    return macroWeightedF1


# In[198]:


def compute_metrics(y_true, y_pred,_verbose = False):
    accuracy = np.sum(y_pred == y_true)/np.shape(y_true)[0]
    f1mw = macroweighted_f1(y_true,y_pred)
    confusion_matrix = metrics.confusion_matrix(y_true, y_pred)
    
    classification_report = metrics.classification_report(y_true, y_pred, digits=6,target_names = CLASSES_DESC.values(),output_dict=True)
    printable_classification_report = metrics.classification_report(y_true, y_pred, digits=4,target_names = CLASSES_DESC.values())

    if _verbose:
        print("Test Accuracy: " + str(accuracy) + "\nTest macro_weighted_avg f1-score: " + str(f1mw)+'\n'+str(confusion_matrix)+'\n'+str(printable_classification_report))

    return accuracy, f1mw, confusion_matrix, classification_report, printable_classification_report


# # Prepare TFLite conversion and evaluation

# In[199]:


# TFLite conversion function
def convert2tflite(tf_model_dir,tflite_model_dir = None,model_name="model",quantization=None,dataset=None):
    assert (quantization==None or quantization=="dynamic" or quantization=="float-fallback" or quantization=="full")
    # Convert the model saved in the previous step.
    converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_dir)
    if quantization is not None:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        if quantization == "full" or quantization=="float-fallback":
            assert dataset is not None
            def representative_dataset():
                for data in tf.data.Dataset.from_tensor_slices((dataset)).batch(1).take(100):
                    yield [tf.dtypes.cast(data, tf.float32)]
            converter.representative_dataset = representative_dataset
        if quantization == "full":
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8  # or tf.uint8
            converter.inference_output_type = tf.int8  # or tf.uint8
        if quantization == "dynamic":
            assert dataset is None

    tflite_model = converter.convert()

    # Save the TF Lite model.
    if tflite_model_dir is None:
        TF_MODEL_PATH = tf_model_dir + "/" + model_name + '.tflite'
    else:
        TF_MODEL_PATH = tflite_model_dir + "/" + model_name + '.tflite'

    with tf.io.gfile.GFile(TF_MODEL_PATH, 'wb') as f:
        f.write(tflite_model)

## USAGE
# model_path = MODELFOLDER + "/" + RUN_NAME + "/fold_1"
# convert2tflite(model_path)


# In[200]:


def test_tflite_model(model_path,X_test,y_test,first_layer_is_conv,verbose_test = False):
    tflite_interpreter = tf.lite.Interpreter(model_path=model_path)
    input_details = tflite_interpreter.get_input_details()[0]
    output_details = tflite_interpreter.get_output_details()[0]
    
    if verbose_test:
        print("+--------------------------------------------+\n| Testing the TF lite model saved            |\n+--------------------------------------------+\n")
        print("[Model loaded]\n")
        print("\n== Input details ==\nname:"+ str(input_details['name']) + "\nshape:"+str(input_details['shape']) +  "\ntype:"+str(input_details['dtype']))
        print("\n== Output details ==\nname:"+str(output_details['name']) + "\nshape:"+str(output_details['shape']) + "\ntype:"+str(output_details['dtype']))
        print("+--------------------------------------------+\n| Testing on TEST set...                     |\n+--------------------------------------------+\n")
    
    tflite_interpreter.allocate_tensors()
    y_pred = list()
    for i in range(X_test.shape[0]):
        extracted_test_sample = np.array(X_test[i:i+1]).astype(np.float32)
        
        # Quantize inputs if necessary (full uint model)
        if input_details['dtype'] is np.int8:
            input_scale, input_zero_point = input_details["quantization"]
            extracted_test_sample = (extracted_test_sample / input_scale + input_zero_point).astype(np.int8)

        if first_layer_is_conv:
            input_tensor = np.expand_dims(extracted_test_sample,axis=2).astype(input_details["dtype"])
        else:
            input_tensor = extracted_test_sample

        if verbose_test:
            print("Setting "+str(input_tensor.shape)+" "+str(input_tensor.dtype)+" as input")

        tflite_interpreter.set_tensor(input_details['index'], input_tensor)
        tflite_interpreter.invoke()
        prediction_vec = tflite_interpreter.get_tensor(output_details['index'])

        if verbose_test:
            print("Getting "+str(prediction_vec.shape)+" "+str(prediction_vec.dtype)+" as output")

        if output_details['dtype'] is np.int8:
            output_scale, output_zero_point = output_details["quantization"]
            prediction_vec = (prediction_vec + output_zero_point) * output_scale

        if verbose_test:
            print(prediction_vec)
        y_pred.append(np.argmax(prediction_vec))
    return y_pred

def test_regulartf_model(model_path,X_test,y_test,first_layer_is_conv,verbose_test = False):
    imported = tf.keras.models.load_model(model_path)
    if first_layer_is_conv:
        test_set = np.expand_dims(X_test,axis=2)
    else:
        test_set = X_test
    _, accuracy = imported.evaluate(test_set,  y_test, verbose=2)
    return accuracy


# # k-Fold Cross Validation
# 

# In[201]:


# --> Epochs / Batches
print('Using training epochs: ', args['epochs'])
print('Using batch size: ', args['batchsize'])

# --> Quantize (Dynamic) and test the TF Lite model obtained (quicker but lower accuracy)
TEST_QUANTIZATION = True
# --> Early Stopping
use_early_stopping = True

# --> OVERSAMPLING ##############################################
DO_OVERSAMPLING = args['oversampling']                          #
                                                                #
unique, counts = np.unique(dataset_labels, return_counts=True)  #
smote_mask = dict(zip(unique, counts))                          #
                                                                #
smote_mask[4] = int(smote_mask[5] * 2.5)                        #
#################################################################

# --> KFOLD RUN #################################################
K_SPLITS = args['k_folds']
USE_CROSS_VALIDATION = K_SPLITS > 1 # Activate K-Fold Cross Validation only if K_SPLITS > 1
val_split_size = 0.1                                            # percentage of total entries going into the validation set
random_state = global_random_state                              # seed for pseudo random generator
#################################################################

# --> SINGLE RUN ################################################
SAVE_MODEL_INFO = True                                          #
test_split_size = 0.2                                           #
#################################################################

DO_TEST = False

# optimizer = { "method" : "sgd", "learning_rate" : args['learning_rate'], "momentum" : 0.7 }
optimizer = { "method" : "adam", "learning_rate" : args['learning_rate'] }


# In[202]:


prefix = "CrossValidated" if USE_CROSS_VALIDATION else "Single"
RUN_NAME = prefix + "Run_"+time.strftime("%Y%m%d-%H%M%S")
cv = StratifiedKFold(n_splits=K_SPLITS,shuffle=True,random_state=random_state)

def main_routine(X,y,train_idx=None,test_idx=None,foldcount=None,is_k_fold=False, eval_metrics=None, quantized_eval_metrics=None):
    if eval_metrics is None:
        raise ValueError("provide a eval_metrics dict")

    current_dir = MODELFOLDER + "/" + RUN_NAME
    # %mkdir -p "$current_dir"
    os.makedirs(current_dir,exist_ok = True) 
    if is_k_fold:
        fold_dir = current_dir + '/Fold_' + str(foldcount)
        # %mkdir -p "$fold_dir"
        os.makedirs(fold_dir,exist_ok = True)

    ### PRINT INFO
    if is_k_fold:
        print("\nFold ["+str(foldcount)+"/"+str(K_SPLITS)+"]")
        #### SPLIT DATA
        print("Selecting Split Data...")
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        # printSupport(y_train)                                                       # Verify that the split is STRATIFIED
    else:
        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=test_split_size,random_state=random_state, shuffle=True, stratify = y)
        # printSupport(y_train)                                                       # Verify that the split is STRATIFIED
        
    X_train,X_valid,y_train,y_valid = train_test_split(X_train,y_train,test_size=val_split_size,random_state=random_state, shuffle=True, stratify = y_train)
    # printSupport(y_train)                                                       # Verify that the split is STRATIFIED

    ### OVERSAMPLE
    if DO_OVERSAMPLING:
        print("Oversampling...")
        VERBOSE_OVERSAMPLING = True
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if VERBOSE_OVERSAMPLING:
                prev_len = y_train.shape[0]
            X_train, y_train = SMOTE(smote_mask).fit_sample(X_train, y_train)
            if VERBOSE_OVERSAMPLING:
                print("Increased training samples from " + str(prev_len) + " to " + str(y_train.shape[0]))
                printSupport(y_train)

    ### DEFINE MODEL
    model = define_model_architecture(_verbose = True)
    
    ### DEFINE LOSS
    loss_fn = get_loss()

    ### PREPARE DATA IN CASE OF A FIRST CONV LAYER IN THE NET
    if type(model.layers[0]) == tf.keras.layers.Conv1D:
        X_train = np.expand_dims(X_train,axis = 2) # Adapt data for Conv1d ([batch_shape, steps, input_dim] -> in our case indim = 1, steps = features, batchshape = train datset size)
        X_valid= np.expand_dims(X_valid,axis = 2)  # Adapt data for Conv1d
        X_test= np.expand_dims(X_test,axis = 2)    # Adapt data for Conv1d

    ### PERFORM TEST (**OPTIONAL)
    if DO_TEST:
        predictions = model(X_test[:1].astype('float32')).numpy()
        print("Predictions: " + str(predictions) + "\nWith Softmax: " + str(tf.nn.softmax(predictions).numpy()) + "\nLoss: " + str(loss_fn(y_test[:1], predictions).numpy()))

    ### COMPILE MODEL
    compile_model(model,optimizer,loss_fn,_verbose = True)

    ### SETUP TENSORBOARD
    tensorboard_callback = start_tensorboard(tb_dir,"Fold_"+str(foldcount) if is_k_fold else None)
    callbacks=[tensorboard_callback,]
    
    ### SETUP EARLY STOPPING (only if not in K-fold mode)
    if is_k_fold is False and use_early_stopping:
        callbacks.append(tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=200))

    # * FIT MODEL *
    history = model.fit(X_train, y_train, epochs=args['epochs'], validation_data = (X_valid,y_valid),
                        callbacks=callbacks,
                        batch_size=args['batchsize'])
    # Plot history
    plot_folder = fold_dir if is_k_fold else current_dir
    getval = lambda metric: history.history[metric]
    plot_history(getval('accuracy'),getval('val_accuracy'), "Training and validation accuracy","Epochs","Accuracy", filename = plot_folder + "/AccuracyPlot")
    plot_history(getval('loss'),    getval('val_loss'),     "Training and validation loss","Epochs","Accuracy",     filename = plot_folder + "/LossPlot")
    plt.close()
    plt.ioff()

    # * TEST MODEL *
    # keras_test_loss, keras_test_accuracy = model.evaluate(X_test,  y_test, verbose=2) # Keras solution, might not be needed
    y_true = np.squeeze(y_test)
    y_pred = np.argmax(model(X_test),axis=1)
    cm_acc, f1mw, cm_conf_matrix, cm_classf_report, cm_printable_classf_report = compute_metrics(y_true,                                                                                                  y_pred,                                                                                                  _verbose=True)
    eval_metrics["accuracy"].append(cm_acc)
    eval_metrics["f1_weightedmacroavg"].append(f1mw)
    eval_metrics["confusion_matrix"].append(cm_conf_matrix)
    eval_metrics["classification_report"].append(cm_classf_report)
    eval_metrics["printable_classification_report"].append(cm_printable_classf_report)

    SAVED_MODEL_PATH = None
    if is_k_fold:
        # Save fold history
        with open(fold_dir+"/history_fold_"+str(foldcount)+".pickle",'wb') as picklefile:
            pickle.dump(history.history,picklefile)

        # Save the fold models only if we want them, or need them to test (in the second case they will be deleted after test)        
        if TEST_QUANTIZATION or DO_SAVE_FOLD_MODELS:
            model.save(fold_dir)
            SAVED_MODEL_PATH = fold_dir
    else:
        assert len(eval_metrics['accuracy']) == 1
        
        # Save the entire model as a SavedModel.
        if TEST_QUANTIZATION or DO_SAVE_FOLD_MODELS:
            model.save(current_dir)
            SAVED_MODEL_PATH = current_dir
        with open(current_dir+"/history.pickle",'wb') as picklefile:
            pickle.dump(history.history,picklefile)

    if TEST_QUANTIZATION:
        assert quantized_eval_metrics is not None
        model_filename = 'partially_quantized_test_model'
        # Convert and save lite model
        convert2tflite(SAVED_MODEL_PATH,model_name=model_filename,quantization="dynamic")
        # Load and Test lite model
        y_quant_pred = test_tflite_model(SAVED_MODEL_PATH+'/'+model_filename+'.tflite',X_test,y_test,type(model.layers[0]) == tf.keras.layers.Conv1D,verbose_test = False)
        # Compute Test Metrics
        cm_acc, f1mw, cm_conf_matrix, cm_classf_report, cm_printable_classf_report = compute_metrics(y_true,                                                                                                      y_quant_pred,                                                                                                      _verbose=True)
        quantized_eval_metrics["accuracy"].append(cm_acc)
        quantized_eval_metrics["f1_weightedmacroavg"].append(f1mw)
        quantized_eval_metrics["confusion_matrix"].append(cm_conf_matrix)
        quantized_eval_metrics["classification_report"].append(cm_classf_report)
        quantized_eval_metrics["printable_classification_report"].append(cm_printable_classf_report)

    if not DO_SAVE_FOLD_MODELS:
        shutil.rmtree(os.path.join(fold_dir,'assets'))
        shutil.rmtree(os.path.join(fold_dir,'variables'))
        if os.path.exists(os.path.join(fold_dir,'savedmodel.pb')):
            os.remove(os.path.join(fold_dir,'savedmodel.pb'))
        if os.path.exists(os.path.join(fold_dir,'keras_metadata.pb')):
            os.remove(os.path.join(fold_dir,'keras_metadata.pb'))
        if TEST_QUANTIZATION and os.path.exists(os.path.join(fold_dir,'partially_quantized_test_model.tflite')):
            os.remove(os.path.join(fold_dir,'partially_quantized_test_model.tflite'))

    # Now that all the tests are performed, all the info can be saved

    if is_k_fold:
        metrics_to_save = {}
        metrics_to_save.update({'def_model_'+key:value for (key,value) in eval_metrics.items()})
        if TEST_QUANTIZATION:
            metrics_to_save.update({'quant_model_'+key:value for (key,value) in quantized_eval_metrics.items()})

        # save_fold_info(model,optimizer,foldcount,K_SPLITS,X_test.shape[0],eval_metrics,list(dataset_features.columns),fold_dir)
        save_model_info(model,
                        optimizer,
                        False,K_SPLITS,
                        metrics_to_save,
                        fold_dir,
                        fold_zerobased=foldcount-1)

    return model


# In[203]:


'''Call the main routine for each fold'''
result_model = []

with tf.device('/gpu:0'):

    evaluation_metrics = { "accuracy" : [], "f1_weightedmacroavg" : [], "confusion_matrix" : [],"classification_report" : [],"printable_classification_report" : [] }
    quantized_model_evaluation_metrics = { "accuracy" : [], "f1_weightedmacroavg" : [], "confusion_matrix" : [],"classification_report" : [],"printable_classification_report" : [] }

    X = dataset_features.to_numpy()
    y = dataset_labels.to_numpy()

    if USE_CROSS_VALIDATION:
        foldcount = 1
        for train_idx, test_idx in cv.split(X, y):
            result_model.append(main_routine(dataset_features.to_numpy(),                                 dataset_labels.to_numpy(),                                 train_idx, test_idx,                                 foldcount,                                  USE_CROSS_VALIDATION,                                 evaluation_metrics,                                  quantized_eval_metrics = quantized_model_evaluation_metrics))
            foldcount += 1
    else:
        result_model = main_routine(X,                                     y,                                     eval_metrics = evaluation_metrics,                                     quantized_eval_metrics = quantized_model_evaluation_metrics)


# # Cross Validation average results

# ## Utilities for reports and metrics

# In[204]:


def report_average(reports):
    mean_dict = dict()
    for label in reports[0].keys():
        dictionary = dict()

        if label in 'accuracy':
            mean_dict[label] = sum(d[label] for d in reports) / len(reports)
            continue

        for key in reports[0][label].keys():
            dictionary[key] = sum(d[label][key] for d in reports) / len(reports)
        mean_dict[label] = dictionary

    return mean_dict

def classification_report_dict2print(report):
    ret = ""
    classes = list(report.keys())[0:-3]
    summary_metrics = list(report.keys())[-3:]
    longest_1st_column_name = max([len(key) for key in report.keys()])
    ret = ' ' * longest_1st_column_name
    ret += '  precision    recall  f1-score   support\n\n'

    METRIC_DECIMAL_DIGITS = 4
    metric_digits = METRIC_DECIMAL_DIGITS + 2 # add 0 and dot

    header_spacing = 1
    metrics = list(report[classes[0]].keys())
    longest_1st_row_name = max([len(key) for key in report[classes[0]].keys()]) + header_spacing

    for classname in classes:
        ret += (' '*(longest_1st_column_name-len(classname))) + classname + ' '
        for metric in metrics:
            if metric != "support":
                ret += (' '*(longest_1st_row_name-metric_digits))
                ret += "%.4f" % round(report[classname][metric],METRIC_DECIMAL_DIGITS)
            else:
                current_support_digits = len(str(int(report[classname][metric])))
                ret += (' '*(longest_1st_row_name-current_support_digits))
                ret += "%d" % round(report[classname][metric],0)
        ret += '\n'
    ret += '\n'

    # Accuracy
    ret += (' '*(longest_1st_column_name-len(summary_metrics[0]))) + summary_metrics[0] + ' '
    ret += 2* (' '*longest_1st_row_name)
    ret += (' '*(longest_1st_row_name-metric_digits))
    ret += "%.4f" % round(report["accuracy"],METRIC_DECIMAL_DIGITS)
    current_support_digits = len(str(int(report[summary_metrics[-1]]['support'])))
    ret += (' '*(longest_1st_row_name-current_support_digits))
    ret += "%d" % round(report[summary_metrics[-1]]['support'],0)
    ret += '\n'
  
  
    for classname in summary_metrics[1:]:
        ret += (' '*(longest_1st_column_name-len(classname))) + classname + ' '
        for metric in metrics:
            if metric != "support":
                ret += (' '*(longest_1st_row_name-metric_digits))
                ret += "%.4f" % round(report[classname][metric],METRIC_DECIMAL_DIGITS)
            else:
                current_support_digits = len(str(int(report[classname][metric])))
                ret += (' '*(longest_1st_row_name-current_support_digits))
                ret += "%d" % round(report[classname][metric],0)
        ret += '\n'
    ret += '\n'

    return ret


# ## Compute average

# In[205]:


if USE_CROSS_VALIDATION:
    assert len(evaluation_metrics['accuracy']) == K_SPLITS
    
    printable_avg_report = classification_report_dict2print(report_average(evaluation_metrics["classification_report"]))
    qm_printable_avg_report = classification_report_dict2print(report_average(quantized_model_evaluation_metrics["classification_report"]))
    metrics_to_save = {"avg_classification_report" : printable_avg_report, "avg_classification_report_for_quantized_model" : qm_printable_avg_report}
else:
    assert len(evaluation_metrics['accuracy']) == 1
    metrics_to_save = {}
    for metric in evaluation_metrics.keys():
        metrics_to_save[metric] = evaluation_metrics[metric][0]
    for metric in quantized_model_evaluation_metrics.keys():
        metrics_to_save['quantizedmod_'+str(metric)] = quantized_model_evaluation_metrics[metric][0]


# # Save Model Info

# In[206]:


if SAVE_MODEL_INFO:
    current_dir = MODELFOLDER + "/" + RUN_NAME
    # %mkdir -p "$current_dir"
    os.makedirs(current_dir,exist_ok = True)

    save_model_info(result_model[0] if type(result_model) == list else result_model,
                    optimizer,
                    USE_CROSS_VALIDATION,K_SPLITS,
                    metrics_to_save,
                    current_dir)
else:
    print("RESULTS\n\n" + metrics_to_save)


# # Train final model on the entire dataset

# In[207]:


use_early_stopping = False

### DEFINE MODEL
final_model = define_model_architecture(_verbose = True)
loss_fn = get_loss()

### PREPARE DATA IN CASE OF A FIRST CONV LAYER IN THE NET
if type(final_model.layers[0]) == tf.keras.layers.Conv1D:
    X_all = np.expand_dims(X,axis = 2) # Adapt data for Conv1d ([batch_shape, steps, input_dim] -> in our case indim = 1, steps = features, batchshape = train datset size)
else:
    X_all = X

### COMPILE MODEL
compile_model(final_model,optimizer,loss_fn,_verbose = True)

### SETUP TENSORBOARD
tensorboard_callback = start_tensorboard(tb_dir,None)
callbacks=[tensorboard_callback,]

### SETUP EARLY STOPPING (only if not in K-fold mode)
if use_early_stopping:
    callbacks.append(tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', mode='min', verbose=1, patience=200))

# * FIT MODEL *
final_model.fit(X_all, y, epochs=args['epochs'],
                callbacks=callbacks,
                batch_size=args['batchsize'])


# In[208]:


final_model_dir = MODELFOLDER + "/" + RUN_NAME + "/finalModel"
# %mkdir -p "$final_model_dir"
os.makedirs(final_model_dir,exist_ok = True)

final_model.save(final_model_dir)

# Convert and save lite model (Non quantized)
convert2tflite(final_model_dir,model_name='final_model',quantization=None)
# Convert and save lite model (Dynamically quantized)
convert2tflite(final_model_dir,model_name='final_model_dynquant',quantization="dynamic")


# # Save the model for TF Lite
# ## *(Only if not a Cross Validated run)*

# In[209]:


# if USE_CROSS_VALIDATION is False:
#     model_path = MODELFOLDER + "/" + RUN_NAME
#     convert2tflite(model_path)                                                # standard TFLITE model
#     convert2tflite(model_path,model_name="model_partially_quantized",quantization="dynamic")   # Partial quantization  https://www.tensorflow.org/lite/performance/post_training_quantization#dynamic_range_quantization
    
#     quantization_dataset = X
#     if type(result_model.layers[0]) == tf.keras.layers.Conv1D:
#         quantization_dataset = np.expand_dims(X,axis = 2) # Adapt data for Conv1d
    
#     convert2tflite(model_path,model_name="model_float_fallback",quantization="float-fallback",dataset=quantization_dataset) # https://www.tensorflow.org/lite/performance/post_training_integer_quant#convert_using_float_fallback_quantization
#     convert2tflite(model_path,model_name="model_fully_quantized",quantization="full",dataset=quantization_dataset)          # FULL uint8 quantization https://www.tensorflow.org/lite/performance/post_training_quantization#full_integer_quantization


# In[210]:


# first_layer_is_conv = (type(result_model.layers[0]) == tf.keras.layers.Conv1D)


# In[211]:


# TEST_SAVED_MODEL = None
# # TEST_SAVED_MODEL = 'model.tflite'
# # TEST_SAVED_MODEL = 'model_partially_quantized.tflite'
# # TEST_SAVED_MODEL = 'model_float_fallback.tflite'
# # TEST_SAVED_MODEL = 'model_fully_quantized.tflite'
# # TEST_SAVED_MODEL = 'quant_aware_model.tflite'
# # TEST_SAVED_MODEL = 'saved_model.pb'
# verbose_test = False

# def test_generic_model(model_filename,model_path,X_test,Y_test,first_layer_is_conv,verbose_test = False):
#     if model_filename.split('.')[-1] == 'tflite':
#         y_pred = test_tflite_model(model_path+'/'+model_filename,X_test,y_test,first_layer_is_conv,verbose_test = verbose_test)
#         correct = np.count_nonzero((np.array(y_pred) == np.ravel(y_test)).astype(int))
#         total = np.shape(y_test)[0]
#         accuracy = round(correct/total,4)
#     elif model_filename.split('.')[-1] == 'pb':
#         accuracy = test_regulartf_model(model_path,X_test,y_test,first_layer_is_conv,verbose_test = verbose_test)
#     else:
#         raise ValueError("")

#     return accuracy

# if USE_CROSS_VALIDATION is False and TEST_SAVED_MODEL is not None:
#     assert np.max([len(ev_metric) for ev_metric in evaluation_metrics]) == K_SPLITS

#     target_accuracy = evaluation_metrics['accuracy'][0]
#     accuracy = test_generic_model(TEST_SAVED_MODEL,model_path,X_test,Y_test,first_layer_is_conv)

#     epsilon = 1e-4
#     EQUAL_ACCURACY = abs(target_accuracy - accuracy) < epsilon

#     print("accuracy: " + str(accuracy))

#     if EQUAL_ACCURACY:
#         print("Accuracy of the original model and the saved TF model correspond(on same test set)")
#     else:
#         raise ValueError('Accuracy does not match target (Target: '+str(target_accuracy)+' but got '+str(accuracy)+' instead)')
# else:
#     print("TF model testing is disabled")


# # Quantization aware fine-tuning

# In[212]:


#################################################
PERFORM_QUANZATION_AWARE_TRAINING = False       #
#################################################
if PERFORM_QUANZATION_AWARE_TRAINING:
    pip_install('tensorflow_model_optimization')


# In[213]:


if PERFORM_QUANZATION_AWARE_TRAINING:
    imported_model = tf.keras.models.load_model(model_path)

    import tensorflow_model_optimization as tfmot

    quantize_model = tfmot.quantization.keras.quantize_model

    # q_aware stands for for quantization aware.
    q_aware_model = None
    q_aware_model = quantize_model(imported_model)

    # `quantize_model` requires a recompile.
    _,loss_fn = define_model_architecture(_verbose = True)  # Get only the loss function
    compile_model(q_aware_model,optimizer,loss_fn,_verbose = True)  # Recompile the quantization aware model

    q_aware_model.summary()


# In[214]:


if PERFORM_QUANZATION_AWARE_TRAINING:
    tb_dir = "logs2/fit/"
#<redacted ipython line>    get_ipython().run_line_magic('tensorboard', '--logdir $tb_dir')
    None


# In[215]:


if PERFORM_QUANZATION_AWARE_TRAINING:
    finetuning_epochs = 50
    tensorboard_callback = start_tensorboard(tb_dir,None)

    q_history = q_aware_model.fit(X_train, y_train, epochs=finetuning_epochs, validation_data = (X_valid,y_valid),
                                callbacks=[tensorboard_callback],
                                batch_size=args['batchsize'])


# In[216]:


if PERFORM_QUANZATION_AWARE_TRAINING:
    quant_model_path = MODELFOLDER + "/" + RUN_NAME + "/quant_aware_model.tflite"

    converter = tf.lite.TFLiteConverter.from_keras_model(q_aware_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    quantized_tflite_model = converter.convert()

    with tf.io.gfile.GFile(quant_model_path, 'wb') as f:
        f.write(quantized_tflite_model)


# ## Rename all output folders by prefixing the accuracy value

# In[217]:


# from glob import glob
# import re

# for dirpattern, newprefix in [('SingleRun*','acc'),('CrossValidated*','c_acc')]:
#     for resdir in glob(os.path.join(MODELFOLDER,dirpattern)):
#         filesindir = glob(os.path.join(resdir,'*'))
#         if len(filesindir) == 0:
#             os.rename(resdir,os.path.join(MODELFOLDER,'todelete',os.path.basename(resdir)))
#         else:
#             if os.path.exists(os.path.join(resdir,'info.txt')):
#                 with open(os.path.join(resdir,'info.txt')) as infof:
#                     acclines = []
#                     for e in infof.readlines():
#                         acclines += re.findall('accuracy[ ]+0\.\d+',e)
#                     assert len(acclines) > 0
#                     accstr = re.findall('0\.\d+',acclines[0])
#                     assert len(accstr) == 1
#                     newfoldername = os.path.join(os.path.dirname(resdir),newprefix+accstr[0]+'_'+os.path.basename(resdir))
#                     os.rename(resdir,newfoldername)


# ## Rename current output folder by prefixing the accuracy value

# In[220]:


run_dir = os.path.join(MODELFOLDER,RUN_NAME)
assert os.path.exists(run_dir)

if os.path.exists(os.path.join(run_dir,'info.txt')):
    with open(os.path.join(run_dir,'info.txt')) as infof:
        acclines = []
        for e in infof.readlines():
            acclines += re.findall('accuracy[ ]+0\.\d+',e)
        assert len(acclines) > 0
        accstr = re.findall('0\.\d+',acclines[0])
        assert len(accstr) == 1
        if 'CrossValidated' in os.path.basename(run_dir):
            newprefix = 'c_acc'
        elif 'SingleRun' in os.path.basename(run_dir):
            newprefix = 'acc'
        else:
            raise Exception('Bad folder name '+str(os.path.basename(run_dir)))
            
        newfoldername = os.path.join(os.path.dirname(run_dir),newprefix+accstr[0]+'_'+os.path.basename(run_dir))
        # print('Renaming "'+run_dir+'" to "'+newfoldername+'"')
        os.rename(run_dir,newfoldername)
else:
    errfoldername = os.path.join(os.path.dirname(run_dir),'ERR_'+os.path.basename(run_dir))
    os.rename(run_dir,errfoldername)

