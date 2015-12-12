import data_module as dm
import os
import numpy as np
import theano as th

# Variables (adapted from demo code of the paper)
rf_size = 6
step_size = 1
num_centroids=1600
whitening='true'
num_patches = 400000
CIFAR_DIM = 32 * 32 * 3

# Check if there already exists the extracted files
if not os.path.isdir(os.path.join(os.path.realpath(''), "cifar-10-batches-py")):
    dm.extract_file()

# Unpickle batches.meta file and save variables
meta = dm.unpickle(['batches.meta'])
label_names = meta[0]['label_names']
num_cases_per_batch = meta[0]['num_cases_per_batch']

# Unpickle data_batch files
data_batches = dm.unpickle(['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5'])

# load all train data
trainX, trainY = dm.load_train_data_all(data_batches)

# extract all patches
patches = dm.extract_all_patches(trainX, rf_size, step_size)

print("break")



