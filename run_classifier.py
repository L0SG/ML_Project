import data_module as dm
import os
import numpy as np
import theano as th
import learn_module as lm
import sklearn.cluster as cl
import sklearn.preprocessing as pp
from scipy.cluster.vq import whiten

# Variables (adapted from demo code of the paper)
rf_size = 6
step_size = 6
num_centroids=100
whitening = 1
num_patches = 10000
CIFAR_DIM = 32 * 32 * 3

# Check if there already exists the extracted files
if not os.path.isdir(os.path.join(os.path.realpath(''), "cifar-10-batches-py")):
    dm.extract_file()

# Unpickle batches.meta file and save variables
meta = dm.unpickle(['batches.meta'])
label_names = meta[0]['label_names']
num_cases_per_batch = meta[0]['num_cases_per_batch']

# Unpickle data_batch files
#data_batches = dm.unpickle(['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5'])
data_batches = dm.unpickle(['data_batch_1'])

# load all train data
trainX, trainY = dm.load_train_data_all(data_batches)

# extract all patches
patches = dm.extract_all_patches(trainX, rf_size, step_size, num_patches)


# standardize patches
print("normalizing patches....")
patches_normalized = pp.scale(patches)

# whitening
if whitening:
    whiten(patches_normalized)

# K-means clustering
print("clustering with kmeans...")
kmeans= cl.KMeans(num_centroids, n_init=1, max_iter=10)
kmeans_centroids = kmeans.fit(patches_normalized)


# extract feature vector using kmeans centroids
trainXC = lm.extract_features(trainX, kmeans_centroids, rf_size, step_size, whitening)





print("break")



