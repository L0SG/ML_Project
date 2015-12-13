import data_module as dm
import os
import numpy as np
import learn_module as lm
import sklearn.cluster as cl
import sklearn.preprocessing as pp
from scipy.cluster.vq import whiten
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import confusion_matrix
import time
import itertools
from multiprocessing import Pool, freeze_support
from sklearn import grid_search

tic=time.clock()

# Variables (adapted from demo code of the paper)
CIFAR_DIM = 32 * 32 * 3
whitening = 1
num_patches = 1000
num_centroids = 200
rf_size = 6
step_size = 2
pooling_dim = 2

# Check if there already exists the extracted files
if not os.path.isdir(os.path.join(os.path.realpath(''), "cifar-10-batches-py")):
    dm.extract_file()

# Unpickle batches.meta file and save variables
meta = dm.unpickle(['batches.meta'])
label_names = meta[0]['label_names']
num_cases_per_batch = meta[0]['num_cases_per_batch']

# Unpickle data_batch files
data_batches = dm.unpickle(['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5'])
#data_batches = dm.unpickle(['data_batch_1'])

# load all train data
trainX, trainY = dm.load_train_data_all(data_batches)

# extract patches
# sequential method
"""
patches = dm.extract_all_patches(trainX, rf_size, step_size, num_patches)
"""
# random method
patches = dm.extract_random_patches(trainX, rf_size, step_size, num_patches)

# standardize patches
print("normalizing patches....")
scaler = pp.StandardScaler()
scaler.fit(patches)
patches=scaler.transform(patches)

# whitening
if whitening:
    whiten(patches)


# K-means clustering
print("clustering with kmeans...")
tic_cluster = time.clock()
kmeans= cl.KMeans(num_centroids, n_jobs=-1, n_init=12, max_iter=100)
kmeans_centroids = kmeans.fit(patches)
toc_cluster = time.clock()
print("clustering time : "+str(toc_cluster-tic_cluster))

# extract feature vector using kmeans centroids
# trainXC is now (# of imput images) * 4K vector
print("extracting feature vector...")
def multi_wrapper(args):
    return lm.extract_features(*args)
if __name__=='__main__':
    freeze_support()
    pool = Pool(12)
    numimages=500
    trainXC = pool.map(multi_wrapper, itertools.izip(
                       (trainX[i:i+numimages] for i in range(0, len(trainX), numimages)),
                        itertools.repeat([kmeans_centroids, rf_size, step_size, whitening, pooling_dim])))
    trainXC = np.concatenate(trainXC)

# standardize data
scaler = pp.StandardScaler()
scaler.fit(trainXC)
trainXC=scaler.transform(trainXC)

#classification
print("training classifier...")
tic_classifier = time.clock()
classifier_svm = OneVsRestClassifier(LinearSVC(), n_jobs=-1)
classifier_svm.fit(trainXC, trainY)
toc_classifier = time.clock()
print("training time : "+str(toc_classifier-tic_classifier))
print("training done")

# testing
print("\ntesting")
test_batch = dm.unpickle(['test_batch'])
testX, testY = dm.load_train_data_all(test_batch)
# extract feature vector
print("extracting feature vector...")
if __name__=='__main__':
    freeze_support()
    pool = Pool(12)
    numimages=500
    testXC = pool.map(multi_wrapper, itertools.izip(
                       (testX[i:i+numimages] for i in range(0, len(testX), numimages)),
                        itertools.repeat([kmeans_centroids, rf_size, step_size, whitening, pooling_dim])))
    testXC = np.concatenate(testXC)

testXC=scaler.transform(testXC)
# predict class labels for test data
predictY = classifier_svm.predict(testXC)
confusion = confusion_matrix(testY, predictY)
print(confusion)
print("accuracy : "+str(float(np.sum(np.diag(confusion)))*100/np.sum(confusion)))

toc=time.clock()
print("elapsed time : "+str(toc-tic))