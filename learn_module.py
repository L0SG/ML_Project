

def extract_features(trainX, args):
    import data_module as dm
    import sklearn.preprocessing as pp
    from scipy.cluster.vq import whiten
    import numpy as np
    from scipy.spatial.distance import cdist

    # unpack args (for pool.map)
    kmeans_centroids, rf_size, step_size, whitening, pooling_dim = args

    trainXC=[]
    for image in trainX:
        # perform same process as sampled patches for kmeans
        patches = dm.extract_all_patches_by_image(image, rf_size, step_size)
        scaler = pp.StandardScaler()
        scaler.fit(patches)
        patches = scaler.transform(patches)
        if whitening:
            whiten(patches)

        # hard kmeans activation function
        dist=cdist(patches, kmeans_centroids.cluster_centers_)
        f_k=[]
        for i in range(0, len(dist)):
            index=np.argmin(dist[i])
            hard_assignment = np.zeros((1, len(kmeans_centroids.cluster_centers_)))
            hard_assignment[0, index]=1
            f_k.append(hard_assignment)
        f_k=np.concatenate(f_k)

        # pooling
        prows = np.sqrt(len(f_k))
        pcols = np.sqrt(len(f_k))
        f_k=np.reshape(f_k, (prows, pcols, len(kmeans_centroids.cluster_centers_)))
        f_k=sum_pooling(f_k, pooling_dim)

        #f_k is now 4K vector, append to trainXC
        trainXC.append(f_k)

    return trainXC


def sum_pooling(f_k, pooling_dim):
    import numpy as np
    row=np.round(len(f_k)/pooling_dim)
    col=np.round(len(f_k)/pooling_dim)
    result=[]
    for i in range(0, pooling_dim):
        for j in range(0, pooling_dim):
            patch = f_k[i:i+row, j:j+col]
            patch= np.reshape(patch, (row*col, len(patch[0][0])))
            sum = np.sum(patch, axis=0)
            result.append(sum)
    result = np.concatenate(result)
    return result


def max_pooling(f_k, pooling_dim):
    import numpy as np
    row=np.round(len(f_k)/pooling_dim)
    col=np.round(len(f_k)/pooling_dim)
    result=[]
    for i in range(0, pooling_dim):
        for j in range(0, pooling_dim):
            patch = f_k[i:i+row, j:j+col]
            patch= np.reshape(patch, (row*col, len(patch[0][0])))
            sum = np.amax(patch, axis=0)
            result.append(sum)
    result = np.concatenate(result)
    return result

