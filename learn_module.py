def extract_features(trainX, kmeans_centroids, rf_size, step_size, whitening):
    print("extracting feature vector...")
    import data_module as dm
    import sklearn.preprocessing as pp
    from numpy import linalg as la
    from scipy.cluster.vq import whiten
    import numpy as np
    from scipy.spatial.distance import cdist

    trainXC=[]
    for image in trainX:
        # perform same process as sampled patches for kmeans
        patches = dm.extract_all_patches_by_image(image, rf_size, step_size)
        patches_normalized = pp.scale(patches)
        if whitening:
            whiten(patches_normalized)

        # hard kmeans activation function
        dist=cdist(patches_normalized, kmeans_centroids.cluster_centers_)
        f_k=[]
        for i in range(0, len(dist)):
            index=np.argmin(dist[i])
            hard_assignment = np.zeros((1, len(kmeans_centroids.cluster_centers_)))
            hard_assignment[0, index]=1
            f_k.append(hard_assignment)
        trainXC.append(f_k)


    return trainXC
