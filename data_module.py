def extract_file():
    import os
    import tarfile
    filepath = os.path.join(os.path.realpath('..'), "data", 'cifar-10-python.tar.gz')
    file = tarfile.open(filepath)
    file.extractall()


def unpickle(filename):
    import os
    import cPickle
    dict=[]
    for name in filename:
        filepath=os.path.join(os.path.realpath(''), "cifar-10-batches-py", name)
        fo=open(filepath, 'rb')
        dict.append(cPickle.load(fo))
        fo.close()
    return dict


def load_train_data_all(data_batches):
    import numpy as np
    trainX=[]
    trainY=[]
    for i in range(0, 5):
        trainX.append(data_batches[i]['data'])
        trainY.append(data_batches[i]['labels'])
    trainX=np.concatenate(trainX)
    trainY=np.concatenate(trainY)
    return trainX, trainY


def extract_all_patches(trainX, rf_size, step_size):
    import numpy as np
    print("extracting all possible patches from data...")
    patches = []
    iter = 0
    for image in trainX:
        image_reshaped = np.reshape(image, (3, 32, 32))
        for i in range(0, 32-rf_size, step_size):
            for j in range(0, 32-rf_size, step_size):
                patches.append(image_reshaped[:][i:i+rf_size][j:j+rf_size])
        iter+=1
        if divmod(iter, 10000)[1] == 0:
            print(str(iter)+" of "+str(len(trainX))+" data extracted...")
    return patches