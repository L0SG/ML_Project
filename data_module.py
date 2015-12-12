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
    for i in range(0, len(data_batches)):
        trainX.append(data_batches[i]['data'])
        trainY.append(data_batches[i]['labels'])
    trainX=np.concatenate(trainX)
    trainY=np.concatenate(trainY)
    return trainX, trainY


def extract_all_patches(trainX, rf_size, step_size, num_patches):
    import numpy as np
    print("extracting all possible patches from data...")
    patches = []
    iter = 0
    for image in trainX:
        image_reshaped = np.reshape(image, (3, 32, 32))
        for i in range(0, 32-rf_size, step_size):
            for j in range(0, 32-rf_size, step_size):
                extracted_patch=[]
                for k in range(0, 3):
                    image_temp=image_reshaped[k][i:i+rf_size, j:j+rf_size]
                    extracted_patch.append(image_temp)
                extracted_patch_reshaped = np.reshape(extracted_patch, (3*rf_size*rf_size))
                patches.append(extracted_patch_reshaped)
                iter+=1
                if iter == num_patches:
                    break
            if iter == num_patches:
                    break
        if iter == num_patches:
                    break
        if divmod(iter, 10000)[1] == 0:
            print(str(iter)+" of "+str(len(trainX))+" data extracted...")
    return patches