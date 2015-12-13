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


def extract_all_patches(trainX, rf_size, step_size, num_patches=-1, num_images=-1):
    import numpy as np
    print("extracting all possible patches from data...")
    patches = []
    iter_patch = 0
    iter_image = 0
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
                iter_patch+=1
                if iter_patch == num_patches:
                    break
            if iter_patch == num_patches:
                    break
        iter_image+=1
        if iter_patch == num_patches:
            break
        if iter_image == num_images:
            break
        if divmod(iter_image, 1000)[1] == 0:
            print(str(iter_image)+" of "+str(len(trainX))+" data extracted...")
    print("total extracted # data : "+str(iter_image))
    print("total extracted # patch : "+str(iter_patch))
    return patches


def extract_random_patches(trainX, rf_size, step_size, num_patches=-1, num_images=-1):
    import numpy as np
    import random
    print("extracting random possible patches from data...")
    patches = []
    iter_patch = 0
    while iter_patch<=num_patches:
        for image in trainX:
            rnd = random.randrange(0, 100)
            if rnd is not 0:
                continue
            image_reshaped = np.reshape(image, (3, 32, 32))
            for i in range(0, 32-rf_size, step_size):
                for j in range(0, 32-rf_size, step_size):
                    rnd = random.randrange(0, 100)
                    if rnd is not 0:
                        continue
                    extracted_patch=[]
                    for k in range(0, 3):
                        image_temp=image_reshaped[k][i:i+rf_size, j:j+rf_size]
                        extracted_patch.append(image_temp)
                    extracted_patch_reshaped = np.reshape(extracted_patch, (3*rf_size*rf_size))
                    patches.append(extracted_patch_reshaped)
                    iter_patch+=1
                    if iter_patch == num_patches:
                        break
                if iter_patch == num_patches:
                    break
            if iter_patch == num_patches:
                break
        if iter_patch == num_patches:
            break
    print("total extracted # patch : "+str(iter_patch))
    return patches


def extract_all_patches_by_image(image, rf_size, step_size):
    import numpy as np
    patches = []
    image_reshaped = np.reshape(image, (3, 32, 32))
    for i in range(0, 32-rf_size, step_size):
        for j in range(0, 32-rf_size, step_size):
            extracted_patch=[]
            for k in range(0, 3):
                image_temp=image_reshaped[k][i:i+rf_size, j:j+rf_size]
                extracted_patch.append(image_temp)
            extracted_patch_reshaped = np.reshape(extracted_patch, (3*rf_size*rf_size))
            patches.append(extracted_patch_reshaped)
    return patches