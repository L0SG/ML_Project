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

