def unpickle(file):
    import pickle
    with open(file,'rb') as fo:
        dict=pickle.load(fo,encoding='bytes')
        return dict

def obtain_batch(batch,usage):
    import random
    import numpy as np
    if usage=='train':
        dict_1=unpickle('/home/yanwu9887/PycharmProjects/project_1/cifar-10-batches-py/data_batch_1')
        dict_2= unpickle('/home/yanwu9887/PycharmProjects/project_1/cifar-10-batches-py/data_batch_2')
        data = np.vstack((dict_1[b'data'],dict_2[b'data']))
        dict_3= unpickle('/home/yanwu9887/PycharmProjects/project_1/cifar-10-batches-py/data_batch_3')
        data = np.vstack((data, dict_3[b'data']))
        dict_4= unpickle('/home/yanwu9887/PycharmProjects/project_1/cifar-10-batches-py/data_batch_4')
        data = np.vstack((data, dict_4[b'data']))
        dict_5 = unpickle('/home/yanwu9887/PycharmProjects/project_1/cifar-10-batches-py/data_batch_5')
        data = np.vstack((data, dict_5[b'data']))
        label = dict_1[b'labels'] + dict_2[b'labels']+dict_3[b'labels']+dict_4[b'labels']
        a=np.random.random_integers(0,49999,batch)
        return data[a,:] ,random.sample(label,batch)
    elif usage=='test':
        dict=unpickle('/home/yanwu9887/PycharmProjects/project_1/cifar-10-batches-py/test_batch')
        data=dict[b'data']
        label=dict[b'labels']
        a = np.random.random_integers(0, 9999, batch)
        return data[a, :], random.sample(label, batch)
