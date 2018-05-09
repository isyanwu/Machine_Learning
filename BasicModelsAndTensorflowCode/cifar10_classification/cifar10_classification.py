
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

#x,y=obtain_batch(10)
#print(y.shape)

import tensorflow as tf
import numpy as np

max_iteration=1000

x=tf.placeholder(tf.float32,[None,3072])
y_=tf.placeholder(tf.int32,[None])
x_image=tf.reshape(x,[-1,32,32,3])
y=tf.cast(tf.one_hot(y_,10,1,0,axis=1),dtype=tf.float32)

#conv_1
w_conv_1=tf.Variable(tf.truncated_normal([5,5,3,64],stddev=0.1))#卷积核大小，通道，卷积核数量
b_conv_1=tf.Variable(tf.constant(0.1,shape=[64]))
conv_1=tf.nn.relu(tf.nn.conv2d(x_image,w_conv_1,strides=[1,1,1,1],padding='SAME')+b_conv_1)#strides代表卷积步长
pool_1=tf.nn.max_pool(conv_1,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME')#ksize代表filter大小
norm_1 = tf.nn.lrn(input=pool_1, depth_radius=4, bias=1.0, alpha=0.001/9.0, beta=0.75)

#conv_2
w_conv_2=tf.Variable(tf.truncated_normal([5,5,64,64],stddev=0.1))
b_conv_2=tf.Variable(tf.constant(0.1,shape=[64]))
conv_2=tf.nn.relu(tf.nn.conv2d(norm_1,w_conv_2,strides=[1,1,1,1],padding='SAME')+b_conv_2)
pool_2=tf.nn.max_pool(conv_2,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME')
norm_2 = tf.nn.lrn(input=pool_2, depth_radius=4, bias=1.0, alpha=0.001/9.0, beta=0.75)

#fully connected，缩写fc
w_fc_1=tf.Variable(tf.truncated_normal([8*8*64,512],stddev=0.1))
b_fc_1=tf.Variable(0.1,[512])
temp=tf.reshape(norm_2,[-1,8*8*64])
fc_1=tf.nn.relu(tf.matmul(temp,w_fc_1)+b_fc_1)

#droout，防止过拟合
keep_prob=tf.placeholder(tf.float32)
fc_1_drop=tf.nn.dropout(fc_1,keep_prob)

#分类结果
w_fc_2=tf.Variable(tf.truncated_normal([512,10],stddev=0.1))
b_fc_2=tf.Variable(tf.constant(0.1,shape=[10]))
y_prediction=tf.nn.softmax(tf.matmul(fc_1_drop,w_fc_2)+b_fc_2)

#交叉熵和模型评估
cross_entropy=-tf.reduce_sum(y*tf.log(y_prediction))
optimizer=tf.train.GradientDescentOptimizer(0.001).minimize(cross_entropy)
correct_prediction=tf.equal(tf.argmax(y_prediction,axis=1),tf.argmax(y,axis=1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
init=tf.global_variables_initializer()

#开始训练
with tf.Session() as sess:
    sess.run(init)
    for i in range(max_iteration):
        batch_x,batch_y=obtain_batch(50,'train')
        if i%20==0:
            print(sess.run(accuracy, feed_dict={x: batch_x, y_: batch_y, keep_prob: 1.0}))
        sess.run(optimizer,feed_dict={x:batch_x,y_:batch_y,keep_prob:0.5})
    x_data,y_data= obtain_batch(50,'test')
    print(sess.run(accuracy,feed_dict={x:x_data,y_:y_data,keep_prob:1.0}))

