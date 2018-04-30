import tensorflow as tf
import numpy as np
import input_data
mnist=input_data.read_data_sets('MNIST_data/',one_hot=True)

#parameters
descent_rate=0.01
max_iteration=1000

#
x=tf.placeholder(tf.float32,[None,784])
y=tf.placeholder(tf.float32,[None,10])
x_image=tf.reshape(x,[-1,28,28,1])

#cnn1
w_conv_1=tf.Variable(tf.truncated_normal([5,5,1,32],stddev=0.1))
b_conv_1=tf.Variable(tf.constant(0.1,shape=[32]))
conv_1=tf.nn.relu(tf.nn.conv2d(x_image,w_conv_1,strides=[1,1,1,1],padding='SAME')+b_conv_1)
pool_1=tf.nn.max_pool(conv_1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

#cnn2
w_conv_2=tf.Variable(tf.truncated_normal([5,5,32,64],stddev=0.1))
b_conv_2=tf.Variable(tf.constant(0.1,shape=[64]))
conv_2=tf.nn.relu(tf.nn.conv2d(pool_1,w_conv_2,strides=[1,1,1,1],padding='SAME')+b_conv_2)
pool_2=tf.nn.max_pool(conv_2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

#fc
w_fc_1=tf.Variable(tf.truncated_normal([7*7*64,1024],stddev=0.1))
b_fc_1=tf.Variable(0.1,[1024])
temp=tf.reshape(pool_2,[-1,7*7*64])
fc_1=tf.nn.relu(tf.matmul(temp,w_fc_1)+b_fc_1)

#droout
keep_prob=tf.placeholder(tf.float32)
fc_1_drop=tf.nn.dropout(fc_1,keep_prob)

#out
w_fc_2=tf.Variable(tf.truncated_normal([1024,10],stddev=0.1))
b_fc_2=tf.Variable(tf.constant(0.1,shape=[10]))
y_prediction=tf.nn.softmax(tf.matmul(fc_1_drop,w_fc_2)+b_fc_2)

cross_entropy=-tf.reduce_sum(y*tf.log(y_prediction))
optimizer=tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction=tf.equal(tf.argmax(y_prediction,axis=1),tf.argmax(y,axis=1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
init=tf.global_variables_initializer()

#train
with tf.Session() as sess:
    sess.run(init)
    for i in range(max_iteration):
        batch_x,batch_y=mnist.train.next_batch(50)
        sess.run(optimizer,feed_dict={x:batch_x,y:batch_y,keep_prob:0.5})
    print(sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels,keep_prob:1.0}))







