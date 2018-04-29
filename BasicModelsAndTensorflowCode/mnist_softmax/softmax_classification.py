import input_data
import tensorflow as tf
mnist=input_data.read_data_sets('MNIST_data/',one_hot=True)

#梯度下降参数
descent_rate=0.01
max_iteration=1000

#TensorFlow流程图
X=tf.placeholder(tf.float32,[None,784])
Y=tf.placeholder(tf.float32,[None,10])

W=tf.Variable(tf.zeros([784,10]))
b=tf.Variable(tf.zeros([1,10]))

Y_prediction=tf.nn.softmax(tf.matmul(X,W)+b)
cross_entropy=-tf.reduce_sum(Y*tf.log(Y_prediction))
optimizer=tf.train.GradientDescentOptimizer(descent_rate).minimize(cross_entropy)
init=tf.global_variables_initializer()

#评估模型
#tf.argmax(input,axis),axis=1返回input每行的最大值所对应的索引值，axis则是列
correct_prediction = tf.equal(tf.argmax(Y, axis=1), tf.argmax(Y_prediction, axis=1))
#tf.cast(input, dtype) 强制类型转换
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#train
with tf.Session() as sess:
    sess.run(init)
    for i in range(max_iteration):
        batch_x,batch_y=mnist.train.next_batch(100)
        sess.run(optimizer,feed_dict={X:batch_x,Y:batch_y})

    print(sess.run(accuracy,feed_dict={X:mnist.test.images,Y:mnist.test.labels}))
