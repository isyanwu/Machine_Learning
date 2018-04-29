import input_data
import tensorflow as tf
mnist=input_data.read_data_sets('MNIST_data/',one_hot=True)

train_X=mnist.train.images
train_Y=mnist.train.labels
#print(train_X.shape)
#print(train_Y.shape)
batch_x,batch_y=mnist.train.next_batch(100)
print(batch_x.shape)

descent_rate=0.01
max_iteration=1000

X=tf.placeholder(tf.float32)
Y=tf.placeholder(tf.float32)

W=tf.Variable(tf.zeros([784,10]))
b=tf.Variable(tf.zeros([1,10]))

Y_prediction=tf.nn.softmax(tf.matmul(X,W)+b)

cross_entropy=-tf.reduce_sum(Y*tf.log(Y_prediction))
optimizer=tf.train.GradientDescentOptimizer(descent_rate).minimize(cross_entropy)
init=tf.global_variables_initializer()

correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_prediction, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(init)
    for i in range(max_iteration):
        batch_x,batch_y=mnist.train.next_batch(100)
        sess.run(optimizer,feed_dict={X:batch_x,Y:batch_y})

    print(sess.run(accuracy,feed_dict={X:train_X,Y:train_Y}))
