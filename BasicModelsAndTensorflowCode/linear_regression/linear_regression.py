import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#生成训练数据
number_of_sample=50
temp=np.arange(number_of_sample).reshape(1,number_of_sample)
train_X_origin=temp+np.random.random((1,number_of_sample))
train_Y_origin=3*temp+6+np.random.random((1,number_of_sample))

#归一化
temp=np.vstack([train_X_origin,train_Y_origin])
maxx=np.max(temp)
minn=np.min(temp)
alp=maxx-minn
train_X=(train_X_origin-minn)/(maxx-minn)
train_Y=(train_Y_origin-minn)/(maxx-minn)


#梯度下降参数
descent_rate=0.1
max_iteration=2000

#TensorFlow
X=tf.placeholder(tf.float32)
Y=tf.placeholder(tf.float32)

w=tf.Variable(tf.random_uniform([1,1],2,3))
b=tf.Variable(tf.zeros([1,1]))

prediction=tf.add(tf.matmul(w,X),b)
loss=tf.reduce_mean(tf.square(Y-prediction))/2

optimizer=tf.train.GradientDescentOptimizer(descent_rate).minimize(loss)
init=tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for i in range(max_iteration):
        sess.run(optimizer,feed_dict={X:train_X,Y:train_Y})
        if i%50==0:
            temp=sess.run(loss,feed_dict={X:train_X,Y:train_Y})
            print('iteration:','%04d'%i,'loss:','{:.9f}'.format(temp), \
                  'w=',sess.run(w),'b=',alp*sess.run(b)+minn+minn*(1-sess.run(w)))#反归一化
    plt.plot(train_X_origin,train_Y_origin,'ro')
    plt.plot(train_X_origin.T, sess.run(w) * train_X_origin.T + alp*sess.run(b)+minn+minn*(1-sess.run(w)))
    plt.show()

