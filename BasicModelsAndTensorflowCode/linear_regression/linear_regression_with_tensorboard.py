import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#
number_of_sample=50
temp=np.arange(number_of_sample).reshape(1,number_of_sample)
train_X_origin=temp+np.random.random((1,number_of_sample))
train_Y_origin=3*temp+6+np.random.random((1,number_of_sample))

temp=np.vstack([train_X_origin,train_Y_origin])
mu=np.mean(temp)
sigma=np.std(temp)
#train_X=(train_X_origin-mu)/sigma
#train_Y=(train_Y_origin-mu)/sigma
maxx=np.max(temp)
minn=np.min(temp)
alp=maxx-minn
train_X=(train_X_origin-minn)/(maxx-minn)
train_Y=(train_Y_origin-minn)/(maxx-minn)


#parameter
descent_rate=0.1
max_iteration=2000

with tf.name_scope('data'):
    X=tf.placeholder(tf.float32,name='x_data')
    Y=tf.placeholder(tf.float32,name='y_data')

with tf.name_scope('parameters'):
    w=tf.Variable(tf.random_uniform([1,1],2,3),name='weight')
    b=tf.Variable(tf.zeros([1,1]),name='bias')

with tf.name_scope('prediction'):
    prediction=tf.add(tf.matmul(w,X),b,name='prediction')

with tf.name_scope('loss'):
    loss=tf.reduce_mean(tf.square(Y-prediction))/2
    tf.summary.scalar('loss',loss)

with tf.name_scope('train'):
    optimizer=tf.train.GradientDescentOptimizer(descent_rate).minimize(loss)

with tf.name_scope('init'):
    init=tf.global_variables_initializer()

merged=tf.summary.merge_all()
writer=tf.summary.FileWriter('logs/',tf.Session().graph)

with tf.Session() as sess:
    sess.run(init)
    for i in range(max_iteration):
        sess.run(optimizer,feed_dict={X:train_X,Y:train_Y})
        rs=sess.run(merged,feed_dict={X:train_X,Y:train_Y})
        writer.add_summary(rs,i)
        if i%50==0:
            temp=sess.run(loss,feed_dict={X:train_X,Y:train_Y})
            print('iteration:','%04d'%i,'loss:','{:.9f}'.format(temp), \
                  'w=',sess.run(w),'b=',alp*sess.run(b)+minn+minn*(1-sess.run(w)))
    plt.plot(train_X_origin,train_Y_origin,'ro')
    #plt.plot(train_X, train_Y, 'ro')
    #plt.plot(train_X_origin.T,sess.run(w)*train_X_origin.T+sess.run(b))
    plt.plot(train_X_origin.T, sess.run(w) * train_X_origin.T + alp*sess.run(b)+minn+minn*(1-sess.run(w)))
    plt.show()



