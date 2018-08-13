import tensorflow as tf
from numpy import *
import matplotlib.pyplot as plt
#数据
data = mat([
    [1, 1, 1, 1, 1, 1, 0.697, 0.460, 1],
    [2, 1, 2, 1, 1, 1, 0.774, 0.376, 1],
    [2, 1, 1, 1, 1, 1, 0.634, 0.264, 1],
    [1, 1, 2, 1, 1, 1, 0.608, 0.318, 1],
    [3, 1, 1, 1, 1, 1, 0.556, 0.215, 1],
    [1, 2, 1, 1, 2, 2, 0.403, 0.237, 1],
    [2, 2, 1, 2, 2, 2, 0.481, 0.149, 1],
    [2, 2, 1, 1, 2, 1, 0.437, 0.211, 1],
    [2, 2, 2, 2, 2, 1, 0.666, 0.091, 0],
    [1, 3, 3, 1, 3, 2, 0.243, 0.267, 0],
    [3, 3, 3, 3, 3, 1, 0.245, 0.057, 0],
    [3, 1, 1, 3, 3, 2, 0.343, 0.099, 0],
    [1, 2, 1, 2, 1, 1, 0.639, 0.161, 0],
    [3, 2, 2, 2, 1, 1, 0.657, 0.198, 0],
    [2, 2, 1, 1, 2, 2, 0.360, 0.370, 0],
    [3, 1, 1, 3, 3, 1, 0.593, 0.042, 0],
    [1, 1, 2, 2, 2, 1, 0.719, 0.103, 0]])
x_data=data[:,0:8].astype(float32)
y_data=data[:,8].astype(float32)
#搭建网络
x=tf.placeholder(tf.float32,[None,8])
y=tf.placeholder(tf.float32,[None,1])
def addlayer(data,insize,outsize,active=None):
    w=tf.Variable(tf.random_normal([insize,outsize]))
    b=tf.Variable(tf.zeros([1,outsize])+0.1)
    w_b=tf.matmul(data,w)+b
    return active(w_b)
layer=addlayer(x,8,20,tf.nn.sigmoid)
output=addlayer(layer,20,1,tf.nn.sigmoid)
loss=tf.reduce_mean(tf.reduce_sum(tf.square(y-output),reduction_indices=[1]))
train=tf.train.GradientDescentOptimizer(0.5).minimize(loss)
init=tf.initialize_all_variables()
#定义精度
def accuracy(l1,l2):
    for i in range(len(l2)):
        if l2[i] > 0.5:
            l2[i] = 1
        else:
            l2[i] = 0
    s = abs(l1 - l2)
    return 1-sum(s) / 17
#训练网络
with tf.Session() as sess:
    sess.run(init)
    for i in range(2000):
         sess.run(train,feed_dict={x:x_data,y:y_data})
    #print(array(y_data).flatten(),sess.run(output,feed_dict={x:x_data,y:y_data}).flatten())
         l1 = array(y_data).flatten()
         l2 = sess.run(output, feed_dict={x: x_data, y: y_data}).flatten()
         # if i%200==0:
         # print(accuracy(l1,l2))
         if (accuracy(l1, l2) > 0.9):
             print(i)
             break











