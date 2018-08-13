import tensorflow as tf
from numpy import *
import matplotlib.pyplot as plt

data = mat([[0.697,0.460,1],
        [0.774,0.376,1],
        [0.634,0.264,1],
        [0.608,0.318,1],
        [0.556,0.215,1],
        [0.403,0.237,1],
        [0.481,0.149,1],
        [0.437,0.211,1],
        [0.666,0.091,0],
        [0.243,0.267,0],
        [0.245,0.057,0],
        [0.343,0.099,0],
        [0.639,0.161,0],
        [0.657,0.198,0],
        [0.360,0.370,0],
        [0.593,0.042,0],
        [0.719,0.103,0]])
#数据类型转换
x_train = data[:,0:2].astype(float32)
y_train = data[:,2].astype(float32)
#定义计算w,b的参数
y_train = mat(y_train)
w = tf.Variable(tf.zeros([2, 1]))
b= tf.Variable(tf.zeros([1, 1]))
y = 1 / (1 + tf.exp(-(tf.matmul(x_train, w) + b)))
loss = tf.reduce_mean(- y_train.reshape(-1, 1) * tf.log(y) - (1 - y_train.reshape(-1, 1)) * tf.log(1 - y))
train = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
#运行并显示w,b的运行结果
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
for step in range(10000):
    sess.run(train)
print(step, sess.run(w).flatten(), sess.run(b).flatten())
#画图
X0 = array(data[:8])
X1 = array(data[8:])
ax = plt.subplot(111)
ax.scatter(X0[:,0],X0[:,1],c='r',label='+')
ax.scatter(X1[:,0],X1[:,1],c='b',label='-')
x1 = arange(-0.2,1.0,0.1)
y1 = (-b-w[0]*x1)/w[1]
#print(shape(sess.run(y1).flatten()))
plt.sca(ax)
plt.plot(x1,sess.run(y1).flatten())
plt.xlabel('density')
plt.ylabel('ratio_sugar')
plt.legend()
plt.show()







