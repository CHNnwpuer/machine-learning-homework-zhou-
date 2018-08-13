from numpy import *
import tensorflow as tf
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
x=data[:,0:2].astype(float32)
y=data[:,2].astype(float32)
#print(x[:,0])
#print(shape(y))
sigma = 0.5
kkx = square(tile(x[:,0].T,[x.shape[0],1])-tile(x[:,0],[1,x.shape[0]]))
#print(shape(kkx))
kkx += square(tile(x[:,1].T,[x.shape[0],1])-tile(x[:,1],[1,x.shape[0]]))
kkx = sqrt(kkx)
KX = exp(-sigma * kkx )
lam = 1./2.
batch = x.shape[0]
alpha = tf.Variable(tf.random_uniform([batch,1],-1.0,1.0))
alpha = tf.maximum(0.,alpha)
loss = lam*tf.reduce_sum(tf.matmul(alpha,tf.transpose(alpha))*KX)
tmp = tf.matmul(KX.astype(float32), alpha)
tmp = y*tmp
tmp = 1. - tmp
tmp = tf.maximum(0.,tmp)
tmp = 1./batch*tf.reduce_sum(tmp)
loss += tmp
optimizer = tf.train.GradientDescentOptimizer(0.2)
train = optimizer.minimize(loss)
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
for step in range(10000):
    sess.run(train)
resA = sess.run(alpha)
#print(resA.shape)
predict=multiply(resA,y)
#print(predict.shape)
predict=sum(multiply(predict,kkx),axis=0)
predict = predict.T
predict=tile(predict,[1,3])
#print(predict>0.1)
ax = array(x)
predictSet1=ax[predict>0.1].reshape([-1,3])
predictSet2=ax[predict<0.0].reshape([-1,3])
fig = plt.figure()
ax = fig.add_subplot(211)
ax.scatter(x=data[:,0],y=data[:,1])
ax.scatter(x=data[:,0],y=data[:,1])
ax = fig.add_subplot(212)
ax.scatter(x=predictSet1[:,0],y=predictSet1[:,1])
ax.scatter(x=predictSet2[:,0],y=predictSet2[:,1])
fig.show()






