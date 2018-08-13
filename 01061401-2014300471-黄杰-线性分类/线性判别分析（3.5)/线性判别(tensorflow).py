from numpy import *
import matplotlib.pyplot as plt
import tensorflow as tf

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

X0 = array(data[:8,0:2])
X1 = array(data[8:,0:2])
u0=mean(X0,axis=0)
u1=mean(X1,axis=0)
cov1=cov(X0-u0,rowvar=0)
cov2=cov(X1-u1,rowvar=0)
s_w=(cov1+cov2).astype(float32)
s_b=((u0-u1).reshape(-1,1)*(u0-u1)).astype(float32)
#print(s_w)
w=tf.Variable(tf.zeros([2,1]))
wT=tf.transpose(w)
s1=tf.matmul(wT,s_b)
s2=tf.matmul(s1,w)
s3=tf.matmul(wT,s_w)
s4=tf.matmul(s3,w)
loss =  tf.reduce_mean(tf.clip_by_value(s2/s4, 100,tf.reduce_max(s2/s4)))
train = tf.train.GradientDescentOptimizer(0.3).minimize(1/loss)
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
for step in range(10000):
    sess.run(train)
print(step, sess.run(w).flatten())