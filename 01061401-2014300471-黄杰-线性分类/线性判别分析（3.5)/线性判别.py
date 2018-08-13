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

X0 = array(data[:8,0:2])
X1 = array(data[8:,0:2])
u0=mean(X0,axis=0)
u1=mean(X1,axis=0)
cov1=cov(X0-u0,rowvar=0)
cov2=cov(X1-u1,rowvar=0)
w=mat(cov1+cov2).I*(u0-u1).reshape(-1,1)
print(w)
x=arange(0,1.0,0.1)
y=-w[0]*x/w[1]
#print(array(y[0]).flatten())
plt.plot(x,array(y[0]).flatten())
ax = plt.subplot(111)
ax.scatter(X0[:,0],X0[:,1],c='r',label='+')
ax.scatter(X1[:,0],X1[:,1],c='b',label='-')