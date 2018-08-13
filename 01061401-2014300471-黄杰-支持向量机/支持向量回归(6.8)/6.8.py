from sklearn.svm import SVR
from numpy import *
import matplotlib.pyplot as plt
#data introduce
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
x=data[:,0:2]
y=data[:,2]
clf=SVR(kernel='linear',C=1000)
clf.fit(x,y)
x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
xx=arange(x_min, x_max, 0.02)
w = clf.coef_[0]
#print(w)
t=square(xx)
a = -w[0] / w[1]
y1 =sqrt( a * t- clf.intercept_[0] / w[1])
X0=data[0:8,:]
X1=data[8:17,:]
plt.scatter(X0[:,0],X0[:,1],c='r',label='+')
plt.scatter(X1[:,0],X1[:,1],c='g',label='-')
plt.plot(t,y1)
plt.xlabel('density')
plt.ylabel('ratio_sugar')
plt.legend()
plt.show()


