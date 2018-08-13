from numpy import *
from sklearn import svm
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
#linear kernal
clf1=svm.SVC(kernel='linear',C=1000)
clf1.fit(x,y)
#RBF kernal
clf2=svm.SVC(kernel='rbf',gamma=0.7, C=1000)
clf2.fit(x,y)
#set test data
x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
xx, yy = meshgrid(arange(x_min, x_max, 0.02),
                     arange(y_min, y_max, 0.02))
t=arange(x_min, x_max, 0.02)
#draw picture
X0=data[0:8,:]
X1=data[8:17,:]
#draw linear kernel picture
ax=plt.subplot(1,2,1)
ax.scatter(X0[:,0],X0[:,1],c='r',label='+')
ax.scatter(X1[:,0],X1[:,1],c='g',label='-')
w = clf1.coef_[0]
a = -w[0] / w[1]
y1 = a * t- clf1.intercept_[0] / w[1]
plt.sca(ax)
plt.plot(t,y1)
plt.xlabel('density')
plt.ylabel('ratio_sugar')
plt.legend()
#draw rbf kernel picture
ax1=plt.subplot(1,2,2)
Z=clf2.predict(c_[xx.ravel(),yy.ravel()]).reshape(xx.shape)
plt.contour(xx, yy, Z)
ax1.scatter(X0[:,0],X0[:,1],c='r',label='+')
ax1.scatter(X1[:,0],X1[:,1],c='g',label='-')
plt.sca(ax1)
plt.xlabel('density')
plt.ylabel('ratio_sugar')
plt.legend()
plt.show()


