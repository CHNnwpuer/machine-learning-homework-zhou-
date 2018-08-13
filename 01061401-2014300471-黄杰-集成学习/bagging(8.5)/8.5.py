import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

file1 = open('w3.csv','r')
data = [line.strip('\n').split(',') for line in file1]
data = np.array(data)
#X = [[float(raw[-7]),float(raw[-6]),float(raw[-5]),float(raw[-4]),float(raw[-3]), float(raw[-2])] for raw in data[1:,1:-1]]

X = [[float(raw[-3]), float(raw[-2])] for raw in data[1:]]
y = [1 if raw[-1]=='1' else 0 for raw in data[1:]]
X = np.array(X)
y = np.array(y)


# Create and fit an AdaBoosted decision tree,不剪枝决策树
bdt = BaggingClassifier(DecisionTreeClassifier(),

                         n_estimators=10)#最后这个参数表示基学习器的个数

bdt.fit(X, y)

plot_colors = "br"
plot_step = 0.02
class_names = "AB"

plt.figure(figsize=(10, 5)) #调图的尺寸

# Plot the decision boundaries
plt.subplot(121)
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                     np.arange(y_min, y_max, plot_step))

Z = bdt.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
cs = plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)
plt.axis("tight")

# Plot the training points
for i, n, c in zip(range(2), class_names, plot_colors):
    idx = np.where(y == i)
    plt.scatter(X[idx, 0], X[idx, 1],
                c=c, cmap=plt.cm.Paired,
                label="Class %s" % n)
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.legend(loc='upper right')
plt.xlabel('Sugar rate')
plt.ylabel('Density')
plt.title('Decision Boundary')


plt.tight_layout()
plt.subplots_adjust(wspace=0.35)
plt.show()
