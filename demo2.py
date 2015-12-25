
"""
Created on Thu Dec 24 16:53:32 2015
This demonstrates Kernel Nearest-Neighbor Algorithm using the cosine kernel  
@author: Joseph
"""

import numpy as np
import matplotlib.pyplot as plt
import KernelKNNClassifier as KernelKNN
from matplotlib.colors import ListedColormap
from sklearn import datasets


n_neighbors = 1

# import some data to play with
iris = datasets.load_iris()
X = iris.data[:, :2]  # we only take the first two features. We could
                      # avoid this ugly slicing by using a two-dim dataset
y = iris.target

h = .02  # step size in the mesh

# Create color maps
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

Knn=KernelKNN.KernelKNNClassifier(['linear'],0)
Knn=KernelKNN.KernelKNNClassifier(['cosine'],1)
Knn.fit(y,X)
yhat=Knn.predicte(np.c_[xx.ravel(), yy.ravel()],n_neighbors)

yhat = yhat.reshape(xx.shape)

# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, m_max]x[y_min, y_max].

    # Put the result into a color plot

plt.figure()
plt.pcolormesh(xx, yy, yhat, cmap=cmap_light)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("3-Class Cosine Kernel KNN Classifier")
plt.show()
