from sklearn import datasets
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from pca import PCA


data = datasets.load_iris()
X = data.data
y = data.target

pca = PCA(2)
pca.fit(X)

X_projected = pca.transform(X)

print("Shape of X:", X.shape)
print("Shape of tranform X:", X_projected.shape)

x1 = X_projected[:,0]
x2 = X_projected[:,1]


plt.scatter(x1,x2, c=y, edgecolors='none',alpha=0.8, cmap= matplotlib.colormaps.get_cmap('viridis'))

plt.xlabel('Principal component 1')
plt.ylabel('Principal component 2')
plt.colorbar
plt.show()


