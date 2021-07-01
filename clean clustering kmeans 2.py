# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 21:38:47 2020

@author: layto
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler

data = load_wine()
wine = pd.DataFrame(data.data, columns=data.feature_names)
print(wine.head()) 
X = wine[['alcohol', 'flavanoids']] 

scale = StandardScaler()
scale.fit(X)
X_scaled = scale.transform(X)


from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3)
kmeans.fit(X_scaled)
y_pred = kmeans.predict(X_scaled)

plt.scatter(X_scaled[:,0], 
            X_scaled[:,1],
            c= y_pred)
plt.scatter(kmeans.cluster_centers_[:, 0],
            kmeans.cluster_centers_[:, 1], 
            marker="*",
            s = 250, 
            c = [0,1,2], 
            edgecolors='k')
plt.xlabel('alcohol'); plt.ylabel('flavanoids')
plt.title('k-means (k=3)')
plt.show()

#elbow method
import numpy as np
# calculate distortion for a range of number of cluster
inertia = []
for i in np.arange(1, 11):
    km = KMeans(
        n_clusters=i
    )
    km.fit(X_scaled)
    inertia.append(km.inertia_)

# plot
plt.plot(np.arange(1, 11), inertia, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()
