# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 17:17:10 2022

@author: forma
"""
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.cluster import KMeans
import os
from scipy.linalg import svd
from sklearn.manifold import TSNE

# loading mat file
data = scipy.io.loadmat('girosmallveryslow2.mp4_features.mat')
# features mat
features = data['features']


#%% centering

mu = np.mean(features,axis=0)
sig = np.std(features,axis=0)

x_norm = (features - mu)/sig

#%% PCA

U, S, V = svd(x_norm)
S = np.diag(S)
lowdim = S[0:100,0:100]@V[:,0:100].T;
plt.figure()
plt.scatter(lowdim[0,:], lowdim[1,:]);

#%% kmeans
lowdim = lowdim.T

kmeans = KMeans(n_clusters=20)
kmeans.fit(lowdim)
y_kmeans = kmeans.predict(lowdim)
plt.figure()
plt.scatter(lowdim[:,0], lowdim[:,1], c=y_kmeans, cmap='viridis')

#%% TSNE
# this is taking long time 
n_components = 2
tsne = TSNE(n_components)
tsne_result = tsne.fit_transform(lowdim)
tsne_result.shape

#%% plot k means

plt.figure()
plt.scatter(tsne_result[:,0], tsne_result[:,1], c=y_kmeans, cmap='viridis')



