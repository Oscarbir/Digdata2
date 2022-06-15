# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 08:41:54 2022

@author: forma
"""

#In[]
import numpy as np
from pandas import DataFrame
import pandas as pd
import scipy.io as sio
import cv2 
from tqdm import tqdm
import time
import math
import matplotlib.pyplot as plt
from yaml import load
import random
from numpy import argwhere, linalg as LNG
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_samples,silhouette_score
import numpy as np
import matplotlib.style as style
import functions
mat = sio.loadmat('esqueletosveryslow.mat')
mat_complete = sio.loadmat('esqueletosveryslow_complete.mat')
data = sio.loadmat('girosmallveryslow2.mp4_features.mat')
features = np.double(data['features'])
filename = "girosmallveryslow2.mp4"


#%%

skeletons = mat["skeldata"]
base = np.double(features[:,5895:5907])
frame_num = skeletons[0,:]
x = skeletons[1::3,:]
y = skeletons[2::3,:]
scelFrames=list(range(10482))

skeletons_c = mat_complete["skeldata"]
frame_num_c = skeletons[0,:]
x_c = skeletons[1::3,:]*640
y_c = skeletons[2::3,:]*360
hist = plt.hist(frame_num,bins=np.shape(features)[1],color='red')
skel_per_frame = hist[0]
# indexes of frames with no skeletons
ind, = np.where(skel_per_frame==0)
for i in ind:
    scelFrames.remove(i)
sFeatures = functions.sceletonFeatures(x,y,frame_num,scelFrames)

# similar,outliers=findx_reduced[i].count()Similar(base,features,1)
# plotimages(similar,10)

#%% Analyse/find required number of clusters
nrOfClusters = 20
lowdim = functions.PCA(sFeatures) #Do the PCA and lower the dimension to 100

functions.elbow(nrOfClusters,lowdim) #Check dimension with elbow approach 

functions.silhouette(nrOfClusters,lowdim) #Check dimension with silhouette approach

#%% Specify needed cluster and calculate TSNE
clusters = 10
y_kmeans,y_tsne = functions.kmeans_tsne(clusters,lowdim) 

#%% agglomerative clustering
from sklearn.cluster import AgglomerativeClustering
lowdimT = lowdim.T
model = AgglomerativeClustering(
                linkage='average', n_clusters=7,affinity="euclidean")
model.fit(lowdimT)
y_agg = model.fit_predict(lowdimT)

plt.figure()
plt.scatter(y_tsne[:,0],y_tsne[:,1],c=y_agg,cmap='viridis')
plt.title('Agglomerative Clustering')
#%% DBSCAN
from sklearn.cluster import DBSCAN, SpectralClustering

clust_DBSCAN = DBSCAN(eps=4, min_samples=2).fit(lowdimT)
print('DBSCAN done')
plt.figure()
plt.scatter(y_tsne[:,0],y_tsne[:,1],c=clust_DBSCAN.labels_,cmap='viridis')
plt.title('DBSCAN')


#%% SpectralClustering
spect_clust = SpectralClustering(n_clusters=7,
                                  assign_labels='discretize',
                                  random_state=0).fit(lowdimT)
print('SpectralClustering done')
plt.scatter(y_tsne[:,0],y_tsne[:,1],c=spect_clust.labels_,cmap='viridis')
plt.title('Spectral Clustering')

#%% ward
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import ward,fcluster
y = pdist(lowdimT)
Z = ward(y)
#%%
clust_ward = fcluster(Z, 10, criterion='maxclust')
plt.scatter(y_tsne[:,0],y_tsne[:,1],c=clust_ward,cmap='viridis')
plt.title('ward Clustering')

#%% Plot random pics from clusters
clust=[]
for i in range(7):
    clust.append(np.argwhere(y_kmeans==i))

test=clust[3] #Select cluster
pics=np.zeros(16)
for i in range(16):
    randnr=random.randint(0,len(test)-1)
    pics[i]=test[randnr]

functions.plotimages(pics,4,filename)
    