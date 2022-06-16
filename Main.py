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
from sklearn.cluster import SpectralClustering
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
sFeatures,f_reduced,x_reduced,y_reduced=sceletonFeatures(x,y,frame_num,scelFrames)
# f_reduced=f_reduced[sFeatures[0,:]!=0]
# sFeatures=sFeatures[sFeatures!=0]
# sFeatures=sFeatures.reshape(3,7069)
# similar,outliers=findx_reduced[i].count()Similar(base,features,1)
# plotimages(similar,10)

#%% Analyse/find required number of clusters
nrOfClusters=20
lowdim=PCA(sFeatures) #Do the PCA and lower the dimension to 100

elbow(nrOfClusters,lowdim) #Check dimension with elbow approach 

silhouette(nrOfClusters,lowdim) #Check dimension with silhouette approach

#%% Specify needed cluster and calculate TSNE
clusters=3
y_kmeans=tsne(clusters,lowdim) 

#%% Plot random pics from clusters
clust=[]

for i in range(clusters):
    clust.append(np.argwhere(y_kmeans==i))

test=clust[0] #Select cluster
pics=np.zeros(64)
for i in range(64):
    randnr=random.randint(0,len(test)-1)
    pics[i]=scelFrames[int(test[randnr])]

plotimages(pics,8)
#%%
plot_scelframe(1,f_reduced)
    