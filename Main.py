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
mat = sio.loadmat('esqueletosveryslow.mat')
mat_complete = sio.loadmat('esqueletosveryslow_complete.mat')
data = sio.loadmat('girosmallveryslow2.mp4_features.mat')
features = np.double(data['features'])
filename = "girosmallveryslow2.mp4"

#%% Define functions

def findSimilar(base,data,case):
    u, s, v = np.linalg.svd(base)
    print(u.shape)
    Pi=u[:,0:3]@u[:,0:3].T
    if case==1:
        fi=Pi@data
    else:
        fi = (np.identity(len(data)) - Pi)@data
    di=np.sqrt(np.sum(np.multiply(fi,fi),axis=0))
    dd=np.sqrt(np.sum(np.multiply(data,data),axis=0))
    ri=np.divide(di,dd)
    outliers=np.argwhere(ri>0.9)
    sorted=np.argsort(ri)
    if case==1:
        similar=sorted[-100:]
    else:
        similar=sorted[:100]
    return similar, outliers

def plotimages(similar,grid):
    cap = cv2.VideoCapture(filename) #video_name is the video being called
    plt.figure(figsize = (20,20))
    i=1
    for frames in similar:
        cap.set(1,frames); # Where frame_no is the frame you want
        ret, frame = cap.read() # Read the frame
        plt.subplot(grid, grid, i)
        plt.imshow(frame)
        
        i +=1
        plt.axis('off')

def plot_scelframe(frame_nr):
    get_start_nr=np.argwhere(frame_num_c==frame_nr)
    i=get_start_nr[0]
    #i=11
    while frame_num_c[i]==frame_nr:    
        cap = cv2.VideoCapture(filename)
        cap.set(1,int(frame_num_c[i]))
        ret, frame = cap.read()
        plt.imshow(frame)
        plt.scatter(x_c[:,i],y_c[:,i],c='r')
        plt.show()
        plt.pause(0.3)
        i+=1

def elbow(clusters,lowdim):
    distortions = []
    K = range(1,clusters)
    for k in K:
        kmeanModel = KMeans(n_clusters=k)
        kmeanModel.fit(lowdim.T)
        distortions.append(kmeanModel.inertia_)
        print(k)
    plt.figure(figsize=(16,8))
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method showing the optimal k')
    plt.show()

def silhouette(clusters,lowdim):
    range_n_clusters = range(2,clusters)
    silhouette_avg_n_clusters = []

    for n_clusters in range_n_clusters:

        clusterer = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = clusterer.fit_predict(lowdim.T)

        silhouette_avg = silhouette_score(lowdim.T, cluster_labels)
        print("For n_clusters =", n_clusters,
            "The average silhouette_score is :", silhouette_avg)

        silhouette_avg_n_clusters.append(silhouette_avg)

        sample_silhouette_values = silhouette_samples(lowdim.T, cluster_labels)



    style.use("fivethirtyeight")
    plt.plot(range_n_clusters, silhouette_avg_n_clusters)
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("silhouette score")
    plt.show()

def PCA(data):
    featcentered=sFeatures-np.mean(sFeatures.T)*np.ones(len(sFeatures[0]))
    u, s, v=LNG.svd(sFeatures,full_matrices=False)
    sx= np.diag(s)
    v=v.T
    lowdim=sx[0:100,0:100]@v[:,0:100].T
    return lowdim

def tsne(clusters,lowdim):
    c=KMeans(n_clusters=clusters).fit(lowdim.T)
    kmeans = KMeans(n_clusters=7)
    kmeans.fit(lowdim.T)
    y_kmeans = kmeans.predict(lowdim.T)

    tsne = TSNE(n_components=2)
    y = tsne.fit_transform(lowdim.T) 
    plt.scatter(y[:,0],y[:,1],c=y_kmeans,s=10,cmap='viridis')
    return y_kmeans

def sceletonFeatures(x,y,frame_num,scelFrames):
    x_reduced=[]
    y_reduced=[]
    f_reduced=[]
    for i in range(len(x[0])):
        points=np.count_nonzero(x[:,i])
        if points>5:
            x_reduced.append(x[:,i])
            y_reduced.append(y[:,i])
            f_reduced.append(int(frame_num[i]))

    sFeatures=np.zeros((3,len(scelFrames)))
    j=0
    for i in scelFrames:
       
        sFeatures[0,j]=f_reduced.count(i) #Number of sceletons in frame
        if sFeatures[0,j]>0:
            indexes=[element for element, x in enumerate(f_reduced) if x == i]
            countpoint=0
            for frame in indexes:
                countpoint+=np.count_nonzero(x_reduced[frame]) #Total number of sceleton points in a frame
            sFeatures[1,j]=countpoint
            sFeatures[2,j]=sFeatures[1,j]/sFeatures[0,j]
        else:
            sFeatures[1,j]=0
            sFeatures[2,j]=0
        j=j+1
    return sFeatures


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
sFeatures=sceletonFeatures(x,y,frame_num,scelFrames)

# similar,outliers=findx_reduced[i].count()Similar(base,features,1)
# plotimages(similar,10)

#%% Analyse/find required number of clusters
nrOfClusters=20
lowdim=PCA(sFeatures) #Do the PCA and lower the dimension to 100

elbow(nrOfClusters,lowdim) #Check dimension with elbow approach 

silhouette(nrOfClusters,lowdim) #Check dimension with silhouette approach

#%% Specify needed cluster and calculate TSNE
clusters=10
y_kmeans=tsne(clusters,lowdim) 

#%% Plot random pics from clusters
clust=[]
for i in range(7):
    clust.append(np.argwhere(y_kmeans==i))

test=clust[3] #Select cluster
pics=np.zeros(16)
for i in range(16):
    randnr=random.randint(0,len(test)-1)
    pics[i]=test[randnr]

plotimages(pics,4)
    