# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 13:48:45 2022

@author: forma
"""
import numpy as np
import cv2 

import matplotlib.pyplot as plt

from numpy import linalg
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_samples,silhouette_score
import matplotlib.style as style
#%% Define functions

def findSimilar(base,data,case):
    u, s, v = linalg.svd(base)
    print(u.shape)
    Pi = u[:,0:3]@u[:,0:3].T
    if case == 1:
        fi = Pi@data
    else:
        fi = (np.identity(len(data)) - Pi)@data
    di = np.sqrt(np.sum(np.multiply(fi,fi),axis=0))
    dd = np.sqrt(np.sum(np.multiply(data,data),axis=0))
    ri = np.divide(di,dd)
    outliers = np.argwhere(ri>0.9)
    sort = np.argsort(ri)
    if case == 1:
        similar = sort[-100:]
    else:
        similar = sort[:100]
    return similar, outliers

def plotimages(similar,grid,filename):
    cap = cv2.VideoCapture(filename) #video_name is the video being called
    plt.figure(figsize = (20,20))
    i = 1
    for frames in similar:
        cap.set(1,frames); # Where frame_no is the frame you want
        ret, frame = cap.read() # Read the frame
        plt.subplot(grid, grid, i)
        plt.imshow(frame)
        
        i +=1
        plt.axis('off')

def plot_scelframe(frame_nr,frame_num,f_reduced,filename,x_reduced,y_reduced):
    get_start_nr=np.argwhere(frame_num==frame_nr)
    i=get_start_nr[0][0]
    #i=11
    while f_reduced[i]==frame_nr:    
        cap = cv2.VideoCapture(filename)
        cap.set(1,int(f_reduced[i]))
        ret, frame = cap.read()
        plt.imshow(frame)
        plt.scatter(x_reduced[:][i]*640,y_reduced[:][i]*360,c='r')
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
    plt.figure()
    plt.plot(range_n_clusters, silhouette_avg_n_clusters)
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("silhouette score")
    plt.show()

def PCA(sFeatures):
    featcentered = sFeatures-np.mean(sFeatures.T)*np.ones(len(sFeatures[0]))
    u, s, v = linalg.svd(featcentered,full_matrices=False)
    sx = np.diag(s)
    v = v.T
    lowdim = sx[0:100,0:100]@v[:,0:100].T
    return lowdim


def kmeans_tsne(clusters,lowdim):
    kmeans = KMeans(n_clusters=clusters)
    kmeans.fit(lowdim.T)
    y_kmeans = kmeans.predict(lowdim.T)

    tsne = TSNE(n_components=2)
    y = tsne.fit_transform(lowdim.T) 
    plt.figure()
    plt.scatter(y[:,0],y[:,1],c=y_kmeans,s=10,cmap='viridis')
    
    # legend1 = ax.legend(*scatter.legend_elements(),loc="lower left", title="Classes")
    # ax.add_artist(legend1)
    return y_kmeans,y

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

    sFeatures=np.zeros((4,len(scelFrames)))
    j=0
    for i in scelFrames:
       
        sFeatures[0,j]=f_reduced.count(i) #Number of sceletons in frame
        if sFeatures[0,j]>0:
            indexes=[element for element, x in enumerate(f_reduced) if x == i]
            countpoint=0
            val1=0
            for frame in indexes:
                count=np.count_nonzero(x_reduced[frame])
                countpoint+=count #Total number of sceleton points in a frame
                val1+=sum(x_reduced[frame])/count

            sFeatures[1,j]=countpoint
            sFeatures[2,j]=sFeatures[1,j]/sFeatures[0,j]
            sFeatures[3,j]=val1/len(indexes)

        else:
            sFeatures[1,j]=0
            sFeatures[2,j]=0
            sFeatures[3,j]=0
        j=j+1
   

    return sFeatures,f_reduced,x_reduced,y_reduced