#In[]
import numpy as np
from pandas import DataFrame
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

mat = sio.loadmat('esqueletosveryslow.mat')
mat_complete = sio.loadmat('esqueletosveryslow_complete.mat')
data = sio.loadmat('girosmallveryslow2.mp4_features.mat')
data = np.double(data['features'])
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

def plotimages(similar):
    cap = cv2.VideoCapture(filename) #video_name is the video being called
    plt.figure(figsize = (20,20))
    i=1
    for frames in similar:
        cap.set(1,frames); # Where frame_no is the frame you want
        ret, frame = cap.read() # Read the frame
        plt.subplot(4, 4, i)
        plt.imshow(frame)
        
        i +=1
        plt.axis('off')

def plot_scelframe(frame_nr):
    get_start_nr=np.argwhere(frame_num_c==frame_nr)
    i=get_start_nr[0]
    #i=11
    while frame_num_c[i]==2:    
        cap = cv2.VideoCapture(filename)
        cap.set(1,frame_num_c[i])
        ret, frame = cap.read()
        plt.imshow(frame)
        plt.scatter(x_c[:,i],y_c[:,i],c='r')
        plt.show()
        plt.pause(0.3)
        i+=1


#%%

skeletons = mat["skeldata"]
base = np.double(data[:,5895:5907])
frame_num = skeletons[0,:]
x = skeletons[1::3,:]
y = skeletons[2::3,:]


skeletons_c = mat_complete["skeldata"]
frame_num_c = skeletons[0,:]
x_c = skeletons[1::3,:]*640
y_c = skeletons[2::3,:]*360


similar,outliers=findSimilar(base,data,1)
plotimages(similar,filename)


#In[]
fcentered=data-np.mean(data.T)*np.ones(10482)
u, s, v=LNG.svd(fcentered,full_matrices=False)
sx= np.diag(s)
v=v.T
lowdim=sx[0:100,0:100]@v[:,0:100].T
c=KMeans(n_clusters=20).fit(lowdim.T)
kmeans = KMeans(n_clusters=20)
kmeans.fit(lowdim.T)
y_kmeans = kmeans.predict(lowdim.T)

tsne = TSNE(n_components=2)
y = tsne.fit_transform(lowdim.T) 
plt.scatter(y[:,0],y[:,1],c=y_kmeans,s=10,cmap='viridis')

#%%
clust=[]
for i in range(20):
    clust.append(np.argwhere(y_kmeans==i))
randnr=random.randint(0,19)
for i in range(16):
    
test=clust[randnr]
randnr=random.randint(0,100)

plotimages(test[randnr:randnr+16,0])