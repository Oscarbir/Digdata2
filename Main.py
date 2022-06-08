import numpy as np
from pandas import DataFrame
import scipy.io as sio
import cv2 
from tqdm import tqdm
import time
import math
import matplotlib.pyplot as plt
from yaml import load

mat = sio.loadmat('PBD_Share/EurosportAll/esqueletosveryslow.mat')
mat_complete = sio.loadmat('PBD_Share/EurosportAll/esqueletosveryslow_complete.mat')

# Missing data
skeletons = mat["skeldata"]
frame_num = skeletons[0,:]
x = skeletons[1::3,:]
y = skeletons[2::3,:]

# Complete data
skeletons_c = mat_complete["skeldata"]
frame_num_c = skeletons[0,:]
x_c = skeletons[1::3,:]
y_c = skeletons[2::3,:]