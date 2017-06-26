import os
import glob
import functions


import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

"""
import cv2
import time
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
"""

# LOCATE AND LOAD ALL TRAINING IMAGES

# Look in defined folder and all present subfolders for vehicle images
basedir = 'vehicles/'
subfolders = os.listdir(basedir)
cars = []
for imtype in subfolders:
    cars.extend(glob.glob(basedir + imtype + '/*'))

print ('Number of vehicle images found:', len(cars))
with open("cars.txt", 'w') as f:
    for fn in cars:
        f.write(fn + '\n')

# Look in defined folder and all present subfolders for non-vehicle images
basedir = 'non-vehicles/'
subfolders = os.listdir(basedir)
notcars = []
for imtype in subfolders:
    notcars.extend(glob.glob(basedir + imtype + '/*'))

print ('Number of non-vehicle images found:', len(notcars))
with open("notcars.txt", 'w') as f:
    for fn in notcars:
        f.write(fn + '\n')


# Chose random car / not-car indices
car_ind = np.random.randint(0, len(cars))
notcar_ind = np.random.randint(0, len(notcars))

# Read in car / not-car images
car_image = mpimg.imread(cars[car_ind])
notcar_image = mpimg.imread(notcars[notcar_ind])

# Define feature parameters
color_space = 'RGB' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 6  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = 0 # Can be 0, 1, 2, or "ALL"
spatial_size = (16, 16) # Spatial binning dimensions
hist_bins = 16    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off

#y_start_stop = [480, 700] # Min and max in y to search in slide_window()







