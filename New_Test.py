import glob
import functions
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import time
from sklearn.externals import joblib

"""
def convert_color(img, conv='RGB2YCrCb'):
    if conv == 'RGB2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    if conv == 'BGR2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    if conv == 'RGB2LUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
"""

# Load trained SVC from pickled file
svc = joblib.load('trained_svc.pkl') 
# Load scaler from training data
X_scaler = joblib.load('x_scaler.pkl')


# Define feature parameters
color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
spatial_size = (16, 16) # Spatial binning dimensions
hist_bins = 32    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off


from scipy.ndimage.measurements import label

# path to images
searchpath = 'test_images/*'
example_images = glob.glob(searchpath)

# define empty lists for images and titles
images = []
titles = []

# threshold for heatmap
threshold = 2


for img_src in example_images:
    t1 = time.time()
    searched_windows = 0
    img = mpimg.imread(img_src)
    draw_img = np.copy(img)

    # Search for smaller appearing vehicles in the distance
    ystart = 414
    ystop = 414 + 96
    scale = 1
    cells_per_step = 2
    windows_far, heatmap_far, searched_windows_far = functions.find_cars(img, cells_per_step, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)

    # Search for middle of the visual field with medium window size
    ystart = 350
    ystop = 350 + 192
    scale = 2
    cells_per_step = 2
    windows_middle, heatmap_middle, searched_windows_middle = functions.find_cars(img, cells_per_step, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)

    # Search for close to the camera for vehicles that appear large
    ystart = 350
    ystop = 670
    scale = 3
    cells_per_step = 2
    windows_near, heatmap_near, searched_windows_near = functions.find_cars(img, cells_per_step, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)

    # combine all windows
    windows = windows_far + windows_middle + windows_near

    # combine all heatmaps
    heatmap = heatmap_far + heatmap_middle + heatmap_near

    # apply threshold to the heatmap
    heatmap = functions.apply_threshold(heatmap, threshold)
    # apply labels to the heatmap
    labels = label(heatmap)


    # Sum up all windows that were searched
    searched_windows = searched_windows_far + searched_windows_middle + searched_windows_near

    # Draw boxes in the image around all hot windows
    window_img = functions.draw_boxes(draw_img, windows, color=(0, 255, 0), thick=6)

    # Append titles
    titles.append(img_src[-12:])
    titles.append(img_src[-12:])
    titles.append(img_src[-12:])

    # Append Images with windows to image list
    images.append(window_img)
    # Append Images with windows to image list
    images.append(heatmap)
    # Append Images with windows to image list
    images.append(labels[0])

    # Show how many windows were searched and how many were detected to conatin cars 
    print(round(time.time()-t1, 2), "seconds to process one image searching", searched_windows, "windows. Found", labels[1], "cars in", len(windows), "windows.")


fig = plt.figure(figsize=(17, 18))#, dpi=300)
functions.visualize(fig, 6, 3, images, titles)
