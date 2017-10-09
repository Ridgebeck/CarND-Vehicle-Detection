import glob
import functions
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import time
from sklearn.externals import joblib
from scipy.ndimage.measurements import label


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


# path to images
#searchpath = 'test_images/*'
#example_images = glob.glob(searchpath)

# define empty lists for images and titles
#images = []
#titles = []


def process_image(img):
    t1 = time.time()
    #searched_windows = 0
    # threshold for heatmap
    threshold = 2
    #img = mpimg.imread(img_src)
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
    scale = 1.5
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
    # draw labels on a copy of the image
    labeled_image = functions.draw_labeled_bboxes(draw_img, labels)

    return labeled_image



from moviepy.editor import VideoFileClip
from IPython.display import HTML

test_output = 'test.mp4'

#clip = VideoFileClip("test_video.mp4")
clip = VideoFileClip("project_video.mp4")
test_clip = clip.fl_image(process_image)

test_clip.write_videofile(test_output, audio = False)
