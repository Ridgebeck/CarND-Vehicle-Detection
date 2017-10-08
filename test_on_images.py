import glob
import functions
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import time
from sklearn.externals import joblib


def convert_color(img, conv='RGB2YCrCb'):
    if conv == 'RGB2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    if conv == 'BGR2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    if conv == 'RGB2LUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)


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

# define large search windows for close vehicles 
xy_window_near = (256, 256)
y_start_stop_near = [350, 670]
overlap_near = 0.75

# define medium search windows for mid range vehicles 
xy_window_middle = (128, 128)
y_start_stop_middle = [350, 542]
overlap_middle = 0.5

# define small search windows for further away vehicles 
xy_window_far = (64, 64)
y_start_stop_far = [414, 478]
overlap_far = 0.5

searchpath = 'test_images/*'
example_images = glob.glob(searchpath)
images = []
titles = []
#y_start_stop = [400, 656]
#overlap = 0.5
for img_src in example_images:
    t1 = time.time()
    img = mpimg.imread(img_src)
    draw_img = np.copy(img)
    img = img.astype(np.float32)/255 # Convert from jpg to png values
    print(np.min(img), np.max(img))

    #windows = functions.slide_window(img, x_start_stop=[None, None], y_start_stop=y_start_stop, 
    #                xy_window=(128, 128), xy_overlap=(overlap, overlap))

    windows_near = functions.slide_window(img, x_start_stop=[None, None], y_start_stop=y_start_stop_near, 
                    xy_window=xy_window_near, xy_overlap=(overlap_near, overlap_near))
    
    windows_middle = functions.slide_window(img, x_start_stop=[None, None], y_start_stop=y_start_stop_middle, 
                    xy_window=xy_window_middle, xy_overlap=(overlap_middle, overlap_middle))

    windows_far = functions.slide_window(img, x_start_stop=[None, None], y_start_stop=y_start_stop_far, 
                    xy_window=xy_window_far, xy_overlap=(overlap_far, overlap_far))

    windows = windows_near + windows_middle + windows_far
    
    hot_windows = functions.search_windows(img, windows, svc, X_scaler, color_space=color_space, 
                    spatial_size=spatial_size, hist_bins=hist_bins, 
                    orient=orient, pix_per_cell=pix_per_cell,
                    cell_per_block=cell_per_block, 
                    hog_channel=hog_channel, spatial_feat=spatial_feat, 
                    hist_feat=hist_feat, hog_feat=hog_feat)

    window_img = functions.draw_boxes(draw_img, hot_windows, color=(0, 0, 255), thick=6)
    images.append(window_img)
    titles.append('')
    print(round(time.time()-t1, 2), "seconds to process one image searching", len(windows), "windows.")

fig = plt.figure(figsize=(17, 18))#, dpi=300)
functions.visualize(fig, 3, 2, images, titles)


# CALCULATE HOG FEATURES ONLY ONCE

out_images = []
out_maps = []
out_titles = []
out_boxes = []
ystart = 350
ystop = 670 #542
scale = 1 #1.5

#pix_per_cell = 6
#cell_per_block = 2

for img_src in example_images:
    img_boxes = []
    t = time.time()
    count = 0
    img = mpimg.imread(img_src)
    draw_img = np.copy(img)
    heatmap = np.zeros_like(img[:,:,0])
    img = img.astype(np.float32)/255 # Convert from jpg to png values

    img_tosearch = img[ystart:ystop,:,:]
    ctrans_tosearch = convert_color(img_tosearch, conv='RGB2YCrCb')
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
    
    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]                                
    ch3 = ctrans_tosearch[:,:,2]
    
    nxblocks = (ch1.shape[1] // pix_per_cell) - 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - 1
    nfeat_per_block = orient * cell_per_block ** 2
    window = 64
    nblocks_per_window = (window // pix_per_cell) - 1
    cells_per_step = 2
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step

    hog1 = functions.get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = functions.get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = functions.get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
                                                
    for xb in range(nxsteps):
        for yb in range(nysteps):
            count += 1
            ypos = yb * cells_per_step
            xpos = xb * cells_per_step

            hog_feat1 = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feat2 = hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feat3 = hog3[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos * pix_per_cell
            ytop = ypos * pix_per_cell

            subimg = cv2.resize(ctrans_tosearch[ytop:ytop + window, xleft:xleft + window], (64, 64))

            spatial_features = functions.bin_spatial(subimg, size=spatial_size)
            hist_features = functions.color_hist(subimg, nbins=hist_bins)

            test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))

            test_prediction = svc.predict(test_features)

            if test_prediction == 1:
                xbox_left = np.int(xleft * scale)
                ytop_draw = np.int(ytop * scale)
                win_draw = np.int(window * scale)
                cv2.rectangle(draw_img, (xbox_left, ytop_draw + ystart), (xbox_left + win_draw, ytop_draw + win_draw + ystart),(0, 0, 255))
                img_boxes.append(((xbox_left, ytop_draw + ystart), (xbox_left + win_draw, ytop_draw + win_draw + ystart)))
                heatmap[ytop_draw + ystart:ytop_draw + win_draw + ystart, xbox_left:xbox_left + win_draw] += 1


    print(time.time()-t, "seconds to run. Total windows:", count)

    out_images.append(draw_img)

    out_titles.append(img_src[-12:])
    out_titles.append(img_src[-12:])

    out_images.append(heatmap)
    out_maps.append(heatmap)
    out_boxes.append(img_boxes)

fig = plt.figure(figsize = (17, 20))
functions.visualize(fig, 6, 2, out_images, out_titles)
