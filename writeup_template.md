
---

**Vehicle Detection Project**

The overall goal of this project was to detect cars in a video and mark them visually with bounding boxes. In order to achieve that I started with a couple test images derived from that video and built a pipeline to process those images in multiple steps. The steps are summarized as follows:

* A labeled training set of images containing cars and no cars was used to train a linear SVC with the help of a Histogram of Oriented Grandients (HOG) feature extraction.
* The test images were analyzed via a sliding window technique and a prediction on those windows was performed by the trained SVC in order to indentify cars in the image. I used different sizes of windows (or different scales on the image) in order to increase accuracy.
* Those detected areas ("hot pixels") were then applied onto a heatmap and a threshold was used to remove false positives.
* The thresholded heatmap was then labeled in order to calculate the number and location of the cars in the image.
* The label data was then used to draw bounding boxes in the detected locations over a copy of the original image to demonstrate that a detection of the cars in the image was successful.
* The pipeline was then tested on the images and the parameters were tuned to achieve good results with a high reliability and less outliers.
* After that the pipline was modified and applied to the project video. The result was the video "output_video.mp4".

[//]: # (Image References)
[image1]: ./examples/data_set_example.jpg
[image2]: ./examples/HOG_features.jpg
[image3]: ./examples/boxes_heatmap_label.jpg
[image4]: ./examples/sliding_windows.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4 

---

### Train a SVC with the help of Histogram of Oriented Gradients (HOG) feature extraction

The code for this step is contained in the file `train_SVC.py`. All help functions can be found in `functions.py`.

#### Reading in the training data images

I started by reading in all the `vehicle` and `non-vehicle` images. I then verified the total amount of vehicle and non-vehicle images and looked randomly at some of the data to get an impresion of the quality of the data set. Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters, especially `orient`, `pixels_per_cell`, `cells_per_block`, and `hog_channel`.  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like and what influence each parameter has on the output. I found out that I was able to achieve the best performance with the following parameters:
`color_space = YCrCb`
`orient = 9`
`pixels_per_cell = 8`
`cell_per_block = 2`
`hog_channel = 0`
`spatial_size = (16,16)`
`hist_bins = (32)`
`spatial_feat = True`
`hist_feat = True`
`hog_feat = True`

I tried the different color spaces and it seemed that RGB and YCrCb delivered the best results, while YCrCb was a bit more reliable. It seems that the built-in algorithms for the HOG feature extraction work a bit better in YCrCb. I set the orientations to the maximum recommended amount of 9. I could see a noticebale difference for values above 9. The hog channel 0 seemed to be working better than 1 and 2, but "ALL" delivered the best performance. Looking at all channels also adds more reliability if a feature was not shown well in one specific channel. All features were enabled. `spatial_size`, `pixels_per_cell`, `cell_per_block`, and `hist_bins` were selected based on how much they increase the result and how much the influenced the performance. For that purpose I stopped the time for the calcuation of the features and compared it against each other.

Here is an example of an image using the above stated parameters:


![alt text][image2]

Then, I had to stack the features and apply a (per column) X scaler and create a features vector.

#### Training the SVC

I split up the data into randomized training and test sets and fitted it to the linear SVC. I then trained with the same parameters a couple of times and looked at the resulting test accuracy. I started with around 70% and ended (with the above mentioned parameters) at roughly 99% accuracy. The trained SVC as well as the X scaler was then saved in a pickle file.

### Sliding Window Search

The code for this step is contained in the file `test_on_images_1.py` and `test_on_images_2.py`. All help functions can be found in `functions.py`.

#### Sliding windows on the test images

First, I loaded the trained SVC and the X scaler. Then I applied the `slide_window()` function in `functions.py` to create windows in a specific size in a specific area with a defined overlap.  I then visualized the windows and defined a suitable size based on the camera position used in the sample images.

I decided to look closer to the horizon with small windows of 64 by 64 pixels and an overlap of 50%. In the middle range I decided to go with 128 by 128 pixels and an overlap of also 50%. Clsoe to the front of the  car I searched in 256 by 256 windows with an overlap of 75%. I achieved slightly better results with an overlap of 75% for the medium and smaller images, but decided to keep the total amount of windows down to reduce processing time. All windows were then searched for possibly containing cars by the trained linear SVC. I used the helper function `search_windows` which returned the "hot windows" where cars were detected.

In order to optimize performance I decided to calculate the HOG once per image and then apply the sliding window approach. This was first done with only one window size (one scale) to verify performance. I then added the feature of a heatmap showing all pixels with positive car detections. I then applied a threshold to the heatmap to only show the pixels where at least 2 windows were overlapping in order to reduce false positives. After that, I applied labels to the heatmap to see how many cars were found in the picture. Finally, I drew a single box around the found spots in a copy of the original image. After I verified the performance by looking at the results of each step, I went back and implemented the three different window sizes by applying different scaling factors to the image. The implementation for processing the sample images was put into the function `find_cars()`. This function returns the windows where the SVC detected a car, the heatmap of the image and the total amount of windows searched.

You can see here the different steps performed on the image on one of the sample images:

![alt text][image3]


### Video Implementation

The code for this step is contained in the file `vehicle_detection.py`. All help functions can be found in `functions.py`.

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

The function `process_image()` was created in order to process only one image at a time and returns an image with bounding boxes around the detected cars. The images were then saved as a video that can be found here: [link to my video result](./project_video.mp4)

The same strategy as before was applied (heatmap of positive detections, apply threshold, apply `scipy.ndimage.measurements.label()`) in order to remove false positives and to combine overlapping bounding boxes.

#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The pipeline detects cars reliably. The processing time per image is fairly high and would not be suitable for real time processing. 

