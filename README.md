# Udacity Self-Driving Car Nanodegree Program 

# **5. Vehicle Detection **

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_not_car.png
[image2]: ./output_images/HOG_example.png
[image3]: ./output_images/sliding_windows.png
[image4]: ./output_images/find_cars.png
[image5]: ./output_images/bboxes_and_heat5.png
[image6]: ./output_images/bboxes_and_heat4.png
[image7]: ./output_images/bboxes_and_heat2.png
[image8]: ./output_images/bboxes_and_heat6.png
[image9]: ./output_images/bboxes_and_heat3.png
[image10]: ./output_images/bboxes_and_heat1.png

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the section "scikit-image HOG" of the IPython notebook in the function `get_hog_features()`.  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

A linear SVC classifier has been defined in the section `HOG Classify` of the IPython notebook. 

I started with `color_space = 'RGB'` and `hog_channel = 0`. The resulted SVM model did not predict well on the test images and produced too many false positives. 

I ended up with `color_space = 'YCrCb'` and `hog_channel = 'ALL'`, which reduced substantially the false positives.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVC using the two given datasets: `vehicles` and `non-vehicles`. The datasets were randomnized and split by 80/20% as training and testing datasets. The resulted model was saved in a file `svc_pickle.p`, along with the hyper-parameters in the following:

```
color_space = 'YCrCb'
orient = 9
pix_per_cell = 8
cell_per_block = 2
hog_channel = 'ALL'
spatial_size = (32, 32)
hist_bins = 32
```

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

A `slide-window()` function has been defined I to search  window positions over the image systematically - from left to right and top to bottom, as illustrated in the following:

![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on three scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here is an example image:

![alt text][image4]

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/lEoP62bZaQY/0.jpg)](https://www.youtube.com/watch?v=lEoP62bZaQY)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

Here are six frames and their corresponding heatmaps, with the resulting bounding boxes labelled by `scipy.ndimage.measurements.label()`:

![alt text][image5]
![alt text][image6]
![alt text][image7]
![alt text][image8]
![alt text][image9]
![alt text][image10]

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The solution works when the cars are visually distinguishable from the backgrounds by the combinations of colors and gradients. This pure computer vision approach, where thresholds and HOG parameters have to be carefully chosen, is not robust for practical self-driving, especially under complex road or weather conditions. I experimented the `Tensorflow Object Detection API` and fed the images to an `ssd_mobilenet_v1_coco` pre-trained model. The  results are more accurate, and the detection speed is fast. Deep learning is a way to go for the needed robustness.
