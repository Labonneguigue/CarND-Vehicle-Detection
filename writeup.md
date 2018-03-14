
# Vehicle Detection Project

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[uml]: ./plantuml/vehicle_detection.png
[sliding]: ./output_images/sliding_window.png
[hog]: ./output_images/hog.png
[hog_car]: ./output_images/hog_car.png
[hog_ncar]: ./output_images/hog_ncar.png
[hog_car16]: ./output_images/hog_car16.png
[scaler]: ./output_images/scaler.png
[result]: ./output_images/result.png

[one]: ./output_images/1.jpg
[four]: ./output_images/4.jpg
[five]: ./output_images/5.jpg
[six]: ./output_images/6.jpg
[heat1]: ./output_images/heat1.jpg
[heat4]: ./output_images/heat4.jpg
[heat5]: ./output_images/heat5.jpg
[heat6]: ./output_images/heat6.jpg

[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points

##### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Code Architecture

My code is separated in different files. Here is a diagram to represent
the interactions between them.

![alt text][uml]

### Image features

#### Histogram of Oriented Gradients (HOG)

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][hog]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the grayscale conversion from RGB color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][hog_car]

and without car

![alt text][hog_ncar]

We can clearly observe different pattern in the orientation of the gradient
of each cell.

I also tried to change the number of pixels per cell to 16.

![alt text][hog_car16]

But we see that it is not clear whether we have a car here or not by just
looking at the HOG features.

After testing, I didn't experience any improvement by increasing the number of orientations but I felt a loss in accuracy when I downscaled the cells per block
to 1.

#### Spatial binning

By resizing the image into a 32 by 32 image and unravelling the 3D array that
constitute the image, I obtain color bins.



#### Normalization

Since I am aggregation all of these different features, I need to scale them
so that they are comparable. To do so, I use the `StandardScaler()` provided
by the `sklearn.preprocessing` library.

I first need to `fit()` a scaler with the training data and then for each images
I `transform()` the extracted features to scale them correctly before classify
the image.

In the `Utils.py` file:

```python
@staticmethod
    def NormalizeFeatures(car_features, notcar_features, scaler):
        '''
        Normalize each set of features so that they have equal variance
        and 0 mean.
        '''
```

![alt text][scaler]


#### Classifier

I trained a linear SVM using the `LinearSVC()` class provided by the sklearn.svm
library.

I also tried to use a non linear classifier that way :

```python
    parameters = {'kernel':('linear', 'rbf'), 'C':[0.8,0.9, 1, 1.1]}
    svr = svm.SVC()
    self.classifier = GridSearchCV(svr, parameters)
```

I obtained great results but the time to `predict()` exploded. It wasn't a
viable option. The GridSearchCV returned the optinal kernel to be `rbf` and
the `C` argument to be 0.9.

### Sliding Window Search

The code for this step is contained in the `Utils.py` file as a static method of the Utils class:

```python
@staticmethod
    def GetSlidingWindows(img, x_start_stop=[None, None], y_start_stop=[None, None],
                        xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
        pass
```
This function is mainly the one given by Udacity.

I decided to start with only 50% of overlapping between 2 windows but I then felt like I didn't have enough windows to detect accurately the vehicles. I then augmented the overlapping in both x and y directions to 75% then slowed down the detection to 20 seconds per images with about 2000 sliding windows.
I finally tried to reduce the x overlapping to 50% and ended up with 256 sliding windows and performing the detection on these was reasonably slow.

![alt text][sliding]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched HOG features using YUV color space but removed removed the  spatially binned color and histograms of color in the feature vector because the size of the feature vector was too big and was crashing the program. It still provided acceptable results. Here are some example images:

* First

![alt text][one]

* Second

![alt text][four]

* Third

![alt text][five]

* Fourth

![alt text][six]

---

### Video Implementation

My pipeline can be applied to videos. The only difference is that I apply some
filtering on the detections to reduce the instability and flickering.

Here's a [link to my video result](./output_videos/project_video.mp4)

### Filtering

#### Heat Map


I recorded the positions of positive detections in each frame of the video. From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.


#### Labelling


I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap on 4 frames:

![alt text][heat1]

![alt text][heat4]

![alt text][heat5]

![alt text][heat6]

---

### Discussion

The HOG technique is quite unstable and although it works ok with this video,
I would suspect this implementation to generalize pretty badly.

The region used for creating the bounding boxes is hard-coded to optimize the
success of this video but would probably fail if used outside the highway.

If I had to do it again, I would investigate a deep learning approach that would learn much more complex classification boundary. I have been impressed by the YOLO and SDD methods.

---
---

#### Personal Notes

Histogram of colors : https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/2b62a1c3-e151-4a0e-b6b6-e424fa46ceab/lessons/fd66c083-4ccb-4fe3-bda1-c29db76f50a0/concepts/4f0692c6-e22d-4f28-b5d0-7990a4d8de86

HOG: https://www.youtube.com/watch?v=7S5qXET179I

Udacity dataset : https://github.com/udacity/self-driving-car/tree/master/annotations
