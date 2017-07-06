## Writeup Advanced Lane Finding

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image0]: ./writeup_images/test1.jpg "Original test image"
[image1]: ./writeup_images/undistort_output.jpg "Undistorted"
[image2]: ./writeup_images/test1.jpg "Road Transformed"
[image3]: ./writeup_images/binary0.jpg "Binary Example"
[image4]: ./writeup_images/warped_straight_lines.jpg "Warp Example"
[image5]: ./writeup_images/color_fit_lines.jpg "Fit Visual"
[image6]: ./writeup_images/example_output.jpg "Output"
[video1]: ./output1_tracked.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the first code cell of the IPython notebook located in lines 7 through 55 of the file called `camera_cal.py`.  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:

![test_image][image0]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

I applied the described distortion correction to the test image using the `cv2.undistort()` function and obtained this result:

![alt text][image1]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines 120 through 129 in `video_gen.py`).  Here's an example of my output for this step.

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes the cv2 function called `cv2.getPerspectiveTransform(src, dst)`, which appears in line 158 in the file `video_gen.py`.  The function takes as inputs source (`src`) and destination (`dst`) points and returns a transformation matrix M. This matrix can be used to transform an input image (`img`) with the open cv function  cv2.warpPerspective(img, M, img_size) which was used in line 161. I chose to hardcode the source and destination points in the following manner:

```python
# four source points
src = np.float32([[img.shape[1]*(mid_top-mid_width/2), img.shape[0]*height_pct], /
                  [img.shape[1]*(mid_top+mid_width/2), img.shape[0]*height_pct], /
                  [img.shape[1]*(mid_bot+bot_width/2), img.shape[0]*bottom_trim],/
                  [img.shape[1]*(mid_bot-bot_width/2), img.shape[0]*bottom_trim]])
offset = img_size[0]*.25
# four destination points
dst = np.float32([[offset, 0], /
                  [img_size[0]-offset, 0], /
                  [img_size[0]-offset, img_size[1]], /
                  [offset, img_size[1]]])
```

This resulted in the following source and destination points:

| Source        | Destination   |
|:-------------:|:-------------:|
| 585, 460      | 320, 0        |
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

I used the convolution or sliding window method describe in the lessons (second method). For this I use a tracker class in the file `tracker.py`. It finds the right and left lane center points by using a window template that convolutes the left and right part of the image on 9 different layers, which are the vertical slices. Using the found averaged left and right centroids, the numpy method polyfit is used to find the coefficients of a 2nd order polynomial (lines 175 to 233). This resulted in an image like this:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines 256 through 269 in my code in `video_gen.py`. For this I followed the tutorial http://www.intmath.com/applications-differentiation/8-radius-curvature.php and used ratios between image pixels and real world distances in meters.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines 235 through 276 in my code in `vide_gen.py`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./output1_tracked.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here are the steps I took to process the raw input video and find the lane lines:

1. **Camera calibration (`camera_cal.py`)**
The information of camera distortions were obtained using a chessboard images and the open cv functions.

2. **Distortion correction (`video_gen.py`)**
The distortion correction is required to get the correct curvature of the lane lines in final processing steps.
Therefore, the distortion caused by the camera lense was corrected using open cv functions and the calibration data from the first step.

3. **Color/gradient threshold (`video_gen.py`)**
A binary threshold image is the first important step to detect the lane lines. It should clearly show the left and right lines which is not always possible under certain wheater conditions (changing sunlight/shadow). To achieve good results I used the x and y gradients using the `cv2.sobel`. I also used a color threshold, where the S channel of the HLS color space and the V channel of the HSV color space provide the best detection of the lane lines (yellow and white).

4. **Perspective transform (`video_gen.py`)**
To find a perspective transform that yields parallel lines on a straight lanes it is important to find "good" source and destination points. If the image size changes these points need to be adjusted.

5. **Detect lane lines using a sliding window approach (`tracker.py`)**
For the convolution of the sliding window approach the starting positions of the bottom layer are not shared among the video frames. Storing the locations could speed up the whole process.
It is important that the two lane lines (right and left) are parallel over all frames and that the locations are not chaning too much. Otherwise the hard coded regions for the sliding window will fail.

6. **Determine the lane curvature (`video_gen.py`)**
The lane curvature is the result of a ratio between the pixel values of the image (width and height of the lane in pixels) and the lane's acutal size in real world (meters).

```python
# get curvature in meters
# tutorial: http://www.intmath.com/applications-differentiation/8-radius-curvature.php
curveFitCr = np.polyfit(np.array(resYvals, np.float32)*ymPerPixel, np.array(leftx, np.float32)*xmPerPixel, 2)
# for the curvature radius only the left line is checked (Todo: averaging of both lines)
curverad = ((1 + (2*curveFitCr[0]*yvals[-1]*ymPerPixel + curveFitCr[1])**2)**1.5) / np.absolute(2*curveFitCr[0])
```
