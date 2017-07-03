import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import pickle
#from tracker import tracker


# Read in the saved object points and image points
dist_pickle = pickle.load(open("camera_cal/calibration_pickle.p", "rb"))
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]

# Gradient functions
# Define a function that takes an image, gradient orientation,
# and threshold min / max values.
def abs_sobel_thresh(image, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    # Return the result
    return binary_output


# Define a function to return the magnitude of the gradient
# for a given sobel kernel size and threshold values
def mag_thresh(image, sobel_kernel=3, mag_thresh=(0, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255
    gradmag = (gradmag/scale_factor).astype(np.uint8)
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

    # Return the binary image
    return binary_output


# Define a function to threshold an image for a given range and Sobel kernel
def dir_threshold(image, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction,
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    # Return the binary image
    return binary_output


# Color function
# Define a function that thresholds the S-channel of HLS
# and the V-channel of HSV color space
# Use exclusive lower bound (>) and inclusive upper (<=)
def color_threshold(image, sthresh=(0, 255), vthresh=(0, 255)):
    # 1) Convert to HLS color space
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    # separate the S channel
    s_channel = hls[:,:,2]
    # 2) Apply a threshold to the S channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel > sthresh[0]) & (s_channel <= sthresh[1])] = 1

    # 1) Convert to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    # separate the V channel
    v_channel = hsv[:,:,2]
    # 2) Apply a threshold to the V channel
    v_binary = np.zeros_like(v_channel)
    v_binary[(v_channel > vthresh[0]) & (v_channel <= vthresh[1])] = 1

    # 3) Return a binary image of threshold result
    binary_output = np.zeros_like(s_channel)
    # 4) Combine S and V channels
    binary_output[(s_binary == 1) & (v_binary == 1)] = 1
    return binary_output

def window_mask(width, height, img_ref, center, level):
    output = np.zeros_like(img_ref)
    output[int(img_ref.shape[0]-(level+1)*height):int(img_ref.shape[0]-level*height),max(0,int(center-width))]


# Make a list of test images
images = glob.glob('./test_images/test*.jpg')

for idx, fname in enumerate(images):
    # read in image
    img = cv2.imread(fname)
    # undistort the image
    img = cv2.undistort(img, mtx, dist, None, mtx)

    # preprocess image and generate binary pixel of interests
    # kernel size of the sobel filter
    ksize = 3
    # gradient thresholds
    gradx = abs_sobel_thresh(img, orient='x', sobel_kernel=ksize, thresh=(30, 80)) #12 255
    grady = abs_sobel_thresh(img, orient='y', sobel_kernel=ksize, thresh=(20, 100)) #25 255
    # color threshold
    c_binary = color_threshold(img, sthresh=(100, 255), vthresh=(50, 255))
    # combine the gradient and color thresholds
    preprocessImage = np.zeros_like(img[:,:,0])
    # set the indexed pixels to 255 because we store a jpg (no png)
    preprocessImage[((gradx == 1) & (grady == 1)) | (c_binary == 1)] = 255


    # perspective transform
    # define perspective transform area (trapezoid)
    # Grab the image shape
    img_size = (img.shape[1], img.shape[0])
    bot_width = .76 # percent of bottom trapezoid height
    mid_width = .08 # percent of middle trapezoid height
    height_pct = .62 # percent for trapezoid height (from top to bottom)
    bottom_trim = .935 # percent from top to bottom to avoid car hood

    mid_bot = 0.5 # 0.55
    mid_top = 0.5 # 0.52
    # four source points in the original camera image
    src = np.float32([[img.shape[1]*(mid_top-mid_width/2), img.shape[0]*height_pct],[img.shape[1]*(mid_top+mid_width/2), img.shape[0]*height_pct],[img.shape[1]*(mid_bot+bot_width/2), img.shape[0]*bottom_trim],[img.shape[1]*(mid_bot-bot_width/2), img.shape[0]*bottom_trim]])
    offset = img_size[0]*.25
    # four destination points
    dst = np.float32([[offset, 0],[img_size[0]-offset, 0],[img_size[0]-offset, img_size[1]],[offset, img_size[1]]])

    # DEBUG show the produced vertices
    #vertices_src = np.array([[(img.shape[1]*(mid_top-mid_width/2), img.shape[0]*height_pct),(img.shape[1]*(mid_top+mid_width/2), img.shape[0]*height_pct), (img.shape[1]*(mid_bot+bot_width/2), img.shape[0]*bottom_trim), (img.shape[1]*(mid_bot-bot_width/2), img.shape[0]*bottom_trim)]], dtype=np.int32)
    #vertices_dst = np.array([[(offset, 0),(img_size[0]-offset, 0), (img_size[0]-offset, img_size[1]), (offset, img_size[1])]], dtype=np.int32)
    #image_roi = np.copy(img)
    #cv2.polylines(image_roi, vertices_dst, 1, (255, 0, 0), thickness=3)
    #plt.imshow(image_roi)
    #plt.show()

    # Given src and dst points, calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(preprocessImage, M, img_size)


    histogram = np.sum(warped[img.shape[0]//2:,:], axis=0)
    plt.plot(histogram)
    plt.show()

    # write the undistored images
    result = warped
    write_name = './test_images/tracked'+str(idx)+'.jpg'
    cv2.imwrite(write_name, result)




# # Read in an image
# image = mpimg.imread('./test_images/signs_vehicles_xygrad.png')
#
# # Choose a Sobel kernel size
# ksize = 7 # Choose a larger odd number to smooth gradient measurements
#
# # Apply each of the thresholding functions
# gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=ksize, thresh=(30, 80))
# grady = abs_sobel_thresh(image, orient='y', sobel_kernel=ksize, thresh=(20, 100))
# mag_binary = mag_thresh(image, sobel_kernel=ksize, mag_thresh=(30, 100))
# #dir_binary = dir_threshold(image, sobel_kernel=ksize, thresh=(0, np.pi/2))
# dir_binary = dir_threshold(image, sobel_kernel=15, thresh=(0.7, 1.3))
# dir_binary_left = dir_threshold(image, sobel_kernel=15, thresh=(0.8, 2.44))
# dir_binary_right = dir_threshold(image, sobel_kernel=15, thresh=(2.4, 1.3))
#
# #thresh = (90, 255)
# hls_binary = hls_select(image, thresh=(90, 255))
#
#
# # selection for pixels where both the x and y gradients meet the threshold criteria,
# # or the gradient magnitude and direction are both within their threshold values.
# combined = np.zeros_like(grady)
# combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
# #combined[((gradx == 1) & (mag_binary == 1)) & ((dir_binary_left == 1) | (dir_binary_right == 1))] = 1
# #combined[(grady == 1)] = 1
#
#
# # Plot the result
# f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
# f.tight_layout()
# ax1.imshow(image)
# ax1.set_title('Original Image', fontsize=50)
# ax2.imshow(combined, cmap='gray')
# ax2.set_title('Combined', fontsize=50)
# plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
# plt.show()
