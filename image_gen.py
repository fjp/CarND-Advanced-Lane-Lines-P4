import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import pickle
from tracker import tracker


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

# WindowMask is a function to draw window areas
def windowMask(width, height, img_ref, center, level):
    output = np.zeros_like(img_ref)
    output[int(img_ref.shape[0]-(level+1)*height):int(img_ref.shape[0]-level*height),max(0,int(center-width))]

def window_mask(width, height, img_ref, center, level):
    output = np.zeros_like(img_ref)
    output[int(img_ref.shape[0]-(level+1)*height):int(img_ref.shape[0]-level*height),max(0,int(center-width/2)):min(int(center+width/2),img_ref.shape[1])] = 1
    return output

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
    colorbinary = color_threshold(img, sthresh=(100, 255), vthresh=(50, 255))
    # combine the gradient and color thresholds
    preprocessImage = np.zeros_like(img[:,:,0])
    # set the indexed pixels to 255 because we store a jpg (no png)
    preprocessImage[((gradx == 1) & (grady == 1)) | (colorbinary == 1)] = 255


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
    Minv = cv2.getPerspectiveTransform(dst, src)
    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(preprocessImage, M, img_size)

    # Define the sliding window size
    windowWidth = 25
    windowHeight = 80
    margin = 25

    # create an instance of the tracker class with pixel conversion ym and xm to the real world space
    # this is use to measure the curvature. It defines a ratio (10 meters corresponds to 720 pixels
    # and 4 meters corresponds to 384 pixels)
    curveCenters = tracker(m_windowWidth = windowWidth, m_windowHeight = windowHeight, m_margin = margin, m_ym = 10/720, m_xm = 4/1280, m_smoothFactor = 15)



    # call the tracking function itself
    windowCentroids = curveCenters.findWindowCentroids(warped)

    print(windowCentroids)

    # draw the found left and right center points
    # If we found any window centers
    if len(windowCentroids) > 0:

        # Points used to draw all the left and right windows
        lPoints = np.zeros_like(warped)
        rPoints = np.zeros_like(warped)

        # Points used to find the left and right lanes
        leftx = []
        rightx = []

        # Go through each level and draw the windows
        for level in range(0,len(windowCentroids)):
            # add found window values
            leftx.append(windowCentroids[level][0])
            rightx.append(windowCentroids[level][1])
            # WindowMask is a function to draw window areas
            lMask = window_mask(windowWidth, windowHeight, warped, windowCentroids[level][0], level)
            rMask = window_mask(windowWidth, windowHeight, warped, windowCentroids[level][1], level)
            # Add graphic points from window mask here to total pixels found
            lPoints[(lPoints == 255) | ((lMask == 1) ) ] = 255
            rPoints[(rPoints == 255) | ((rMask == 1) ) ] = 255

        # Draw the results
        # add both left and right window pixels together
        template = np.array(rPoints + lPoints, np.uint8)
        # create a zero color channel
        zeroChannel = np.zeros_like(template)
        # make window pixels green (other channels are black = red and blue)
        template = np.array(cv2.merge((zeroChannel, template, zeroChannel)), np.uint8)
        # making the original road pixels 3 color channels
        warpage = np.array(cv2.merge((warped, warped, warped)), np.uint8)
        # overlay the orignal road image with window results
        # (100 percent for the warpage image and 50 percent for the green windows)
        result = cv2.addWeighted(warpage, 1, template, 0.5, 0.0)

    # If no window centers found, just display orginal road image
    #else:
    #    result = np.array(cv2.merge((warped, warped, warped)),np.uint8)

    # fit the lane boundaries to the left, right center positions found
    yvals = range(0, warped.shape[0])

    resYvals = np.arange(warped.shape[0] - (windowHeight/2), 0, -windowHeight)

    # find the polynomial coefficients of degree 2 polynomial
    # a*x^2 + b*x + c
    leftFit = np.polyfit(resYvals, leftx, 2)
    leftFitx = leftFit[0]*yvals*yvals + leftFit[1]*yvals + leftFit[2]
    leftFitx = np.array(leftFitx, np.int32)

    rightFit = np.polyfit(resYvals, rightx, 2)
    rightFitx = rightFit[0]*yvals*yvals + rightFit[1]*yvals + rightFit[2]
    rightFitx = np.array(rightFitx, np.int32)

    # fancy array magic to encapsulate the list values
    leftLane = np.array(list(zip(np.concatenate((leftFitx - windowWidth/2, leftFitx[::-1]+windowWidth/2), axis=0), np.concatenate((yvals, yvals[::-1]), axis=0))), np.int32)
    rightLane = np.array(list(zip(np.concatenate((rightFitx - windowWidth/2, rightFitx[::-1]+windowWidth/2), axis=0), np.concatenate((yvals, yvals[::-1]), axis=0))), np.int32)
    middleMarker = np.array(list(zip(np.concatenate((rightFitx - windowWidth/2, rightFitx[::-1]+windowWidth/2), axis=0), np.concatenate((yvals, yvals[::-1]), axis=0))), np.int32)

    road = np.zeros_like(img)
    roadBkg = np.zeros_like(img)
    cv2.fillPoly(road, [leftLane], color=[255, 0, 0])
    cv2.fillPoly(road, [rightLane], color=[0, 0, 255])
    cv2.fillPoly(roadBkg, [leftLane], color=[255, 255, 255])
    cv2.fillPoly(roadBkg, [rightLane], color=[255, 255, 255])


    # transform the warped lines back to the original image
    roadWarped = cv2.warpPerspective(road, Minv, img_size, flags=cv2.INTER_LINEAR)
    roadWarpedBkg = cv2.warpPerspective(roadBkg, Minv, img_size, flags=cv2.INTER_LINEAR)

    base = cv2.addWeighted(img, 1.0, roadWarpedBkg, -1.0, 0.0)
    result = cv2.addWeighted(base, 1.0, roadWarped, 1.0, 0.0)

    # meters per pixel in y dimension
    ymPerPixel = curveCenters.ymPerPixel
    # meters per pixel in x dimension
    xmPerPixel = curveCenters.ymPerPixel

    # get curvature in meters
    # tutorial: http://www.intmath.com/applications-differentiation/8-radius-curvature.php
    curveFitCr = np.polyfit(np.array(resYvals, np.float32)*ymPerPixel, np.array(leftx, np.float32)*xmPerPixel, 2)
    # for the curvature radius only the left line is checked (Todo: averaging of both lines)
    curverad = ((1 + (2*curveFitCr[0]*yvals[-1]*ymPerPixel + curveFitCr[1])**2)**1.5) / np.absolute(2*curveFitCr[0])

    # calculate the offset of the car on the road
    cameraCenter = (leftFitx[-1] + rightFitx[-1])/2
    centerDiff = (cameraCenter - warped.shape[1]/2)*xmPerPixel
    sidePos = 'left'
    if centerDiff <= 0:
        sidePos = 'right'

    # draw the text showing curvature, offset and speed
    cv2.putText(result, 'Radius of curvature = ' + str(round(curverad,3))+'(m)', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(result, 'Vehicle is ' + str(round(centerDiff,3))+' m' + sidePos + ' of center', (50,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Display the final results
    plt.imshow(result)
    plt.title('window fitting results')
    plt.show()

    #histogram = np.sum(warped[img.shape[0]//2:,:], axis=0)
    #plt.plot(histogram)
    #plt.show()

    # write the undistored images
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
