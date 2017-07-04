import numpy as np
import cv2
class tracker():

    # when starting a new instance please be sure to specify all unasigned variables
    def __init__(self, m_windowWidth, m_windowHeight, m_margin, m_ym = 1, m_xm = 1, m_smoothFactor = 15):
        # list that stores all the past (left,right) center set values used for smoothing the output
        self.recentCenters = []

        # the window pixel width of the center values, used to count pixels inside
        # center windows to determine curve values
        self.windowWidth = m_windowWidth


        # the window pixel height of the center values, used to count pixels inside
        # center windows to determine curve values
        # breaks the image into vertical levels
        self.windowHeight = m_windowHeight

        # the pixel distance in both directions to slide (left_window + right_window)
        # template for searching
        self.margin = m_margin

        # meters per pixel in vertical axis
        self.ymPerPixel = m_ym

        # meters per pixel in horizontal axis
        self.xmPerPixel = m_xm

        self.smoothFactor = m_smoothFactor

    # the main tracking function for finding and storing lane segment positions
    def findWindowCentroids(self, warped):
        windowWidth = self.windowWidth
        windowHeight = self.windowHeight
        margin = self.margin

        # store the (left, right) window centroid positions per level (stored as pairs)
        windowCentroids = []
        # create 1-D window template that is used for convolution _____-----_____
        window = np.ones(windowWidth)

        # First find the two starting positions for the left and right lane by
        # using np.sum to get the vertical image slice and then np.convolve
        # the vertical image slice with the window template

        # create a histogram by summing over columns of quarter bottom slice of image
        # vertical pixel arrays are squashed to a 1-D array over the entire slice
        lSum = np.sum(warped[int(3*warped.shape[0]/4):, :int(warped.shape[1]/2)], axis=0)
        # take the convolution of the window template and the squashed image slice
        # - windowWidth/2 is necessary to get the center of the window template
        # Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window
        lCenter = np.argmax(np.convolve(window, lSum)) - windowWidth/2
        rSum = np.sum(warped[int(3*warped.shape[0]/4):, int(warped.shape[1]/2):], axis=0)
        # we need to add the starting position back (int(warped.shape[1]/2))
        rCenter = np.argmax(np.convolve(window, rSum)) - windowWidth/2 + int(warped.shape[1]/2)

        # Add what we found for the first layer
        windowCentroids.append((lCenter, rCenter))

        # After finding the lane centers (layer 0)
        # Go through each layer looking for max pixel locations
        for level in range(1,(int)(warped.shape[0]/windowHeight)):
    	    # convolve the window into the vertical slice of the image
            # sum each colum/vertical slice (window height) to get a 1-D array
    	    imageLayer = np.sum(warped[int(warped.shape[0]-(level+1)*windowHeight):int(warped.shape[0]-level*windowHeight), :], axis=0)
    	    convSignal = np.convolve(window, imageLayer)
    	    # Find the best left centroid by using past left center as a reference
    	    # Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window
    	    offset = windowWidth/2
            # limit the region of the convolution by using l_min_index and l_max_index
    	    lMinIndex = int(max(lCenter + offset - margin, 0))
    	    lMaxIndex = int(min(lCenter + offset + margin, warped.shape[1]))
            # find the max pixel density in a local region
            # find the center (index where convolution has its maximum)
            # use l_min_index to get the right starting point
    	    lCenter = np.argmax(convSignal[lMinIndex:lMaxIndex]) + lMinIndex - offset
    	    # Find the best right centroid by using past right center as a reference
    	    rMinIndex = int(max(rCenter + offset - margin, 0))
    	    rMaxIndex = int(min(rCenter + offset + margin, warped.shape[1]))
    	    rCenter = np.argmax(convSignal[rMinIndex:rMaxIndex]) + rMinIndex - offset
    	    # Add what we found for that layer to the window centroid list
            # this list will contain the x locations for the left and right center points
            # for each of the layers (y is defined by the vertical slices or window height)
    	    windowCentroids.append((lCenter, rCenter))

        # smoothing function
        self.recentCenters.append(windowCentroids)
        # return averaged values of the line centers, helps from keeping the
        # markers from jumping around too much
        # average over the past smoothFactor (number = 15) values
        return np.average(self.recentCenters[-self.smoothFactor:], axis=0)



#windowCentroids = findWindowCentroids(warped, windowWidth, windowHeight, margin)
