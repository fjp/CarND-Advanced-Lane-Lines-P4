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
        windowWidth = self.m_windowWidth
        windowHeight = self.m_windowHeight
        margin = self.m_margin

        # store the (left, right)
        windowCentroids = []
