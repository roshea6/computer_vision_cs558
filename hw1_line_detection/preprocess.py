"""
	Description: Part 1 of hw1, preprocess the image by detecting potential points on the lines.
	Apply a Gaussian filter first and use the Sobel filters as derivative operators.
	Threshold the determinant of the Hessian and then apply non-maximum suppression in 3 × 3
	neighborhoods. Ignore pixels for which any of the filters falls even partially out of the image
	boundaries.

"""

import cv2 # Image loading, writing, and displaying tools
import numpy as np # Matrix functions 

# Applies a guassian filter to the input image and returns the filtered image
def gaussianFilter(img)
	pass

# Applies a sobel filter to the input image and returns the filtered image
def sobelFilter(arr)
	pass

# Return the Hessian of the passed in arr
def getHessian(arr):
	pass

# Applies non maximum suppression to the passed in array
def nonMaxSuppression(arr):
	pass


if __name__ == "__main__":
	# Get the image
	img = cv2.imread("road.png")

	cv2.imshow("Image", img)

	cv2.waitKey(0)