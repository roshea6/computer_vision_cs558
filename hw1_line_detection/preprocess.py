"""
	Description: Part 1 of hw1, preprocess the image by detecting potential points on the lines.
	Apply a Gaussian filter first and use the Sobel filters as derivative operators.
	Threshold the determinant of the Hessian and then apply non-maximum suppression in 3 × 3
	neighborhoods. Ignore pixels for which any of the filters falls even partially out of the image
	boundaries.

"""

import cv2 # Image loading, writing, and displaying tools
import numpy as np # Matrix functions 
import math

# Function to apply a filter to an image
def convolute(img, filt):
	# Get offset of starting position based on filter
	offset = int(math.floor(filt.shape[0])/2)

	# Matrix to store result of filtering in
	h = np.zeros((img.shape[0], img.shape[1]))

	# Loop through the pixels in the image and apply the filter
	for i in range(0, img.shape[0]):
		for j in range(0, img.shape[1]):
			# If the filter will be out of bounds just take the original pixel value
			if(i < offset or j < offset or (i + offset > img.shape[0] - 1) or (j + offset > img.shape[1] - 1)):
				h[i][j] = img[i][j]
			# The filter and image perfectly overlap so apply the filter
			else:
				sum = 0.0
				# Loop through overlapping pixels on image and filter
				for k in range(0, filt.shape[0]):
					for g in range(0, filt.shape[1]):
						sum += filt[k][g] * img[i + k - offset][j + g - offset]
				
				# Put the sum into the correct spot in the matrix
				h[i][j] = int(sum)

	return h

# Gets the value at a point (x, y) in a gaussian distribution
def guassianValues(x, y, std_dev):
	g = (1/(2 * math.pi * std_dev**2))
	g *= math.e**(-((x**2 + y**2)/(2*std_dev**2)))

	return g

# Applies a guassian filter to the input image and returns the filtered image
def gaussianFilter(img, size, std_dev):
	# Conver the image to a numpy array
	arr = np.array(img)

	# Initialize the filter as an nxn matrix of 0s
	gaus_filt = np.zeros((size, size))

	# Get the index of the central point in the array. For example the center of a 
	# 5x5 array would be the point arr[2,2]
	center = int(math.floor(size/2))

	# Populate the filter with the proper values
	for i in range(size):
		for j in range(size):
			# The x and y values to pass into the Guassian value function are the x and 
			gaus_filt[i][j] = guassianValues(abs(center - i), abs(center - j), std_dev)

	print(gaus_filt)

	# Apply the filter to the image
	# cv_filtered = cv2.filter2D(img, -1, gaus_filt)
	# cv2.imwrite("cv_filtered.png", cv_filtered)

	filtered = convolute(arr, gaus_filt)

	cv2.imwrite("guassian_filtered.png", filtered)

	return filtered

# Applies a sobel filter to the input image to get the vertical edges
def sobelFilterVert(img):
	sobel_y = np.array([[1, 0, -1],
						[2, 0, -2],
						[1, 0, -1]])

	sobel_y_img = convolute(img, sobel_y)

	return sobel_y_img

# Applies a sobel filter to the input image to get the horizantal edges
def sobelFilterHoriz(img):
	sobel_x = np.array([[1, 2, 1],
						[0, 0, 0],
						[-1, -2, -1]])

	sobel_x_img = convolute(img, sobel_x)

	return sobel_x_img

# Return the Hessian of the passed in arr
def getHessian(arr):
	pass

# Applies non maximum suppression to the passed in array
def nonMaxSuppression(arr):
	pass


if __name__ == "__main__":
	# Get the image
	img = cv2.imread("road.png")

	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	cv2.imshow("Image", img)

	cv2.waitKey(0)

	guas_img = gaussianFilter(img, 5, 1)

	cv2.imshow("Guassian filtered", guas_img/255)
	cv2.waitKey(0)

	sobel_y_img = sobelFilterVert(guas_img)

	cv2.imshow("Sobel vert", sobel_y_img/255)
	cv2.waitKey(0)

	sobel_x_img = sobelFilterHoriz(guas_img)

	cv2.imshow("Sobel horiz", sobel_x_img/255)
	cv2.waitKey(0)