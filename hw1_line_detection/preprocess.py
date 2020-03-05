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
import random
import time

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
				h[i][j] = sum

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

# Calcultes the determinate of the Hessian of the passed in image and thresholds it
def getHessian(img):
	# Create the output image
	ouput = np.zeros((img.shape[0], img.shape[1]))

	# Get the second derivatives of the image
	img_xx = sobelFilterHoriz(sobelFilterHoriz(img))
	img_yy = sobelFilterVert(sobelFilterVert(img))
	img_xy = sobelFilterHoriz(sobelFilterVert(img))

	# cv2.imshow("XX", img_xx/255)
	# cv2.waitKey(0)
	# cv2.imshow("YY", img_yy/255)
	# cv2.waitKey(0)
	# cv2.imshow("XY", img_xy/255)
	# cv2.waitKey(0)

	# Hessian matrix (Just for visual reference)
	# hess = [[img_xx, img_xy],
	# 		[img_xy, img_yy]]

	# Get determinant of the Hessian matrix
	det = np.zeros((img.shape[0], img.shape[1]))
	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			det[i][j] = img_xx[i][j]*img_yy[i][j] - img_xy[i][j]*img_xy[i][j]

	# cv2.imshow("Hessian Det", det/255)
	# cv2.waitKey(0)

	# Get max and min values to use for thresholding
	max_val = np.amax(det)
	min_val = np.amin(det)

	val_range = max_val - min_val

	# Nomalize image
	for i in range(det.shape[0]):
		for j in range(det.shape[1]):
			det[i][j] = (det[i][j] - min_val) * (255/val_range)

	# cv2.imshow("Norm Hessian Det", det/255)
	# cv2.waitKey(0)

	# Threshold the determinant
	for i in range(det.shape[0]):
		for j in range(det.shape[1]):
			# If the pixel value is lower than the threshold set it to 0
			if det[i][j] < 150:
				det[i][j] = 0
			else:
				det[i][j] = 255

	# cv2.imshow("Thresh Hessian Det", det/255)
	# cv2.waitKey(0)

	suppressed = nonMaxSuppression(det)

	cv2.imshow("Suppressed", suppressed)
	cv2.waitKey(0)

	return suppressed

# Applies non maximum suppression to the passed in array
def nonMaxSuppression(img):
	output = img

	# Loop through the pixels in the image
	for i in range(1, img.shape[0] - 1):
		for j in range(1, img.shape[1] - 1):
			# Check if the current pixel is not the largest in the surrounding 3x3 area
			if img[i][j] != max(img[i-1][j-1], img[i-1][j], img[i-1][j+1], # First row
								img[i][j-1], img[i][j], img[i][j+1], # Second row
								img[i+1][j-1], img[i+1][j], img[i+1][j+1]): # Third row
				# Pixel is not the largest so set it to 0
				output[i][j] = 0
			else:
				# It is the max so set the rest to 0
				for k in range(0, 3):
					for g in range(0, 3):
						if not (k == 1 and g == 1):
							output[i + k -1][[j + g -1]] = 0
	return output

# Function to get a list of all the pixels with values above a certain threshold
def getPoints(img, threshold):
	# List of x, y pairs stored as tuples
	points = []

	# Loop through all pixels in image and get all coordinates of pixels with value above threshold
	for i in range(0, img.shape[0]):
		for j in range(0, img.shape[1]):
			if img[i][j] > threshold:
				points.append((i, j))

	return points

# Returns two random points from the list of passed in points
def pickRandPoints(pointsList):
	# Get the random index of the first point
	idx1 = random.randint(0, len(pointsList) - 1)

	# Pick another radom index that id different from the first one
	idx2 = random.randint(0, len(pointsList) - 1)

	# Ensure that the two indexes are not the same
	while(idx2 == idx1):
		idx2 = random.randint(0, len(pointsList) - 1)

	# Get the random points
	temp_point1 = pointsList[idx1]
	temp_point2 = pointsList[idx2]

	# Rearrange so the points are in x, y
	point1 = (temp_point1[1], temp_point1[0])
	point2 = (temp_point2[1], temp_point2[0])

	return point1, point2

# Get the slope m and intercept c of a line based on the two passed in points
def getLine(point1, point2):
	# Get x1 and y1 from the correct indexes
	x1 = point1[0]
	y1 = point1[1]

	# Get x2 and y2 from the correct indexes
	x2 = point2[0]
	y2 = point2[1]

	# Check if x1 and x2 and the same (AKA points make a vertical line)
	if(x1 == x2):
		# Set slope to infinity
		m = math.inf
	else:
		m = (y2 - y1)/(x2 - x1)

	# Find the intercept c using the point slope intercept formula
	# y - y1 = m(x - x1)
	c = -m*x1 + y1

	print("Line: y = " + str(m) + "x + " + str(c))

	return m, c

# Gets the perpendicular distance between a point and a line
def getDistFromLine(m, c, x, y):
	# Get the point on the line where the line from the point to the line intersects
	line_x = (x + (m*y) - (m*c))/(1 + m**2)
	line_y = ((m*x) + ((m**2)*y) - ((m**2)*c))/(1 + m**2) + c

	# Calculate the distance between the point on the line (line_x, line_y) and point (x, y)
	dist = math.sqrt(((line_x - x)**2) + ((line_y - y)**2))

	return dist, line_x, line_y

# Run the RANSAC algorithm on the passed in image to determine the 4 best lines in the image
def RANSAC(img):
	# Get the list of points to pick from
	points = getPoints(img, 0)

	print("Number of point: " + str(len(points)))

	# Get 2 random points from the list
	point1, point2 = pickRandPoints(points)

	print(point1)
	print(point2)

	cv2.line(img, point1, point2, (255, 255, 255), thickness=3)
	cv2.imshow("img", img)
	cv2.waitKey(0)

	m, c = getLine(point1, point2)

	for point in points:
		# Get distance between line and point
		# print(point)
		dist, line_x, line_y = getDistFromLine(m, c, point[1], point[0])

		if dist < 100:
			print("Distance: " + str(dist))
			print(point)
			cv2.line(img, (int(line_x), int(line_y)) , (point[1], point[0]), (255, 255, 255), thickness=1)
			cv2.imshow("Lines", img)
			key = cv2.waitKey(0)

			if key == 113:
				exit()



	# point3, point4 = pickRandPoints(points)

	# print(point3)

	# dist = getDistFromLine(m, c, point3[1], point3[0])

	print(dist)
	

if __name__ == "__main__":
	# Initialize the random seed
	random.seed(time.time())
	
	# Get the image
	# img = cv2.imread("road.png")

	# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	# cv2.imshow("Image", img)

	# cv2.waitKey(0)

	# # sup = cv2.imread("Suppressed_hessian.png")
	# # cv2.imshow("Sup", sup)
	# # cv2.waitKey(0)

	# guas_img = gaussianFilter(img, 5, 1)

	# pot_points = getHessian(guas_img)

	pot_points = cv2.imread("Suppressed_hessian.png")
	pot_points = cv2.cvtColor(pot_points, cv2.COLOR_BGR2GRAY)

	cv2.imshow("Supressed", pot_points)
	cv2.waitKey(0)

	RANSAC(pot_points)

	# cv2.imshow("Guassian filtered", guas_img/255)
	# cv2.waitKey(0)

	# sobel_y_img = sobelFilterVert(guas_img)

	# cv2.imshow("Sobel vert", sobel_y_img/255)
	# cv2.waitKey(0)

	# sobel_yy_img = sobelFilterVert(sobel_y_img)

	# cv2.imshow("Sobel vert 2", sobel_yy_img/255)
	# cv2.waitKey(0)

	# sobel_x_img = sobelFilterHoriz(guas_img)

	# cv2.imshow("Sobel horiz", sobel_x_img/255)
	# cv2.waitKey(0)

	# sobel_xx_img = sobelFilterHoriz(sobel_x_img)

	# cv2.imshow("Sobel horiz 2", sobel_xx_img/255)
	# cv2.waitKey(0)

	# sobel_xy_img = sobelFilterHoriz(sobel_y_img)

	# cv2.imshow("Sobel xy", sobel_xy_img/255)
	# cv2.waitKey(0)