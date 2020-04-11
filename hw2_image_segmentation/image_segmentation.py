"""
	Ryan O'Shea

	Description: Computer Vision Homework 2 Image segmentation. 
	1. Apply k-means clustering to an image with K = 10
	2. Apply the SLIC super pixel algorithm to an image 

	I pledge my honor that I have abided by the Stevens Honor System

"""

import cv2
import random
import numpy as np
import math
import time


# Returns the Euclidean distance between two RGB triplets
def getColorDist(r1, g1, b1, r2, g2, b2):
	# Get individual color distances squared
	red_dist = (float(r1)-float(r2))**2
	green_dist = (float(g1)-float(g2))**2
	blue_dist = (float(b1)-float(b2))**2

	distance = math.sqrt(red_dist + green_dist + blue_dist)

	return distance

# Performs K means clustering on an image in order to segment it into 10 distinct colors
def kMeansClustering(img, k=10):
	print("Segmenting image into " + str(k) + " colors")

	# Get 10 random pixels from the image and store them in a list
	centers = []

	# Loop until 10 points have been collected. These are our starting cluster centers
	while len(centers) < 10:
		# First random number
		x = random.randint(0, img.shape[1] - 1)

		# Second random number
		y = random.randint(0, img.shape[0] - 1)

		# If the point picked has not yet been picked as a center add it to the list
		if [x, y, img[y][x][2], img[y][x][1], img[y][x][0]] not in centers:
			# Append the coordinate pair and its RGB value to the list
			centers.append([x, y, img[y][x][2], img[y][x][1], img[y][x][0]])

	# print(centers)

	# List of lists to hold the K clusters
	clusters = []

	converged = False

	# Loop until clusters have converged
	while(converged == False):
		# Clear clusters
		clusters = []

		for i in range(k):
			# Append an empty list
			clusters.append([])

		# Loop through the image and assign each pixel to its closest cluster center according to color
		for i in range(img.shape[0]):
			for j in range(img.shape[1]):
				# Distance variable for holding distance between pixel color and cluster center color
				# Reset to very high value for each pixel
				dist = 10000000

				# Index variable to keep track of which cluster the pixel best fits in
				idx = 0

				# Check distance between pixel color and colors of each color center
				for x in range(len(clusters)):
					# Get coordinates of a cluster center
					center = centers[x]

					# Get the individual colors values for the center.
					cent_r = center[2]
					cent_g = center[3]
					cent_b = center[4]

					# Get the individual colors values for the pixel. OpenCV stores images as BGR by default
					# Index images using the coordinate pair as (y, x) or (rows, cols)
					pix_r = img[i][j][2]
					pix_g = img[i][j][1]
					pix_b = img[i][j][0]

					new_dist = getColorDist(cent_r, cent_g, cent_b, pix_r, pix_g, pix_b)

					# Check if new color distance is smaller than current smallest color distance to a center
					if new_dist < dist:
						# Set new smallest distance and new cluster index that the pixel best belongs to
						dist = new_dist
						idx = x

				# Add the pixel to the cluster that it was closest to
				clusters[idx].append([j, i, img[i][j][2], img[i][j][1], img[i][j][0]])
		
		# Loop through clusters to find the new ideal cluster center
		for i in range(len(clusters)):
			cluster = clusters[i]

			# Variables to hold average pixel colors
			avg_r = 0
			avg_g = 0
			avg_b = 0

			# Find the average of each color in the cluster
			for j in range(len(cluster)):
				avg_r += cluster[j][2]
				avg_g += cluster[j][3]
				avg_b += cluster[j][4]

			avg_r = int(avg_r/len(cluster))
			avg_g = int(avg_g/len(cluster))
			avg_b = int(avg_b/len(cluster))

			# Check if the average color for the cluster changed
			if avg_r != centers[i][2] or avg_g != centers[i][3] or avg_b != centers[i][4]:
				centers[i][2] = avg_r
				centers[i][3] = avg_g
				centers[i][4] = avg_b
			# If the center didn't change then he clusters have converged
			else:
				converged = True

	# Segmented image to return
	output = img.copy()

	print("Final centers: " + str(centers))

	# Loop through clusters and change value of all pixels in that cluster to the cluster center
	for x in range(len(clusters)):
		cluster = clusters[x]
		center = centers[x]

		# Loop through points in cluster
		for point in cluster:
			output[point[1]][point[0]][2] = center[2] # Set Red
			output[point[1]][point[0]][1] = center[3] # Set Green
			output[point[1]][point[0]][0] = center[4] # Set Blue

	return output

# Run the SLIC super pixel algorithm on the passed in image
# Segments the image into block_size*block_size segments
def SLIC(img, block_size=50):
	# Break the image into blocks and find the centroids of the blocks
	centroids = []

	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			# Check if we're at a centroid. Centroids will appear every block_size/2 pixels in both the x and y direction
			if ((i - block_size/2) % block_size) == 0 and ((j - block_size/2) % block_size) == 0 and i != 0 and j != 0:
				centroids.append([i, j])

	# print(centroids)

	# Find ideal centroid position based on color gradient magnitudes
	for z in range(len(centroids)):
		# Set gradient magnitude to be extremely high to begin with
		mag = 10000000
		new_coord = [0, 0]

		y = centroids[z][0]
		x = centroids[z][1]

		# Loop through the 3x3 area around the centroid
		for i in range(-1, 1):
			for j in range(-1, 1):
				# Calculate magnitude of the color gradient between two points
				new_mag = math.sqrt(float(int(img[y+i+1][x+j+1][0]) - int(img[y+i][x+j][0]))**2 + float(int(img[y+i+1][x+j+1][1]) - int(img[y+i][x+j][1]))**2 + float(int(img[y+i+1][x+j+1][2]) - int(img[y+i][x+j][2]))**2)

				# If the magnitude is smaller then we have a new best point
				if new_mag < mag:
					mag = new_mag
					new_coord = [y+i, x+j]

		# Update with the new ideal centroid
		centroids[z] = new_coord

	# print(centroids)

	clustered_img = SLICkMeans(img, centroids)

	return clustered_img

# Returns the Euclidean distance between two pixels
def getEuclidDist(x1, y1, r1, g1, b1, x2, y2, r2, g2, b2):
	# Get individual color distances squared
	red_dist = (float(r1)-float(r2))**2
	green_dist = (float(g1)-float(g2))**2
	blue_dist = (float(b1)-float(b2))**2

	x_dist = ((float(x1)-float(x2))**2)/2
	y_dist = ((float(y1)-float(y2))**2)/2

	distance = math.sqrt(red_dist + green_dist + blue_dist + x_dist + y_dist)

	return distance


# Performs K means clustering on the input image using the passed in centroid as the centers
# Also uses 5D space [x, y, r, g, b] for distance calculation
def SLICkMeans(img, centroids):
	print("Segmenting image into " + str(len(centroids)) + " sections")

	converged = False

	centers = []

	# Convert the centroids into cluster centers
	for i in range(len(centroids)):
		centroid = centroids[i]

		x = centroid[1]
		y = centroid[0]

		# Center has form [x, y, r, g, b]
		centers.append([x, y, img[y][x][2], img[y][x][1], img[y][x][0]])

	# Loop until clusters have converged
	while(converged == False):
		# Clear clusters
		clusters = []

		for i in range(len(centroids)):
			# Append an empty list
			clusters.append([])

		# print(clusters)

		# Loop through the image and assign each pixel to its closest cluster center according to color
		for i in range(img.shape[0]):
			for j in range(img.shape[1]):
				# Distance variable for holding distance between pixel color and cluster center color
				# Reset to very high value for each pixel
				dist = 10000000

				# Index variable to keep track of which cluster the pixel best fits in
				idx = 0

				# Check distance between pixel color and colors of each color center
				for x in range(len(clusters)):
					# Get coordinates of a cluster center
					center = centers[x]

					# Get the individual colors values for the center.
					cent_r = center[2]
					cent_g = center[3]
					cent_b = center[4]

					# Get the individual colors values for the pixel. OpenCV stores images as BGR by default
					# Index images using the coordinate pair as (y, x) or (rows, cols)
					pix_r = img[i][j][2]
					pix_g = img[i][j][1]
					pix_b = img[i][j][0]

					new_dist = getEuclidDist(center[0], center[1], cent_r, cent_g, cent_b, j, i, pix_r, pix_g, pix_b)

					# Check if new color distance is smaller than current smallest color distance to a center
					if new_dist < dist:
						# Set new smallest distance and new cluster index that the pixel best belongs to
						dist = new_dist
						idx = x

				# Add the pixel to the cluster that it was closest to
				clusters[idx].append([j, i, img[i][j][2], img[i][j][1], img[i][j][0]])
		
		# Loop through clusters to find the new ideal cluster center
		for i in range(len(clusters)):
			cluster = clusters[i]

			if len(cluster) == 0:
				continue

			# Variables to hold average pixel colors
			avg_r = 0
			avg_g = 0
			avg_b = 0

			# Find the average of each color in the cluster
			for j in range(len(cluster)):
				avg_r += cluster[j][2]
				avg_g += cluster[j][3]
				avg_b += cluster[j][4]

			avg_r = int(avg_r/len(cluster))
			avg_g = int(avg_g/len(cluster))
			avg_b = int(avg_b/len(cluster))

			# Check if the average color for the cluster changed
			if avg_r != centers[i][2] or avg_g != centers[i][3] or avg_b != centers[i][4]:
				centers[i][2] = avg_r
				centers[i][3] = avg_g
				centers[i][4] = avg_b
			# If the center didn't change then he clusters have converged
			else:
				converged = True

	# Segmented image to return
	output = img.copy()

	print("Final centers: " + str(centers))

	# Loop through clusters and change value of all pixels in that cluster to the cluster center
	for x in range(len(clusters)):
		cluster = clusters[x]
		center = centers[x]

		# Loop through points in cluster
		for point in cluster:
			output[point[1]][point[0]][2] = center[2] # Set Red
			output[point[1]][point[0]][1] = center[3] # Set Green
			output[point[1]][point[0]][0] = center[4] # Set Blue

	return output

# Adds black borders to all segments by turning every pixel that touches two or more segments black
def addBorders(img):
	output = img.copy()

	# Loop through the pixels in the image and find which ones touch more than one segment
	for i in range(img.shape[0] - 1):
		for j in range(img.shape[1] - 1):
			# Check if we are at the edge of the image
			if i == 0 or j == 0:
				continue
			else:
				# Check the 3x3 area around the pixel to see if any of them are different
				for x in range(-1, 1):
					for y in range(-1, 1):
						# If the pixel differs from one of its neighbors turn it black
						if (img[i][j][0] == img[i+x][j+y][0]) and (img[i][j][1] == img[i+x][j+y][1]) and (img[i][j][2] == img[i+x][j+y][2]):
							continue
						else:
							output[i][j] = [0, 0, 0]

	return output
	
if __name__ == "__main__":

	# Read in the image to perform K means clustering on
	k_means_img = cv2.imread("white-tower.png")

	cv2.imshow("Before K means", k_means_img)
	cv2.waitKey(0)

	# print(k_means_img.shape)

	k_clustered_img = kMeansClustering(k_means_img, 10)

	cv2.imshow("After K means", k_clustered_img)
	cv2.waitKey(0)

	# Read in the image to perform SLIC on
	pre_slic_img = cv2.imread("wt_slic.png")

	cv2.imshow("Before SLIC", pre_slic_img)
	cv2.waitKey(0)

	post_slic_img = SLIC(pre_slic_img, 50)

	cv2.imshow("Post SLIC", post_slic_img)
	cv2.waitKey(0)

	post_borders = addBorders(post_slic_img)

	cv2.imshow("Post Borders", post_borders)
	cv2.waitKey(0)