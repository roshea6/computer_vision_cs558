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
	red_dist = (int(r1)-int(r2))**2
	green_dist = (int(g1)-int(g2))**2
	blue_dist = (int(b1)-int(b2))**2

	distance = math.sqrt(red_dist + green_dist + blue_dist)

	return distance

# Performs K means clustering on an image in order to segment it into 10 distinct colors
def kMeansClustering(img, k=10):
	# Get 10 random pixels from the image and store them in a list
	centers = []

	# Loop until 10 points have been collected. These are our cluster centers
	for i in range(10):
		# First random number
		x = random.randint(0, img.shape[1] - 1)

		# Second random number
		y = random.randint(0, img.shape[0] - 1)

		# Append the coordinate pair to the list
		centers.append((x, y))

	print(centers)

	# List of lists to hold the K clusters
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

				# print(center[0])
				# print(center[1])

				# Get the individual colors values for the center. OpenCV stores images as BGR by default
				# Index images using the coordinate pair as (y, x) or (rows, cols)
				cent_r = img[center[1]][center[0]][2]
				cent_g = img[center[1]][center[0]][1]
				cent_b = img[center[1]][center[0]][0]

				# print(str(cent_r) + ' ' + str(cent_g) + ' ' + str(cent_b))

				# Get the individual colors values for the pixel
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
			clusters[idx].append(((i, j)))
	
	for clust in clusters:
		print(clust)
		cv2.imshow("Reeeee", img)
		cv2.waitKey(0)


if __name__ == "__main__":

	# Read in the image to perform K means clustering on
	k_means_img = cv2.imread("white-tower.png")

	cv2.imshow("Before K means", k_means_img)
	cv2.waitKey(0)

	print(k_means_img.shape)

	kMeansClustering(k_means_img)
	

