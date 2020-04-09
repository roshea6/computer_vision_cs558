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

	print(centers)

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
	ouput = img.copy()

	print("New centers: " + str(centers))

	# Loop through clusters and change value of all pixels in that cluster to the cluster center
	for x in range(len(clusters)):
		cluster = clusters[x]
		center = centers[x]

		# Loop through points in cluster
		for point in cluster:
			ouput[point[1]][point[0]][2] = center[2] # Set Red
			ouput[point[1]][point[0]][1] = center[3] # Set Green
			ouput[point[1]][point[0]][0] = center[4] # Set Blue

	cv2.imshow("Done?", ouput)
	cv2.waitKey(0)

	



if __name__ == "__main__":

	# Read in the image to perform K means clustering on
	k_means_img = cv2.imread("white-tower.png")

	cv2.imshow("Before K means", k_means_img)
	cv2.waitKey(0)

	print(k_means_img.shape)

	kMeansClustering(k_means_img)
	

