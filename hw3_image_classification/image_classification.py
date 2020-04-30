"""
	Ryan O'Shea

	Description: Computer Vision Homework 3 Image classification. 
	1. classify each image of the test set into one of three classes: coast, forest or “insidecity” using histograms
	2. Classify individual pixels in an image as sky or non sky

	I pledge my honor that I have abided by the Stevens Honor System
"""

import cv2
import glob
import math
import random
from sklearn import cluster

# Gets the histogram of the passed in image
def getHist(img, bins):
	# Histogram for holding the number of pixels in each bin
	# Structured as a list of lists where each entry in the main list is the 
	# number of pixels in the B G and R color channels in that bin
	# For example hist[0] = [b, g, r] where b is the number of blue channel pixels in the first bin
	# g is the number green channel pixels in the first bin, and r is the number of red channel pixels in the first bin
	hist = []

	for i in range(bins):
		hist.append([0, 0, 0])

	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			# Pixel three color value stored in BGR
			pixel = img[i][j]

			blue = pixel[0]
			green = pixel[1]
			red = pixel[2]

			# Increment the counts for the B, G, and R bins for a single pixel
			# The bin index is found by dividing the color calue by the bin legth in values which
			# is found by dividing 256 by the number of bins
			hist[int(blue // (256/bins))][0] += 1
			hist[int(green // (256/bins))][1] += 1
			hist[int(red // (256/bins))][2] += 1

	# Check to make sure there are the proper amount of histogram entries
	if verifyHist(hist, img):
		return hist
	else:
		print("Error: Histogram did not verify")
		return None

# Verifies that the histogram of the passed in image has the same number of pixels in the image times 3
def verifyHist(hist, img):
	# Get amount of pixels in image
	img_pix_count = img.shape[0] * img.shape[1]
	

	hist_pix_count = 0

	# Get the amount of entries in the histogram
	# This should be 3x the number of image pixels
	for i in range(len(hist)):
		for j in range(len(hist[i])):
			hist_pix_count += hist[i][j]

	# print("Hist: " + str(int(hist_pix_count/3)))
	# print("Img: " + str(img_pix_count))

	if int(hist_pix_count/3) == img_pix_count:
		return True
	else:
		return False

# Gets the N dimensional Euclidian distance between two histograms where N is the number 
# of bins in the histogram
def getEuclidDist(hist1, hist2):
	bins = len(hist1)

	dist = 0

	# Loop through all the elements to determine the distance
	for i in range(len(hist1)):
		for j in range(len(hist1[i])):
			dist += (float(hist1[i][j] - hist2[i][j]))**2

	dist = math.sqrt(dist)

	return dist

# Classifies the histograms in the ImClass directory 
def nearestNeighborHistogram(NUM_BINS = 8):
	# Lists to hold the histograms for the images and their respective class labels
	train_img_histograms = []
	train_img_labels = []

	# Get a list of all file paths to training images in ImClass directory
	train_img_dir = glob.glob('ImClass/*train*.jpg')

	print("Getting training data histograms")

	# Loop through the training images in the directory
	for filename in train_img_dir:
		# Read in the image
		img = cv2.imread(filename)

		# Compute the histogram
		hist = getHist(img, NUM_BINS)

		# Append the histogram to the training img list
		train_img_histograms.append(hist)

		# Check which class the image is and append that class to labels list
		if "coast" in filename:
			train_img_labels.append("coast")
			# print("coast")
		elif "forest" in filename:
			train_img_labels.append("forest")
			# print("forest")
		elif "insidecity" in filename:
			train_img_labels.append("insidecity")
			# print("insidecity")

	# Get a list of all file paths to test images in ImClass directory
	test_img_dir = glob.glob('ImClass/*test*.jpg')

	# Lists to hold the file names for the images and their respective class labels
	test_img_names = []
	test_img_labels = []
	num_correct = 0

	print("Classifying test images")

	# Loop through the testing images in the directory
	for filename in test_img_dir:
		# Read in the image
		img = cv2.imread(filename)

		# Compute the histogram
		hist = getHist(img, NUM_BINS)

		# Figure out which histogram in the training list has the smallest distance to the current image
		best_dist = 9999999999 # Very large best distance value to start
		idx = 0

		for i in range(len(train_img_histograms)):
			new_dist = getEuclidDist(hist, train_img_histograms[i])

			# If a new smallest distance has been found save the index of the image and update best_dist
			if new_dist < best_dist:
				best_dist = new_dist
				idx = i
		
		# Get the label from the training label list
		label = train_img_labels[idx]

		# Store filename and its classification
		test_img_names.append(filename)
		test_img_labels.append(label)

		print(str(filename[8:]) + " has been assigned to class " + label)

		# Check if the label is correct
		if label in filename:
			num_correct += 1

		cv2.imshow(filename[8:], img)
		cv2.waitKey(0)

	print("Accuracy = " + str(num_correct/len(test_img_dir)))

# Classifies the histograms in the ImClass directory using the 3 nearest neighbors
def threeNearestNeighborHistogram(NUM_BINS = 8):
	# Lists to hold the histograms for the images and their respective class labels
	train_img_histograms = []
	train_img_labels = []

	# Get a list of all file paths to training images in ImClass directory
	train_img_dir = glob.glob('ImClass/*train*.jpg')

	print("Getting training data histograms")

	# Loop through the training images in the directory
	for filename in train_img_dir:
		# Read in the image
		img = cv2.imread(filename)

		# Compute the histogram
		hist = getHist(img, NUM_BINS)

		# Append the histogram to the training img list
		train_img_histograms.append(hist)

		# Check which class the image is and append that class to labels list
		if "coast" in filename:
			train_img_labels.append("coast")
			# print("coast")
		elif "forest" in filename:
			train_img_labels.append("forest")
			# print("forest")
		elif "insidecity" in filename:
			train_img_labels.append("insidecity")
			# print("insidecity")

	# Get a list of all file paths to test images in ImClass directory
	test_img_dir = glob.glob('ImClass/*test*.jpg')

	# Lists to hold the file names for the images and their respective class labels
	test_img_names = []
	test_img_labels = []
	num_correct = 0

	print("Classifying test images based on 3 nearest neighbors")

	# Loop through the testing images in the directory
	for filename in test_img_dir:
		# Read in the image
		img = cv2.imread(filename)

		# Compute the histogram
		hist = getHist(img, NUM_BINS)

		# Figure out which histogram in the training list has the smallest distance to the current image
		first_dist = 9999999999 # Very large best distance value to start
		second_dist = 9999999999 # Very large best distance value to start
		third_dist = 9999999999 # Very large best distance value to start
		idx_1 = 0
		idx_2 = 0
		idx_3 = 0

		for i in range(len(train_img_histograms)):
			new_dist = getEuclidDist(hist, train_img_histograms[i])

			# Check if the dist is better than the 1st, 2nd, and 3rd best distances 
			if new_dist < first_dist:
				first_dist = new_dist
				idx_1 = i
			elif new_dist < second_dist:
				second_dist = new_dist
				idx_2 = i
			elif new_dist < third_dist:
				third_dist = new_dist
				idx_3 = i

		
		# Get the labels from the training label list
		label_1 = train_img_labels[idx_1]
		label_2 = train_img_labels[idx_2]
		label_3 = train_img_labels[idx_3]

		print("Class 1: " + label_1)
		print("Class 2: " + label_2)
		print("Class 3: " + label_3)

		# Determine which class the histogram should be 
		if label_1 == label_2 or label_1 == label_3:
			final_label = label_1
		# Can only not be label 1 if label 2 and label 3 are the same and label 1 is not
		elif label_2 == label_3:
			final_label = label_2
		else:
			final_label = label_1

		# Store filename and its classification
		test_img_names.append(filename)
		test_img_labels.append(final_label)

		print(str(filename[8:]) + " has been assigned to class " + final_label + "\n")

		# Check if the label is correct
		if final_label in filename:
			num_correct += 1

		cv2.imshow(filename[8:], img)
		cv2.waitKey(0)

	print("Accuracy = " + str(num_correct/len(test_img_dir)))

# Returns the Euclidean distance between two RGB triplets
def getColorDist(r1, g1, b1, r2, g2, b2):
	# Get individual color distances squared
	red_dist = (float(r1)-float(r2))**2
	green_dist = (float(g1)-float(g2))**2
	blue_dist = (float(b1)-float(b2))**2

	distance = math.sqrt(red_dist + green_dist + blue_dist)

	return distance

# Performs K means clustering on an image in order to segment it into 10 distinct colors
def kMeansClustering(pixel_set, k=10):
	print("Segmenting image into " + str(k) + " colors")

	# Get k random pixels from the image and store them in a list
	centers = []

	# Loop until k points have been collected. These are our starting cluster centers
	while len(centers) < k:
		# First random number
		x = random.randint(0, len(pixel_set) - 1)

		# If the point picked has not yet been picked as a center add it to the list
		if [pixel_set[x][0], pixel_set[x][1], pixel_set[x][2]] not in centers:
			# Append the coordinate pair and its RGB value to the list
			centers.append([pixel_set[x][0], pixel_set[x][1], pixel_set[x][2]])

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
		for i in range(len(pixel_set)):
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
				cent_r = center[0]
				cent_g = center[1]
				cent_b = center[2]

				# Get the individual colors values for the pixel. OpenCV stores images as BGR by default
				# Index images using the coordinate pair as (y, x) or (rows, cols)
				pix_r = pixel_set[i][0]
				pix_g = pixel_set[i][1]
				pix_b = pixel_set[i][2]

				new_dist = getColorDist(cent_r, cent_g, cent_b, pix_r, pix_g, pix_b)

				# Check if new color distance is smaller than current smallest color distance to a center
				if new_dist < dist:
					# Set new smallest distance and new cluster index that the pixel best belongs to
					dist = new_dist
					idx = x

			# Add the pixel to the cluster that it was closest to
			clusters[idx].append([pixel_set[i][0], pixel_set[i][1], pixel_set[i][2]])
	
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
				avg_r += cluster[j][0]
				avg_g += cluster[j][1]
				avg_b += cluster[j][2]

			avg_r = int(avg_r/len(cluster))
			avg_g = int(avg_g/len(cluster))
			avg_b = int(avg_b/len(cluster))

			# Check if the average color for the cluster changed
			if avg_r != centers[i][0] or avg_g != centers[i][1] or avg_b != centers[i][2]:
				centers[i][0] = avg_r
				centers[i][1] = avg_g
				centers[i][2] = avg_b
			# If the center didn't change then he clusters have converged
			else:
				converged = True

	print("Final centers: " + str(centers))

	return centers

# Returns the Euclidean distance between two pixels
def getPixelEuclidDist(r1, g1, b1, r2, g2, b2):
	# Get individual color distances squared
	red_dist = ((float(r1)-float(r2))**2)
	green_dist = ((float(g1)-float(g2))**2)
	blue_dist = ((float(b1)-float(b2))**2)

	distance = math.sqrt(red_dist + green_dist + blue_dist)

	return distance


if __name__ == "__main__":

	NUM_BINS = 8

	sky_img = cv2.imread("sky/sky_train.jpg")

	no_sky_img = cv2.imread("sky/no_sky_train.jpg")

	cv2.imshow("sky", sky_img)
	cv2.waitKey(0)

	cv2.imshow("no sky", no_sky_img)
	cv2.waitKey(0)

	mask_color = no_sky_img[0][0]

	sky_set = []
	no_sky_set = []

	# Create sets of the sky pixels and non sky pixels
	for i in range(no_sky_img.shape[0]):
		for j in range(no_sky_img.shape[1]):
			# Check if the pixel color matches the mask color
			if no_sky_img[i][j][0] == mask_color[0] and no_sky_img[i][j][1] == mask_color[1] and no_sky_img[i][j][2] == mask_color[2]:
				b = sky_img[i][j][0]
				g = sky_img[i][j][1]
				r = sky_img[i][j][2]

				sky_set.append([r, g, b])
			else:
				b = sky_img[i][j][0]
				g = sky_img[i][j][1]
				r = sky_img[i][j][2]

				no_sky_set.append([r, g, b])

	print("Getting no sky words")
	no_sky_centers = kMeansClustering(no_sky_set, 10)
	# temp_sky_centers = cluster.k_means(no_sky_set, 10)[0]

	# no_sky_centers = []

	# for pixel in temp_sky_centers:
	# 	pix_list = []
	# 	for color in pixel:
	# 		pix_list.append(int(color))

	# 	no_sky_centers.append(pix_list)

	# print(no_sky_centers)

	print("Getting sky words")
	sky_centers = kMeansClustering(sky_set, 10)
	# temp_sky_centers = cluster.k_means(sky_set, 10)[0]

	# sky_centers = []

	# for pixel in temp_sky_centers:
	# 	pix_list = []
	# 	for color in pixel:
	# 		pix_list.append(int(color))

	# 	sky_centers.append(pix_list)

	# print(sky_centers)

	# Get a list of all paths to test images in the sky directory
	test_img_dir = glob.glob("sky/*test*.jpg")

	# Loop through all the test images
	for filename in test_img_dir:
		test_img = cv2.imread(filename)

		cv2.imshow(filename, test_img)
		cv2.waitKey(0)

		output = test_img.copy()

		for i in range(test_img.shape[0]):
			for j in range(test_img.shape[1]):
				pixel = test_img[i][j]

				# Get the rgb values for the pixel
				img_blue = pixel[0]
				img_green = pixel[1]
				img_red = pixel[2]

				# Distance between pixel in image and cluster center
				dist = 999999999999

				# Determine the distance between the pixel and the closest non sky center
				for z in range(len(no_sky_centers)):
					# Get rgb values for the center
					cent_red = no_sky_centers[z][0]
					cent_green = no_sky_centers[z][1]
					cent_blue = no_sky_centers[z][2]

					# Find the euclidean distance between the pixel and center
					new_dist = getPixelEuclidDist(img_red, img_green, img_blue, cent_red, cent_green, cent_blue)

					# Check if new distance is smaller
					if new_dist < dist:
						dist = new_dist

				# Determine the distance between the pixel and the closest sky center
				# If any sky center is found to be closer than the non sky centers then classify it as a sky pixel and 
				# change the color on the output image
				for z in range(len(sky_centers)):
					# Get the rgb values for the center
					cent_red = sky_centers[z][0]
					cent_green = sky_centers[z][1]
					cent_blue = sky_centers[z][1]

					# Find the euclidean distance between the pixel and center
					new_dist = getPixelEuclidDist(img_red, img_green, img_blue, cent_red, cent_green, cent_blue)

					# Check if new distance is smaller
					if new_dist < dist:
						output[i][j] = [0, 0, 255]
						break

		cv2.imshow(filename + ' red', output)
		cv2.waitKey(0)





	


