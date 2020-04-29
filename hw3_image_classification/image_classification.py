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


if __name__ == "__main__":

	NUM_BINS = 8

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

		# cv2.imshow(filename, img)
		# cv2.waitKey(0)

	# Get a list of all file paths to test images in ImClass directory
	test_img_dir = glob.glob('ImClass/*test*.jpg')

	# print(test_img_dir)
	# print(len(test_img_dir))
	# exit()

	# Lists to hold the file names for the images and their respective class labels
	test_img_names = []
	test_img_labels = []

	print("Classifying test images")

	# Loop through the testing images in the directory
	for filename in test_img_dir:
		# Read in the image
		img = cv2.imread(filename)

		# Compute the histogram
		hist = getHist(img, NUM_BINS)

		# Figure out which histogram in the training list has the smallest distance to the current image
		best_dist = 9999999999 # Very large best distance value
		idx = 0
		# print("Yeet: " + str(len(train_img_histograms)))
		# exit()
		for i in range(len(train_img_histograms)):
			new_dist = getEuclidDist(hist, train_img_histograms[i])
			# print(new_dist)

			# If a new smallest distance has been found save the index of the image and update best_dist
			if new_dist < best_dist:
				best_dist = new_dist
				idx = i
		
		test_img_names.append(filename)
		test_img_labels.append(train_img_labels[idx])

		print(str(filename[8:]) + " has been assigned to class " + str(train_img_labels[idx]))

		cv2.imshow(filename[8:], img)
		cv2.waitKey(0)


