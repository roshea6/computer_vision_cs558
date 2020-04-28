"""
	Ryan O'Shea

	Description: Computer Vision Homework 3 Image classification. 
	1. classify each image of the test set into one of three classes: coast, forest or “insidecity” using histograms
	2. Classify individual pixels in an image as sky or non sky

	I pledge my honor that I have abided by the Stevens Honor System
"""

import cv2
import os

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

	return hist

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


if __name__ == "__main__":

	NUM_BINS = 8

	path = 'ImClass/'

	img = cv2.imread('ImClass/coast_test1.jpg')

	hist = getHist(img, NUM_BINS)	
	print(verifyHist(hist, img))

	cv2.imshow("test", img)
	cv2.waitKey(0)