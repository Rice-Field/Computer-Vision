# Jonathon Rice
# python 2.7 for opencv

from PIL import Image
from numpy import *
import sys
import math
import cv2

# Calculates gaussian values
def gaussian(sigma, x, y):
	n = pow(e, (-(x*x+y*y)/(2*sigma*sigma)))
	return (n/(sqrt(2*pi)*sigma))

def plotcorners(image, corner):
	# number of features for the image
	num = len(corner)
	# x and y coordinates of the features and gaussian weight
	x = 0
	y = 0
	gauss = 0

	# plot the features on the image
	for i in range(num):
		x = int(corner[i][0][0])
		y = int(corner[i][0][1])
		image[y][x] = 255

	return image

# image1 = x gradient, image2 = y gradient, image3 = time gradient
# kernel creates nxn mask, corners are our features we are tracking
# puts values in matrices to setup for calculating u,v
def lkmatrices(image1, image2, image3, corner, kernel):
	# number of features for the image
	num = len(corner)
	# x and y coordinates of the features
	x = 0
	y = 0

	space = [[0, 0], [0, 0]]
	time = [[0], [0]]
	vector = [[0], [0]]
	vlist = []

	# plot the features on the image
	for i in range(num):
		vlist.append([])
		x = int(corner[i][0][0])
		y = int(corner[i][0][1])

		# calculate radius around center pixel for kernel
		size = kernel/2

		# move through window size
		for j in range(-size,size+1): #rows
			for k in range(-size,size+1): #columns

				try:
					image1[y+j][x+k]
				except IndexError:
					continue

				else:
					# calculate our weight for pixel k,j distance from center
					# using gaussian values, apply it to gradients
					gauss = gaussian(1, k, j)

					space[0][0] = space[0][0] + pow(gauss*image1[y+j][x+k], 2)
					space[0][1] = space[0][1] + gauss*image1[y+j][x+k] * gauss*image2[y+j][x+k]
					space[1][0] = space[1][0] + gauss*space[0][1]
					space[1][1] = space[1][1] + pow(gauss*image2[y+j][x+k], 2)

					time[0][0] = time[0][0] + -gauss*image1[y+j][x+k] * gauss*image3[y+j][x+k]
					time[1][0] = time[1][0] + -gauss*image2[y+j][x+k] * gauss*image3[y+j][x+k]

		space = linalg.inv(space)
		vector = matmul(space, time)
		vlist[i] = vector

		space = [[0, 0], [0, 0]]
		time = [[0], [0]]

	return vlist
				

def fixlist(list):
	num = len(list)
	list2 = []

	for i in range(num):
		list2.append([])

	return list2

def plotcolor(image, corner, vector):
	num = len(corner)
	# x and y coordinates of the features and gaussian weight
	x = 0
	y = 0
	gauss = 0

	max_x = 0
	max_y = 0

	for i in range(num):
		if vector[i][0] > max_x:
			max_x = vector[i][0]

		if vector[i][1] > max_y:
			max_y = vector[i][1]

	# plot color intensity in a local block around the feature
	# green is change in x direction
	# red is change in y direction
	# yellow is change in both
	for i in range(num):
		x = int(corner[i][0][0])
		y = int(corner[i][0][1])
		for j in range(-2,3):
			for k in range(-2,3):
				try:
					image1[y+j][x+k]
				except IndexError:
					continue

				else:
					image[y+j][x+k][0] = 0
					image[y+j][x+k][1] = abs(vector[i][0])/max_x*255
					image[y+j][x+k][2] = abs(vector[i][1])/max_y*255
			

	return image

def avgvectors(v1, v2, v3):
	num = len(v1)

	v4 = []

	for i in range(num):
		v4.append([])
		v4[i] = (v1[i] + v2[i] + v3[i])/3

	return v4


# read in the two images to compare
image1 = cv2.imread('basketball1.png',0)
image2 = cv2.imread('basketball2.png',0)

# we use this to find corners to track through the images
# so we can calculate the gradients
features1 = cv2.goodFeaturesToTrack(image1, maxCorners=500, qualityLevel=0.001, minDistance=10)
features2 = cv2.goodFeaturesToTrack(image2, maxCorners=500, qualityLevel=0.001, minDistance=10)

# detects if missing features 
if (features1.all() != features2.all()):
	print('missing points')

# plots features
mask1 = plotcorners(image1, features1)
mask2 = plotcorners(image2, features2)

# window size
kernel = 5

# calculute x, y and t gradients with gaussian smoothing
sobelx = cv2.Sobel(image1,cv2.CV_64F,1,0,ksize=5)
sobely = cv2.Sobel(image1,cv2.CV_64F,0,1,ksize=5)
tdiv = cv2.GaussianBlur(image2,(5,5),1) - cv2.GaussianBlur(image1,(5,5),1)

# create vectors by calculating lucas-kanade
vectors = lkmatrices(sobelx, sobely, tdiv, features1, kernel)

# convert image to color to add color intensity based on vector strength
color1 = cv2.cvtColor(image1, cv2.COLOR_GRAY2RGB)
color1 = plotcolor(color1, features1, vectors)

cv2.imshow('optical flow', color1)
cv2.waitKey(0)
cv2.destroyAllWindows()

###############
# image set 2

# read in the two images to compare
image3 = cv2.imread('grove1.png',0)
image4 = cv2.imread('grove2.png',0)

# we use this to find corners to track through the images
# so we can calculate the gradients
features3 = cv2.goodFeaturesToTrack(image3, maxCorners=200, qualityLevel=0.01, minDistance=10)
features4 = cv2.goodFeaturesToTrack(image4, maxCorners=200, qualityLevel=0.01, minDistance=10)

# detects if missing features 
if (features3.all() != features4.all()):
	print('missing points')

# plots features
mask1 = plotcorners(image3, features3)
mask2 = plotcorners(image4, features4)

# window size
kernel = 5

# calculute x, y and t gradients with gaussian smoothing
sobelx = cv2.Sobel(image3,cv2.CV_64F,1,0,ksize=5)
sobely = cv2.Sobel(image3,cv2.CV_64F,0,1,ksize=5)
tdiv = cv2.GaussianBlur(image4,(5,5),1) - cv2.GaussianBlur(image3,(5,5),1)

# create vectors by calculating lucas-kanade
vectors = lkmatrices(sobelx, sobely, tdiv, features3, kernel)

# convert image to color to add color intensity based on vector strength
color2 = cv2.cvtColor(image3, cv2.COLOR_GRAY2RGB)
color2 = plotcolor(color2, features3, vectors)

cv2.imshow('optical flow', color2)
cv2.waitKey(0)
cv2.destroyAllWindows()

#################################
# pyramid multiscale ############

# read in the two images to compare
full1 = cv2.imread('basketball1.png', 0)
full2 = cv2.imread('basketball2.png', 0)

# half size
half1 = cv2.resize(full1, (0,0), fx=0.5, fy=0.5)
half2 = cv2.resize(full2, (0,0), fx=0.5, fy=0.5)

# quarter size
quart1 = cv2.resize(full1, (0,0), fx=0.25, fy=0.25)
quart2 = cv2.resize(full2, (0,0), fx=0.25, fy=0.25)

# we use this to find corners to track through the images
# so we can calculate the gradients
# we lower the max corners to have similar features across all 3 layers
featuresf1 = cv2.goodFeaturesToTrack(full1, maxCorners=50, qualityLevel=0.01, minDistance=10)
featuresf2 = cv2.goodFeaturesToTrack(full2, maxCorners=50, qualityLevel=0.01, minDistance=10)

featuresh1 = cv2.goodFeaturesToTrack(half1, maxCorners=50, qualityLevel=0.01, minDistance=10)
featuresh2 = cv2.goodFeaturesToTrack(half2, maxCorners=50, qualityLevel=0.01, minDistance=10)

featuresq1 = cv2.goodFeaturesToTrack(quart1, maxCorners=50, qualityLevel=0.01, minDistance=10)
featuresq2 = cv2.goodFeaturesToTrack(quart2, maxCorners=50, qualityLevel=0.01, minDistance=10)

# detects if missing features 
if (features1.all() != features2.all()):
	print('missing points')

# plots features
maskf1 = plotcorners(full1, featuresf1)
maskf2 = plotcorners(full2, featuresf2)

# plots features for half
maskh1 = plotcorners(half1, featuresh1)
maskh2 = plotcorners(half2, featuresh2)

# plots features for quarter
maskq1 = plotcorners(quart1, featuresq1)
maskq2 = plotcorners(quart2, featuresq2)


# window size
kernel = 5

# calculute x, y and t gradients with gaussian smoothing
fullx = cv2.Sobel(full1,cv2.CV_64F,1,0,ksize=5)
fully = cv2.Sobel(full1,cv2.CV_64F,0,1,ksize=5)
fullt = cv2.GaussianBlur(full2,(5,5),1) - cv2.GaussianBlur(full1,(5,5),1)

# calculute x, y and t gradients with gaussian smoothing
halfx = cv2.Sobel(half1,cv2.CV_64F,1,0,ksize=5)
halfy = cv2.Sobel(half1,cv2.CV_64F,0,1,ksize=5)
halft = cv2.GaussianBlur(half2,(5,5),1) - cv2.GaussianBlur(half1,(5,5),1)

# calculute x, y and t gradients with gaussian smoothing
quartx = cv2.Sobel(quart1,cv2.CV_64F,1,0,ksize=5)
quarty = cv2.Sobel(quart1,cv2.CV_64F,0,1,ksize=5)
quartt = cv2.GaussianBlur(quart2,(5,5),1) - cv2.GaussianBlur(quart1,(5,5),1)


vectorsf = lkmatrices(fullx, fully, fullt, featuresf1, kernel)
vectorsh = lkmatrices(halfx, halfy, halft, featuresh1, kernel)
vectorsq = lkmatrices(quartx, quarty, quartt, featuresq1, kernel)

avgvect = avgvectors(vectorsf, vectorsh, vectorsq)

color3 = cv2.cvtColor(full1, cv2.COLOR_GRAY2RGB)
color3 = plotcolor(color3, featuresf1, avgvect)

cv2.imshow('optical flow', color3)
cv2.waitKey(0)
cv2.destroyAllWindows()

##########
# pyramid 2

# read in the two images to compare
full3 = cv2.imread('grove1.png', 0)
full4 = cv2.imread('grove2.png', 0)

# half size
half3 = cv2.resize(full3, (0,0), fx=0.5, fy=0.5)
half4 = cv2.resize(full4, (0,0), fx=0.5, fy=0.5)

# quarter size
quart3 = cv2.resize(full3, (0,0), fx=0.25, fy=0.25)
quart4 = cv2.resize(full4, (0,0), fx=0.25, fy=0.25)

# we use this to find corners to track through the images
# so we can calculate the gradients
# we lower the max corners to have similar features across all 3 layers
featuresf3 = cv2.goodFeaturesToTrack(full3, maxCorners=50, qualityLevel=0.01, minDistance=10)
featuresf4 = cv2.goodFeaturesToTrack(full4, maxCorners=50, qualityLevel=0.01, minDistance=10)

featuresh3 = cv2.goodFeaturesToTrack(half3, maxCorners=50, qualityLevel=0.01, minDistance=10)
featuresh4 = cv2.goodFeaturesToTrack(half4, maxCorners=50, qualityLevel=0.01, minDistance=10)

featuresq3 = cv2.goodFeaturesToTrack(quart3, maxCorners=50, qualityLevel=0.01, minDistance=10)
featuresq4 = cv2.goodFeaturesToTrack(quart4, maxCorners=50, qualityLevel=0.01, minDistance=10)

# detects if missing features 
if (features1.all() != features2.all()):
	print('missing points')

# plots features
maskf3 = plotcorners(full3, featuresf3)
maskf4 = plotcorners(full4, featuresf4)

# plots features for half
maskh3 = plotcorners(half3, featuresh3)
maskh4 = plotcorners(half4, featuresh4)

# plots features for quarter
maskq3 = plotcorners(quart3, featuresq3)
maskq4 = plotcorners(quart4, featuresq4)


# window size
kernel = 5

# calculute x, y and t gradients with gaussian smoothing
fullx = cv2.Sobel(full3,cv2.CV_64F,1,0,ksize=5)
fully = cv2.Sobel(full3,cv2.CV_64F,0,1,ksize=5)
fullt = cv2.GaussianBlur(full4,(5,5),1) - cv2.GaussianBlur(full3,(5,5),1)

# calculute x, y and t gradients with gaussian smoothing
halfx = cv2.Sobel(half3,cv2.CV_64F,1,0,ksize=5)
halfy = cv2.Sobel(half3,cv2.CV_64F,0,1,ksize=5)
halft = cv2.GaussianBlur(half4,(5,5),1) - cv2.GaussianBlur(half3,(5,5),1)

# calculute x, y and t gradients with gaussian smoothing
quartx = cv2.Sobel(quart3,cv2.CV_64F,1,0,ksize=5)
quarty = cv2.Sobel(quart3,cv2.CV_64F,0,1,ksize=5)
quartt = cv2.GaussianBlur(quart4,(5,5),1) - cv2.GaussianBlur(quart3,(5,5),1)


vectorsf = lkmatrices(fullx, fully, fullt, featuresf1, kernel)
vectorsh = lkmatrices(halfx, halfy, halft, featuresh1, kernel)
vectorsq = lkmatrices(quartx, quarty, quartt, featuresq1, kernel)

avgvect = avgvectors(vectorsf, vectorsh, vectorsq)

color4 = cv2.cvtColor(full3, cv2.COLOR_GRAY2RGB)
color4 = plotcolor(color4, featuresf3, avgvect)

cv2.imshow('optical flow', color4)
cv2.waitKey(0)
cv2.destroyAllWindows()
