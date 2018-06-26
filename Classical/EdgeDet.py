# Jonathon Rice
from PIL import Image
from pylab import *
import numpy
import sys
import math

# Calculates gaussian values
def gaussian(sigma, x):
	n = pow(numpy.e, (-x*x/(2*sigma*sigma)))
	return (n/(sqrt(2*numpy.pi)*sigma))

# normalize kernel to total to 1
def normalize(array):
	sum = 0
	for i in range(0,3):
		sum += G[i]

	for i in range(0,3):
		G[i] = G[i]/sum

# Convolute horizontally, image matrix needs to be 1st parameter
def convoluteX(list1, list2):
	numofrow = len(list1)
	numofcol = len(list1[0])
	list3 = []
	for i in range(0, numofrow):
		# add a new row
		list3.append([])
		for j in range(0, numofcol):
			#check if out of bounds, treat as 0
			if(j-1 < 0):
				list3[i].append(list2[1]*list1[i][j] + list2[2]*list1[i][j+1])
			elif(j+1 > numofcol-1):
				list3[i].append(list2[0]*list1[i][j-1] + list2[1]*list1[i][j])
			else:
				list3[i].append(list2[0]*list1[i][j-1] + list2[1]*list1[i][j] + list2[2]*list1[i][j+1])
	return list3
	
# Convolute Vertically, image matrix needs to be 1st parameter
def convoluteY(list1, list2):
	numofrow = len(list1)
	numofcol = len(list1[0])
	list3 = []
	for i in range(0, numofrow):
		# add a new row
		list3.append([])
		for j in range(0, numofcol):
		# check if out of bounds, treat as 0
			if(i-1 < 0):
				list3[i].append(list2[1]*list1[i][j] + list2[2]*list1[i+1][j])
			elif(i+1 > numofrow-1):
				list3[i].append(list2[0]*list1[i-1][j] + list2[1]*list1[i][j])
			else:
				list3[i].append(list2[0]*list1[i-1][j] + list2[1]*list1[i][j] + list2[2]*list1[i+1][j])
	return list3

# Magnitude of two images together
def magnitude(list1, list2):
	numofrow = len(list1)
	numofcol = len(list1[0])
	list3 = []
	for i in range(0, numofrow):
		# add a new row
		list3.append([])
		for j in range(0, numofcol):
			list3[i].append(sqrt(pow(list1[i][j], 2) + pow(list2[i][j], 2)))
	return list3

# Non-maximum Suppression, parameter 3 should be image magnitude
def suppression(list1, list2, list3):
	numofrow = len(list1)
	numofcol = len(list1[0])
	angle = 0
	for i in range(0, numofrow):
		for j in range(0, numofcol):
			# angle of the gradient
			angle = math.atan2(list1[i][j],list2[i][j])
			# check north and south
			if(angle > -3*numpy.pi/4 and angle < 3*numpy.pi/4):
				# check if out of bounds
				if(i+1 <= numofrow-1):
					# pixel is less than northern 1
					if(list3[i][j] < list3[i+1][j]-15):
						list3[i][j] = 0
				# check if out of bounds
				if(i-1 >= 0):
					# pixel is less than southern 1
					if(list3[i][j] < list3[i-1][j]-15):
						list3[i][j] = 0
			#other angles
			#elif():

			#elif():
			# check east and west
			else:
				# check if out of bounds
				if(j-1 >= 0):
					# pixel is less than western 1
					if(list3[i][j] < list3[i][j-1]-15):
						list3[i][j] = 0
				# check if out of bounds
				if(j+1 <= numofcol-1):
					# pixel is less than eastern 1
					if(list3[i][j] < list3[i][j+1]-15):
						list3[i][j] = 0
	return list3

# Determine if pixel is an edge through hi and lo thresholds
def hysteresis(list1):
	numofrow = len(list1)
	numofcol = len(list1[0])
	total = numofrow*numofcol
	top30 = numpy.percentile(list1, 90)
	bot20 = numpy.percentile(list1, 20)
	for i in range(0, numofrow):
		for j in range(0, numofcol):
			if(list1[i][j] >= top30):
				list1[i][j] = 255
			elif(list1[i][j] <= bot20):
				list1[i][j] = 0

	return list1

# Determine standard deviation for gaussian filtering
sigma = .5

# Grab an image and convert to a 2D array
I = array(Image.open('56028.jpg').convert("L"))

# Create 1D Gaussian mask
G = []
# Use negatives to have sigma moving away from center
for x in range(-1,2):
	G.append(gaussian(sigma, x))
normalize(G)
#print(G)

# derivative of G, Gx
Gx = []
for x in range(-1,2):
	Gx.append(-x/(sigma*sigma)*G[x+1])

#print(Gx)

# Gy is the same as Gx but applied vertically to the image
Gy = Gx

# Convolve G*I = Ix
Ix = convoluteX(I, G)
figure()
imshow(Ix, cmap='gray')

# Convolve G*I = Iy
Iy = convoluteY(I, G)
figure()
imshow(Iy, cmap='gray')

# Convolve Gx*Ix
Idx = convoluteX(Ix, Gx)
figure()
imshow(Idx, cmap='gray')

# Convolve Gy*Iy
Idy = convoluteY(Iy, Gy)
figure()
imshow(Idy, cmap='gray')

# Magnitude of I'x and I'y
Im = magnitude(Idx, Idy)
figure()
imshow(Im, cmap='gray')

# Non-maximum suppression, remove false-positive edges
Is = suppression(Idx, Idy, Im)
figure()
imshow(Is, cmap='gray')

# Hysteresis thresholding
Ih = hysteresis(Is)
figure()
imshow(Ih, cmap='gray')
show()

# Determine standard deviation for gaussian filtering
sigma = 1

# Grab an image and convert to a 2D array
I2 = array(Image.open('65019.jpg').convert("L"))

# Convolve G*I = Ix
I2x = convoluteX(I2, G)
figure()
imshow(I2x, cmap='gray')

# Convolve G*I = Iy
I2y = convoluteY(I2, G)
figure()
imshow(I2y, cmap='gray')

# Convolve Gx*Ix
I2dx = convoluteX(I2x, Gx)
figure()
imshow(I2dx, cmap='gray')

# Convolve Gy*Iy
I2dy = convoluteY(I2y, Gy)
figure()
imshow(I2dy, cmap='gray')

# Magnitude of I'x and I'y
I2m = magnitude(I2dx, I2dy)
figure()
imshow(I2m, cmap='gray')

# Non-maximum suppression, remove false-positive edges
I2s = suppression(I2dx, I2dy, I2m)
figure()
imshow(I2s, cmap='gray')

# Hysteresis thresholding
I2h = hysteresis(I2s)
figure()
imshow(I2h, cmap='gray')
show()

# Determine standard deviation for gaussian filtering
sigma = 5

# Grab an image and convert to a 2D array
I3 = array(Image.open('42044.jpg').convert("L"))

# Convolve G*I = Ix
I3x = convoluteX(I3, G)
figure()
imshow(I3x, cmap='gray')

# Convolve G*I = Iy
I3y = convoluteY(I3, G)
figure()
imshow(I3y, cmap='gray')

# Convolve Gx*Ix
I3dx = convoluteX(I3x, Gx)
figure()
imshow(I3dx, cmap='gray')

# Convolve Gy*Iy
I3dy = convoluteY(I3y, Gy)
figure()
imshow(I3dy, cmap='gray')

# Magnitude of I'x and I'y
I3m = magnitude(I3dx, I3dy)
figure()
imshow(I3m, cmap='gray')

# Non-maximum suppression, remove false-positive edges
I3s = suppression(I3dx, I3dy, I3m)
figure()
imshow(I3s, cmap='gray')

# Hysteresis thresholding
I3h = hysteresis(I3s)
figure()
imshow(I3h, cmap='gray')
show()

# comments
# the best sigma value kind of depends on the image but I seem to
# get the best results from a sigma of 1
