# Jonathon Rice
from PIL import Image
from pylab import *
import numpy
import sys
import math

def hessian(list):
	# hessian matrix
	H = numpy.zeros((2, 2))
	# holds eigenvalues
	v = []
	# grab number of col and rows
	# subtract 1 as we are ignoring borders
	numofrow = len(list) - 1
	numofcol = len(list[0]) - 1
	for i in range(1, numofrow):
		for j in range(1, numofcol):
			# calculate our second order central derivatives
			H[0][0] = list[i][j+1] - 2*list[i][j] + list[i][j-1]
			H[0][1] = int(list[i+1][j+1]) - int(list[i-1][j+1]) - int(list[i+1][j-1]) + int(list[i-1][j-1])
			H[1][0] = H[0][1]
			H[1][1] = list[i+1][j] - 2*list[i][j] + list[i-1][j]
			# grab our eignevalues
			v = linalg.eigvals(H)
			# if both eigenvalues exceed the threshold then color it
			if(v[0] > 10 and v[1] > 10):
				list[i][j] = 255

	return list

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

def harris(list):
	# harris matrix
	H = numpy.zeros((2, 2))
	# trace, determinate and cornerness place holders
	trace = 0
	determ = 0
	corner = 0
	# grab number of col and rows
	# subtract 1 as we are ignoring borders
	numofrow = len(list) - 1
	numofcol = len(list[0]) - 1
	for i in range(1, numofrow):
		for j in range(1, numofcol):
			# calculate single derivatives and square
			H[0][0] = pow(list[i][j+1] - list[i][j-1], 2)
			H[0][1] = list[i+1][j+1] - list[i-1][j+1] - list[i+1][j-1] + list[i-1][j-1]
			H[1][0] = H[0][1]
			H[1][1] = pow(list[i+1][j] - list[i-1][j], 2)
			# get trace and determinate
			trace = H[0][0] + H[1][1]
			determ = H[0][0] * H[1][1] - pow(H[0][1],2)
			# calculate cornerness
			corner = determ - (1/float(25))*pow(trace, 2)
			# if both eigenvalues exceed the threshold then color it
			if(corner > 925000):
				list[i][j] = 255

	return list

def harriseigen(list):
	# harris matrix
	H = numpy.zeros((2, 2))
	# holds eigenvalues
	v = []
	# grab number of col and rows
	# subtract 1 as we are ignoring borders
	numofrow = len(list) - 1
	numofcol = len(list[0]) - 1
	for i in range(1, numofrow):
		for j in range(1, numofcol):
			# calculate single derivatives and square
			H[0][0] = pow(list[i][j+1] - list[i][j-1], 2)
			H[0][1] = list[i+1][j+1] - list[i-1][j+1] - list[i+1][j-1] + list[i-1][j-1]
			H[1][0] = H[0][1]
			H[1][1] = pow(list[i+1][j] - list[i-1][j], 2)
			# grab our eignevalues
			v = linalg.eigvals(H)
			# calculate cornerness
			corner = v[0]*v[1] - (1/float(25))*(v[0]+v[1])
			# if both eigenvalues exceed the threshold then color it
			# cannot find a good threshold
			if(corner > 925000):
				list[i][j] = 255

	return list

#Part 1
# pull the image
I = array(Image.open('input1.png').convert("L"))

# use hessian for corner detection
C = hessian(I)
imshow(C, cmap='gray')
show()

########
#Part 2
# pull the image
I2 = array(Image.open('input2.png').convert("L"))
imshow(I2, cmap='gray')

sigma = 1
# Create 1D Gaussian mask
G = []
# Use negatives to have sigma moving away from center
for x in range(-1,2):
	G.append(gaussian(sigma, x))
normalize(G)

# Convolve G*I = Ix
I2x = convoluteX(I2, G)

# Convolve G*Ix = Ixy smoothing
# equivalent to 2D Gauss smoothing after both
I2g = convoluteY(I2x, G)

# use harris for corner detection
C2 = harris(I2g)
figure()
imshow(C2, cmap='gray')
show()

########
#Part 3
# pull the image
I3 = array(Image.open('input3.png').convert("L"))

# Convolve G*I = Ix
I3x = convoluteX(I3, G)

# Convolve G*Ix = Ixy smoothing
# equivalent to 2D Gauss smoothing after both
I3g = convoluteY(I3x, G)

# use harris for corner detection
C3 = harris(I3g)
figure()
imshow(C3, cmap='gray')
show()
