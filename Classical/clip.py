# Jonathon Rice
from PIL import Image
from pylab import *
import numpy
import sys
import math

# creates a histogram of an image
def histogram(list1):
	# construct histogram
	H = []
	for i in range(256):
		H.append(0)
	# count total num of pixels
	total = 0
	# grab number of col and rows
	numofrow = len(list1)
	numofcol = len(list1[0])
	for i in range(0, numofrow):
		for j in range(0, numofcol):
			# count frequency of each grey scale
			H[list1[i][j]] += 1
	return H

# counts total pixels from histogram
def total(list):
	total = 0
	num = len(list)
	for i in range(num):
		total += list[i]
	return total

# calculates the cumulative distributive function for each grey scale level
def cdf(list, total):
	num = len(list)
	prev = 0
	for i in range(num):
		list[i] = list[i]/float(total) + prev
		prev = list[i]
	return list

# equalize to create a mapping of new pixel level values to new ones
def equalize(list):
	l = 255
	num = len(list)
	for i in range(num):
		list[i] = floor(list[i] * l)
	return list

# list is the image, list2 is the equalized mapping
def map(list, list2):
	# grab number of col and rows
	numofrow = len(list)
	numofcol = len(list[0])
	for i in range(0, numofrow):
		for j in range(0, numofcol):
			list[i][j] = list2[list[i][j]]
	return list

def clip(list):
	alpha = 50
	beta = 2
	# grab number of col and rows
	numofrow = len(list)
	numofcol = len(list[0])
	for i in range(0, numofrow):
		for j in range(0, numofcol):
			if(list[i][j] < 51):
				list[i][j] = 0
			elif(list[i][j] < 151):
				list[i][j] = beta * (list[i][j] - alpha)
			else:
				list[i][j] = 200
	return list

def rangecomp(list, c):
	# grab number of col and rows
	numofrow = len(list)
	numofcol = len(list[0])
	for i in range(0, numofrow):
		for j in range(0, numofcol):
			list[i][j] = c * math.log(1 + list[i][j], 10)
	return list

#Part 1
# Grab an image and convert to a 2D array
I = array(Image.open('299091.jpg').convert("L"))

figure()
imshow(I, cmap='gray')
figure()
hist(I.flatten(), 256)
show()

h = histogram(I)
total = total(h)
cdf = cdf(h, total)
e = equalize(cdf)
Ie = map(I, e)

figure()
imshow(Ie, cmap='gray')
figure()
hist(Ie.flatten(), 256)
show()

########
#Part 2

I2 = array(Image.open('227046.jpg').convert("L"))

figure()
imshow(I2, cmap='gray')
figure()
hist(I2.flatten(), 256)
show()

Ic = clip(I2)

figure()
imshow(Ic, cmap='gray')
figure()
hist(Ic.flatten(), 256)
show()

# Comments
# The original image has 2 bell curves of pixel frequency at both
# ends of the grey scale level. The mountain and columns blend into
# each other. The sky is mostly white with grey clouds.
# After clipping, the objects around the mountain become more visible.
# Contrast on the buildings increase with sunlit areas becoming white.
# The sky has become completely white with almost all traces of the
# clouds gone. The frequency of pixels is level until value 200, with
# an immense increase in the amount of pixels chunked there.

########
#Part 3

I3 = array(Image.open('374020.jpg').convert("L"))

figure()
imshow(I3, cmap='gray')
figure()
hist(I3.flatten(), 256)
show()

Ir = rangecomp(I3, 1)

figure()
imshow(Ir, cmap='gray')
figure()
hist(Ir.flatten(), 256)
show()

Ir2 = rangecomp(I3, 10)

figure()
imshow(Ir2, cmap='gray')
figure()
hist(Ir2.flatten(), 256)
show()

Ir3 = rangecomp(I3, 100)

figure()
imshow(Ir3, cmap='gray')
figure()
hist(Ir3.flatten(), 256)
show()

Ir4 = rangecomp(I3, 1000)

figure()
imshow(Ir4, cmap='gray')
figure()
hist(Ir4.flatten(), 256)
show()

# Comments
# The original image has a gaussian bell curve like frequency.
# 3 main objects in the image; mountain, aquaduct and the sky.
# All 3 are similar in darkness. After the 1st compression
# There are only 3 pixel levels of grey scale. The image becomes
# white with grey to fill in the details. After the 2nd compression
# the image looks the same but become brighter. The 3rd is the
# same making it even brighter and hard to make out details.
# However, after the 4th the colors invert compared to the 1st
# compression and white becomes dark grey and grey becomes white.
