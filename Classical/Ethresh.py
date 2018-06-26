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
def totalpix(list):
	total = 0
	num = len(list)
	for i in range(num):
		total += list[i]
	return total

def findthreshold(H, total):
	# search for maximum entropy between foreground and background
	maxE = 0.0
	# our index for the threshold of max entropy
	thresh = 0
	A = 0.0
	B = 0.0
	# loop through all grey scale levels to find max thresh
	for i in range(256):
		for j in range(i):
			A += H[j]/float(total)
		for k in range(i, 256):
			B += float(H[k]/float(total))

		if(maxE < A+B):
			maxE = A+B
			thresh = i
		# reset A and B for next loop
		A = 0
		B = 0
	return thresh

def threshold2(H, total):
	# search for maximum entropy between foreground and background
	maxE = 0.0
	# our index for the threshold of max entropy
	thresh = 0
	A = 0.0
	B = 0.0
	# loop through all grey scale levels to find max thresh
	for i in range(256):
		for j in range(i):
			#print(H[j]/float(total))
			# cannot do log of 0
			if(H[j]/float(total) == 0):
				continue
			A += (H[j]/float(total)) * math.log(H[j]/float(total))

		for k in range(i, 256):
			# cannot do log of 0
			if(H[k]/float(total) == 0):
				continue
			B += (H[k]/float(total)) * math.log(H[k]/float(total))

		if(maxE < A+B):
			maxE = A+B
			thresh = i
		# reset A and B for next loop
		A = 0
		B = 0
	return thresh

# looks through pixels and sets to 0 if below thresh and 255 else
def applythresh(list1, thresh):
	numofrow = len(list1)
	numofcol = len(list1[0])
	for i in range(0, numofrow):
		for j in range(0, numofcol):
			if(list1[i][j] < thresh):
				list1[i][j] = 0
			else:
				list1[i][j] = 255
	return list1

# pull the image
I = array(Image.open('106025.jpg').convert("L"))
figure()
imshow(I, cmap='gray')

H = histogram(I)
#print(H)
total = totalpix(H)

threshold = findthreshold(H, total)
#print(threshold)

#threshold2 = threshold2(H, total)
#print(threshold2)

T = applythresh(I, threshold)
figure()
imshow(T, cmap='gray')
show()

####################
# pull the image
I2 = array(Image.open('156079.jpg').convert("L"))
figure()
imshow(I2, cmap='gray')

H2 = histogram(I2)
#print(H)
total2 = totalpix(H2)

threshold2 = findthreshold(H2, total2)
#print(threshold2)

#threshold2 = threshold2(H, total)
#print(threshold2)

T2 = applythresh(I2, threshold2)
figure()
imshow(T2, cmap='gray')
show()

####################
# pull the image
I3 = array(Image.open('41004.jpg').convert("L"))
figure()
imshow(I3, cmap='gray')

H3 = histogram(I3)
#print(H)
total3 = totalpix(H3)

threshold3 = findthreshold(H3, total3)
#print(threshold3)

#threshold2 = threshold2(H, total)
#print(threshold2)

T3 = applythresh(I3, threshold3)
figure()
imshow(T3, cmap='gray')
show()
