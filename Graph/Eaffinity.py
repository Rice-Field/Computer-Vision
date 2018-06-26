from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import scipy as sp
import tensorflow as tf
from tensorflow import losses
import os
import math

from detgenhist import detGen

# passes 2 images and returns entropy of images stacked
def entropy(I1, I2):
    # total entropy
    entropy = 0.0

    # turn into numpy arrays
    n1 = I1
    n2 = I2

    # stack
    # n3 = np.concatenate((n1, n2), axis=0)
    n3 = n1 - n2

    hist, _ = np.histogram(n3,bins=np.arange(256), density=True)

    # image3 = Image.fromarray(n3)
    # plt.imshow(image3)
    # plt.pause(1)
    # plt.show()

    # loop through each cell and calc entropy
    for i in range(len(hist)):
        # cannot do log of 0
        if(hist[i] == 0):
            continue
        entropy += (hist[i]) * math.log(hist[i], 2)

    return -entropy


F = np.loadtxt('pickframe.txt')
# for i in range():
startframe = int(F[0])
endframe = int(F[1])
j = 0
for cframe, cBB, cXY, chist, frame in detGen(int(1), int(2)):
    for k in range(len(cframe[0])):
        imgtable[(j*4)+k] = cframe[0,k]
    j += 1

for i in range(4):
    for j in range(4):
        A[i,j] = entropy(imgtable[i],imgtable[4+j])

hungA = sp.optimize.linear_sum_assignment(A)
for j in range(4):
    H[j][hungA[1][j]] = 1

print(A)
print(H)

np.save('affinity12', A)
