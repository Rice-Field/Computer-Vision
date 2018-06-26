# Jonathon Rice
# Matrix creation assuring there is only 1 hungarian solution for given matrix
# Returns random generated matrices in affinity file and their solutions in solutions

import numpy as np
import scipy as sp
from scipy import optimize
from hungarian import hungarian, allSolutions

# select number of num to generate
amount = 100000
dim = 4
matrix_dim = 16 

# random and permutation num
# X = np.random.random((num, 12, 10))
# X = np.random.randint(0, 99, size=[num,dim,dim])
H = np.zeros((dim, dim), dtype=int)
A = [] # selected affinity matrices
P = [] # selected permutation matrices

# H = np.negative(H)

total = 0
while(total < amount):
    X = np.random.randint(0, 99, size=[dim,dim])
    cost, hung = hungarian(X)
    solutions = []
    cost, solutions = allSolutions(cost, solutions)
    num = len(solutions)
    if num == 1:
        for j in range(dim):
            H[j,solutions[0][j]] = 1
        A.append(X)
        P.append(H)
        total += 1
        H = np.zeros((dim, dim), dtype=int)

print("Number of matrices with 1 solution: %d" % len(P))

np.save('affinity', A)
np.save('solutions', P)

##############################
# Creates test matrices

# X = np.random.randint(0, 99, size=[100,dim,dim])
# H = np.zeros((40, dim, dim), dtype=int)

# # create permutation matrices
# for i in range(0, 40):
#     for j in range(0, dim):
#         sum = 0
#         for k in range(0, dim):
#             sum += X[i][j][k]

#         X[i][j] = np.exp(X[i][j])/np.sum(np.exp(X[i][j]))

#     hungA = sp.optimize.linear_sum_assignment(X[i])
#     for j in range(0, dim):
#         H[i][j][hungA[1][j]] = 1

# np.save('testaffinity', X)
# np.save('testsolutions', H)
