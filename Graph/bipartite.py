# Jonathon Rice
# Build graphs to visualize solving hungarian assignment

import numpy as np
import scipy as sp
import os
import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms import bipartite
from scipy import optimize
from hungarian import hungarian, allSolutions


def buildAffGraph(matrix, k):
	if not os.path.exists('graphs/'):
		os.makedirs('graphs/')

	num = len(matrix)
	B = nx.Graph()
	# Add nodes with the node attribute "bipartite"
	B.add_nodes_from(np.arange(1, num+1), bipartite=0)
	B.add_nodes_from(np.arange(num+1,num*2+1), bipartite=1)
	# Add edges only between nodes of opposite node sets
	for i in range(num):
		for j in range(num):
				B.add_edges_from([(i+1, num+j+1, {'weight': matrix[i,j]})])

	color_map = []
	for node in B:
		if node < num+1:
			color_map.append('blue')
		else: color_map.append('red')

	X, Y = bipartite.sets(B)
	pos = dict()
	pos.update( (n, (1, i)) for i, n in enumerate(X) ) # put nodes from X at x=1
	pos.update( (n, (2, i)) for i, n in enumerate(Y) ) # put nodes from Y at x=2

	nx.draw(B, pos, node_color=color_map, with_labels=True, font_weight='bold')
	labels = nx.get_edge_attributes(B,'weight')
	nx.draw_networkx_edge_labels(B, pos, edge_labels=labels)
	plt.savefig("graphs/aff{}.png".format(k+1))
	plt.close()

	E = nx.algorithms.bipartite.matrix.biadjacency_matrix(B,np.arange(1, num+1)).todense()
	# print(E)

def buildPermGraph(matrix, k):
	if not os.path.exists('graphs/'):
		os.makedirs('graphs/')

	num = len(matrix)
	B = nx.Graph()
	# Add nodes with the node attribute "bipartite"
	B.add_nodes_from(np.arange(1, num+1), bipartite=0)
	B.add_nodes_from(np.arange(num+1,num*2+1), bipartite=1)
	# Add edges only between nodes of opposite node sets
	for i in range(num):
		for j in range(num):
			if matrix[i,j] > 0:
				B.add_edges_from([(i+1, num+j+1, {'weight': matrix[i,j]})])

	color_map = []
	for node in B:
		if node < num+1:
			color_map.append('blue')
		else: color_map.append('red')

	nx.draw(B,node_color=color_map, with_labels=True, font_weight='bold')
	plt.savefig("graphs/perm{}.png".format(k+1))
	plt.close()

	E = nx.algorithms.bipartite.matrix.biadjacency_matrix(B,np.arange(1, num+1)).todense()
	# print(E)

# A = np.load('affinity.npy')
# B = np.load('solutions.npy')

# buildAffGraph(A[0],0)
# buildPermGraph(B[0],1)