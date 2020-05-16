'''
Created on Mar 24, 2020

@author: DanyK
'''
import networkx as nx
from loadNetwork import readEdgeList
from centroid_detection import preprocessG, neighborSampling

G1 = readEdgeList('../data/CCD/subgraphs/seedsLJ1/'+ str(1000584) + ".txt", delm="\t")
G = preprocessG(G1)
print(sorted([v for v in nx.neighbors(G1, 1000584)]))
print(sorted(G[1000584]))
sample = neighborSampling(G, 1000584, 10, "minDegree")
print(sample)