'''
Created on 29 Dec 2017

@author: DanyK
'''
import numpy as np
import networkx as nx

def readEdgeList(filename, delm):
    G = nx.read_edgelist(filename, delimiter=delm, nodetype=int)
    return G

#get x,y columns
def readGraph(filename,delm):
    x, y = np.loadtxt(filename, delimiter=delm, unpack=True, usecols=range(2))
    #plot the graph
    nodes = [int(i) for i in set(x).union(y)]
    edges = zip(x,y)
    network =  nx.Graph()
    network.add_nodes_from(nodes)
    network.add_edges_from(edges)
    return network   # x,y

def getGraphFromAdjacencyMatrix(adjacency_matrix):
    rows, cols = np.where(adjacency_matrix >= 1)
    edges = zip(rows.tolist(), cols.tolist())
    gr = nx.Graph()
    gr.add_edges_from(edges)
    return gr

# G = readEdgeList("../data/graphs/1000001.txt", "\t")
# 
# neighbors = nx.neighbors(G, 48637)
# print([v for v in neighbors])
#     
   
