'''
Created on 5 Apr 2018

@author: DanyK
'''

#import networkx as nx
import numpy as np
from operator import itemgetter
import networkx as nx
import sys
np.set_printoptions(threshold=sys.maxsize)

# collabPath = "/content/drive/My Drive/Community/"
collabPath = "../"

def getClustersMine(G):
    from NMFdecomposition import decomposition
    from getMembership import getMembership
    
#     print("Filtered: ", sorted(G.nodes))
    A = nx.adjacency_matrix(G).todense()
#     print(A.shape)
    limitN = len(sorted(G.nodes))
    # estimate the number of communities starting from 1
#     print("limit: ", limitN)
    for k in range(2,int(limitN)):
#         print("iter: ", k)
        _, H = decomposition(A, k)
        M = getMembership(H)
#         print(np.transpose(np.round(M,2)))

        #we assume a  community should have at least one centroid
        # Therefore, we filter only those nodes with 1's and make others 0's
        M[M < 1] = 0
#         print(np.transpose(np.round(M,2)))
        
#         print(M.any(axis=0))
        
        if (~M.any(axis=0)).any():
            #index = np.where(~M.any(axis=0))[0]
#             print("We estimated " + str(k) + " communities because #" + 
#                   str(k + 1) + " yields communities without cluster centers")
            break
    return k-1


def getCommunities(G, clusters, alpha):
    from NMFdecomposition import decomposition
    from getMembership import getMembership
    
    detectedCommunities = []
    A = nx.adjacency_matrix(G).todense()
    nodes = sorted(nx.nodes(G))
    _, H = decomposition(A, clusters)
    M = getMembership(H)
    #print(np.transpose(np.round(M,2)))
    M[M >= alpha] = 1
    # we assume the probability of 0.75 is enough to include a node in a community
    M[M < 1] = 0
    #print(np.transpose(np.round(M,2)))
       
    # map indices of hyperedges in M to the corresponding nodes
    numbering = list(range(len(nodes)))  # create indices 
    for k in range(0, clusters):
        # extract indices of nodes with community membership = 1
        c_indices = [j for i, j in zip(M[:, k], numbering) if i == 1]
        if len(c_indices) == 1:
            c_nodes = [nodes[c_indices[0]]]
        else:
            # convert tuple to list
            c_nodes = list(itemgetter(*c_indices)(nodes))
        
        # Do a 1 step BSF to include the neighbours of the centroids
        temp = []
        for node in c_nodes:
            neighbours = nx.neighbors(G, node)
            if node not in temp:
                temp.append(node)
            temp.extend([int(v) for v in neighbours if v not in temp])
        temp = sorted(temp)
        detectedCommunities.append(temp)
    return detectedCommunities
