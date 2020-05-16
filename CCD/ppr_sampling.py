import collections
import sys

# setup the graph
# G = {
#   1:set([ 2, 3, 5, 6,]),
#   2:set([ 1, 4,]),
#   3:set([ 1, 6, 7,]),
#   4:set([ 2, 5, 7, 8,]),
#   5:set([ 1, 4, 6, 8, 9, 10,]),
#   6:set([ 1, 3, 5, 7,]),
#   7:set([ 3, 4, 6, 9,]),
#   8:set([ 4, 5, 9,]),
#   9:set([ 5, 7, 8, 20,]),
#   10:set([ 5, 11, 12, 14, 15,]),
#   11:set([ 10, 12, 13, 14,]),
#   12:set([ 10, 11, 13, 14, 15,]),
#   13:set([ 11, 12, 15,]),
#   14:set([ 10, 11, 12, 25,]),
#   15:set([ 10, 12, 13,]),
#   16:set([ 17, 19, 20, 21, 22,]),
#   17:set([ 16, 18, 19, 20,]),
#   18:set([ 17, 20, 21, 22,]),
#   19:set([ 16, 17,]),
#   20:set([ 9, 16, 17, 18,]),
#   21:set([ 16, 18,]),
#   22:set([ 16, 18, 23,]),
#   23:set([ 22, 24, 25, 26, 27,]),
#   24:set([ 23, 25, 26, 27,]),
#   25:set([ 14, 23, 24, 26, 27,]),
#   26:set([ 23, 24, 25,]),
#   27:set([ 23, 24, 25,]),
# }
# Gvol = 102

import networkx as nx
from loadNetwork import readGraph
import numpy as np

def getNeighbourhood(G, seedset):
    if type(seedset) is not list:
        seedset = [seedset]
    neighborhood = []
    for v in seedset:
        neighborhood.extend([v]) # v is also a member
        v_neighbors = nx.neighbors(G, v) # and v's neighbors
        neighborhood.extend([u for u in v_neighbors if u not in neighborhood])
    return sorted(neighborhood)

def nextVertex(network, seed):
    
    if type(seed) is not list:
        seed = [seed]
    seed_neighborhood1 = getNeighbourhood(network, seed)
    seed_neighborhood2 = getNeighbourhood(network, seed_neighborhood1)
    seed_neighborhood = getNeighbourhood(network, seed_neighborhood2)
    
    subG = nx.subgraph(network, seed_neighborhood)
    #working on the neighbourhood graph
    
    G = {}
    for node in subG.nodes:
        node_neighbors = set([int(node) for node in nx.neighbors(subG, node)])
        G.update({int(node):node_neighbors})
    Gvol_all = dict(subG.degree)
    Gvol = sum(Gvol_all.values()) #node degrees
#     print(Gvol)
        
    # G is graph as dictionary-of-sets
    alpha = 0.99
    tol = 1e-3
#     seed=[8]
    
    x = {} # Store x, r as dictionaries
    r = {} # initialize residual
    Q = collections.deque() # initialize queue
    for s in seed: 
        r[s] = 1/len(seed)
        Q.append(s)
    while len(Q) > 0:
        v = Q.popleft() # v has r[v] > tol*deg(v)
        if v not in x: x[v] = 0
        x[v] += (1-alpha)*r[v]
        mass = alpha*r[v]/(2*len(G[v])) 
        for u in G[v]: # for neighbors of u
            assert u is not v, "Error"
            if u not in r: r[u] = 0.
            if r[u] < len(G[u])*tol and \
               r[u] + mass >= len(G[u])*tol:
                Q.append(u) # add u to queue if large
            r[u] = r[u] + mass
        r[v] = mass*len(G[v]) 
        if r[v] >= len(G[v])*tol: Q.append(v)
#     print(x)
      
#     # Find cluster, first normalize by degree
    for v in x: x[v] = x[v]/len(G[v])
    # now sort x's keys by value, decreasing
    sv_init = sorted(x.items(), key=lambda x: x[1], reverse=True)
    sv = [(v, np.round(p,2)) for v, p in sv_init]
    print("sv: ", sv)
    if len(sv) > 1:
        if int(sv[0][0]) == int(seed[0]):
            topNode = sv[1][0] #the second after the seed
            topProb = sv[1][1]
        else:
            topNode = sv[0][0] #the second after the seed
            topProb = sv[0][1]
    else:
        topNode = sv[0][0] #the second after the seed
        topProb = sv[0][1]
    print("Next node: ", topNode)
    return topNode, topProb

def ppr_cd(network, seed):
    G = {}
    for node in network.nodes:
        node_neighbors = set([int(node) for node in nx.neighbors(network, node)])
        G.update({int(node):node_neighbors})
#     Gvol_all = dict(network.degree)
#     Gvol = sum(Gvol_all.values()) #node degrees
#     print(Gvol)
        
    # G is graph as dictionary-of-sets
    alpha = 0.99
    tol = 0.001
#     seed=[8]
    
    x = {} # Store x, r as dictionaries
    r = {} # initialize residual
    Q = collections.deque() # initialize queue
    for s in seed: 
        r[s] = 1/len(seed)
        Q.append(s)
    while len(Q) > 0:
        v = Q.popleft() # v has r[v] > tol*deg(v)
        if v not in x: x[v] = 0
        x[v] += (1-alpha)*r[v]
        mass = alpha*r[v]/(2*len(G[v])) 
        for u in G[v]: # for neighbors of u
            assert u is not v, "Error"
            if u not in r: r[u] = 0.
            if r[u] < len(G[u])*tol and \
               r[u] + mass >= len(G[u])*tol:
                Q.append(u) # add u to queue if large
            r[u] = r[u] + mass
        r[v] = mass*len(G[v]) 
        if r[v] >= len(G[v])*tol: Q.append(v)
#     print(x)
         
#     # Find cluster, first normalize by degree
    for v in x: x[v] = x[v]/len(G[v])
    # now sort x's keys by value, decreasing
    sv = sorted(x.items(), key=lambda x: x[1], reverse=True)
    supportN = [v for v, p in sv if p > 0] 
#     if len(supportN) < 5000:
#         return supportN
#     else:
#         return supportN[:5000]
#     S = set()
#     volS = 0.
#     cutS = 0.
#     bestcond = 1.
#     bestset = sv[0]
#     for p in sv:
#         s = p[0] # get the vertex
#         volS += len(G[s]) # add degree to volume
#         for v in G[s]:
#             if v in S:
#                 cutS -= 1
#             else:
#                 cutS += 1
# # #         print("v: %4i  cut: %4f  vol: %4f"%(s, cutS,volS))
#         S.add(s)
#         denom = min(volS,Gvol-volS)
#         if cutS == denom:
#             bestset = supportN
#             bestcond = 1
#         else:   
#             if cutS/denom < bestcond:
#                 bestcond = cutS/denom
#                 bestset = set(S) # make a copy
    return supportN#bestset #, bestcond 


def ppr_sampling(network, seedset):
    localComs = {}
    sampled = ppr_cd(network, seedset)
#     print(sampled)
#     print("Seed: ", seed)
#     print("Best set conductance: %f"%(bestcond))
#     print("  set = ", str(bestset))
    
#     initial sampling
#     if bestcond < 1:
#         sampled = [node for node in bestset]
#     else: 
#         sampled = [bestset[0]]
#     localComs.update({seed:sampled})
    
#     seed_neighbors = network.neighbors(seed)
#     sorted_neighbors = sortNodesByDegree(network, seed_neighbors)
#     topN = sorted_neighbors[:5]
#     
#     #add neighbhood
#     for node in topN:
#         node =  int(node)
# #         node_degree = nx.degree(network, node)
# #         if node_degree > 0:
# #             print("Node: ", node)
#         bestset, bestcond = ppr_cd(network, [node])
# #         print("Best set conductance: %f"%(bestcond))
# #         print("  set = ", str(bestset))
#         if bestcond < 1:
#             newCom = [node for node in bestset]
#             sampled.extend(newCom)
#         else: 
#             newCom = bestset[0]
#             sampled.append(newCom)
#         localComs.update({node:newCom})
#     sampled = sorted(list(set(sampled)))
#     print("sampled: ", sampled)
    return sampled

#run the function

# network = nx.karate_club_graph()
# 
# # graphFile = 'graphA.txt'
# # network = readGraph('../data/graphs/' + graphFile, delm="\t")
# # seed = 100800
# seed = 8
# sampled = localSampling(network, seed)
# print(sampled)

#testing demon
# import demon as d
#  
# g = nx.karate_club_graph()
# dm = d.Demon(graph=g, epsilon=0.25, min_community_size=3)
# coms = dm.execute()
# print(coms)
    
