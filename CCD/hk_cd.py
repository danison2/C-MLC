"""
hkrelax.py
A demonstration of a relaxation method for computing a heat-kernel based
community that implements the algorith from "Heat-kernel based community
detection" by Kloster & Gleich.
Written by Kyle Kloster and David F. Gleich
"""

import collections
import math
import networkx as nx
 

def compute_psis(N,t):
    psis = {}
    psis[N] = 1.
    for i in range(N-1,0,-1):
        psis[i] = psis[i+1]*t/(float(i+1.))+1.
    return psis
    
def hk_cd(network, seed):
    
    G = {}
    for node in network.nodes:
        node_neighbors = set([int(node) for node in nx.neighbors(network, node)])
        G.update({int(node):node_neighbors})
        
    Gvol_all = dict(network.degree)
    Gvol = sum(Gvol_all.values()) #node degrees
    
    ## Setup parameters that can be computed automatically
    # see paper for how to set this automatically no greater than: 2t * log(1/eps)
    t = 80.
    eps = 0.01
    N = int((2 * t) * math.log(1/eps, 10))
#     print(N)
#     seed = [1]
    psis = compute_psis(N,t)
        
    ## Estimate hkpr vector 
    # G is graph as dictionary-of-sets, 
    # t, tol, N, psis are precomputed
    x = {} # Store x, r as dictionaries
    r = {} # initialize residual
    Q = collections.deque() # initialize queue
    for s in seed: 
        r[(s,0)] = 1./len(seed)
        Q.append((s,0))
    while len(Q) > 0:
        (v,j) = Q.popleft() # v has r[(v,j)] ...
        rvj = r[(v,j)]
        # perform the hk-relax step
        if v not in x: x[v] = 0
        x[v] += rvj 
        r[(v,j)] = 0
        mass = (t*rvj/(float(j)+1.))/len(G[v])
        for u in G[v]:   # for neighbors of v
            next_item = (u,j+1) # in the next block
            if j+1 == N: 
                x[u] += rvj/len(G[v])
                continue
            if next_item not in r: r[next_item] = 0
            thresh = math.exp(t)*eps*len(G[u])
            thresh = thresh/(N*psis[j+1])
            if r[next_item] < thresh and \
            r[next_item] + mass >= thresh:
                Q.append(next_item) # add u to queue
            r[next_item] = r[next_item] + mass
    
    # Find cluster, first normalize by degree
    for v in x: x[v] = x[v]/len(G[v])  
      
#     for v in range(1,len(G)+1):
#         if v in x:
#             print("hk[%2i] = %.16lf"%(v, x[v]))
#         else:
#             print("hk[%2i] = -0."%(v))
#     print(x)
        
    ## Step 2 do a sweep cut based on this vector 
      
    # now sort x's keys by value, decreasing
    sv = sorted(x.items(), key=lambda x: x[1], reverse=True)
#     supportN = [v for v, p in sv if p > 0] 
    S = set()
    volS = 0.
    cutS = 0.
    bestcond = 1.
    bestset = sv[0]
    for p in sv:
        s = p[0] # get the vertex
        volS += len(G[s]) # add degree to volume
        for v in G[s]:
            if v in S:
                cutS -= 1
            else:
                cutS += 1
#         print("v: %4i  cut: %4f  vol: %4f"%(s, cutS,volS))
        S.add(s)
        denom = min(volS,Gvol-volS)
        if cutS == denom:
            bestset = [v for (v, p) in sv]
            bestcond = 1
        else:   
            if cutS/denom < bestcond:
                bestcond = cutS/denom
                bestset = set(S) # make a copy
#     print("Best set conductance: %f"%(bestcond))
#     print("  set = ", str(bestset))
    return list(bestset)
#     return supportN


def hk_sampling(network, seed):
    
    G = {}
    for node in network.nodes:
        node_neighbors = set([int(node) for node in nx.neighbors(network, node)])
        G.update({int(node):node_neighbors})
    
    ## Setup parameters that can be computed automatically
    # see paper for how to set this automatically no greater than: 2t * log(1/eps)
    t = 80.
    eps = 0.01
    N = int((2 * t) * math.log(1/eps, 10))
#     print(N)
#     seed = [1]
    psis = compute_psis(N,t)
        
    ## Estimate hkpr vector 
    # G is graph as dictionary-of-sets, 
    # t, tol, N, psis are precomputed
    x = {} # Store x, r as dictionaries
    r = {} # initialize residual
    Q = collections.deque() # initialize queue
    for s in seed: 
        r[(s,0)] = 1./len(seed)
        Q.append((s,0))
    while len(Q) > 0:
        (v,j) = Q.popleft() # v has r[(v,j)] ...
        rvj = r[(v,j)]
        # perform the hk-relax step
        if v not in x: x[v] = 0
        x[v] += rvj 
        r[(v,j)] = 0
        mass = (t*rvj/(float(j)+1.))/len(G[v])
        for u in G[v]:   # for neighbors of v
            next_item = (u,j+1) # in the next block
            if j+1 == N: 
                x[u] += rvj/len(G[v])
                continue
            if next_item not in r: r[next_item] = 0
            thresh = math.exp(t)*eps*len(G[u])
            thresh = thresh/(N*psis[j+1])
            if r[next_item] < thresh and \
            r[next_item] + mass >= thresh:
                Q.append(next_item) # add u to queue
            r[next_item] = r[next_item] + mass
    
    # Find cluster, first normalize by degree
    for v in x: x[v] = x[v]/len(G[v])  
      
#     for v in range(1,len(G)+1):
#         if v in x:
#             print("hk[%2i] = %.16lf"%(v, x[v]))
#         else:
#             print("hk[%2i] = -0."%(v))
#     print(x)
        
    ## Step 2 do a sweep cut based on this vector 
      
    # now sort x's keys by value, decreasing
    sv = sorted(x.items(), key=lambda x: x[1], reverse=True)
    supportN = [v for v, p in sv if p > 0] 
    
    return supportN
    