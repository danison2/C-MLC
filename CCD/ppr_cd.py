import collections

import networkx as nx


def ppr_cd(network, seed):
    G = {}
    for node in network.nodes:
        node_neighbors = set([int(node) for node in nx.neighbors(network, node)])
        G.update({int(node):node_neighbors})
    Gvol_all = dict(network.degree)
    Gvol = sum(Gvol_all.values()) #node degrees
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
    S = set()
    volS = 0.
    cutS = 0.
    bestcond = 1.
    bestset = sv
    for p in sv:
        s = p[0] # get the vertex
        volS += len(G[s]) # add degree to volume
        for v in G[s]:
            if v in S:
                cutS -= 1
            else:
                cutS += 1
# #         print("v: %4i  cut: %4f  vol: %4f"%(s, cutS,volS))
        S.add(s)
        denom = min(volS,Gvol-volS)
        if cutS == denom:
            bestset = [v for (v, p) in sv]
            bestcond = 1
        else:   
            if cutS/denom < bestcond:
                bestcond = cutS/denom
                bestset = set(S) # make a copy
    return bestset #, bestcond 


def localSampling(network, seedset):
    sampled = list(ppr_cd(network, seedset))
    return sampled