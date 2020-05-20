'''
Created on Dec 1, 2019

@author: DanyK
'''
import networkx as nx
from ppr_cd import localSampling
from hk_cd import hk_cd


def getNeighbours(G, seedset):
    if type(seedset) is not list:
        seedset = [seedset]
    neighbors = []
    for v in seedset:
        v_neighbors = [n for n in G[v] if n not in seedset] # and v's neighbors
        neighbors.extend([u for u in v_neighbors if u not in neighbors])
    return sorted(neighbors)

#friends relation score: 1 per friends
def F(u,v, G):
    if type(u) is not list:
        u = [u]
    if type(v) is not list:
        v = [v]
#     print("F...")
    total_score = sum([1 if(x in G[y] or int(x) == int(y)) else 0 for x in u for y in v])
#     print("Done F.")
    
    return total_score
        
          
#Friend of Friend: score 0.5
def FOF(u,v, G):
    if type(u) is not list:
        u = [u]
    if type(v) is not list:
        v = [v]
        
#     avg_deg_uv = getAvgDegree(G, u, v)
       
    total_score = 0
    for x in u:
        score = 0
        for y in v:
            score += len([i for i in set(G[x]).intersection(set(G[y])) if i not in u and i not in v])
        total_score += score
    return total_score * 0.5

#Our Friends are Friends relation score: 0.25
def OFF(u_list, v_list, G):
#     print("Computing OFF")
    if type(u_list) is not list:
        u_list = [u_list]
    if type(v_list) is not list:
        v_list = [v_list] 
    
    all_nodes_total = 0 #all nodes of u over all nodes of v
    for u in u_list:
        single_node_score = 0 #u score over all v nodes
        u_neighbors = neighborSampling(G, u, 100, "random")
        N_u = [n for n in u_neighbors if n not in v_list] #n==v becomes common friends
#         print("N_u: ", len(N_u))
        for v in v_list:
            v_neighbors = neighborSampling(G, v, 100, "random")
            N_v = [n for n in v_neighbors if n not in u_list]#n==u becomes common friends
#             print("N_v: ", len(N_v))
            
            single_node_score += F(N_u, N_v, G)
#             print("score: ", single_node_score)
            
        all_nodes_total += single_node_score
    all_nodes_total = all_nodes_total * 0.25
#     print("Done computing OFF")
    return all_nodes_total
        

def nextNode(G, start, seed):
    start_neighbors = neighborSampling(G, start, 10, "minDegree")
#     print("Deciding between: ", len(start_neighbors), " nodes")
    max_score = 0
    next_node  = start
    for _, v in enumerate(start_neighbors):
#         print("Node: ", v)
#         score = crs(v, start, G)
        score = FOF(start, v, G) + OFF(start, v, G)
#         score = FOF(start, v, G)
#         print("Score: ", score)
#         if int(start) ==  int(seed):
#             seed_score = 1 #self loop
#         else:
#             seed_score = F(seed, v, G) + FOF(seed, v, G) + OFF(seed, v, G)
# # #         print("Seed score: ", seed_score)
        if score > max_score:
            max_score = score
            next_node = v
    return next_node, max_score

def biased_DFS(G, start, seed):
    queue = [start]
    explored = [start]
    scores = [1 * len(G[start])]
#     i = 1
    while(queue):
        start = queue.pop()
#         print("start node: ", start)
        next_node, score = nextNode(G, start, seed)
#         next_node, score = nextVertex(G, start)
#         print("Next node: ", next_node)
#         print("Score: ", score)
        if next_node not in explored:
            queue.append(next_node)
            explored.append(next_node)
            scores.append(score)
#         i += 1        
    prev_score = scores[-1] #previous node
    if prev_score > score:
        centroid = explored[-1]
    elif score > prev_score:
        centroid = next_node 
    else: #compare degrees
        #check if it is the same node
        if len(explored) == 1:
            centroid = explored[0] 
        else:     
            node_degrees = sorted([(v, len(G[v])) for v in explored[-2:]])
#             print("node degrees: ", node_degrees)
            if node_degrees[0][1] > node_degrees[1][1]: #
                centroid = node_degrees[0][0]
            else:
                centroid = node_degrees[1][0]

#         centroid = max(node_degrees, key=lambda x: x[1])[0]
#     print("Found a centroid: ", centroid)
    return centroid

def BFS(graph, start, max_level = 2):
#     print("BFS...")
    # keep track of all visited nodes
    explored = []
    # keep track of nodes to be checked
    queue = [start]
    levels = {}         # this dict keeps track of levels
    levels[start] = 0    # depth of start node is 0
    # to avoid inserting the same node twice into the queue
    visited = [start]
    # keep looping until there are nodes still to be checked
    while queue:
        node = queue.pop(0)
        if levels[node] <= max_level:
            explored.append(node)
            neighbours = graph.neighbors(node)
            # add neighbours of node to queue
            for neighbour in neighbours:
                if neighbour not in visited:
                    queue.append(neighbour)
                    visited.append(neighbour)
                    levels[neighbour] = levels[node] + 1
                    # print(neighbour, ">>", levels[neighbour]
                    
#     print("END BFS")
    return explored

def extendSeed(G, seed):
    C = [seed]
    neibourhood_C = getNeighbours(G, C)
    explored = [seed]
    while(True):
        com_size = len(C)
#         print("C: ", C)
        for v in [n for n in neibourhood_C if n not in C and n not in explored]:
#             print("v: ", v)
            centroid = biased_DFS(G, v, v)
#             print("centroid: ", centroid)
            explored.append(v)
            if(centroid in C or seed in G[centroid]):
                if(v not in C):
                    C.append(v)
#             score =  F(C, v, G) + FOF(C, v, G)
# #             print("score: ", score, "F: ", F(C, v, G), "FOF: ", FOF(C, v, G))
# #             print("degree: ", G.degree(v))
#             score =  score / G.degree(v)
#             print("Norm Score: ", score)
#             if score >= 0.5:
#                 C.append(v)
                
#             else:
#                 rel_C = F(v, C, G) + FOF(v, C, G) + OFF(v, C, G)
#                 rest = [n for n in G.neighbors(v) if n not in C]
#                 rel_rest = F(v, rest, G) + FOF(v, rest, G) + OFF(v, rest, G)
#                 if(rel_C > rel_rest):
#                     if(v not in C):
#                         C.append(v)
        if len(C) == com_size:
            break
        else:
#             break
            neibourhood_C = getNeighbours(G, C)
#         print(sorted(C))
    return sorted([int(v) for v in C])

def neighborSampling(G, seed, max_nodes, samplingMethod):
    import random
#     print("Sampling...")
    seed_neighbors = G[seed]
    neighbor_degrees = [(v, len(G[v])) for v in seed_neighbors]
    sorted_asc = sorted(neighbor_degrees, key=lambda x:x[1])
    len_sorted = len(sorted_asc)
    nber = min(len_sorted, max_nodes) #nodes to return

    if samplingMethod == "maxDegree":
        sampled_neighbors = [v for v,_ in sorted_asc[-nber:]] #max degree
    elif samplingMethod == "minDegree":
        sampled_neighbors = [v for v,_ in sorted_asc[:nber]] #min degree
    else:
        sampled_neighbors = [v for v in random.sample(list(seed_neighbors), nber)]
#     print("Done Samppling...")
    return sampled_neighbors

def findCentroids(G, sampled_neighbors, seed):
#     print("Finding centroids")
#     print("Finding centroids of ", len(sampled_neighbors), " nodes: ", sampled_neighbors)
    centroids = []
    for _, start in enumerate(sampled_neighbors):
#         print("Finding a centroid for ", start)
        centroid = biased_DFS(G, start, seed)
#         centroid = getCore(seed, G1)
        centroids.append(int(centroid))
#         print(centroid)
    centroids = list(set(centroids))
    return centroids

def preprocessG(network):
#     print("Preprocessing G...")
    G = {}
    for node in network.nodes:
        node_neighbors = set([node for node in nx.neighbors(network, node)])
#         print("sizeG", len(network.nodes))
        G.update({node:node_neighbors})
#     print("Done preprocessing")
    return G

def detectComs(G1, G, seed, sample_size, samplingMethod):
#     n = len(G1.nodes)
    #     print("Detecting Communities of the seed: ", seed) 
    sample = neighborSampling(G, seed, sample_size, samplingMethod)
#     sample.append(seed)
#     print("Sample: ", sample)
    centroids = findCentroids(G, sample, seed)
    filtered_centroids = filterCentroids(G1, centroids)
#     print("centroids: ", centroids)
    #     centroids.append(seed)
         
    coms = []
    for centroid in filtered_centroids:
#         print("Finding a community for: ", centroid, "with degree: ", G.degree(centroid))
#         seedset = [centroid, seed]
#         com = sorted(extendSeed(G, centroid))
        com = localSampling(G1, [centroid])
        #com = hk_cd(G1, [centroid])
        
#         com  = LOSP(G1, [centroid])
#         com.extend([seed])
     
        coms.append(list(set(com)))
#     return coms
    return coms, filtered_centroids

def filterCentroids(G, centroids):
    from itertools import combinations
    filtered_centroids = [c for c in centroids]
    possible_edges = list(combinations(sorted(centroids), 2))
    for u,v in possible_edges:
#         print(u,"_", v)
        if u in G[v]: #they are connected then choose one
#             print("connected")
            if len(G[u]) < len(G[v]):
                if u in filtered_centroids:
                    filtered_centroids.remove(u) 
            else:
                if v in filtered_centroids:
                    filtered_centroids.remove(v)  
    return filtered_centroids
