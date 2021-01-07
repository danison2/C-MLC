'''
Created on Dec 1, 2019

@author: DanyK
'''
import networkx as nx
from CCD.ppr_cd import ppr_cd
from hk_cd import hk_cd

def keys_exists(element, *keys):
    '''
    Check if *keys (nested) exists in `element` (dict).
    '''
    if not isinstance(element, dict):
        raise AttributeError('keys_exists() expects dict as first argument.')
    if len(keys) == 0:
        raise AttributeError('keys_exists() expects at least two arguments, one given.')

    _element = element
    for key in keys:
        try:
            _element = _element[key]
        except KeyError:
            return False
    return True

def choose_next_node(G, current_node, similarity_scores):
    max_score = -1
    next_node = -1
    for u in list(nx.neighbors(G, current_node)):
        if keys_exists(similarity_scores, current_node):
            if keys_exists(similarity_scores[current_node], u):
                score = similarity_scores[current_node][u]
            else:
                score = FOF(current_node, u, G) + OFF(current_node, u, G)
    #             similarity_scores[current_node][u] = score
    #             similarity_scores[u][current_node] = score
                similarity_scores[current_node] = {}
                similarity_scores[current_node][u] = score
                similarity_scores[u] = {}
                similarity_scores[u][current_node] = score
            if score > max_score:
                max_score = score
                next_node = u
        else:
            score = FOF(current_node, u, G) + OFF(current_node, u, G)
#             similarity_scores[current_node][u] = score
#             similarity_scores[u][current_node] = score
            similarity_scores[current_node] = {}
            similarity_scores[current_node][u] = score
            similarity_scores[u] = {}
            similarity_scores[u][current_node] = score
        if score > max_score:
            max_score = score
            next_node = u
    return next_node
                          

def choose_centroid_node(G, exploring_path):
    if nx.degree(G, exploring_path[-2]) > nx.degree(G, exploring_path[-1]):
        centroid_node = exploring_path[-2]
    elif nx.degree(G, exploring_path[-2]) < nx.degree(G, exploring_path[-1]):
        centroid_node = exploring_path[-1]
    else:
        centroid_node = max(exploring_path[-1], exploring_path[-2])
    return centroid_node


def findCentroids(G,sampled_neighbors):
    centroid_set = []
    explored_nodes = {}
    similarity_scores = {}
    for u in sampled_neighbors:
        current_node = u
        exploring_path = [u]
        while(True):
            if current_node in explored_nodes.keys():
                centroid_node = explored_nodes[current_node]
                for v in exploring_path:
                    explored_nodes[v] = centroid_node
                break
            else:
                next_node = choose_next_node(G, current_node, similarity_scores)
                if next_node not in exploring_path:
                    current_node = next_node
                    exploring_path.append(next_node)
                else:
                    if len(exploring_path) == 1:
                        centroid_node = current_node
                    else:
                        centroid_node = choose_centroid_node(G, exploring_path)
                    centroid_set.append(centroid_node)
                    for v in exploring_path:
                        explored_nodes[v] = centroid_node
                    break
    return list(set(centroid_set))
                        
                        

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

def preprocessG(network):
#     print("Preprocessing G...")
    G = {}
    for node in network.nodes:
        node_neighbors = set([node for node in nx.neighbors(network, node)])
#         print("sizeG", len(network.nodes))
        G.update({node:node_neighbors})
#     print("Done preprocessing")
    return G

def getShortestPathNodes(G, fromNode, toNode):
    return nx.shortest_path(G, fromNode, toNode)

def detectComs(G, G_preprocessed, seed, sample_size, samplingMethod):
    sample = neighborSampling(G_preprocessed, seed, sample_size, samplingMethod)
    centroids = findCentroids(G, sample)
         
    coms = []
    for centroid in centroids:
        strongSeedset = getShortestPathNodes(G, seed, centroid)
#         print("Strong seedset", strongSeedset)
        com = ppr_cd(G, strongSeedset, seed, centroids)
#         com = hk_cd(G1, strongSeedset, seed, centroids)
        coms.append(list(set(com)))
    return coms, []


# sample usage:

# G = an undirected graph: read it using any method provided in loadNetwork.py
# G_preprocessed = preprocessG(G)
# seed = some_seed_from_G
# coms, _ = detectComs(G, G_preprocessed, seed, 10, "minDegree")#
