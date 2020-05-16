'''
Created on 5 Apr 2018

@author: DanyK
'''

#import networkx as nx
from getMembership import getMembershipH
from NMFdecomposition import NMFModel
import numpy as np
import matplotlib.pyplot as plt
import pickle
from operator import itemgetter
import networkx as nx
import sys
np.set_printoptions(threshold=sys.maxsize)

# collabPath = "/content/drive/My Drive/Community/"
collabPath = "../"


def plotSparsenesses(sparsenesses, highSparsenesses):
    n = len(highSparsenesses)
    plt.plot(np.arange(2,n+2), sparsenesses, label="sparseness")
    plt.plot(np.arange(2,n+2), highSparsenesses, label="max sparseness")
    plt.title("Network", fontsize=10)
    plt.xlabel("# Communities", fontsize=10)
#     plt.xticks(np.arange(2,n+2, step=4))
#     plt.yticks(np.arange(0, 1.2, step=0.2))
    plt.ylabel("Sparseness/Max Sparseness", fontsize=10)
    plt.tight_layout()
    plt.legend()
    plt.show()
    
def estimateClusters(A, P):
    from operator import itemgetter 
    from models.SparseNMF import SNMF
    highSparsenesses = []
    sparsenessHs = []
    (_, n) = A.shape
    tempS = 0
    tempK = 2
#     P = 31
    errors = []
    ks = []
    # estimate the number of communities starting from 1
    counter = {}
    for k in range(2,P):
#         print("\n[INFO] Iteration {} of {}...".format(k, P))
        _, _, error, _, sparsenessH = SNMF(A, k)
#         print("k:{} \t sparseness: {}".format(k, sparsenessH))
        
        if sparsenessH > tempS:
            tempK = k
            tempS = sparsenessH
#         print("K: ", tempK)
        counts = counter.get(tempK, 0)
        counts += 1
        counter.update({tempK:counts})
#         if counts == 10:
#             break
#         print("S: ", tempS)
#         print("S2: ", sparsenessH)
        highSparsenesses.append(tempS)
        sparsenessHs.append(sparsenessH)
#         errors.append(error)
#         ks.append(k)
#     lis = list(zip(clusters, sparsenessHs))
#     k, s = max(lis,key=itemgetter(1)) # get k with max sparseness in a list of tuples (k, sparseness)
#     k2, s2 = min(lis,key=itemgetter(1)) # get k with max sparseness in a list of tuples (k, sparseness)
#     plotSparsenesses(ks, errors)
#     plotSparsenesses(sparsenessHs, highSparsenesses)
    return tempK, tempS, highSparsenesses, sparsenessHs
    
def estimateClusters2(A, P):
    from models.SparseNMF import SNMF
    n, m =  A.shape
    h_s = 0.8
    counter = 1
    final_k = 1
    final_H = np.ones((1,n))
    for k in range(2, P):
#         print("step: ", k)
        _, H, error, _, avg_s = SNMF(A, k) 
#         print("avg_s", avg_s)
        if avg_s <= h_s or avg_s <= 0.5:
            counter += 1
        else:
            final_k = k
            final_H = H
            h_s = avg_s  
            counter = 1
        if counter == 10:
            break
    return final_k, final_H, h_s
       

def getNumberBySparseness(subG, maxIter):
    A = nx.adjacency_matrix(subG).todense()
    #execture NMF more than once to estimate k
    c = {}
    ks = []
    spas = []
#     logged = collabPath + "data/plots/AMN2.pickle"
#     f = open(logged, "wb")

    data = {}
    for i in range(0,1):
# #         print("[INFO] iteration {} ...".format(str(i)))
#         k, sparseness, highSparsenesses, sparsenesses = estimateClusters(A, maxIter)
#         data.update({"s":sparsenesses})
#         data.update({"hs":highSparsenesses})
#         f.write(pickle.dumps(data))
#         f.close()
# #         plotSparsenesses(highSparsenesses, sparsenesses)
# #         print("K: {}, Sparseness: {}".format(k, sparseness))
# #         if sparseness < alpha:
# #             k = k-1
#         temp = c.get(k, [])
#         temp.append(sparseness)
#         c[k] = temp
#         ks.append(k)
#         spas.append(np.round(sparseness,2))
#     final_k2 = [k for k, s in zip(ks, spas) if s is max(spas)]
        final_k, final_H, spas = estimateClusters2(A, maxIter)

#     print(final_k)
#     print(final_H.shape)
#     print(spas)
#     if(spas[0] == 0):
#         final_k = 1
#     else:
#         final_k, _ = max([(n,len(s)) for n,s in c.items()], key=lambda x:x[1])
    return final_k

def getClustersMine(G):
    import networkx as nx
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


def detectCommunitiesSNMF(subgraph, clusters):
    A = nx.adjacency_matrix(subgraph).todense()
    filtered_nodes = sorted(subgraph.nodes())
    from models.SparseNMF import SNMF
    if clusters > 1:
        #     clusters = getNumberByMine(A)
        #     print("Number of clusters New: ", clusters1)
    #         print("Number of clusters: ", clusters)
        #     clusters = 2
        limit = 1 / clusters
#         limit = 0.75
#         limit = 0.2
        _, H, _, _, _ = SNMF(A, clusters)
        M = getMembershipH(H)
    #         plotMatrix(M.T, filtered_nodes)
            
#         print(np.round(M,2))
    #         print(filtered_nodes)
    #         print(np.transpose(np.round(M,2)))
        # ##########################################################################
    #         comMemberships = np.argmax(M, axis=1)
    #     #     print(comMemberships)
    #         detectedCommunities = {}
    #         for com_idx, node in zip(comMemberships, filtered_nodes):
    #             idx = np.array(com_idx)[0][0]
    #     #         print(idx)
    #             temp = detectedCommunities.get(idx, [])
    #             temp.append(node)
    #             detectedCommunities.update({idx:temp})
    #         detectedCommunities = list(detectedCommunities.values())
    #     else:
    #         detectedCommunities = [filtered_nodes]
    #     ###########################################################################
    #     print(detectedCommunities)
            
        M[M >= limit] = 1
        #     # replace less than .85 with 0 to activate the community membership
        M[M < 1] = 0
        #     
        #     # map indices of hyperedges in M to the corresponding nodes
        numbering = list(range(len(filtered_nodes)))  # create indices
    #         f = open('communities/' + filename, "w+")
    #         f.close()
        detectedCommunities = []
        for k in range(0, clusters):
            # extract indices of nodes with community membership=1
            c_indices = [j for i, j in zip(M[:, k], numbering) if i == 1]
            if len(c_indices) == 0:
                continue
            if len(c_indices) == 1:
                c_nodes = [filtered_nodes[c_indices[0]]]
            else:
                # convert tuple to list
                c_nodes = list(itemgetter(*c_indices)(filtered_nodes))
            # Do 1 step BSF to include the neighbours of the centroids
            temp = []
            for node in c_nodes:
    #                 neighbours = BFS_steps(subgraph, node, 1)
#                 shortest_path = nx.shortest_path(subgraph, node, seed)
                if node not in temp:
                    temp.append(node)
    #                 temp.extend([int(v) for v in neighbours if v not in temp])
#                 temp.extend([int(v) for v in shortest_path if v not in temp])
            temp = sorted(temp)
    #         print("Before Finetuning: ", temp)
    #         temp = fineTuneCom(subgraph, seed, temp, 0.50)
            detectedCommunities.append(temp)
            # write them to file
    #             nber = countLines('../data/ground_truth/' + filename)
    #             com = "C" + str(nber + 1) + ": "
            # new_communities.append(current_list)
    #             with open('communities/' + filename, "a") as myfile:
    #                 myfile.write(str({com: temp}) + "\n")  # write communities to file
    else:
        detectedCommunities = [filtered_nodes]
    return detectedCommunities


def getCommunities(G, clusters, alpha):
    from models.BNMF import decomposition
    from getMembership import getMembership
    from centroid_detection import BFS
    
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
            neighbours = BFS(G, node, 1)
            if node not in temp:
                temp.append(node)
            temp.extend([int(v) for v in neighbours if v not in temp])
        temp = sorted(temp)
        detectedCommunities.append(temp)
    return detectedCommunities
