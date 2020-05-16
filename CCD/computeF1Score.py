'''
Created on 6 Apr 2018

@author: DanyK
'''
import numpy as np
import difflib
import math
import collections
from collections import Counter
import networkx as nx
from collections import defaultdict
from functools import reduce


def computeF1Score(groundTruth, detectedCommunities):
    #get similar community from the list of communities
    '''
    for com in all_communities:
        similarity=difflib.SequenceMatcher(None, ground_truth, com)
        sim.update({similarity.ratio():com}) # similarity in %
    max_sim,nodes=getMaxSimilarity(sim)'''
    sum_F1 = 0
    sum_F2 = 0
    i = 0
    F1s = []
    F2s = []
    
    for com in [com for com in groundTruth if len(com) >0]:
        sim = {}
        for detected in [com for com in detectedCommunities if len(com) >0]:
            counterA = Counter(detected)
            counterB = Counter(com)
            similarity = counter_cosine_similarity(counterA, counterB)
            sim.update({similarity: detected})
#             print(sim)
    
        if len(sim.items()) > 0:
            od = collections.OrderedDict(sorted(sim.items()))
#             print("sim items: ", sim.items())
            item = od.popitem()  # removes the last item
    #         print("Item: ", item)
            max_sim, nodes = item
#             print("Ground truth: ", com)
#             print("Most similar: ", nodes)
#             print("Similarity: ", max_sim)
            #print("\n")
            #evaluation
            nodes = list(nodes)
            tp = np.intersect1d(com, nodes)  # nodes in common
            # len(nodes)-tp.size #total returned - the number of true returned
            fp = [node for node in nodes if node not in com]
            # len(communities)-tp.size #total correct - correct returned
            fn = [node for node in com if node not in nodes]
            '''print("Community: ",ground_truth)
            print("Returned: ",nodes)
            print("Common nodes: ",tp)
            print("Additional nodes returned: ",fp,len(fp))
            print("Nodes not returned: ",fn, len(fn))
            print("# of community nodes: ", len(ground_truth))
            print("# of common nodes: ",len(tp))'''
            prec = len(tp) / (len(tp) + len(fp))
            rec = len(tp) / (len(tp) + len(fn))
            if len(tp) == 0 and len(fp) == 0 and len(fn) == 0:
                prec = 1
                rec = 1
                F1 = 1
                F2 = 1
            elif prec == 0:
                F1 = 0
                F2 = 0
            else:
                F1 = (2 * prec * rec) / (prec + rec)
                F2 = (1 + 4) * (prec * rec) / ((4 * prec) + rec) #4 is square(2) as beta = 2
            #print("F1: ", F1)
        else: #no similarity
            F1 = 0
            F2 = 0
        sum_F1 += F1
        sum_F2 += F2
            
        F1s.append(F1)
        F2s.append(F2)
        
        i += 1
    if i > 0:
        avg_F1 = sum_F1 / i
        avg_F2 = sum_F2 / i
    else:
        avg_F1 = 0
        avg_F2 = 0
    return avg_F1, avg_F2


def counter_cosine_similarity(c1, c2):
    terms = set(c1).union(c2)
    dotprod = sum(c1.get(k, 0) * c2.get(k, 0) for k in terms)
    magA = math.sqrt(sum(c1.get(k, 0)**2 for k in terms))
    magB = math.sqrt(sum(c2.get(k, 0)**2 for k in terms))
    return dotprod / (magA * magB)


def getMaxSimilarity(similarity_list):
    max_sim = next(iter(similarity_list))
    for item in similarity_list.items():  # iteritems over dictionary
        sim, nodes = item
        if sim > max_sim:
            max_sim = sim
            nodes = nodes
    return max_sim, nodes


def getRecall(com, nodes):
    tp = np.intersect1d(com, nodes)  # nodes in common
    # len(nodes)-tp.size #total returned - the number of true returned
    fp = [node for node in nodes if node not in com]
    # len(communities)-tp.size #total correct - correct returned
    fn = [node for node in com if node not in nodes]
    prec = len(tp) / len(nodes)
    rec = len(tp) / len(com)
    if len(nodes) == 0:
        prec = 0
        rec = 0
    return prec, rec

def getConductance(G,communities):
    sum_cond = 0
#     avg_len = 0
    conductances = {}
    for L in communities:
#         print(L)
#         avg_len += len(L)
        if len(L) == len(G.nodes):
            cond = 1
        else:
            cond = nx.conductance(G, L)
        conductances.update({str(L):cond})
        sum_cond += cond
    avg_cond = sum_cond / len(communities)
#     avg_len = avg_len / len(communities)
    return avg_cond

