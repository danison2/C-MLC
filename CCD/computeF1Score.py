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

def get_node_membership(communities):
    """
    Get the community membership for each node given a list of communities.
    :param communities: list of list of int
        List of communities.
    :return: membership: dict (defaultdict) of set of int
        Dictionary such that, for each node,
        membership[node] is the set of community ids to which the node belongs.
    """
    membership = defaultdict(set)
    for i, community in enumerate(communities):
        for node in community:
            membership[node].add(i)
    return membership

def compute_f1_scores(communities, groundtruth):
    """
    Compute the maximum F1-Score for each community of a list of communities
    with respect to a collection of ground-truth communities.

    :param communities: list of list/set of int
        List of communities.
    :param groundtruth: list of list/set of int
        List of ground-truth communities with respect to which we compute the maximum F1-Score.
    :return: f1_scores: list of float
        List of F1-Scores corresponding to the score of each community given in input.
    """
    groundtruth_inv = get_node_membership(groundtruth)
    communities = [set(community) for community in communities]
    groundtruth = [set(community) for community in groundtruth]
    f1_scores = list()
    f2_scores = list()
    for community in communities:
        groundtruth_indices = reduce(lambda indices, node: groundtruth_inv[node] | indices, community, set())
        max_precision = 0.
        max_recall = 0.
        max_f1 = 0.
        max_f2 = 0.
        for i in groundtruth_indices:
            precision = float(len(community & groundtruth[i])) / float(len(community))
            recall = float(len(community & groundtruth[i])) / float(len(groundtruth[i]))
            f1 = 2 * precision * recall / (precision + recall)
            f2 = (1 + 4) * (precision * recall) / ((4 * precision) + recall) #4 is square(2) as beta = 2
            max_precision = max(precision, max_precision)
            max_recall = max(recall, max_recall)
            max_f1 = max(f1, max_f1)
            max_f2 = max(f2, max_f2)
        f1_scores.append(max_f1)
        f2_scores.append(max_f2)
    return np.average(f1_scores), np.average(f2_scores)


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
# def getModularity(G, communities):
#     #convert array of community to a dictionary of nodes as keys and com nber as value
#     partitions ={}
#     i=0
#     for com in communities:
#         for node in com:
#             partitions.update({node:i})
#         i+=1
#     flat_list = [item for sublist in communities for item in sublist]
#     subG = nx.subgraph(G,flat_list)
#     mod = community.modularity(partitions,subG)
#     return mod

# Variation of information (VI)
#
# Meila, M. (2007). Comparing clusterings-an information
#   based distance. Journal of Multivariate Analysis, 98,
#   873-895. doi:10.1016/j.jmva.2006.11.013
#
# https://en.wikipedia.org/wiki/Variation_of_information



def VI(X, Y): #variation of information
    from math import log, log2
    n1 = float(sum([len(x) for x in X]))
    n2 = float(sum([len(y) for y in Y]))
    if n1 > n2:
        n = n1
    else: n = n2
    
    sigma = 0.0
    for x in X:
        p = len(x) / n
        for y in Y:
            q = len(y) / n
            r = len(set(x) & set(y)) / n
            if r > 0.0:
                sigma += r * (log(r / p, 2) + log(r / q, 2))
    score = abs(sigma)
#     print("n: ", n)
    upperBound = log2(n)
    lowerBound = 0
#     normalized_score = normalizeVI(score, upperBound, lowerBound)
    return score


def normalizeVI(score, upperBound, lowerBound):
    return (score - lowerBound) / (upperBound - lowerBound)

def NMI(X, Y):
    from sklearn.metrics.cluster import normalized_mutual_info_score
    nmi = normalized_mutual_info_score(X, Y)
    return nmi

def main():
    # Identical partitions
    X1 = [ [1,2,3,4,5], [6,7,8,9,10] ]
    Y1 = [ [1,2,3, 4, 5], [6,7,8,9,10] ]
    
    # from math import log10, log, log2
    # print("Upper: ", log(len([i for com in X1 for i in com])))
    # print("Upper2: ", log2(len([i for com in X1 for i in com])))
    # print("Upper10: ", log10(len([i for com in X1 for i in com])))
    
    print(VI(X1, Y1))
    # VI = 0
    
    # Similar partitions
    X2 = [ [1,2,3,4], [5,6,7,8,9,10] ]
    Y2 = [ [1,2,3,4,5,6], [7,8,9,10] ]
#     print(VI(X2, Y2))
    # VI = 1.102
    
    # Dissimilar partitions
    X3 = [ [1,2], [3,4,5], [6,7,8], [9,10] ]
    Y3 = [ [10,2,3], [4,5,6,7], [8,9,1] ]
    print(VI(X3, Y3))
    # VI = 2.302
    
    # Totally different partitions
    X4 = [ [1,2,3,4,5,6,7,8,9,10] ]
    Y4 = [ [1], [2], [3], [4], [5], [6], [7], [8], [9], [10] ]
    print(VI(X4, Y4))
#     VI = 3.322 (maximum VI is log(N) = log(10) = 3.322)
# main()
