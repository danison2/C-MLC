'''
Created on 4 Apr 2018

@author: DanyK
'''
from centroid_detection import detectComs, preprocessG
from computeF1Score import computeF1Score, getRecall, getConductance
import numpy as np
from loadNetwork import readGraph, readEdgeList
import networkx as nx
import os
import time
from CCD.ppr_sampling import ppr_sampling
from hk_cd import hk_cd, hk_sampling

path_folder = "../data/graphs/"

# graphFiles = []
# graphFiles.append('graphA.txt')
# graphFiles.append('graphD.txt')
# 
# # #    
# allCommunitiesFile = {}
# allCommunitiesFile.update({'graphA.txt': 'newComA.txt'})
# allCommunitiesFile.update({'graphD.txt': 'newComD.txt'})
# 
# 
# seedsFiles = {}
# seedsFiles.update({'graphA.txt': [
#                      'seedsA1.txt', 'seedsA2.txt', 'seedsA3.txt']})
# seedsFiles.update({'graphD.txt': [
#                      'seedsD1.txt', 'seedsD2.txt','seedsD3.txt']})

# graphFiles = []
# graphFiles.append("LiveJ.txt")
# graphFiles.append("Orkut.txt")
#  
#     
# allCommunitiesFile = {}
# allCommunitiesFile.update({'LiveJ.txt': 'newComLJ.txt'})
# allCommunitiesFile.update({'Orkut.txt': 'newComO.txt'})
#     
# seedsFiles = {}
# seedsFiles.update({'LiveJ.txt': [
#                     'seedsLJ1.txt', 'seedsLJ2.txt', 'seedsLJ3.txt']})
# seedsFiles.update({'Orkut.txt': [
#                     'seedsO1.txt', 'seedsO2.txt', 'seedsO3.txt']})

# graphFiles = []
# graphFiles.append('network01_10_30-57.dat')
# graphFiles.append('network01_100_200-6.dat')
# graphFiles.append('network02_3_20-112.dat')
# graphFiles.append('network02_20_200-9.dat')
# graphFiles.append('network12_20_200-13.dat')
# graphFiles.append('network13_20_200-10.dat')
# graphFiles.append('network14_20_200-11.dat')
# graphFiles.append('network15_20_200-18.dat')
# # # # # # 
# allCommunitiesFile = {}
# allCommunitiesFile.update({'network01_10_30-57.dat': 'lfr01_10_30-57.txt'})
# allCommunitiesFile.update({'network01_100_200-6.dat': 'lfr01_100_200-6.txt'})
# allCommunitiesFile.update({'network02_3_20-112.dat': 'lfr02_3_20-112.txt'})
# allCommunitiesFile.update({'network02_20_200-9.dat': 'lfr02_20_200-9.txt'})
# allCommunitiesFile.update({'network12_20_200-13.dat': 'lfr12_20_200-13.txt'})
# allCommunitiesFile.update({'network13_20_200-10.dat': 'lfr13_20_200-10.txt'})
# allCommunitiesFile.update({'network14_20_200-11.dat': 'lfr14_20_200-11.txt'})
# allCommunitiesFile.update({'network15_20_200-18.dat': 'lfr15_20_200-18.txt'})
# # # # #  
# seedsFiles = {}
# seedsFiles.update({'network01_10_30-57.dat': [
#                   'seeds01_10_30-57.txt']})
# seedsFiles.update({'network01_100_200-6.dat': [
#                   'seeds01_100_200-6.txt']})
# seedsFiles.update({'network02_3_20-112.dat': [
#                   'seeds02_3_20-112.txt']})
# seedsFiles.update({'network02_20_200-9.dat': [
#                   'seeds02_20_200-9.txt']})
# seedsFiles.update({'network12_20_200-13.dat': [
#                   'seeds12_20_200-13.txt']})
# seedsFiles.update({'network13_20_200-10.dat': [
#                   'seeds13_20_200-10.txt']})
# seedsFiles.update({'network14_20_200-11.dat': [
#                   'seeds14_20_200-11.txt']})
# seedsFiles.update({'network15_20_200-18.dat': [
#                   'seeds15_20_200-18.txt']})

def getCommunities(filename, seed):
#     print("file: ", filename)
    coms = []
    with open('../data/ground_truth/' + filename) as f:
        lines = list(filter(None, f.readlines()))  # remove invalid strings
        for line in lines:
            com = []
            for word in line.split():
                # set your node separator as an argument of split
                # IF it is not a blank space or tab
                com.append(int(word))
            if seed in com:
                coms.append(sorted(com))
    return coms

GN1 = nx.random_partition_graph([32,32,32, 32],.48,.01) #intdegree = 15
partition1 = GN1.graph['partition']
GN2 = nx.random_partition_graph([32,32,32, 32],.41,.03) #intdegree = 13
partition2 = GN2.graph['partition']
GN3 = nx.random_partition_graph([32,32,32, 32],.35,.05) #intdegree = 11
partition3 = GN3.graph['partition']
GN4 = nx.random_partition_graph([32,32,32, 32],.29,.07) #intdegree = 9
partition4 = GN4.graph['partition']

graphs = [GN1, GN2, GN3, GN4]
graphNames = ["G1", "G2", "G3", "G4"]
partitions = [partition1, partition2, partition3, partition4]
# for graphFile, seedsFile in zip(graphFile, seedsFile):
for G1, allCommunities, graph in zip(graphs, partitions, graphNames):
     
    print(graph)
    seeds = np.random.choice(G1.nodes, 100)
 
    Precisions = []
    Recalls = []
    F1s = []
    VIs = []

    condsSample = []
    condsFlatGT = []
    condsDetected = []
    condsGT = []

    Sum_Prec = 0
    Sum_Recall = 0
    Sum_F1 = 0
    Sum_VI = 0

    Sum_CondS = 0
    Sum_CondF = 0
    Sum_CondD = 0
    Sum_CondG = 0
    
    allDiffTime = 0 #time used in seconds to run the algorithm on all seeds

    i = 0
#     minF1 = 0
#     maxF1 = 0
#     goodF1s = 0
#     badF1s = 0
    dirs  = ["VI", "F1", "precision", "recall", "timeUsed", "condDetected", "condGT", "Seeds_used"]
    for dir1 in dirs:
        directory = os.path.dirname("../data/CCD/" + graph + "/" + dir1)
        if not os.path.exists(directory):
            os.makedirs(directory)

    f = open('../data/CCD/' + graph + '/VI', "w+")
    f.close()
    f = open('../data/CCD/' + graph + '/F1', "w+")
    f.close()
    f = open('../data/CCD/' + graph + '/precision', "w+")
    f.close()
    f = open('../data/CCD/' + graph + '/recall', "w+")
    f.close()

    f = open('../data/CCD/' + graph + '/condDetected', "w+")
    f.close()
    f = open('../data/CCD/' + graph + '/condGT', "w+")
    f.close()
    seeds_used = []
    for seed in seeds:  # loop over the seeds
        G = preprocessG(G1)
        
        groundTruth = [com for com in allCommunities if seed in com]

        flatGT = [int(node) for com in groundTruth for node in com]
        flatGT = sorted(set(flatGT))
        lenGT = len(flatGT)
        if lenGT <= 100:  # work with communities not bigger than 100 nodes
#                 print("used: ", seed)
            seeds_used.append(seed) #store used seeds
#                 sample_size = 10
            startTime = time.time() #time.perf_counter
#                 coms          = LDEMON(G1, seed)
#             coms, _ = MLC(G1, seed)
#             coms = MLOSP(G1, seed)
#             coms = SMLC(G1, seed)
            coms = run_multicom(G1, seed)
#             coms, _ = detectComs(G1, G, seed, 10, "minDegree")#                 
            detectedCommunities = [com for com in coms if seed in com]
#                 print("Detected coms: ", detectedCommunities)
#                 detectedCommunities = getSeedCommunities(allDetected, seed)
            endTime = time.time()
            diffTime = endTime - startTime
            allDiffTime += diffTime
            #filter those containing the seed
#                 print("Detected: ", detectedCommunities)
          
            if len(detectedCommunities) < 1:
                f1, vi = (0,0)
                prec, recall = (0,0)
                condDetected = 1
                continue
            else:
                f1, vi = compute_f1_scores(detectedCommunities, groundTruth)
                             
#                     flatDetected = list(set([int(v) for com in detectedCommunities for v in com]))
#                     f1, vi = computeF1Score([flatGT], [flatDetected])
#                     f1, vi = computeF1Score(detectedCommunities, groundTruth)
                
#                     print("Computing conductances...")
                condDetected = getConductance(G1, detectedCommunities)
#                    
            
            condGT = getConductance(G1, groundTruth)
            
            prec, recall = getRecall(flatGT, list(set(G1.nodes)))
#                 print("Prec: ", prec)
#                 print("Recall: ", recall)
#                 print("F1: ", f1)
            VIs.append(vi)
            Sum_VI += vi

            F1s.append(f1)
            Sum_F1 += f1

            Precisions.append(prec)
            Sum_Prec += prec

            Recalls.append(recall)
            Sum_Recall += recall


            condsDetected.append(condDetected)
            Sum_CondD += condDetected
# 
            condsGT.append(condGT)
            Sum_CondG += condGT

            i += 1
            Avg_VI = np.round(Sum_VI / i, 2)
            Avg_F1 = np.round(Sum_F1 / i, 2)
            Avg_Prec = np.round(Sum_Prec / i, 2)
            Avg_Recall = np.round(Sum_Recall / i, 2)
# 

            Avg_CondD = np.round(Sum_CondD / i, 2)
            Avg_CondG = np.round(Sum_CondG / i, 2)

            #print("Avg com: ",avg_size)
            with open('../data/CCD/' + graph +'/VI', "a") as myfile:
                # write communities to file
                myfile.write(str(vi) + "\n")
            with open('../data/CCD/' + graph + '/F1', "a") as myfile:
                # write communities to file
                myfile.write(str(f1) + "\n")
                 
            with open('../data/CCD/' + graph + '/Seeds_used', "a") as myfile:
                # write communities to file
                myfile.write(str(seed) + "\n")

            with open('../data/CCD/' + graph + '/precision', "a") as myfile:
                # write communities to file
                myfile.write(str(prec) + "\n")
#                 #print("Avg com: ",avg_size)
            with open('../data/CCD/' + graph + '/recall', "a") as myfile:
                # write communities to file
                myfile.write(str(recall) + "\n")
# 
#                     
            with open('../data/CCD/' + graph + '/timeUsed', "a") as myfile:
                # write communities to file
                myfile.write(str(diffTime) + "\n")
# 
            with open('../data/CCD/' + graph + '/condDetected', "a") as myfile:
                # write communities to file
                myfile.write(str(condDetected) + "\n")
#                 #print("Avg com: ",avg_size)
            with open('../data/CCD/' + graph + '/condGT', "a") as myfile:
                # write communities to file
                myfile.write(str(condGT) + "\n")
            if i == 100:
                break

    with open('../data/CCD/' + graph + '/VI', "a") as myfile:
        myfile.write("Average VI: " + str(Avg_VI) + "\n")
    
    with open('../data/CCD/' + graph + '/F1', "a") as myfile:
        myfile.write("Average F1: " + str(Avg_F1) + "\n")
        
    with open('../data/CCD/' + graph + '/timeUsed', "a") as myfile:
               
        myfile.write("Entire graph time: " + str(allDiffTime) + "\n")
    
    with open('../data/CCD/' + graph + '/precision', "a") as myfile:
        myfile.write("Average Precision: " + str(Avg_Prec) + "\n")
    
    with open('../data/CCD/' + graph + '/recall', "a") as myfile:
        myfile.write("Average Recall: " + str(Avg_Recall) + "\n")
    # 
    # 
    with open('../data/CCD/' + graph + '/condDetected', "a") as myfile:
        myfile.write("Average Cond Detected: " + str(Avg_CondD) + "\n")
    #  
    with open('../data/CCD/' + graph + '/condGT', "a") as myfile:
        myfile.write("Average Cond GT: " + str(Avg_CondG) + "\n")


