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
from ppr_sampling import ppr_sampling
from hk_cd import hk_cd, hk_sampling

path_folder = "../data/graphs/"
graphFiles = []
graphFiles.append("karate.txt")
graphFiles.append("simple1.txt")
graphFiles.append("edge_dolphins_numbered.txt")
graphFiles.append('footballTSEinput_numbered.txt')


# graphFiles.append('graphA.txt')
# graphFiles.append('graphD.txt')
# # # # 
# # # # # #    
# allCommunitiesFile = {}
# allCommunitiesFile.update({'graphA.txt': 'newComA.txt'})
# allCommunitiesFile.update({'graphD.txt': 'newComD.txt'})
# #   
# #   
# seedsFiles = {}
# seedsFiles.update({'graphA.txt': [
#                      'seedsA1.txt', 'seedsA2.txt', 'seedsA3.txt']})
# seedsFiles.update({'graphD.txt': [
#                      'seedsD1.txt', 'seedsD2.txt','seedsD3.txt']})

# graphFiles = []
# graphFiles.append("LiveJ.txt")
# graphFiles.append("Orkut.txt")
# #   
# #      
allCommunitiesFile = {}
# allCommunitiesFile.update({'LiveJ.txt': 'newComLJ.txt'})
# allCommunitiesFile.update({'Orkut.txt': 'newComO.txt'})

allCommunitiesFile.update({'karate.txt': 'com-karate.txt'})
allCommunitiesFile.update({'simple1.txt': 'com-simple1.txt'})
allCommunitiesFile.update({'edge_dolphins_numbered.txt': 'comDolphinsNumbered.txt'})
allCommunitiesFile.update({'footballTSEinput_numbered.txt': 'comFootballTSEinputNumbered.txt'})
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
# # # # # # # 
# allCommunitiesFile = {}
# allCommunitiesFile.update({'network01_10_30-57.dat': 'lfr01_10_30-57.txt'})
# allCommunitiesFile.update({'network01_100_200-6.dat': 'lfr01_100_200-6.txt'})
# allCommunitiesFile.update({'network02_3_20-112.dat': 'lfr02_3_20-112.txt'})
# allCommunitiesFile.update({'network02_20_200-9.dat': 'lfr02_20_200-9.txt'})
# allCommunitiesFile.update({'network12_20_200-13.dat': 'lfr12_20_200-13.txt'})
# allCommunitiesFile.update({'network13_20_200-10.dat': 'lfr13_20_200-10.txt'})
# allCommunitiesFile.update({'network14_20_200-11.dat': 'lfr14_20_200-11.txt'})
# allCommunitiesFile.update({'network15_20_200-18.dat': 'lfr15_20_200-18.txt'})
# # # # # #  
seedsFiles = {}
seedsFiles.update({'karate.txt': [
                     'seedsKarate.txt']})
seedsFiles.update({'simple1.txt': [
                     'seedsSimple1.txt']})
seedsFiles.update({'edge_dolphins_numbered.txt': [
                     'seedsDolphins.txt']})
seedsFiles.update({'footballTSEinput_numbered.txt': [
                     'seedsFootball.txt']})

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


# for graphFile, seedsFile in zip(graphFile, seedsFile):
for graphFile in graphFiles:

    allCommunities = allCommunitiesFile[graphFile]

    for seedsFile in seedsFiles[graphFile]:
#         if(seedsFile == "seedsLJ3.txt"):
#             graphFolder = seedsFile.split(".")[0]
#         else:
#             graphFolder = seedsFile.split(".")[0] + " 1000"
        graphFolder = seedsFile.split(".")[0]
        
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
            directory = os.path.dirname('../data/CCD/' + dir1 + "/" + seedsFile)
            if not os.path.exists(directory):
                os.makedirs(directory)

        f = open('../data/CCD/VI/' + seedsFile, "w+")
        f.close()
        f = open('../data/CCD/F1/' + seedsFile, "w+")
        f.close()
        f = open('../data/CCD/precision/' + seedsFile, "w+")
        f.close()
        f = open('../data/CCD/recall/' + seedsFile, "w+")
        f.close()
#         f = open('../data/CCD/condSample/' + seedsFile, "w+")
        f.close()
#         f = open('../data/CCD/condFlatGT/' + seedsFile, "w+")
        f.close()
        f = open('../data/CCD/condDetected/' + seedsFile, "w+")
        f.close()
        f = open('../data/CCD/condGT/' + seedsFile, "w+")
        f.close()

        
        seeds_used = []
        for line in open('../data/seeds/' + seedsFile, 'r'):  # loop over the seeds
            line = [int(word) for word in line.split()]

            seed = int(line[0])
#             print("[INFO] Seed: ", seed)
            if graphFile == "karate.txt":
                G1 = nx.karate_club_graph()
            else:
                G1 = readGraph('../data/graphs/' + graphFile, delm=" ") #artificial, Amazon and DBLP
            
#             G1 = readEdgeList('../data/CCD/subgraphs/' + graphFolder + "/" + str(seed) + ".txt", delm="\t")
            G = preprocessG(G1)
#             print(nx.info(G1))
            
            groundTruth = getCommunities(allCommunities, seed)
#             print("GT", groundTruth)

            flatGT = [int(node) for com in groundTruth for node in com]
            flatGT = sorted(set(flatGT))
            lenGT = len(flatGT)
            if lenGT <= 500:  # work with communities not bigger than 100 nodes
#                 print("used: ", seed)
                seeds_used.append(seed) #store used seeds
#                 sample_size = 10
                startTime = time.time() #time.perf_counter
#                 coms          = LDEMON(G1, seed)
#                 coms, _ = MLC(G1, seed)
#                 coms = MLOSP(G1, seed)
#                 coms = SMLC(G1, seed)
                coms = run_multicom(G1, seed)
#                 coms, _ = detectComs(G1, G, seed, 10, "minDegree")#                 
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
                with open('../data/CCD/VI/' + seedsFile, "a") as myfile:
                    # write communities to file
                    myfile.write(str(vi) + "\n")
                with open('../data/CCD/F1/' + seedsFile, "a") as myfile:
                    # write communities to file
                    myfile.write(str(f1) + "\n")
                     
                with open('../data/CCD/Seeds_used/' + seedsFile, "a") as myfile:
                    # write communities to file
                    myfile.write(str(seed) + "\n")
 
                with open('../data/CCD/precision/' + seedsFile, "a") as myfile:
                    # write communities to file
                    myfile.write(str(prec) + "\n")
#                 #print("Avg com: ",avg_size)
                with open('../data/CCD/recall/' + seedsFile, "a") as myfile:
                    # write communities to file
                    myfile.write(str(recall) + "\n")
# 
#                     
                with open('../data/CCD/timeUsed/' + seedsFile, "a") as myfile:
                    # write communities to file
                    myfile.write(str(diffTime) + "\n")
# 
                with open('../data/CCD/condDetected/' + seedsFile, "a") as myfile:
                    # write communities to file
                    myfile.write(str(condDetected) + "\n")
#                 #print("Avg com: ",avg_size)
                with open('../data/CCD/condGT/' + seedsFile, "a") as myfile:
                    # write communities to file
                    myfile.write(str(condGT) + "\n")
                if i == 100:
                    break

        with open('../data/CCD/VI/' + seedsFile, "a") as myfile:
            myfile.write("Average VI: " + str(Avg_VI) + "\n")
 
        with open('../data/CCD/F1/' + seedsFile, "a") as myfile:
            myfile.write("Average F1: " + str(Avg_F1) + "\n")
            
        with open('../data/CCD/timeUsed/' + seedsFile, "a") as myfile:
                   
            myfile.write("Entire graph time: " + str(allDiffTime) + "\n")

        with open('../data/CCD/precision/' + seedsFile, "a") as myfile:
            myfile.write("Average Precision: " + str(Avg_Prec) + "\n")
 
        with open('../data/CCD/recall/' + seedsFile, "a") as myfile:
            myfile.write("Average Recall: " + str(Avg_Recall) + "\n")
# 
# 
        with open('../data/CCD/condDetected/' + seedsFile, "a") as myfile:
            myfile.write("Average Cond Detected: " + str(Avg_CondD) + "\n")
#  
        with open('../data/CCD/condGT/' + seedsFile, "a") as myfile:
            myfile.write("Average Cond GT: " + str(Avg_CondG) + "\n")


