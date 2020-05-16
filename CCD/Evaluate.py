'''
Created on 4 Apr 2018

@author: DanyK
'''
from centroid_detection import detectComs, preprocessG
from computeF1Score import computeF1Score, getRecall, getConductance
import numpy as np
from loadNetwork import readGraph
import networkx as nx
import os
import time

path_folder = "../data/graphs/"

# graphFile = ['graph_ncp.txt']
# allCommunities = 'comFootballTSEinputNumbered.txt'
# seedsFile = ['seedsKarate.txt']

graphFiles = []
# graphFiles.append("karate.txt")
# graphFiles.append("simple1.txt")
# graphFiles.append("edge_dolphins_numbered.txt")
# graphFiles.append('footballTSEinput_numbered.txt')
graphFiles.append('graphA.txt')
graphFiles.append('graphD.txt')
#graphFiles.append("LiveJ.txt")
#graphFiles.append("Orkut.txt")

# #  
# #    
allCommunitiesFile = {}
# allCommunitiesFile.update({'karate.txt': 'com-karate.txt'})
# allCommunitiesFile.update({'simple1.txt': 'com-simple1.txt'})
# allCommunitiesFile.update({'edge_dolphins_numbered.txt': 'comDolphinsNumbered.txt'})
# allCommunitiesFile.update({'footballTSEinput_numbered.txt': 'comFootballTSEinputNumbered.txt'})
allCommunitiesFile.update({'graphA.txt': 'newComA.txt'})
allCommunitiesFile.update({'graphD.txt': 'newComD.txt'})
# allCommunitiesFile.update({'LiveJ.txt': 'newComLJ.txt'})
# allCommunitiesFile.update({'Orkut.txt': 'newComO.txt'})
# #   
seedsFiles = {}
# seedsFiles.update({'karate.txt': [
#                      'seedsKarate.txt']})
# seedsFiles.update({'simple1.txt': [
#                      'seedsSimple1.txt']})
# seedsFiles.update({'edge_dolphins_numbered.txt': [
#                      'seedsDolphins.txt']})
# seedsFiles.update({'footballTSEinput_numbered.txt': [
#                      'seedsFootball.txt']})
seedsFiles.update({'graphA.txt': [
                     'seedsA1.txt', 'seedsA2.txt', 'seedsA3.txt']})
seedsFiles.update({'graphD.txt': [
                     'seedsD1.txt', 'seedsD2.txt','seedsD3.txt']})
# seedsFiles.update({'LiveJ.txt': [
#                     'seedsLJ1.txt', 'seedsLJ2.txt', 'seedsLJ3.txt']})
# seedsFiles.update({'Orkut.txt': [
#                     'seedsO1.txt', 'seedsO2.txt', 'seedsO3.txt']})

# graphFiles = []
# graphFiles.append('network03.dat')
# graphFiles.append('network05.dat')
# graphFiles.append('network13.dat')
# graphFiles.append('network15.dat')
# graphFiles.append('network23.dat')
# graphFiles.append('network25.dat')
# graphFiles.append('network33.dat')
# graphFiles.append('network35.dat')
# 
# allCommunitiesFile = {}
# allCommunitiesFile.update({'network03.dat': 'lfr03.txt'})
# allCommunitiesFile.update({'network05.dat': 'lfr05.txt'})
# allCommunitiesFile.update({'network13.dat': 'lfr13.txt'})
# allCommunitiesFile.update({'network15.dat': 'lfr15.txt'})
# allCommunitiesFile.update({'network23.dat': 'lfr23.txt'})
# allCommunitiesFile.update({'network25.dat': 'lfr25.txt'})
# allCommunitiesFile.update({'network33.dat': 'lfr33.txt'})
# allCommunitiesFile.update({'network35.dat': 'lfr35.txt'})
# 
# seedsFiles = {}
# seedsFiles.update({'network03.dat': [
#                   'seeds03_1.txt', 'seeds03_3.txt']})
# seedsFiles.update({'network05.dat': [
#                   'seeds05_1.txt', 'seeds05_5.txt']})
# seedsFiles.update({'network13.dat': [
#                   'seeds13_1.txt', 'seeds13_3.txt']})
# seedsFiles.update({'network15.dat': [
#                   'seeds15_1.txt', 'seeds15_5.txt']})
# seedsFiles.update({'network23.dat': [
#                   'seeds23_1.txt', 'seeds23_3.txt']})
# seedsFiles.update({'network25.dat': [
#                   'seeds25_1.txt', 'seeds25_5.txt']})
# seedsFiles.update({'network33.dat': [
#                   'seeds33_1.txt', 'seeds33_3.txt']})
# seedsFiles.update({'network35.dat': [
#                   'seeds35_1.txt', 'seeds35_5.txt']})

# # graphFiles = []
# graphFiles.append('network01_10_30-57.dat')
# graphFiles.append('network01_100_200-6.dat')
# graphFiles.append('network02_3_20-112.dat')
# graphFiles.append('network02_20_200-9.dat')
# graphFiles.append('network12_20_200-13.dat')
# graphFiles.append('network13_20_200-10.dat')
# graphFiles.append('network14_20_200-11.dat')
# graphFiles.append('network15_20_200-18.dat')
# # # # # 
# # allCommunitiesFile = {}
# allCommunitiesFile.update({'network01_10_30-57.dat': 'lfr01_10_30-57.txt'})
# allCommunitiesFile.update({'network01_100_200-6.dat': 'lfr01_100_200-6.txt'})
# allCommunitiesFile.update({'network02_3_20-112.dat': 'lfr02_3_20-112.txt'})
# allCommunitiesFile.update({'network02_20_200-9.dat': 'lfr02_20_200-9.txt'})
# allCommunitiesFile.update({'network12_20_200-13.dat': 'lfr12_20_200-13.txt'})
# allCommunitiesFile.update({'network13_20_200-10.dat': 'lfr13_20_200-10.txt'})
# allCommunitiesFile.update({'network14_20_200-11.dat': 'lfr14_20_200-11.txt'})
# allCommunitiesFile.update({'network15_20_200-18.dat': 'lfr15_20_200-18.txt'})
# # # #  
# # seedsFiles = {}
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

# graphFiles = ['footballTSEinput_numbered.txt']
# allCommunitiesFile = {}
# allCommunitiesFile.update({'footballTSEinput_numbered.txt':'comFootballTSEinputNumbered.txt'})
# seedsFiles = {}
# seedsFiles.update({'footballTSEinput_numbered.txt':['seedsFootball.txt']})


# graphFile = ['simple1.txt']
# allCommunities = 'com-simple1.txt'
# seedsFile = ['seedsSimple1.txt']

# graphFile = ['karate.txt']
# allCommunities = 'com-karate.txt'
# seedsFile = ['seedsKarate.txt']


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
    #     f = open('F1/' + graphFile, "w+")
    #     f.close()
    if graphFile == "karate.txt":
        G1 = nx.karate_club_graph()
#         adj_matrix = nx.adjacency_matrix(G1)
        
    else:
        if graphFile == "LiveJ.txt" or graphFile == "Orkut.txt":
            G1 = readGraph('../data/graphs/' + graphFile, delm="\t")
#             adj_matrix = load_graph('../data/graphs/' + graphFile, delimiter='\t', comment='#')
        else:
            G1 = readGraph('../data/graphs/' + graphFile, delm="\t")
#             adj_matrix = load_graph('../data/graphs/' + graphFile, delimiter='\t', comment='#')
    
    G = preprocessG(G1)
#     allStartTime = time.perf_counter()
#     allDetected = detectAll(G1)
#     allEndTime = time.perf_counter()
#     allDiffTime = allEndTime - allStartTime

    allCommunities = allCommunitiesFile[graphFile]

    for seedsFile in seedsFiles[graphFile]:

        #     print(nx.info(G))
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

#         f = open('../data/results/VI/' + seedsFile, "w+")
#         f.close()
#         f = open('../data/results/F1/' + seedsFile, "w+")
#         f.close()
#         f = open('../data/results/precision/' + seedsFile, "w+")
#         f.close()
#         f = open('../data/results/recall/' + seedsFile, "w+")
#         f.close()
#         f = open('../data/results/condSample/' + seedsFile, "w+")
#         f.close()
#         f = open('../data/results/condFlatGT/' + seedsFile, "w+")
#         f.close()
#         f = open('../data/results/condDetected/' + seedsFile, "w+")
#         f.close()
#         f = open('../data/results/condGT/' + seedsFile, "w+")
#         f.close()

        
        seeds_used = []
        for line in open('../data/seeds/' + seedsFile, 'r'):  # loop over the seeds
            line = [int(word) for word in line.split()]

            seed = line[0]
#             print("seed: ", seed)
            groundTruth = getCommunities(allCommunities, seed)
#             print("GT", groundTruth)

            flatGT = [int(node) for com in groundTruth for node in com]
            flatGT = sorted(set(flatGT))
            lenGT = len(flatGT)
            if lenGT <= 100:  # work with communities not bigger than 100 nodes
                print("used: ", seed)
                seeds_used.append(seed) #store used seeds
#                 sample_size = 10
                startTime = time.time() #time.perf_counter
#                 coms, sampled = LDEMON(G1, seed)
#                 coms, sampled = MLC(G1, seed)
                coms, sampled = detectComs(G1, G, seed, 10, "minDegree")#                 
                detectedCommunities = [com for com in coms if seed in com]
                print("detected: ", len(detectedCommunities))
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
#                     f1, vi = compute_f1_scores(detectedCommunities, groundTruth)
                                 
                    flatDetected = list(set([int(v) for com in detectedCommunities for v in com]))
                    f1, vi = computeF1Score([flatGT], [flatDetected])
                    
#                     print("Computing conductances...")
#                     condDetected = getConductance(G1, detectedCommunities)
#                    
#                  
#                 condGT = getConductance(G1, groundTruth)
                
                prec, recall = getRecall(flatGT, list(set(sampled)))
               
#                 print("ENDED computing conductance")
                
#                 modGT = getModularity(G, groundTruth)
#                 modDetected = getModularity(G, detectedCommunities)

                VIs.append(vi)
                Sum_VI += vi
 
                F1s.append(f1)
                Sum_F1 += f1
 
                Precisions.append(prec)
                Sum_Prec += prec
 
                Recalls.append(recall)
                Sum_Recall += recall
 
#                 condsSample.append(condSample)
#                 Sum_CondS += condSample
 
#                 condsFlatGT.append(condFlatGT)
#                 Sum_CondF += condFlatGT
 
#                 condsDetected.append(condDetected)
#                 Sum_CondD += condDetected
# 
#                 condsGT.append(condGT)
#                 Sum_CondG += condGT

                i += 1
                Avg_VI = np.round(Sum_VI / i, 2)
                Avg_F1 = np.round(Sum_F1 / i, 2)
                Avg_Prec = np.round(Sum_Prec / i, 2)
                Avg_Recall = np.round(Sum_Recall / i, 2)
# 
#                 Avg_CondS = np.round(Sum_CondS / i, 2)
#                 Avg_CondF = np.round(Sum_CondF / i, 2)
                Avg_CondD = np.round(Sum_CondD / i, 2)
                Avg_CondG = np.round(Sum_CondG / i, 2)

#                 print("Average VI: " + str(Avg_VI) + " at i=" + str(i))
#                 print("Average F1: " + str(Avg_F1) + " at i=" + str(i))
#                 print("Average Precision: " + str(Avg_Prec) + " at i=" + str(i))
#                 print("Average Recall: " + str(Avg_Recall) + " at i=" + str(i))

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
#                 with open('../data/results_SMLC/modDetected/' + seedsFile, "a") as myfile:
#                     # write communities to file
#                     myfile.write(str(modDetected) + "\n")
#                     
                with open('../data/CCD/timeUsed/' + seedsFile, "a") as myfile:
                    # write communities to file
                    myfile.write(str(diffTime) + "\n")

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
