'''
Created on Mar 20, 2020

@author: DanyK
'''
# i = 0;
# seed = 1
# for line in open("../data/graphs/LiveJ.txt", "r"):
#     if("#" not in line):
#         print(line)
#         i+= 1
#         if i== 100:
#             break
graphs = ['LiveJ.txt', 'Orkut.txt']
allCommunitiesFile = {}
allCommunitiesFile.update({'LiveJ.txt': 'newComLJ.txt'})
allCommunitiesFile.update({'Orkut.txt': 'newComO.txt'})

seedsFiles = {}
seedsFiles.update({'LiveJ.txt': [
                    'seedsLJ1.txt', 'seedsLJ2.txt', 'seedsLJ3.txt']})
seedsFiles.update({'Orkut.txt': [
                    'seedsO1.txt', 'seedsO2.txt', 'seedsO3.txt']})

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

def generateSeeds(seedsFile, graphFile):
    allCommunities = allCommunitiesFile[graphFile]
    totalSeeds = 0;
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
                with open('../data/CCD/Seeds_used/' + seedsFile, "a") as myfile:
                    # write communities to file
                    myfile.write(str(seed) + "\n")
                totalSeeds += 1;
            if totalSeeds == 100:
                break
            
def generateEdges(seedsFile, graphFile):
    graphFolder = graphFile.split(".")[0]
    totalSeeds = 0;
    for line in open('../data/seeds/' + seedsFile, 'r'):  # loop over the seeds
            line = [int(word) for word in line.split()]
            seed = line[0]
            processedNodes = []
            seedNeighbours = {0: [seed], 1:[], 2: [], 3:[]}
            print("[INFO] processing seed:", seed)
#             scan the graph three times to get neighbors of the seed at level 3 or up to 100,000 nodes
            for i in range(0,3):
                print("Step: ", i+1)
                for line in open("../data/graphs/" + graphFile, "r"):
                    if("#" not in line):
                        edges = [int(word) for word in line.split("\t")]
                        u = edges[0]
                        v = edges[1]
                        validEdge = False;
                        if i == 0:
                            if u in seedNeighbours[0]:
                                seedNeighbours[1].append(v)
                                validEdge = True;
                            if v in seedNeighbours[0]:
                                seedNeighbours[1].append(u)
                                validEdge = True;
                        elif i == 1:    
                            if u in seedNeighbours[1]:
                                if v not in seedNeighbours[0]: #triangle
                                    seedNeighbours[2].append(v)
                                    validEdge = True;
                            if v in seedNeighbours[1]:
                                if u not in seedNeighbours[0]:
                                    seedNeighbours[2].append(u)
                                    validEdge = True;
                        else: 
                            if u in seedNeighbours[2]:
                                if v not in seedNeighbours[1]:
                                    seedNeighbours[3].append(v)
                                    validEdge = True;
                            if v in seedNeighbours[2]:
                                if u not in seedNeighbours[1]:
                                    seedNeighbours[3].append(u)
                                    validEdge = True;
                        if validEdge == True:
                            with open('../data/CCD/subgraphs/' + graphFolder + "/" + str(seed) + ".txt", "a") as myfile:
                                myfile.write(str(u) + "\t")
                                myfile.write(str(v) + "\t")
                                myfile.write("\n")
                            processedNodes.extend([n for n in [u,v] if n not in processedNodes])
                            
                        if len(processedNodes) > 999:
                            break
#                     print("[INFO] current # of neghbours:", len(processedNodes))
                if len(processedNodes) > 999:
                    break
                
# #             write edges to file
#             for edges in seedEdges:
#                 with open('../data/CCD/subgraphs/' + str(seed) + ".txt", "a") as myfile:
#                     myfile.write(str(edges[0]) + "\t")
#                     myfile.write(str(edges[1]) + "\t")
#                     myfile.write("\n")       
            totalSeeds += 1 
            if totalSeeds == 100:
                break;   
                                  
# generateSeeds("seedsO3.txt", "Orkut.txt")
for graph in graphs:
    for seedFile in seedsFiles[graph]:
        generateEdges(seedFile, graph)
    