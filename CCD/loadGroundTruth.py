'''
Created on 19 Sep 2018

@author: DanyK
'''
def readGroundTruth(communityFile):
    coms = []
    with open(communityFile) as f:
        lines = list(filter(None, f.readlines()))  # remove invalid strings
        for line in lines:
            com = []
            for word in line.split():
                # set your node separator as an argument of split
                #IF it is not a blank space or tab
                if word is not None:
                    com.append(int(word))
                    #if seed in com:
            coms.append(com)
    return coms

def readGroundTruthSeed(communityFile, seed):
    coms = []
    with open(communityFile) as f:
        lines = list(filter(None, f.readlines()))  # remove invalid strings
        for line in lines:
            com = []
            for word in line.split():
                # set your node separator as an argument of split
                #IF it is not a blank space or tab
                if word is not None:
                    com.append(int(word))
            if seed in com:
                coms.append(com)
    return coms

def readGroundTruthSeed(communityFile, seed):
    coms = []
    with open(communityFile) as f:
        lines = list(filter(None, f.readlines()))  # remove invalid strings
        for line in lines:
            com = []
            for word in line.split():
                # set your node separator as an argument of split
                #IF it is not a blank space or tab
                if word is not None:
                    com.append(int(word))
            if seed in com:
                coms.append(com)
    return coms

# GT = getGroundTruth("com-simple1.txt", 4)
# print(GT)