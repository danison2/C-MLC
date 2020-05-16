'''
Created on Dec 16, 2019

@author: DanyK
'''
import demon as d
import networkx as nx

def l_demon(Gs):
    dm = d.Demon(graph=Gs, epsilon=0.25, min_community_size=3)
    allDetectedCommunities = dm.execute()
    return allDetectedCommunities

def detectAll(G):
    dm = d.Demon(graph=G, epsilon=0.25, min_community_size=3)
    allDetectedCommunities = dm.execute()
    return allDetectedCommunities

def getSeedCommunities(allDetectedCommunities, seed):
    detectedCommunities = []
    for i in range(len(allDetectedCommunities)):
        if seed in allDetectedCommunities[i]:
            detectedCommunities.append(sorted(list(allDetectedCommunities[i])))
    return detectedCommunities