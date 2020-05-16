'''
Created on 29 Dec 2017

@author: DanyK
'''

def getMembershipH(H):
    import numpy as np
    HT = H.T
    vector = np.sum(HT, axis=1)
#     print(vector)
    M = H / vector.T
    return M.T

def getMembershipW(W):
    vector = W.sum(axis=1)
    M = W / vector
    return M

def getMembership(H):
    import numpy as np
    HT = np.transpose(H)
    M = HT / HT.sum(axis=1)[:, None] # rows are node probabilities, columns are communities
    return M
