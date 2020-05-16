'''
Created on 29 Dec 2017

@author: DanyK
'''
from sklearn.decomposition import NMF


def NMFModel(A, nb):
#     model = NMF(n_components=nb,  init = 'nndsvd', random_state=1,alpha=.1, l1_ratio=.5,solver = 'mu', beta_loss ='kullback-leibler')   
    model = NMF(n_components=nb, init = 'random', random_state=0, solver = 'mu', beta_loss ='frobenius', max_iter=200) 
#     model = NMF(n_components=nb, init = 'nndsvd', random_state=1,alpha=.1, l1_ratio=.5)
    W = model.fit_transform(A);
    H = model.components_;
    return W, H

def decomposition2(A):
#     model = NMF(n_components=nb,  init = 'nndsvd', random_state=1,alpha=.1, l1_ratio=.5,solver = 'mu', beta_loss ='kullback-leibler')   
#     model = NMF(n_components=nb, init = 'random', random_state=0, solver = 'cd', beta_loss ='frobenius', max_iter=200) 
    model = NMF(init = 'nndsvd', random_state=1,alpha=.1, l1_ratio=.5)
    W = model.fit_transform(A);
    H = model.components_;
    return W, H

def decomposition(A, nb):
    model = NMF(n_components=nb, init='random', random_state=0)   
    W = model.fit_transform(A);
    H = model.components_;
    return W, H