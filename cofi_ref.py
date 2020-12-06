'''
Created on May 27, 2013

@author: bchiu
'''
from math import sqrt
from similarity import radialKernel, radialKernel2, sim_euclidean
import numpy as np
from DataStore import *

import cluster2

DataRoot = "/Users/pleiades/Documents/workspace/QRecommender/src/qrecommender"
DFile = "astudentData.csv"
IDMap = {}  # maps effective ID to true ID
NF = 10

def load(input_file=None, nF=NF, verbose=True):
    if input_file is None: input_file = DFile
    ds = DataStore(input_file=input_file, nParams=nF)
    if verbose: ds.summary()
    X = ds.transform(0)   # questions
    Theta = ds.transform(1)  # users
    Y, R = ds.incidence(debug=True)
    IDMap = ds.map_eff_to_true_id()
    assert len(IDMap) == 2, "should have 2 maps: one for questions and one for users"
    return (X, Theta, Y, R) 
 

def eval_shared_score(Y, R, q1, q2):
    vec1, vec2 = ([], [])
    uids1 = (R[q1,:]==1).nonzero()[0]
    uids2 = (R[q2,:]==1).nonzero()[0]
 
    for uid in uids1:
        if uid in uids2:
            # q1 and q2 both was assigned to user uid
            vec1.append(Y[q1, uid])
            vec2.append(Y[q2, uid]) 

    return (np.array(vec1), np.array(vec2))  


def sim_radial_kernel(Y, R, q1, q2, c=0.6, _ord=1):
    """
    Evaluate the similarity between q1 and q2 with 
    the radial kernel (aka square exponential and 
    sometimes Gaussian kernel). 
    
    @input: 
       q1, q2: question indices
       ord: order of norm (type) 
            e.g. 
            inf: infinity norm
            1: 1-norm (manhattan)
            2: 2-norm (euclidean)
    
    """
    v1, v2 = eval_shared_score(Y, R, q1, q2)  
    n = len(v1)
    assert n == len(v2)
    if n == 0: return 0.0  
    
    delta = np.linalg.norm(v1-v2, _ord)/(n+0.0)
#    if q1 % 5 ==0 and q2 % 5 == 0: 
#        print "> v1: %s\n> v2: %s" % (v1.astype('int'), v2.astype('int'))  
#        print "> 1-norm: %s, 2-norm: %s => %s\n" % \
#           (np.linalg.norm(v1-v2, 1)/(n+0.0), np.linalg.norm(v1-v2, 2)/(n+0.0), radialKernel2(delta, c=c))

    # now, ready to compare their scores or labels, etc
    return radialKernel2(delta, c=c)

def sim_euclidean(prefs, Y, q1, q2):
    v1, v2 = eval_shared_score(prefs, Y, q1, q2)
    assert len(v1) == len(v2)
    if len(v1) == 0: return 0.0 
    return sim_euclidean(v1, v2)

# Returns the best matches for user from the prefs dictionary. 
# Number of results and similarity function are optional params.
def topMatches(prefs,user,n=5,similarity=sim_radial_kernel):

    scores=[(similarity(prefs,user,other),other) 
                  for other in prefs if other!=user]
    scores.sort()
    scores.reverse()
    return scores[0:n]


def invPref(prefs):
    result={}
    for u in prefs:
        for q in prefs[u]:
            result.setdefault(q,{})
      
            # Flip question and user
            result[q][u]=prefs[u][q]
    return result


def eval_pref_map(Y, R):
    """
    Build the dictionary which maps each question (ID) to 
    the IDs of the users who had answered the question. 
    """
    nQ, nU = Y.shape
    prefs = {}
    for i in range(nQ):
        idx = (R[i,:]==1).nonzero()[0]
        if len(idx) == 0: 
            print "> question %d has not been answered ..." % i
        prefs[i] = idx.tolist()
    return prefs


def evalQuestionAffinity(X, Y, R, sim_measure=sim_radial_kernel):
    """
    Compute affinity matrix for questions. 
    Affinity of two questions can be compared 
    via the common users who answered them.
    They are most similar when all users 
    share the same evaluation result (i.e. 
    correctness label, 1 or 0).  
    
    [note] 1. In principle, the user affinity can be computed 
              the same way (via applying weighted-average method 
              over common questions that any two users answered).
              However, since number of users are often large, 
              and very sparse, its usefulness can be very limited.
    
    """
    #prefs = eval_pref_map(Y, R)
    #Xp = np.zeros(X.shape)
    nQ, nU = Y.shape
    W = np.zeros((nQ, nQ))
    print "> size(W): %s" % str(W.shape)
    for i in range(nQ):
        for j in range(i, nQ):
            
            W[i,j] = sim_measure(Y, R, i, j)
            if i != j: 
                W[j,i] = W[i,j]  
                  
    return W

def predict(W, Y, R, _binary=True):
    """
    Make prediction of user performance 
    via weighted average of the scores (or correctness)
    of the questions answered. Weights correspond 
    to the similarity between unanswered question and
    answered question. 
    
    @input:
      _binary: convert raw score to correctness label 
               in 1 or 0? 
    """
    def weigted_avg(qid_unknown, uid, qids, epsilon=1e-4):
        acc = 0.0
        weights = []
        for qid in qids: 
            assert R[qid, uid] == 1
            acc += W[qid_unknown, qid] * Y[qid, uid]
            weights.append(W[qid_unknown, qid])
        
        wacc = sum(weights)
        if acc < epsilon: # not similar to answered questions all that much
            return 0.0
        return (acc+0.0)/sum(weights)
            
    nQ, nU = Y.shape
    Ypred = np.zeros(Y.shape)
    for i in range(nQ):
        for j in range(nU):
            if R[i,j] == 0:
                qids = (R[:,j]==1).nonzero()[0]
                Ypred[i,j] = weigted_avg(i, j, qids)  
                if _binary: 
                    if Ypred[i,j] > 0.5: 
                        Ypred[i,j] = 1
                    else: 
                        Ypred[i,j] = 0
            else: 
                Ypred[i,j] = Y[i,j]
    return Ypred
    
def main(input_file=None):
    """
    
    [note] 1. Since questions in the CoFi algorithm 
              based on weighted similarity do not
              have "features" associated with them as in the 
              case of regression-based methods, we may not 
              be able to visualize the question clusters 
              even though their affinity matrix can be defined.
    """
    import pickle
    import matplotlib.pyplot as plt
    from pylab import *
    
    resultset = {}
    
    if input_file is None: input_file = DFile 
    X, Theta, Y, R = load(input_file)
    print "> data loaded ..."
    W = evalQuestionAffinity(X, Y, R)
    print "> W: %s" % W
    
    # make prediction based on answered questions and their 
    # similarity to unanswered questions
    Ypred = predict(W, Y, R)
    
    # can do mutual iteration W0 -> Ypred0 -> W1 -> Ypred1 ?  
    
    # [note] 1.   
    centroids, idx, eVals, eVecs= cluster2.scluster(W=W, glap='rw', nC=10)
    resultset['membership'] = idx
    resultset['eigen_values'] = eVals 
    
    print "> centroids:\n %s\n" % centroids
    print "> scluster(evals):\n%s" % str(eVals)
    
    # not as meaningful as in the case of regression-based CoFi 
    # because both questions and users do not have associated 
    # features
    cluster2.plot_figure(X, eVals, eVecs, idx) 
    
    # plot affinity matrix 
    pcolor(cmap=cm.RdBu, vmax=abs(Y).max(), vmin=-abs(Y).max())
    colorbar()
    axis([0,1,0,1])

    show()

    # save data if you like
    outfile = "cofi_ref_1.pkl"
    fp = open(outfile, 'wb')  
    pickle.dump(resultset, open(outfile, 'wb'))
    fp.close()
    
    return

if __name__ == "__main__":
    main() 
  
  

