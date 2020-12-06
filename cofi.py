'''
Created on May 25, 2013

@author: bchiu
'''

import numpy as np
from DataStore import *
import os, random, pickle
import similarity
#from cluster2 import kcluster

# [configuration]

# 1. root directory in which data and learned parameters are kept
DataRoot = "./"

# 2. id maps; from effective IDs to true IDs; one for questions and the other for users
IDMaps = ({}, {})  

# 3. data file 
DataFile = "astudentData.csv"

# 4. file for learned parameters 
ParamFile = 'qrecommender_2_l10.mat'  # previously trained
NewParamFile = 'qrecommender_params.mat'  # newly trained
ClusterFile = 'qcluster.pkl' # pre-trained clusters

# 5. algorithmic parameters 
#    a. NF: number of features 
#    b. LAMBDA: regularization factor/constant
    
NF = 10
LAMBDA = 10

# helper function that converts flatten array to 
# its original matrix
def rebuild(params, nQ, nU, nF):
    """
    Reconstruct data matrix. 
    """
    div = nQ*nF
    X = params[0:div].reshape(nQ, nF)  # questions (nQ-by-nF)
    Theta = params[div:].reshape(nU, nF)  # users 
    return (X, Theta)

# helper function that flattens matrix
def unroll(X, Theta):
    return np.hstack([np.hstack(X), np.hstack(Theta)])
        
def cost(params, Y, R, nQ, nU, nF, _lambda=None):  
    """
    Cost function of the regression-based collaborative filtering algorithm.
    
    *params: parameters for both questions and users 
             e.g. for 400 questions and 10 features/paramters for each question 
                  we have 400 * 10 = 4000 question-specific parameters 
    *Y: correctness matrix 
        Y[i,j]: correctness (1/0) of ith question answered by jth student
    
    *R: activation matrix 
        R[i,j] = 1 iff ith question was answered by or assigned to jth student
               = 0 otherwise 
    *nQ: number of questions 
    *U: number of users
    *nF: number of features (for both questions and users)
    
    *_lambda: regularization constant
    """ 
  
    if _lambda is None: _lambda = 0

    # reconstruct question and user matrix
    X, Theta = rebuild(params, nQ, nU, nF)
    # compute cost
    C = (np.dot(X, Theta.T) - Y) ** 2 
    assert R.shape == C.shape, \
           "[costFunc] dim mismatch dim(R)=%s, dim(C)=%s" % (str(R.shape), str(C.shape))
           
    # regularization 
    rX = _lambda * sum(np.diag( np.dot(X, X.T))) 
    rTheta = _lambda * sum(np.diag( np.dot(Theta, Theta.T)))
           
    # element-wise mul to extract only the score of answered questions
    return 1.0/2 * (sum(sum(R * C)) + rX + rTheta)

def gradCost(params, Y, R, nQ, nU, nF, _lambda=None):
    """
    Evaluate the gradient of the cost function along each 
    coordinate (i.e. question features and user features) 
    """
    
    if _lambda is None: _lambda = 0
    X, Theta = rebuild(params, nQ, nU, nF)
    
    # compute grad(cost) w.r.t question params (x1, x2, ..., x<nQ>)
    gradJ_X = np.zeros(np.shape(X))      
    for i in range(nQ):
        idx = (R[i,:]==1).nonzero()[0] # select users who answered the ith question
        Thetap = Theta[idx, :]
        Yp = Y[i, idx]
        gradJ_X[i, :] = np.dot( np.dot(X[i], Thetap.T) - Yp, Thetap) + \
                       _lambda * X[i]

    gradJ_Theta = np.zeros(np.shape(Theta))
    for j in range(nU):
        idx = (R[:,j]==1).nonzero()[0] # pull all questions answered by user j
        Xp, Yp = (X[idx], Y[idx, j])
        gradJ_Theta[j,:] = np.dot( (np.dot(Xp, (Theta.T)[:,j]) - Yp).T, Xp ) + \
                           _lambda * Theta[j]

    return np.hstack([np.hstack(gradJ_X), np.hstack(gradJ_Theta)])
 

def checkGradients(_lambda=None):
    from utils import check_gradients
    #from numpy import random
    from functools import partial
    
    if _lambda is None: _lambda = 0
    # create simple test case
    X_p = np.random.rand(4,3)
    Theta_p = np.random.rand(5, 3)
    Y = np.dot(X_p, np.transpose(Theta_p))
    Y[np.random.rand(*np.shape(Y)) > 0.5] = 0
    R = np.zeros(np.shape(Y))
    R[Y!=0] = 1
    # R.astype('int')
    
    # Run Gradient Checking
    X = np.random.randn(*np.shape(X_p))
    Theta = np.random.randn(*np.shape(Theta_p))
    nQ, nU = Y.shape
    nF = Theta_p.shape[1]

    init_params = unroll(X, Theta)
    J = partial(cost, Y=Y, R=R, nQ=nQ, nU=nU, nF=nF, _lambda=_lambda)    
    gradJ = partial(gradCost, Y=Y, R=R, nQ=nQ, nU=nU, nF=nF, _lambda=_lambda)
    check_gradients( J, gradJ, init_params, eps = 1e-4 )
    
    # btw, check normalized score
    #print "> Y:     %s" % Y 
    #Ynorm, Ymean = normalizedScore(Y, R)
    #print "> Ynorm: %s" % Ynorm
    #print "> Ymean: %s" % Ymean

    return

def normalizedScore(Y, R):
    nQ, nU = np.shape(Y)
    Ymean = np.zeros((nQ, 1))
    Ynorm = np.zeros(np.shape(Y))
    np.mean(Y, axis=1)
    for i in range(nQ):
        idx = (R[i,:]==1).nonzero()[0] # idx of users who answered ith question
        Ymean[i] = np.mean(Y[i, idx], axis=0)
        Ynorm[i, idx] = Y[i, idx] - Ymean[i]
    return (Ynorm, Ymean)

def load(input_file=None, nF=NF, verbose=False):
    """
    Load the data set and convert it to the 
    4-tuple (X, Theta, Y, R)
      X: nQ-by-nF matrix representing question set.
         where nQ: number of users and 
               nF: number of features 
      Theta: nU-by-nF matrix representing users 
         where nU: number of users
      Y, R: please refer to cost()
    
    """
    if input_file is None: input_file = DataFile
    ds = DataStore(input_file=input_file, nParams=nF)
    if verbose: ds.summary()
    X = ds.transform(0)   # questions
    Theta = ds.transform(1)  # users
    Y, R = ds.incidence(debug=True)
    IDMap = ds.map_eff_to_true_id()
    assert len(IDMap) == 2, "not two maps? one for questions and one for users"
    return (X, Theta, Y, R)    
    
def learn(input_file=None, _lambda=LAMBDA, _maxiter=None, 
          _norm=True,
          _data=None, 
          _minimizer=None, length=-700,
          output_file=None, 
          verbose=True):
    
    import timer
    from scipy import optimize, io
    from functools import partial
    
    msg = ''
    X, Theta, Y, R = load(input_file)
    nQ = Y.shape[0]
    nU = Y.shape[1]
    nF = Theta.shape[1]
    assert nF == X.shape[1], "# of features does not agree"
    if verbose: 
        msg += "[learn] # questions: %d, # users: %d, # features: %d\n" % (nQ, nU, nF)
    
    # save the learned parameters to file
    if _data is None: _data = {}
 
    # make initial guess
    # init parameters for X and Theta
    # [debug] try starting from learned params
    if _data.has_key('opt_params'):
        init_params = _data['opt_params']
    else: 
        init_params = unroll(np.random.randn(*X.shape), 
                             np.random.randn(*Theta.shape));

    # normalize the "correctness score" in case if any question 
    # was never assigned to any users
    Ynorm = Y
    if _norm: 
        Ynorm, Ymean = normalizedScore(Y, R)  
        _data['Ymean'] = Ymean
        #_data['Ynorm'] = _Y
    _data['Y'], _data['R'] = (Y, R)  # for convenience later    

    # configure cost and gradient-of-cost functions
    J = partial(cost, Y=Ynorm, R=R, nQ=nQ, nU=nU, nF=nF, _lambda=_lambda)    
    gradJ = partial(gradCost, Y=Ynorm, R=R, nQ=nQ, nU=nU, nF=nF, _lambda=_lambda)

    # choose minimizer
    if not _minimizer:  # scipy's CG
        if _maxiter is None: _maxiter = 200 * len(init_params)
        print "> max # of iteration: %d" % _maxiter   
        best_params, fopt, func_calls, grad_calls, warnflag = \
            optimize.fmin_cg(J, init_params, fprime=gradJ, 
                             full_output=True,
                             gtol=1e-5, 
                             maxiter = _maxiter)
        _data['fopt'] = fopt
        _data['func_calls'] = func_calls
        _data['grad_calls'] = grad_calls
        _data['warnflag'] = warnflag 
        
    elif _minimizer in (1, 'cg', ): # Polack-Ribiere flavour of conjugate gradients
        from minimizer import minimize
        if type(length) == type(1): length = [length]    
        best_params, fvec, num_iter, _f, _df = \
              minimize(J, init_params, gradJ, length=length)
        _data['fX'] = fvec  # vec of function values indicating progress made
        _data['nIter'] = num_iter
        _data['J'] = _f
        _data['gradJ'] = _df
    elif _minimizer in (2, 'bfgs', ):
        raise NotImplementedError, "[learn] %s mode not avail" % _minimizer 

    _data['opt_params'] = best_params
    X, Theta = rebuild(best_params, nQ, nU, nF)
    _data['X'] = X
    _data['Theta'] = Theta
    
    # save result 
    # oned_as: default for the way 1-D vectors are stored
    if output_file is None: output_file = NewParamFile
    io.savemat(NewParamFile, _data, oned_as='row')
    if verbose:
        msg += "[learn] lambda: %f, maxiter: %s, cg_mode: %s\n" % \
                (_lambda, _maxiter, _minimizer)
        msg += "[learn] optimal parameters:\n %s\n" % best_params
        msg += "[learn] grad(J) at opt:\n %s\n" % gradJ(best_params)
        print msg
    
    #rebuild(res1, nQ, nU, nF)
    return 

def load_learned_params(_file=None, _rootdir=None):
    import scipy.io as sio
        
    # 'cofi_qrecommender_3_3.mat'
    if _file is None: _file = ParamFile # [hardcode] 
    _data = sio.loadmat(os.path.join(DataRoot, _file))
    return (_data['X'], _data['Theta'], _data['Y'], _data['R'], _data['Ymean'])

def load_clusters(_file=None, _rootdir=None):
    if _file is None: _file = ClusterFile
    pkl = open(_file, 'rb')
    _data = pickle.load(pkl)
    pkl.close()
    return (_data['ctree'], _data['centers'])

def predict(_data=None,
            _file=None, _binary=True):
    """
    Given learned parameters (included in _data), predict 
    user performance for the question set (i.e. Y[i,j]
    for which R[i,j] = 0)
    
    @input 
      *_data: a dictionary containing folowing keys 
              representing learned parameters: 
          X: trained or optimal question-specific parameters  
          Theta: trained user-specific parameters 
          Y, R: 
          Ymean: mean (correctness) score for questions
          
          ... etc. 
      *_file: input file containing trained data
      
      *_binary: if True, then convert predicted score 
                to 1 or 0, reflecting the correctness of 
                the answer; if predicted score > 0.5 
                then predict a correct (or 1); 
                incorrect (or 0) otherwise. 
                
    @output 
      a 3-tuple: (Ypred, Y, R)
          where Ypred: predicted Y 

    [note] 
    """
    def mean_adjust():
        """
        Adjust the predicted score by adding back the mean 
        score for each question (since Y was mean-centered
        earlier)
        """
        nQ, nU = np.shape(Y)
        Ypred = np.zeros(np.shape(Y))
        
        # predicted score/label
        Ymc = np.dot(X, Theta.T)  # mean-centered predicted score
        #print "[prediction] mean-centered score:\n  %s" % Ymc    
        
        assert Ymean.shape[1] == 1
        for i in range(nQ):
            #idx = (R[i,:]==1).nonzero()[0] # indices of questions answered by someone
            for j in range(nU):
                if R[i,j] == 0: # for those who haven't answered it
                    Ypred[i,j] = Ymc[i,j] + Ymean[i] 
                    if _binary: 
                        if Ypred[i,j] > 0.5: 
                            Ypred[i,j] = 1
                        else: 
                            Ypred[i,j] = 0
                else: 
                    Ypred[i,j] = Y[i,j] 
        return Ypred         
                   
    # load learned params
    if _data is None: 
        X, Theta, Y, R, Ymean = load_learned_params(_file)
    else: 
        X, Theta, Y, R, Ymean = (_data['X'], _data['Theta'],
                                 _data['Y'], _data['R'], 
                                 _data['Ymean'])
    # Ypred = mean_adjust()
    Ypred = mean_adjust() 
    
#    for i in range(X.shape[0]):
#        idx = (R[i,:]==1).nonzero()[0]
#        if i % 10 == 0: 
#            print "> Predicted Score:\n  %s" % Ypred[i, idx][:15]
#            print "> Y:\n  %s" % Y[i, idx][:15]
        
    #print "> Ypred[0:20, 0:20] %s" % Ypred[_R==1][0:20, 0:20]
    return (Ypred, Y, R)

def kcluster(rows, distance=None, nC=100, nCycle=10):
    """
    Create clusters using (frequency-balanced) kmeans++
    """
    import cluster2, cluster_util
    if distance == None: distance = similarity.pnorm
    assert hasattr(distance, '__call__'), "Invalid distance function"
    centroids, membership = cluster2.kcluster(rows, distance=distance, k=nC, nCycle=nCycle)
    
    # convert to list of lists representation
    return (cluster_util.ctree(membership), centroids)
        
def scluster(rows, W=None, 
             glap='r',
             nC=100,  
             sim=similarity.radialKernel(c=0.15), 
             outfile=None, _plot=False):
    """
    Create clusters using spectral clustering.
    """
    import cluster2, cluster_util

    W = cluster2.getAffinityMatrix(rows, sim=sim)  # given data and choice of sim graph (e.g. knn, fully-connected)    
    centroids, idx, evals, Y = cluster2.scluster(rows, W, 'rw', nC=10)
    if _plot: 
        cluster2.plot_figure(rows, evals, Y, idx)
        
    if outfile:
        import scipy.io as sio 
        resultset = {}
        resultset['membership'] = idx
        resultset['eigen_values'] = evals
        sio.savemat(outfile, resultset)
   
    return (cluster_util.ctree(idx), centroids)

    
def has_dup(alist): # duplicate questions? 
    aset = set([])
    tval = False
    for _e in alist:
        if not _e in aset:
            aset.add(_e)
        else: 
            tval = True; break
    return tval

def fitness(sol, Ypred, Y, R, gsize=5, 
            qcluster=None, ucluster=None, 
            uset=None,
            dup_discount=0.9, 
            _chain_discount=False):
    """
    Evaluate the fitness of the given solution (*sol). 
    
    @input
       sol:  solution vector of size nU * gsize 
             where nU: number of (sampled or clustered) user IDs
       gsize: number of elements associated with each object
              e.g. number of questions (to be) answered by a 
                   user.
      
      ucluster: If similar users are grouped into 
                    clusters, we need to know the members
                    (user IDs) associated with each cluster
                Each cluster is represented by a list of lists 
                (see ctree() in cluster_util) 
                 
      qcluster: Similar to the case of user clusters above 
                 each question cluster consists of 
                 similar question (IDs). 
      uset:  Clustering user is often very costly (assuming 
              a fairly large set of users); therefore, we can 
              resort to sampling methods to select a subset of
              users to evaluate their performance over a given
              set of questions. Average multiple runs to draw
              conclusions. 
      dup_discount: if True, penalize duplicate questions
      
    [note] 
    
    1. The assignment of questions to users can be represented
       as a vector of question IDs of size n, in which 
       every m question IDs correspond to the questions assigned
       to a particular user: 
       
       e.g. a solution of [2, 4, 10, 6, 7, 1, 3, 4, 11, 9 ...]
            represents: 
               user0 is assigned to questions: {2, 4, 10, 6, 7}
               user1                         : {1, 3, 4, 11, 9}
               user2 ...
    2. 
         
    """
    def select_from_group(group):
        # sampling in proportion to size of cluster
        # assuming that members of the group are similar 
        # alternatively, use the cluster centroid
        try: 
            candidates = random.sample(group, len(group))  # cluster empty? 
        except: 
            pass 
        return candidates

    total, N, dupc = (0.0, len(sol), 1.0)
    #_nQ = Y.shape[0] if qcluster is None else sum([len(g) for g in qcluster])
    #_nU = Y.shape[1] if ucluster is None else sum([len(g) for g in ucluster])

    scores = [] # keep track of # of correct answers for each user
    
    if ucluster:
        raise NotImplementedError, "coming soon but not recommended for use!"
        
    uid = 0
    for d in range(0, N+5, gsize): # foreach user (or user group)
        if d+gsize > N: break 
        qids = [sol[d+i] for i in range(gsize)]
        
        # Questions cannot have duplicates
        # but instead of imposing -inf penalty, relax it a bit
        if has_dup(qids):
            #print "> found dup in qids: %s" % qids
            #return -np.inf 
             
            # continuous dicounting
            if _chain_discount: 
                dupc *= dup_discount
            else: # or just one single discount 
                dupc = dup_discount
        
        nCorrect = 0  # correctness ratio

        if uset: 
            try: 
                _uid = uset[uid]  # find true user ID
            except: 
                print "> N:%d | (uid, d): (%s, %s), len(uset): %d, uset: %s" % (N, uid, d, len(uset), uset[:10])
                raise RuntimeError
        
        # sampling question IDs from cluster
        if qcluster:
            qcandidates = []
            for qid in qids: 
                # sampling without replacement in proportion to 
                # cluster size
                qcandidates.extend(select_from_group(qcluster[qid]))

        # for each questions, estimate/lookup user performance 
        for _qid in qcandidates:               
            if R[_qid, _uid] == 1:  # question was answered by user
                if Y[_qid, _uid] == 1:
                    nCorrect += 1
            else: # question unanswered, need to predict
                # assuming that we have converted 
                # the predicted score
                if Ypred[_qid, _uid] > 0.5: 
                    nCorrect += 1
        # keep track of score of user wrt the current question assignment
        scores.append(nCorrect)
        uid += 1
        
    assert len(scores) == N/gsize, \
           "[genetic_cost] mismatch: # of scores: %s while N/gsize: %s, (d, uid): (%s, %s)" % \
              (len(scores), N/gsize, d, uid)
           
    # if the question selection leads to larger variability 
    # in user performance (# of correct answers or 
    # 1 - error rate, etc), then favor such selection strategy 
    # since it potentially offers more meaning ranking of 
    # user performance => the larger the variance => the better
    # fitness_score = sum((np.array(scores) - np.mean(scores)) ** 2) 
    
    return sum((np.array(scores) - np.mean(scores)) ** 2) * dupc
        
def anti_fitness(sol, Ypred, Y, R, gsize=5, 
                  qcluster=None, ucluster=None, 
                  uset=None):

    fitness_score = fitness(sol, Ypred, Y, R, gsize,
                            qcluster, ucluster, uset)
    print "[anti_fitness] fitness: %s" % fitness_score
    # return 0.0 - sum((np.array(scores) - np.mean(scores)) ** 2) 
    
    # use SE kernel to convert to cost in [0, 1]
    return similarity.radialKernel2(fitness_score, c=10)    
                              
def assignment(_data=None,  
               _file=ParamFile, group_size=5, 
               group_questions=True, 
               group_users=False,
               gen_clusters=kcluster,
               _cluster_file=None,
               ncycle=20,
               upopsize=100,
               qRatio=0.5):
    """
    Find the best question assignment strategy (wrt users).
    
    Optimization Objective: 
    
    Assign questions to users such that their performance 
    is as distinguishable as possible. In other words, 
    the questions that do not lead to larger variability in 
    user performance should potentially be excluded from the exam. 
    
    @input 
      _data: see predict()
      _file: file containing learned parameters 
      group_size: size of each group in the solution vector. 
                   see genetic_cost() for more details.
      group_questions: if true, then group similar questions
      group_users: if True, then group similar users 
      gen_clusters: clustering algorithm
                       cluster(): k-means
                       scluster(): spectral clustering
      _cluster_file: data file containing pre-trained clustering
                     results
      qRatio:       
       
    """
    from evolution import genetic_optimize
    from functools import partial

    # load learned params
    if _data is None: 
        X, Theta, Y, R, Ymean = load_learned_params(_file)
    else: 
        X, Theta, Y, R, Ymean = (_data['X'], _data['Theta'],
                                 _data['Y'], _data['R'], 
                                 _data['Ymean'])    
    nQ, nU = Y.shape
    
    # predict user performance based on learned parameters
    Ypred, Y, R = predict(_data, _file) 
    
    # cluster similar questions and/or users
    # alternatively, use sampling-based method (especially
    # for users)
    _cdata = {}
    qcluster = ucluster = None; qcenters = ucenters = None
    if group_questions: 
        if not _cluster_file is None:
            qcluster, qcenters = load_clusters(_cluster_file) 
            print "[debug] ctree:\n %s" % qcluster
        else: 
            qcluster, qcenters = gen_clusters(X)
    
    # grouping users is not recommended (expensive)
    uset = []
    if group_users: 
        ucluster, ucenters = gen_clusters(Theta)
    else: 
        uset = random.sample(range(0, nU-1), upopsize)  # [todo] 
        
    # evaluate solution vector template
    maxQ = nQ if qcluster is None else len(qcluster)
    maxU = (nU if uset is None else len(uset)) if ucluster is None else len(ucluster)
    domain = [(0, maxQ-1)] * maxU * group_size
    
    print "[assignment] size of domain: %s" % len(domain)    
        
    # configure fitness function 
    fitness_func = partial(fitness, 
                        Ypred=Ypred, Y=Y, R=R,
                        gsize=group_size, 
                        qcluster=qcluster, 
                        ucluster=ucluster,
                        uset=uset)
    
    ranked_solution_set = []
    for k in range(ncycle): 
        ranked_solutions = genetic_optimize(domain,
                                            fitness=fitness_func, 
                                            popsize=100, 
                                            step=1,          
                                            mutprob=0.2, elite=0.2, maxiter=150,
                                            _order='desc',
                                            gsize=group_size, 
                                            _ntrial=k)
    
    # from the ranked solution, rate the question 
    # according to its weighted frequency; i.e. if a question 
    # occurs more often in higher-ranked solution, then we know 
    # that this question should be given higher preference 
    # from the question set because its higher capacity (potentially) 
    # in distinguishing user performance
    qScore = {}
    #wmax = len(ranked_solutions)
    weights = []  # [debug]
    for ranked_solutions in ranked_solution_set: 
        max_weight = max([_w for _w, _ in ranked_solutions])
        for w, _sol in ranked_solutions:
            weights.append(w)  # [debug]
            for qid in _sol: 
                if not qScore.has_key(qid):
                    qScore[qid] = 0.0
                qScore[qid] += w/(max_weight+0.0)
            
    print "[assignment] weights (should be increasing): %s" % weights
           
    return qScore

def plot_rank():
    pass
        
  
def main():
    # first learn the parameters 
    learn(_minimizer='cg')
    X, Theta, Y, R, Ymean = load_learned_params(ParamFile)
    
    # cluster similar questions and users 
    #   to reduce solution space 
    qcluster, qcenters = kcluster(X)
    ucluster, ucenters = kcluster(Theta)
    

if __name__ == "__main__":
    #learn(_minimizer='cg')
    qScore = assignment(_cluster_file="qkcluster.pkl")
    print "question ranked score: %s" % qScore
    # X, Theta, Y, R = load(input_file="astudentData.csv")
    #prediction()
    #checkGradients(1.5)