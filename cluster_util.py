import sys
import numpy
from numpy.linalg import norm

# [seeding]
def kinit(X, k, distfun=None):
    """
    Choose initial seeding for k-means or other 
    instance-based clustering algorithms.
    
    *X*: data in row-vector format
    *k*: number of seeds (i.e. initial cluster centroids)
    
    """
    'init k seeds according to kmeans++'
    
    from numpy.random import randint

    X = numpy.array(X)
    try: 
        n = X.shape[0]   # n training samples
    except: 
        n = len(X)
       
    if n == 0: 
        warn = '[kinit] Empty data set'
        print warn
        return numpy.array([]) 
          

    'choose the 1st seed randomly, and store D(x)^2 in D[]'
    centers = [X[randint(n)]]   # numpy.random.randint()
    D = [norm(x-centers[0], 2)**2 for x in X]  # numpy.linalg.norm(), two-norm

    for _ in range(k-1):
        bestDsum = bestIdx = -1

        for i in range(n):
            'Dsum = sum_{x in X} min(D(x)^2,||x-xi||^2)'
            Dsum = reduce(lambda x,y:x+y,
                          (min(D[j], norm(X[j]-X[i], 2)**2) for j in xrange(n)))

            if bestDsum < 0 or Dsum < bestDsum:
                bestDsum, bestIdx  = Dsum, i

        centers.append(X[bestIdx])
        D = [min(D[i], norm(X[i]-X[bestIdx], 2)**2) for i in xrange(n)]

    return numpy.array(centers)


def choose_initial_plusplus(data, k, distfun=None,  measure_corr=True):
    """
    Choose k initial *different* centroids randomly using the
    k-means++ euristic by David Arthur and Sergei Vassilvitskii.
    This often gives better clustering results, but it is slower than the
    basic choice of starting points."""
    # See article "k-means++: The Advantages of Careful Seeding" by
    #   David Arthur and Sergei Vassilvitskii.
    # See also: http://theory.stanford.edu/~sergei/kmeans
    from random import choice, randrange, random
    from itertools import izip
    from bisect import bisect

    def weigthed_choice(objects, frequences):
        if len(objects) == 1:
            return 0

        addedFreq = []
        lastSum = 0
        for freq in frequences:
            lastSum += freq
            addedFreq.append(lastSum)

        return bisect(addedFreq, random())

    if distfun is None:
        def distfun(p1, p2):
            #return sum( (cp1 - cp2) * (cp1 - cp2) for cp1, cp2 in izip(p1, p2) )
            measure_corr=False
            return norm(numpy.array(p1)-numpy.array(p2), 2) ** 2

    # choose an intial centroid randomly
#    centroids = [tuple(data[numpy.random.randint(len(data))])]
#    print "[debug] centroids: %s" % centroids
#    centroids = set(centroids)
    p_ndarray = True
    if type(data) != type([]):
        data = data.tolist()
        p_ndarray = False 
            
    #print "[debug] given data: %s" % data
    data_backup = []
    pos = randrange(len(data))
    centroids = set([tuple(data[pos])])
    data_backup.append(data[pos]) 
    del data[pos] # slow
   
    ntries = 0
    while len(centroids) < k and ntries < (k * 5):
        if not measure_corr: 
            min_dists = [min( distfun(c, x) for c in centroids) for x in data]
        else: 
            min_dists = [max( distfun(c, x) for c in centroids) for x in data]
            
        tot_dists = float(sum(min_dists))
        probabilities = [min_dists[i] / tot_dists for i, x in enumerate(data)]
        pos = weigthed_choice(data, probabilities) # this can be made faster
        centroids.add(tuple(data[pos]))
        ntries += 1
        data_backup.append(data[pos])
        del data[pos] # slow

    result = map(list, centroids)
    if len(result) < k:
        # Fill in missing centroids
        result.extend( [result[0]] * (k - len(result)) )

    data.extend(data_backup) 
    if p_ndarray: data = numpy.array(data)
    
    return numpy.array(result)
 
def distortion(X, centroids, membership):
    acc = 0.0
    for i, cid in enumerate(membership): 
        #print "[%d] %s - %s => %s" % (i, X[i], centroids[cid], sum((X[i] - centroids[cid]) ** 2))
        acc += sum((X[i] - centroids[cid]) ** 2)
    return acc

def ctree(membership, _exclude_empty=True):
    """
    Create list-of-lists representation for clusters.
    Each nested/inner list represents a cluster with 
       members as the list elements
    e.g. [[2, 3, 7], [12, 9, 5, 6, 8], []]
         => 3 clusters where first cluster contains 3 elements 
                             2nd cluster, 5 elements 
            followed by an empty cluster
    """
    nC = max(membership) + 1
    resultset = [[] for _ in range(nC)]
    for rid, cid in enumerate(membership): 
        resultset[cid].append(rid)
    
    if _exclude_empty: # exclude empty clusters
        removals = []
        for i, group in enumerate(resultset): 
            if not group: 
                removals.append(i)  
        if removals: 
            resultset = [resultset[i] for i, group in enumerate(resultset)
                            if not i in removals]
    return resultset
 
# data generation utilty    
def genData(D=4, k=5, dc_start=1, dc_incr=100):
    """
    *D*: data dimension 
    *k*  : number of clusters 
    *dc_start*: dc value added to noisy data
    """
    from numpy.random import normal, randint
    dataSet = []
    for i in range(k):
        lk = k
        if k-1 > 0: lk = k-1 
        uk = k+30
        
        N = randint(lk, uk)  # number of data in the same cluster
        _set = dc_start * numpy.ones((N,D)) + normal(size=(N,D))
        #print "[debug] set %d ->\n%s" % (i, str(_set))
        for j in range(N): 
            dataSet.append(_set[j].tolist())   # .tolist()
           
    #dataSet.append(dc_start * numpy.ones((k,D)) + normal(size=(k,D)))
        dc_start += dc_incr
    
    dataSet = numpy.array(dataSet)
    print "> All data:\n%s\n" % dataSet  
    print "> dimension of data set: %s" % str(dataSet.shape) 
    print "> number of data set (len): %s" % len(dataSet)
    
     
    return dataSet   

def test():
    from scipy.cluster.vq import kmeans2
    X = genData() 
    print "> data:\n  X" % X.tolist()
    
    centroid_seeds = kinit(X, 4) 
    res,idx = kmeans2(X, centroid_seeds, minit='points')
    
    print "> centroids:\n %s" % res
    print "> membership:\n %s" % idx
    print "> distortion: %s" % distortion(X, res, idx)
    
    print "> cluster-members:\n %s" % ctree(idx)

if __name__ == "__main__": 
    test()
