
import random
import numpy as np
import similarity
#from similarity import pearson
import cluster_util
#from PIL import Image,ImageDraw


def fskmeans(data, k=4, t=1e-5, distfun=similarity.euclidean, maxiter=150, chooser=None, verbose=False):
    """
    frequency-balanced k-means++ algorithm.

    [input]
      data: list of data points
      k: desired number of clusters
      t: error tolerance (double t). Used as the stopping criterion, i.e. when the sum of
        squared euclidean distance (standard error for k-means) of an iteration is within the
        tolerable range from that of the previous iteration, the clusters are considered
        "stable", and the function returns a suggested value would be 0.0001
      distfunc: a optional function that given two points computes their distance, used by the
        k-means clustering. If not specified it's used the (square of) eucledean distance.
      maxiter: maximum number of iterations, another stopping criterion.
      chooser: function that given data and k return k points chosen as starting cluster
               centroids. By default it's used the choose_initial.
      verbose: True if you want a print of the error on each iteration of the algorithm.

    Output  (c, labels):
      c: list of computed centroids.
      labels: list of what cluster (centroid) each point is assigned to.
      
    """
    from numpy.linalg import norm
    # Adapted from a C version by Roger Zhang, <rogerz@cs.dal.ca>
    # http://cs.smu.ca/~r_zhang/code/kmeans.c
    # It seems that Psyco is useless on this function, ShedSkin makes it about 40+ times faster.
    if chooser is None: chooser = cluster_util.kinit

    DOUBLE_MAX = 1.797693e308

    n = len(data)
    m = len(data[0]) # dimension
    print "[fskmeans] data dimension: %d-by-%d" % (n, m)
    #assert data and k > 0 and k <= n and m > 0 and t >= 0
    assert (n > 0) and (k > 0) and (k <= n) and (m > 0) and (t >= 0)

    error = DOUBLE_MAX # sum of squared euclidean distance

    counts = [0] * k # size of each cluster
    labels = [0] * n # output cluster label for each data point
    cfreq  = [(n+0.0)/k] * k

    # c1 is an array of len k of the temp centroids
    c1 = []
    for i in xrange(k):
        c1.append([0.0] * m)

    # choose k initial centroids
    c = chooser(data, k, distfun)
   
    list_type = False
    if type(data) == type([]): list_type = True
    data = np.array(data)

    niter = 0
    # main loop
    while True:
        # save error from last step
        old_error = error
        error = 0

        # clear old counts and temp centroids
        for i in xrange(k):
            counts[i] = 0
            for j in xrange(m):
                c1[i][j] = 0

        # Note: a big block of code is duplicated to keep the program fast
        if distfun:
            for h in xrange(n):  # foreach data (n: number of data)
                # identify the closest cluster
                min_distance = DOUBLE_MAX
                for i in xrange(k):
                    
                    #distance = distfun(data[h], c[i]) #  
                    # frequency-balanced clustering 
                    cfactor = -np.inf if cfreq[i] == 0 else np.log(cfreq[i])
                    distance = cfreq[i] * distfun(data[h], c[i]) - cfactor
                    
                    if distance < min_distance:
                        labels[h] = i
                        min_distance = distance
                
                # update size and temp centroid of the destination cluster
                c1[labels[h]] += data[h]
                counts[labels[h]] += 1
                # update standard error
                error += min_distance
            # update cluster frequency
            for j in xrange(k):
                cfreq[j] = counts[j]
        else:
            # assign each data to the best cluster 
            for h in xrange(n):  # foreach data 
                # identify the closest cluster
                min_distance = DOUBLE_MAX
                for i in xrange(k):   # foreach cluster rep. 
                    #distance = 0
                    #print "[debug] norm of %s and %s => %s" % (data[h], c[i], norm(data[h] - c[i], 2))
                    distance = cfreq[i] * norm(data[h] - c[i], 2) - np.log(cfreq[i])
                    if distance < min_distance:
                        labels[h] = i                # assign h-th data to i-th cluster 
                        min_distance = distance

                # update size and temp centroid of the destination cluster
                #for j in xrange(m):
                #    c1[labels[h]][j] += data[h][j]
                c1[labels[h]] += data[h]
                counts[labels[h]] += 1
                # update standard error
                error += min_distance
                        # update cluster frequency
            for j in xrange(k):
                cfreq[j] = counts[j]

        for i in xrange(k): # update all centroids
            _count = counts[i]
            if _count: 
                c[i] = c1[i]/(_count+0.0)
            else: 
                c[i] = c1[i]
#            for j in xrange(m):
#                c[i][j] = (c1[i][j]+0.0) / _count if _count else c1[i][j]

        niter += 1
        if verbose: print "%d) Error:" % niter, abs(error - old_error)
        if (abs(error - old_error) < t) or (niter > maxiter):
            break
            
    # recover original data type
    if list_type: data = data.tolist()
    c = np.array(c); labels = np.array(labels)

    return (c, labels)

def kmeans(rows, distance=similarity.pearson, k=4, maxIter=150, epsilon=1e-4, 
            balanced=True, 
            debug=True):
  """
    Kmeans algorithm starts by partitioning the input points into k initial sets,
    either at random or using some heuristic. It then calculates the mean point, or
    centroid, of each set. It constructs a new partition by associating each point with the
    closest centroid. Then the centroids are recalculated for the new clusters, and
    algorithm repeated by alternate application of these two steps until convergence, which
    is obtained when the points no longer switch clusters (or alternatively centroids are no
    longer changed).

    @input 
      *rows: data in row-vector format
      *distance: distance function (i.e. dissimilar measure) 
      *balanced: if True, use the objective function taking into account of 
                 cluster size
    @output
      *bestmatches: a list of lists, each of which represents a cluster 
                    containing indices of the data  
  """
  # Determine the minimum and maximum values for each point
  ranges=[(min([row[i] for row in rows]),max([row[i] for row in rows])) 
  for i in range(len(rows[0]))]

  # Create k randomly placed centroids
  clusters=[[random.random()*(ranges[i][1]-ranges[i][0])+ranges[i][0] 
  for i in range(len(rows[0]))] for j in range(k)]
  
  lastmatches=None
  last_distortion = np.inf
  for t in range(maxIter):
    if debug: print 'Iteration %d' % t
    bestmatches=[[] for i in range(k)]
    distortion = 0.0
    #cidx = set([])  # unique cluster indices
    # Find which centroid is the closest for each row

    for j in range(len(rows)):
      row=rows[j]
      bestmatch=0
      for i in range(k):
        d=distance(clusters[i],row)
        if d<distance(clusters[bestmatch],row): 
            bestmatch=i  
      # best cluster assignment for jth data is 
      # the one with index bestmatch
      bestmatches[bestmatch].append(j)
      #print "> clusters[bestmatch]: %s" % clusters[bestmatch]
      #print "> rows[j]:             %s" % rows[j]
      distortion += np.linalg.norm(clusters[bestmatch]-rows[j], 2)
      #cidx.add(bestmatch) 
       
    # If the results are the same as last time, this is complete
    if (bestmatches==lastmatches and 
           abs(distortion-last_distortion) < epsilon) or (t > maxIter): break
    lastmatches=bestmatches
    last_distortion = distortion
    
    # evaluate new clusters: move the centroids to the average of their members
    for i in range(k):
      avgs=[0.0]*len(rows[0])  # init centroid
      if len(bestmatches[i])>0:
        for rowid in bestmatches[i]: # foreach member in ith cluster
          for m in range(len(rows[rowid])):
            avgs[m]+=rows[rowid][m]
        for j in range(len(avgs)):
          avgs[j]/=len(bestmatches[i])
        clusters[i]=avgs
      
  print "last distortion: %s" % last_distortion
  return (bestmatches, distortion, clusters) 
  
  

  

def membership(bestmatches, X=None):
    """
    Covert cluster representation to 
    row-id-to-cluster-id mapping. 
    """
    M = 0
    if not (X is None): 
        M = len(X)
    else: 
        for members in bestmatches:
            M += len(members)
            
    idx = [-1] * M
    for cid, members in enumerate(bestmatches): 
        for rid in members: 
            idx[rid] = cid
    return idx

def test():
    from cluster_util import genData
    X = genData()
    print "1. regular kmeans ..."
    bestmatches, distortion, centroids = kmeans(X, distance=similarity.euclidean)
        
    print "> member-to-cluster-id:\n %s" % bestmatches 
    print "> distortion: %s" % distortion
    print "> centroids:\n %s" % centroids
    
    print "2. fs-kmeans ..."
    c, labels = fskmeans(X, k=4, distfun=similarity.euclidean)
    print "> clusters:\n %s" % c
    print "> labels:\n %s" % labels
    
if __name__ == "__main__":
    test()