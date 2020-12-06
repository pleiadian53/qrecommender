'''
Created on May 26, 2013

@author: bchiu
'''

import sys, warnings, pickle

import numpy as np
import matplotlib.pyplot as plt

from scipy.linalg import eig, inv, norm
from scipy.cluster.vq import kmeans2
from scipy.sparse.linalg import eigen
from scipy.spatial.kdtree import KDTree

# seeding function
import cluster_util, cluster
# distortion function [debug]
from similarity import pearson, radialKernel, cosine, \
                       sim_euclidean
import similarity

# configuration 
DataRoot = "./"
ParamFile = "cofi_qrecommender_3_4.mat"  #"qrecommender_2_l10.mat"
ClusterFile = "qrecommender_cluster.mat"

# global parameters 
debug = 1

# warnings as exceptions 
warnings.filterwarnings("error", category=UserWarning, message="One of the clusters is empty.*")

def get_noise(stddev=0.25, numpoints=150):
    # 2d gaussian random noise
    x = np.random.normal(0, stddev, numpoints)
    y = np.random.normal(0, stddev, numpoints)
    return np.column_stack((x, y))

def get_circle(center=(0.0, 0.0), r=1.0, numpoints=150):
    # use polar coordinates to get uniformly distributed points
    step = np.pi * 2.0 / numpoints
    t = np.arange(0, np.pi * 2.0, step)
    x = center[0] + r * np.cos(t)    # unit: rad
    y = center[1] + r * np.sin(t)
    return np.column_stack((x, y))   # stack 1D array as column vectors into a 2D array

def circle_samples(n=150):
    """
    *n*: number of sample points 
    
    """
    circles = []
    for radius in (1.0, 2.8, 5.0, 10.0,  ):
        circles.append(get_circle(r=radius, numpoints=n) + get_noise(numpoints=n))
    return np.vstack(circles)

def mutual_knn(points, n=10, sim=cosine):
    knn = {}
    kt = KDTree(points)
    for i, point in enumerate(points):
        for neighbour in kt.query(point, n + 1)[1]:
            if i != neighbour:
                #print "> points[neighbor]: %s" % points[neighbour]
                knn.setdefault(i, []).append(
                    (sim(point, points[neighbour]), neighbour))
    return knn

def knn(points, n=10, sim=radialKernel):
    pass

def get_full_matrix(points, sim=None):
    """
    Affinity matrix defined by fully-connected graph 
    """
    if sim is None: 
        sim = sim_euclidean
    try: 
        m, n = points.shape
    except: 
        m, n = np.array(points).shape
    
    W = np.zeros((m, m))
    for i in range(m):
        for j in range(i,m):
            W[i,j] = sim(points[i], points[j])
            if i != j: 
                W[j,i] = W[i,j]
    return W

def get_knn_matrix(knn):   
    """
    Affinity matrix defined by KNN graph 
    """
    n = len(knn)
    W = np.zeros((n, n))
    for point, nearest_neighbours in knn.iteritems():
        for sim, neighbour in nearest_neighbours:
            W[point][neighbour] = sim
    return W

def rename_clusters(idx):
    # so that first cluster has index 0
    num = -1
    seen = {}
    newidx = []
    for id in idx:
        if id not in seen:
            num += 1
            seen[id] = num
        newidx.append(seen[id])
    return np.array(newidx)

def cluster_points_kmeans(X, nC=3):
    """
    Cluster data using kmeans++.
    """
    centroid_seeds = cluster_util.kinit(X, nC)
    _distortion = None
    try: 
        # [note] res: A k-by-n array of centroids 
        #             where n: dimension of data (row vectors/ndarrays)
        #        membership: 
        #             membership[i] is the index of the centroid the ith observation is closest to.
        # frequence-balaced kmeans++
        res, membership = kmeans2(X, centroid_seeds, minit='points')
        _distortion = cluster_util.distortion(X, res, membership)
    except Exception, e:  
        try: 
            # frequence-balaced kmeans++
            res, membership = cluster.fskmeans(X, k=nC)
        except Exception, e: 
            err = "[scluster] Oops! my kmeans also crashed: %s\n" % e
            raise RuntimeError, err
    _distortion = cluster_util.distortion(X, res, membership)
    
    return (res, membership, _distortion)
      

def cluster_points(L, glap='r', num_clusters=3):
    """
    Run frequency-balanced kmean++ in the eigenspace of given graph Laplacian (*L). 
    
    """
    # sparse eigen is a little bit faster than eig
    #evals, evcts = eigen(L, k=15, which="SM")
    print "[scluster] start clustering ..."
    evals, evcts = eig(L)
    evals, evcts = evals.real, evcts.real
    edict = dict(zip(evals, evcts.transpose()))
    evals = sorted(edict.keys())     # sort from smallest e-value to the largest

    if glap.lower().startswith('unnorm'):
        # second and third smallest eigenvalue + vector 
        Y = np.array([edict[k] for k in evals[1:num_clusters]]).transpose() # transpose to col vec format
    elif glap.lower().startswith('sym'):
        Y = np.array([edict[k] for k in evals[0:num_clusters]]).transpose()
        # [todo]
    elif glap.lower().startswith('r'):
        Y = np.array([edict[k] for k in evals[0:num_clusters]]).transpose()     
            
    # [note] 1. choose better seeding using knit (kmeans++) 
    #        2. some clusters may be emtpy, in which case, need to choose diff. seeding
    centroid_seeds = cluster_util.kinit(Y,num_clusters)
    distortion = None
    try: 
        # [note] centroids: a list of coordinates of cluster representatives 
        #        clustermap: data id to clutser id 
        centroids, clustermap = kmeans2(Y, centroid_seeds, minit='points')
    except Exception, e:
        try: 
            centroids, clustermap = cluster.fskmeans(Y, k=num_clusters)
            
        except Exception, e: 
            err = "[scluster] Oops! my kmeans also crashed: %s\n" % e
            raise RuntimeError, err 
    
    if debug > 2: 
        print "> data\\eigenvecs:\n%s" % str(Y[:10])
        print "> size(eigenvecs): %s" % str(Y.shape)
        print "> centroids:\n%s" % str(centroids)  # k by N
    
    if distortion == None: 
        distortion = calc_distortion(Y, clustermap, centroids) 
    
    print "> total distortion: %f" % distortion
    #print "> eigen values: %s" % str(evals[:15])
    #print "> Y1: %s, Y2: %s" % (str(Y[0]), str(Y[1]))
    #centroids, clustermap = kmeans2(Y, 3, minit='random')
   
    return (centroids, rename_clusters(clustermap), evals, Y, distortion)
    
def calc_distortion(X, cid, centroids): 
    """
    [input
      cid: cluster center 
    
    [output] 
      sum over all cluster distortions 
      
      center_dict: 
      a dictionary with key: cluster center, 
                      value: distortion of that cluster
    """
    # assuming that the cluster index starts from 0 
    
    # compute distortion against the centroids 
    center_dict = {}
    for i, c in enumerate(cid):
        if not center_dict.has_key(c): center_dict[c] = 0.0
        else: 
            # cluster_center -> sum of all points in cluster to their centroid
            center_dict[c] += np.linalg.norm(X[i] - centroids[c],2)  # 2-norm
   
    return sum(center_dict.values())

def change_tick_fontsize(ax, size):
    for tl in ax.get_xticklabels():
        tl.set_fontsize(size)
    for tl in ax.get_yticklabels():
        tl.set_fontsize(size)

def get_colormap():
    # map cluster label to color (0, 1, 2) -> (orange, blue, green)
    from matplotlib.colors import ListedColormap
    import random
    orange = (0.918, 0.545, 0.0)
    blue = (0.169, 0.651, 0.914)
    green = (0.0, 0.58, 0.365)
    base = [orange, blue, green]
    cpool = [ '#bd2309', '#bbb12d', '#1480fa', '#14fa2f', '#000000',
              '#faf214', '#2edfea', '#ea2ec4', '#ea2e40', '#cdcdcd',
              '#577a4d', '#2e46c0', '#f59422', '#219774', '#8086d9' ]
    codes = base + cpool
    random.shuffle(codes)
    return ListedColormap(codes)

def plot_circles(ax, points, idx, colormap, 
                 xlabel='x1', ylabel='x2',
                 xscope=(-1.5,1.5), yscope=(-1.5,1.5)):
    
    # [todo] find principle axes for higher-D data
    import re
    pat = re.compile(r'\w+([0-9])')
    axes = []
    for label in (xlabel, ylabel, ):
        mat = pat.match(label)
        if mat:
            try:  
                axes.append(int(mat.group(1)))
            except: 
                msg = "[plot_circles] check x, y-label format. missing number?"
                print msg 
    if not axes: axes = [0, 1]
    
    plt.scatter(points[:,axes[0]], points[:,axes[1]], s=10, c=idx, cmap=colormap,
        alpha=0.9, facecolors="none")
    plt.xlabel(xlabel, fontsize=8)
    plt.ylabel(ylabel, fontsize=8)
    change_tick_fontsize(ax, 8)

    plt.ylim(*yscope)  # 6, 6 
    plt.xlim(*xscope)

def plot_eigenvalues(ax, evals):
    plt.scatter(np.arange(0, len(evals)), evals,
        c=(0.0, 0.58, 0.365), linewidth=0)
    plt.xlabel("Number", fontsize=8)
    plt.ylabel("Eigenvalue", fontsize=8)
    plt.axhline(0, ls="--", c="k")
    change_tick_fontsize(ax, 8 )

def plot_eigenvectors(ax, Y, idx, colormap):
    from matplotlib.ticker import MaxNLocator
    from mpl_toolkits.axes_grid import make_axes_locatable
    divider = make_axes_locatable(ax)
    ax2 = divider.new_vertical(size="100%", pad=0.05)
    fig1 = ax.get_figure()
    fig1.add_axes(ax2)
    ax2.set_title("Eigenvectors", fontsize=10)
    ax2.scatter(np.arange(0, len(Y)), Y[:,0], s=10, c=idx, cmap=colormap,
        alpha=0.9, facecolors="none")
    ax2.axhline(0, ls="--", c="k")
    ax2.yaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.axhline(0, ls="--", c="k")
    ax.scatter(np.arange(0, len(Y)), Y[:,1], s=10, c=idx, cmap=colormap,
        alpha=0.9, facecolors="none")
    ax.set_xlabel("index", fontsize=8)
    ax2.set_ylabel("2nd Smallest", fontsize=8)
    ax.set_ylabel("3nd Smallest", fontsize=8)
    change_tick_fontsize(ax, 8 )
    change_tick_fontsize(ax2, 8 )
    for tl in ax2.get_xticklabels():
        tl.set_visible(False)

def plot_spec_clustering(ax, Y, idx, colormap):
    plt.title("Spectral Clustering", fontsize=10)
    plt.scatter(Y[:,0], Y[:,1], c=idx, cmap=colormap, s=10, alpha=0.9,
        facecolors="none")
    plt.xlabel("Second Smallest Eigenvector", fontsize=8)
    plt.ylabel("Third Smallest Eigenvector", fontsize=8)
    change_tick_fontsize(ax, 8 )

def plot_figure(points, evals, Y, idx):
    colormap = get_colormap()
    fig = plt.figure(figsize=(12, 11.0))   # (6, 5.5)

    fig.subplots_adjust(wspace=0.4, hspace=0.3)  # (0.4, 0.3)
    ax = fig.add_subplot(2, 2, 1)
    plot_circles(ax, points, idx, colormap, xlabel='feature_1', ylabel='feature_2')

    ax = fig.add_subplot(2, 2, 2)
    plot_eigenvalues(ax, evals)

    ax = fig.add_subplot(2, 2, 3)
    plot_eigenvectors(ax, Y, idx, colormap)

    ax = fig.add_subplot(2, 2, 4)
    plot_spec_clustering(ax, Y, idx, colormap)

    plt.show()


def mylog(*args): 
   
    #import sys 
    print "> Output redirected to scluster.log ..."
    saveout = sys.stdout
    fsock = open('scluster.log', 'w')                             
    sys.stdout = fsock 
    
    print "> eigen values:\n %s" % str(args[0])
    print "> eigen vec:  :\n %s" % str(args[1])
    print "> idx         :\n %s" % str(args[2])
    
    sys.stdout = saveout                                     
    fsock.close()
    return 


def genData(n=150, sample_func=None): 
    """
    Generate *n* cycle samples
    """
    #sample_func = circle_samples
    if sample_func == None:    
        points = circle_samples(n)
    else: 
        try: 
            points = sample_func(n)
        except: 
            err = "[genData] Could not generate data using given sample_func" 
            raise RuntimeError, err
    
    print "[genData] first %d points:\n%s" % (2, points[0:2, :])    
    return points 
    
def getAffinityMatrix(rows, graph_type='knn', 
                      sim=cosine): 
    """
    Given data points (in row-vector format), evaluate
    the corresponding affinity matrix.
    
    @input
      *rows: data (row vector rep) 
      *sim_graph: 
          'knn': mutual knn graph
          'full': fully-connected similarity graph 
      *sim: similarity measure
    """
    #points = genData(n)
    #knn_points = sim_graph(rows)
    #W = get_knn_matrix(knn_points)
    W = None
    if graph_type in ('knn', ):
        sim_graph = mutual_knn
        W = get_knn_matrix(sim_graph(rows, sim=sim))  
    else: 
        W = get_full_matrix(rows, sim=sim)
    return W
    
def getGraphLaplacian(W, glap='r'):
    """
    @input 
      W: affinity matrix 
      glap: graph Laplacian type 
      
    @output
      L: graph laplacian matrix
    """
    sum_w = [sum(Wi) for Wi in W]
    D = np.diag(sum_w)
    L = D - W
         
    if glap.startswith('unn'): 
        return L 
    elif glap.startswith('sym'):
        # [note] how to compute D^(-1/2)? 
        # DT^(-1/2) * L * D^(-1/2)
        D_sqr_inv = np.diag(np.power(sum_w, -0.5)) # [todo]
        return np.dot(D_sqr_inv, np.dot(L, D_sqr_inv)) 
    elif glap in ('r', 'rw', 'random_walk', ): 
        #print "[getGraphLaplacian] L: random walk Laplacian"
        try: 
            D_inv = np.diag(np.power(sum_w, -1.0))
        except Exception, e:   # [debug]
            err  = "[getGraphLaplacian] Could not compute D inverse with sum of w\'s: %s" % str(sum_w)
            err += "                    > %s" % e 
            raise RuntimeError, err
        return np.dot(D_inv, L)
       
    else: 
        raise RuntimeError, "[getGraphLaplacian] Method %s not avail!" % glap
       
    return L

##############################################################
#
#   Clustering algorithms starts here
#     1. kmeans++, fskmeans (frequency-balanced kmeans)
#     2. spectral
###############################################################

def kcluster(rows, distance=similarity.euclidean, k=4, maxIter=150, nCycle=3):
    """
    Run k-means++ multiple times and choose the best result.
    """    
    dmin = np.inf
    centroids_opt = membership_opt = None # [row for i, row in enumerate(rows) if i <= k]
    for _ in range(nCycle): 
        centroids, membership, d = cluster_points_kmeans(rows, nC=k)
        if d < dmin: 
            centroids_opt, membership_opt, dmin = \
                (centroids, membership, d)
    return (centroids, membership, dmin)

# spectral clustering algorithm
def scluster(rows=None, 
             W=None, 
             glap='r',
             nC=10,  
             sim=radialKernel(), 
             sim_graph=mutual_knn, 
             nCycles=3, gtype='knn'):
    """
    Spectral clustering with a given graph Laplacian (*glap)
    
    @input
      rows: data in row-vector format 
      W:    similarity matrix (derived from a given similarity graph)    
      glap: the type of graph Laplacians
             'r': random_walk
             'unnorm': unnormalized 
             'sym': symmetric
      gtype: the graph type
             'full': for fully-connected graph 
             'knn': for mutual knn graph 
             'epsilon': epsilon-neighbor graph (todo)
             
    @output
      a 4-tuple: (_res, idx, evals, Y)
        _res: cluster centroids in eigenspace 
        idx: cluster membership
        evals: eigenvalues 
        Y: eigenvectors (which forms an eigenspace)
    """
    if rows is None and W is None: 
        err = "[scluster] Both data and similarity matrix not given!"
        raise RuntimeError, err     
     
    if W is None: 
        W = getAffinityMatrix(rows, graph_type=gtype, sim=sim)

    print "[scluster::debug] Given W shape:\n%s" % str(W.shape)
    #print "> W[0:20, 0:20] %s" % W[0:20, 0:20]
    L = getGraphLaplacian(W, glap)
    
    #history = []
    distortion_ref = np.inf
    
    # run several cycles and choose the best result
    for _ in range(nCycles): 
        #res, rename_clusters(idx), evals[:15], Y, distortion
        _res, _idx, _evals, _Y, distortion = cluster_points(L, glap, nC)   # [engine]
        if distortion < distortion_ref:
            _res, idx, evals, Y = _res, _idx, _evals, _Y
            distortion_ref = distortion
    
    # [debug]
    print "[scluster] cluster idx:\n > %s" % idx
    print "[scluster] eigenvalues:\n > %s" % evals[:50]
         
    # examine the eigen-gap
    gap_0_1 = abs(evals[0]/(evals[1]+0.0))
    if gap_0_1 > 1e-4:   # [todo]
        print "[scluster] Cluster may be less useful!"
       
    print "[scluster] eigengap 0th-to-1st: %f" % gap_0_1
    print "[scluster] final distortion: %f" % distortion_ref           
  
    return (_res, idx, evals, Y)

def test_scluster(args):
    points = circle_samples()
    knn_points = mutual_knn(points)
    W = get_knn_matrix(knn_points)
    D = np.diag([sum(Wi) for Wi in W])

    # unnormalized graph Laplacian
    L = D - W
    print "\n[main] L:\n%s\n" % str(L) 
    
    res, idx, evals, Y, distortion = cluster_points(L)   # [key]
    
    print "> main(evals):\n%s" % str(evals)
    #mylog(evals, Y, idx)  
        
    #plot_figure(points, evals, Y, idx)
    
    return evals, Y, idx, points

def test_kcluster():
    #import cluster_util 
    X = cluster_util.genData()
    centroids, membership, d = kcluster(X, k=4, maxIter=150, nCycle=10)
    print "> centroids:\n %s" % centroids
    print "> membership:\n %s" % membership
    print "> distortion:\n %s" % d
    
def test_affinity(sim=radialKernel(c=0.2)):  # similarity.euclidean
    import scipy.io as sio
    _data = sio.loadmat("qrecommender_2_l10.mat")
    points = _data['X']
    W = getAffinityMatrix(points, graph_type='full', sim=sim)
    N = W.shape[0]
    print "> %d points" % N
    print "> W[0:20, 0:20]: %s" % W[0:20, 0:20]
    
    # average distance per row 
    Wrow = np.zeros((N, 1))
    for i in range(N):
        Wrow[i] = np.mean(W[i,:])
    print "> row mean: %s" % Wrow
    
    _mean = (sum(sum(W))+0.0)/N**2
    print "> mean:     %s" % _mean
    
def evalKCluster(points=None, outfile=None, nC=10, ptype='X', distance=None):
    import scipy.io as sio
    from cluster_util import ctree
    if distance == None: distance = similarity.pnorm
    assert hasattr(distance, '__call__'), "Invalid distance function"

    if points == None: 
        _data = sio.loadmat(ParamFile)   # "qrecommender_2_l10.mat"
        points = _data[ptype]  # try clustering similar objects

        print "> %s-matrix: %s" % (ptype, str(points.shape))
        #points = points.tolist()
        #points = genData(n=150)
        print "> total # of data: %s" % len(points)
    
    resultset = {}
    centroids, membership, distortion = kcluster(points, distance=distance, k=nC, nCycle=3)
    resultset['membership'] = membership
    resultset['ctree'] = ctree(membership)
    print "> membership: %s" % membership
    resultset['centers'] = centroids
    
    # save data 
    if outfile is None: outfile = "qrecommender_kcluster_%s.mat" % ptype  
    fp = open(outfile, 'wb')  
    pickle.dump(resultset, open(outfile, 'wb'))
    fp.close()
    
    return (membership, centroids)
 
def evalSCluster(points=None, outfile=None, nC=10, ptype='X'):
    """
    @input 
      points: data points (e.g. feature vectors) 
              in row-vector format
    """
    import scipy.io as sio
    from cluster_util import ctree
    
    if points == None: 
        _data = sio.loadmat(ParamFile)   # "qrecommender_2_l10.mat"
        points = _data[ptype]  # try clustering similar questions

        print "> %s-matrix: %s" % (ptype, str(points.shape))
        #points = points.tolist()
        #points = genData(n=150)
        print "> total # of data: %s" % len(points)
         
    resultset = {} #radialKernel(c=0.3)
    W = getAffinityMatrix(points, sim=cosine)  # given data and choice of sim graph (e.g. knn, fully-connected)    
    centroids, idx, evals, Y = scluster(points, W, 'rw', nC=nC)
    resultset['membership'] = idx
    #print "idx: %s" % idx
    resultset['ctree'] = ctree(idx)
    resultset['eigen_values'] = evals
    resultset['centers'] = centroids  # really are "eigen-centers"
    
    print "> centroids:\n %s\n" % centroids
    
    print "> scluster(evals):\n%s" % str(evals)
    plot_figure(points, evals, Y, idx)
    
    # save data 
    if outfile is None: outfile = "qrecommender_cluster_%s.mat" % ptype
    fp = open(outfile, 'wb')  
    pickle.dump(resultset, open(outfile, 'wb'))
    fp.close()

    return (idx, centroids) 

def main():
    #evalSCluster(outfile='qcluster.pkl', ptype='X', nC=15)
    
    evalKCluster(outfile='qkcluster.pkl', ptype='X', nC=15)
    
    # very slow! 
    #evalKCluster(outfile='user_kcluster.mat', ptype='Theta', nC=300)
    
    return 
 
if __name__ == "__main__":
    main()
    #test_affinity()

#    # loggging (see mylog(...))    
#    logToFile = 0
#    if logToFile:
#        print "> Output redirected to scluster.log ..."
#        saveout = sys.stdout
#        fsock = open('scluster.log', 'w')                             
#        sys.stdout = fsock 
#    
#    print "> eigen values:\n %s" % str(evals)
#    print "> eigen vec:  :\n %s" % str(Y)
#    print "> idx         :\n %s" % idx
#    
#    if logToFile:
#        sys.stdout = saveout                                     
#        fsock.close()
        
    
    
    
    