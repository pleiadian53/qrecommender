
#from math import sqrt 

import math
import numpy as np
#from numpy import matrix
from numpy.linalg import solve, cholesky
from utils import chol
#from cholesky_solve import chol


########################################
#
#  Distance/Norm/Energy Functions   
#
#
######################################## 

def euclidean(v1,v2):
    """
    Compute euclidean distance 
    """
    #v1 = np.array(v1)
    #v2 = np.array(v2)
    return np.linalg.norm(v1-v2, 2)

def sim_euclidean(v1, v2):
    return 1.0/(1+np.linalg.norm(v1-v2, 2))  

def pnorm(v1, v2, _ord=1.5):
    return np.linalg.norm(v1-v2, _ord)  

def sim_pnorm(v1, v2, _ord=1.5):
    return 1.0/(1+np.linalg.norm(v1-v2, _ord))


###############################################
#
#  Similarity Measures and 
#    distance function
#  
#  note: distance function is essentially a 
#        'dissimilarity' measure
###############################################



def dim(x):
    """
    x: row vector of ndarray type
    """
    assert type(x) == np.ndarray
    
    return len(x.ravel())


def mahalanobis(x1, x2, K):     
    """
    [input] 
      K: data (from which x1, x2 were drawn) covar matrix
    
    Calculate Mahalanobis distance (or quadratic distance) 
    between two points in R^n (i.e. the general dot product of x1 and x2
    wrt matrix K;
    """
    assert type(x1) == type(x2) == np.ndarray
    d = dim(x1)
    if d != dim(x2): 
       raise RuntimeError( 
             """Data points do not have the same dimension: %d and %d"""
         % ( d, dim(x2) ) )
 
    n, m = K.shape
    if (n != m) or (n != d): 
       raise RuntimeError("""Error: Inconsistent dimensionality! """  
                          """Expecting %d-by-%d convariance matrix but given %d-by-%d"""
                          % (d, d, n, m) )
     
    #from numpy.linalg import             
    try: 
       #print "[chol] K: %s" % K
       #print "[chol] chol(K): %s" % chol(K)
    
       L = np.matrix(chol(K))   # result is upper-triangular matrix
       print "[chol] L:\n %s" % L
        
    except:
       from numpy.linalg import eig
       vals = eig(K)[0]
    
       raise RuntimeError( """Cholesky decomposition of covariance """
                          """matrix failed. Your kernel may not be positive """
                 """definite. Eigenvalues of K: %s""" % vals ) 
           
    # alpha = solve(L.T, solve(self.L,x2)); [note] solve returns ndarray type
    
    #print "[chol] L: %s, type(L): %s" % (L, type(L)) 
    print "[chol] K:\n %s" % np.dot(L.T, L)
    print "[chol] inv(K) * (x1 - x2):\n %s" % solve(L, solve(L.T, x1-x2))
    return math.sqrt( np.dot(x1-x2, solve(L, solve(L.T, x1-x2)))  )
    
def radialKernel(c=1.5):
    def inner(a, b):
        d = a - b
        return np.exp((-1 * (np.sqrt(np.dot(d, d.conj()))**2)) / c)
    return inner

def radialKernel2(delta, c=1.5):
    return np.exp((-1.0 * delta) / c)

def gaussianKernel(v1, v2, tau=0.01):
    return np.exp(-1 * (sum((v1 - v2) ** 2)/2 * (tau ** 2)))

def cosine(a, b):
    return np.dot(a, b)/(np.dot(a, a.T) * np.dot(b, b.T))

  
def pearson(v1,v2):
    """
    Compute Pearson correlation score  
    Output range: 1 - [-1, 1]
    """
    # Simple sums
    sum1=sum(v1)
    sum2=sum(v2)
  
    # Sums of the squares
    sum1Sq=sum([pow(v,2) for v in v1])
    sum2Sq=sum([pow(v,2) for v in v2])    
  
    # Sum of the products
    pSum=sum([v1[i]*v2[i] for i in range(len(v1))])
  
    # Calculate r (Pearson score)
    num=pSum-(sum1*sum2/len(v1))
    den=math.sqrt((sum1Sq-pow(sum1,2)/len(v1))*(sum2Sq-pow(sum2,2)/len(v1)))
    if den==0: return 0

    return 1.0-num/den
  
def pseudoShift(v):
    for i, e in enumerate(v): 
        if e == 0: 
            v[i] = e + 0.01
    return v 
    
def euclideanScore(v1, v2, scaledown=1): 
    """
    Compute similarity in euclidean distance 
    Output range: [0, 1]; the larger the more disimilar 
    """
    distance = euclidean(v1, v2) 
    return 1.0 - 1/(1 + distance * scaledown)
       
def tanamoto(v1,v2):
    c1,c2,shr=0,0,0
  
    for i in range(len(v1)):
        if v1[i]!=0: c1+=1 # in v1
        if v2[i]!=0: c2+=1 # in v2
        if v1[i]!=0 and v2[i]!=0: shr+=1 # in both
  
    return 1.0-(float(shr)/(c1+c2-shr))    
  
  
  
if __name__ == "__main__": 
   
   from numpy import array, mat
   
   x1 = array([1.0, 3.0, 5.0])
   x2 = array([1.5, 3.5, 5.0])
   K = mat('[5.0, 0.1, 0.7; 0.3, 4.0, 0.6; 0.2, 0.5, 10.0]')
   
   print "> given x1: %s" % x1
   print "> given x2: %s" % x2
   print "> given K:\n %s" % K
   
   d = mahalanobis(x1, x2, K)
   print "> d(x1, x2, K): %f" % d

   import scipy.spatial.distance as sd
   from numpy.linalg import inv
   d = sd.mahalanobis(x1, x2, inv(K))
   print "> d(x1, x2, K, scipy): %f" % d
   
   #print "> inv(K): %s" % inv(K)
   delta = np.matrix(x1-x2)
   #print "> delta: %s" % delta.T
   #K_inv = numpy.matrix(K)
   
   # [note] K_inv * delta.T  cannot use with %s? 
   print "> inv(K) * (x1 - x2):\n ", inv(K) * delta.T
   
   d = euclidean(x1, x2)
   print "> d(x1, x2, 'euclidean'): %f" % d
   

