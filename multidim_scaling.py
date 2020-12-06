
import random, math, os
from PIL import Image,ImageDraw
import numpy as np
from similarity import pearson, euclidean, cosine
import scipy.io as sio 
from DataStore import DataStore

DataRoot = "/Users/pleiades/Documents/workspace/QRecommender/src/qrecommender"
PFile = 'qrecommender_2_l10.mat'
DFile = "astudentData.csv"


def scaledown(data,distance=pearson,rate=0.01):
  n=len(data)

  # The real distances between every pair of items
  realdist=[[distance(data[i],data[j]) for j in range(n)] 
             for i in range(0,n)]

  # Randomly initialize the starting points of the locations in 2D
  loc=[[random.random(),random.random()] for i in range(n)]
  fakedist=[[0.0 for j in range(n)] for i in range(n)]
  
  lasterror=None
  for m in range(0,1000):
    # Find projected distances
    for i in range(n):
      for j in range(n):
        fakedist[i][j]=math.sqrt(sum([pow(loc[i][x]-loc[j][x],2) 
                                 for x in range(len(loc[i]))]))
  
    # Move points
    grad=[[0.0,0.0] for i in range(n)]
    
    totalerror=0
    for k in range(n):
      for j in range(n):
        if j==k: continue
        # The error is percent difference between the distances
        errorterm=(fakedist[j][k]-realdist[j][k])/realdist[j][k]
        
        # Each point needs to be moved away from or towards the other
        # point in proportion to how much error it has
        grad[k][0]+=((loc[k][0]-loc[j][0])/fakedist[j][k])*errorterm
        grad[k][1]+=((loc[k][1]-loc[j][1])/fakedist[j][k])*errorterm

        # Keep track of the total error
        totalerror+=abs(errorterm)
    print totalerror

    # If the answer got worse by moving the points, we are done
    if lasterror and lasterror<totalerror: break
    lasterror=totalerror
    
    # Move each of the points by the learning rate times the gradient
    for k in range(n):
        loc[k][0]-=rate*grad[k][0]
        loc[k][1]-=rate*grad[k][1]

  return loc

def draw2d(data,labels,jpeg='mds2d.jpg'):
    img=Image.new('RGB',(2000,2000),(255,255,255))
    draw=ImageDraw.Draw(img)
    for i in range(len(data)):
        x=(data[i][0]+0.5)*1000
        y=(data[i][1]+0.5)*1000
        draw.text((x,y),labels[i],(0,0,0))
    img.save(jpeg,'JPEG') 
  
  
def plot_questions(_file=None, _dataset=None):
    """
    Show relationship of questions in 2D.
    """
    from DataStore import DataStore
    if _file is None: _file = PFile # [hardcode] 
    _params = sio.loadmat(os.path.join(DataRoot, _file)) 
    X = _params['X']
    nQ, nF = X.shape
    mapped_loc = scaledown(X)
    print "> # of q-points: %s" % len(mapped_loc)

    # load original data set to obtain ID maps
    if _dataset is None: _dataset = DFile
    ds = DataStore(input_file=_dataset, nParams=nF)
    idmap = ds.map_eff_to_true_id()
    assert len(idmap) == 2, "not two maps? one for questions and one for users"

    try: 
        qidmap = qidmap = idmap[0]
    except: 
        msg = "Could not load question idmap.\n"; print msg
        ds.summary() 
    if qidmap: 
        qIds = ['Q%s' % v for _, v in qidmap.items()]
    else: 
        qIds = ['Q%s' % i for i in range(nQ)]

    draw2d(mapped_loc, qIds, jpeg='questions2d.jpg')
    
    return

def plot_users(_file=None, _dataset=None):
    """
    Show relationship of users in 2D.
    """
    if _file is None: _file = PFile # [hardcode] 
    _data = sio.loadmat(os.path.join(DataRoot, _file)) 
  
    Theta = _data['Theta']
    nU, nF = Theta.shape
    mapped_loc = scaledown(Theta)
    print "> # of users: %s" % len(mapped_loc)

    # load original data set to obtain ID maps
    if _dataset is None: _dataset = DFile
    ds = DataStore(input_file=_dataset, nParams=nF)
    idmap = ds.map_eff_to_true_id()
    assert len(idmap) == 2, "not two maps? one for questions and one for users"

    try: 
        uidmap = idmap[1]
    except: 
        msg = "Could not load user idmap.\n"; print msg
        ds.summary() 
    if uidmap: 
        uIds = ['U%s' % v for _, v in uidmap.items()]
    else: 
        uIds = ['U%s' % i for i in range(nU)]
    
    draw2d(mapped_loc, uIds, jpeg='users2d.jpg')
    
    return
  
if __name__ == "__main__":
    #plot_questions()
    plot_users()
