'''
Created on May 22, 2013

@author: bchiu
'''

import os, re, random
import numpy as np
from collections import OrderedDict
        
ROOTDIR = '/Users/pleiades/Documents/workspace/QRecommender/resource'       
DATA_FILE = "astudentData.csv"        
        
class DataStore(object):
    rootdir = None
    header = []  # question_id,user_id,correct
    rMap = {}
    rMatrix = [] # matrix representation of the data (costly)
    rAdjList = {}  # adjacency list rep. of the data; questions as vertices 
    
    def __init__(self, input_file=None, header=True, delimit=',', key_ids=None, val_ids=None, 
                       nParams=10, debug=0): 
        """
        @input
          *input_file: 
          
          *nParams: number of (hidden) parameters used to model each entity 
                   in collaborative filtering algorithm
                   
        @instance variables
          *self.entitySet: holds entity's unique 
                           IDs (e.g. unique user IDs) in the form of 
                           an ordered map, which maps entity ID to 
                           its internal ID in sorted order
                            
          *self.nEntity: keep track of number of entities (e.g. users, questions, etc)
        """    
        if DataStore.rootdir is None: DataStore.rootdir = ROOTDIR 
        if input_file is None: input_file = DATA_FILE
        self.input_file = os.path.join(DataStore.rootdir, input_file)

        # [debug] 
        # self.dupSet = set([])
        self.errSet = set([])
        self.nDup, self.nErr = (0, 0)
        self.debug = debug  # debug level
        
        # keep track of unique question IDs and user IDs
        # self.questions = set([]); self.users = set([])
        # [note] entity tracking is made generic to deal with more general dataset later 
        self.entitySet = [] 
        self.inv_entitySet = [] # [debug]
        self.nEntity = [] # keep track of number of entities (e.g. users, questions, etc)
        
        self.loadToTable(self.input_file, header, delimit, key_ids, val_ids)
        
        # number of (hidden) parameters 
        self.nParams = nParams
        #self.transform()
        
        
    def loadToTable(self, input_file, header=True, delimit=',', key_ids=None, val_ids=None, track_ids=True):
        """
        Load data to a map in which 
           key: (question id, user id) 
           value: correctness
        Also checks if there are duplicate entries and inconsistent data (e.g. 
           same question-user combination with different correctness scores)
        
        """
        def integrity_check(entry, row):
            err_level = 0 
            if rMap.has_key(entry):
                # has duplicate entry
                self.nDup += 1
                err_level += 1
                #self.dupSet.add(entry)
                if self.debug > 1: 
                    msg = "[loadToTable] found duplicate on %s" % str(entry)
                    print msg 
                    # raise ValueError, "duplicate on %s" % str(entry)
                
                # is result consistent? 
                if rMap[entry] != row[-1]: 
                    # discard inconsistent data 
                    self.nErr += 1
                    self.errSet.add(entry)
                    msg = "[loadToTable] Correctness not consistent for: %s" % str(entry)
                    if self.debug > 0: print msg
                    err_level += 1
    
            return err_level

        if key_ids is None: key_ids = [0, 1]  
        self.entitySet = [OrderedDict({}) for _ in key_ids] 
        
        rMap = DataStore.rMap
        #qset, uset = (set([]), set([]))
        for i, line in enumerate(file(input_file)):
            row = line.split(delimit)
            if header and i == 0: 
                DataStore.header = [e.strip() for e in row]; continue
            else: 
                row = [int(e) for e in row]  # [domain] specific
                
            entry = tuple([row[j] for j in key_ids])
            st = integrity_check(entry, row)
            if st == 0: 
                rMap[entry] = row[-1]
            elif st == 2:  # dup & inconsistent 
                del rMap[entry]   # delete existing entry

            # also keep track of unique ID
            # qset.add(row[0]); uset.add(row[1])
            if track_ids: 
                for k in key_ids: 
                    self.entitySet[k][row[k]] = None
            
        #print "[loadToTable] # rows: %s" % i
        # size of each key/ID representing a particular entity (e.g. user, question)
        self.nEntity = [len(self.entitySet[j]) for j in key_ids] 
        self._transform_sortIDs()
        # remove temp data
        
        return
    
    # [specialized]
    def _transform_sortIDs(self):
        """
        Build a mapping from the true ID of the entity (e.g. users) 
        to the internal ID (in the sorted order of the true ID). 
        
        Note that this operation is unnecessary if the underlying 
        data structure for set (and dictionary) were implemented via 
        balanced search tree as that in C++. 
        
        [reference] bintrees 0.3.0
        """   
        nE = len(self.entitySet) # number of entities
        for i in range(nE): # foreach entity of interest (e.g. question, user)
            idlist = list(self.entitySet[i].keys())
            idlist.sort()
            invMap = OrderedDict({}) 
            self.entitySet[i].clear()
            for j, _id in enumerate(idlist): 
                self.entitySet[i][_id] = j
                invMap[j] = _id
            self.inv_entitySet.append(invMap)
        
        return
    
    def map_eff_to_true_id(self):
        return self.inv_entitySet # from effective ID to true ID
    
    def toTrueId(self, effId, axis=0):
        """
        @input
          *axis: 
          
        Given an effective ID, find its true ID.
        """
        return self.inv_entitySet[axis][effId]
    
    def toEffId(self, trueId, axis=0):
        return self.entitySet[axis][trueId]
            
    def getIDs(self, axis=0):
        return self.entitySet[axis].keys()
    
    def getEffIDs(self, axis=0):
        return self.entitySet[axis].values()

    def summary(self):
        msg = "\n----------- SUMMARY -----------\n"
        msg += "0. header\n"
        msg += "   %s\n" % DataStore.header
        msg += "1. stats\n"
        msg += "   # of questions: %d\n" % self.nEntity[0]
        msg += "   # of users:     %d\n" % self.nEntity[1]
        msg += "2. error report\n"
        msg += "   # of inconsistent data: %d\n" % self.nErr
        msg += "   # of dup: %d\n" % self.nDup
        #msg += "   dup set:\n   %s\n" % str(self.dupSet)
        msg += "   inconsistent set:\n   %s\n" % str(self.errSet)
        print msg
        return
        
    @staticmethod
    def loadToTable2(input_file, header=True, delimit=',', key_ids=None, val_ids=None, track_ids=True):
        """
        Load input file to a dictionary where 
             key: a tuple of (question ID, user ID) 
             value: correctness of the answer (1 or 0)
        """     
        if key_ids is None: key_ids = [0, 1]
        #if val_ids is None: val_ids = [-1]

        for i, line in enumerate(file(input_file)):
            row = line.split(delimit)
            if header and i == 0: 
                DataStore.header = row[:]; continue 
            entry = tuple([int(row[j]) for j in key_ids])

            DataStore.rMap[entry] = row[-1]
        return
       
    # [specialized]
    def transform(self, key_id, init_guess=None, to_array=True):
        """
        Transform data into matrix form. 
        """
        N = self.nEntity[key_id]
        if init_guess is None: 
            init_guess = [0.0] * self.nParams
        X = [init_guess for i in range(N)]
        if to_array: 
            return np.array(X)
        return X
    
    def incidence(self, key_ids=None, to_array=True, debug=False, 
                  _Yna=0.0):
        """
        @input 
          *k1 and *k2 are the entity IDs; together they form 
              the key to the entry of *rMap
              (e.g. a data set that maps question and user IDs to 
                   their corresponding correctness label 
                   will have (question ID, student ID) as key 
                   of *rMap and the value is the correctness)
              In general, the key may not have only two indices
              *k1 and *k2 are therefore used to indicate the 
              entity IDs of interest. 
        
        Create incidence matrix between two entities 
        indexed by (*k1, *k2)
        
        @output 
          a tuple of (R, Y) where
             both R and Y are of the same size 
             
              
             R[i,j] = 1 if ith question was answered by user j  
             Y[i,j] is the corresponding correctness with value
             in {1, 0}
        """
        if key_ids is None: key_ids = [0, 1]
        k1, k2 = key_ids
        
        Y, R = ([], [])
        for i in range(self.nEntity[k1]):
            Y.append([_Yna] * self.nEntity[k2])
            R.append([0] * self.nEntity[k2])
        
        # keep track of questions being answered by users
        for i in range(self.nEntity[k1]):
            ti = self.toTrueId(i, axis=k1)
            for j in range(self.nEntity[k2]):
                tj = self.toTrueId(j, axis=k2)
                if self.rMap.has_key((ti,tj)): 
                    R[i][j] = 1  
                    Y[i][j] = self.rMap[(ti, tj)] 
        
        if debug: 
            self._check_incidence(Y=Y, R=R)             
        if to_array:
            return (np.array(Y), np.array(R))
        return (Y, R)   
    
    # [debug]
    def _check_incidence(self, key_ids=None, Y=None, R=None):
        if key_ids is None: key_ids = [0, 1]
        i, j = key_ids
        
        if None in (Y, R): Y, R = self.incidence(to_array=True)
        else: Y = np.array(Y); R = np.array(R)
        
        for k, v in self.rMap.items(): 
            ei, ej = (self.toEffId(k[i], i), self.toEffId(k[j], j))
            assert v == Y[ei, ej], \
                  "[check_incidence] error in Y: Y[%d,%d]=%s while rMap[%s]=%f" % \
                       (ei, ej, Y[ei, ej], str(k), v)
            if v: 
                assert R[ei, ej] == 1, \
                   "[check_incidence] error in R"
        return        
        
               
    def loadToMatrix(self, input_file, header=True, delimit=',', root_dir=None):
        if root_dir is None: root_dir = DataStore.rootdir
        pass
    
    def save(self):
        """
        Save data to file for later use. 
        """
        import scipy.io as sio
        pat = re.compile('(\w+)\.\w+')
        try: 
            filebase = pat.match(DATA_FILE).group(1)
        except: 
            filebase = "datastore"
        ds = {}
        ds['idmap'] = self.map_eff_to_true_id()
        Y, R = self.incidence()
        ds['Y'] = Y; ds['R'] = R
        sio.savemat(filebase + '.mat', ds)
        
        return
        
    def missing_ID_check(self):
        """
        Check if any {user,question} ID is missing as a result of removing
        inconsistent data set
        """
        pass
    
    def queryRMap(self, qID, uID):
        """
        Look up correctness score
        """
        val = None
        try: 
            val = DataStore.rMap[(qID, uID)]
        except: 
            pass 
#            msg = ""
#            if not uID in self.entitySet[1]: 
#                msg = "[query] user %d does not exist\n"
#            if not qID in self.entitySet[0]: 
#                msg = "[query] question %d does not exist\n"
#            if msg: 
#                raise RuntimeError, msg
#            else: 
#                raise RuntimeError, "[query] Invalid combo (qID, uID): %s\n" % (qID, uID) 
        return val    
        
    def C(self, x, y):
        return self.queryRMap(x, y)

    def CEff(self, x, y):
        return self.queryRMap(self.toTrueId(x, axis=0), 
                              self.toTrueId(y, axis=1))
        
        
# helper class
class QueryTable(object):
    pass
            
  
if __name__ == "__main__": 
    dstore = DataStore(input_file="astudentData.csv")
    dstore.summary()


    qIDs = dstore.getIDs(axis=0)
    uIDs = dstore.getIDs(axis=1)
    
    print "> n(qID): %d, max: %d, qIDs: %s\n" % (len(qIDs), max(qIDs), qIDs[:100])
    print "> eff_qIDs: %s" % dstore.getEffIDs(axis=0)[:100]
    print "> qIDs    : %s" % dstore.getIDs(axis=0)[:100]
    
    print "> n(uID): %d, max: %d, uIDs: %s\n" % (len(uIDs), max(uIDs), uIDs[:100])
    print "> eff_uIDs: %s" % dstore.getEffIDs(axis=1)[:100]
    print "> uIDs    : %s" % dstore.getIDs(axis=1)[:100]

    print "> %s" % dstore.C(1143,20007)
    
    X = dstore.transform(1)
    Xm, Xn = X.shape; k = random.randint(0, Xm)
    print "> dim(X): (%d, %d)" % (Xm, Xn)
    print "> [%d:] = %s\n" % (k, X[k])
    Q = dstore.transform(0)
    Qm, Qn = Q.shape; k = random.randint(0, Qm)
    print "> dim(Q): (%d, %d)" % (Qm, Qn)
    print "> Q[%d:] = %s\n" % (k, Q[k])
    
    Y, R = dstore.incidence(debug=True)
    Ym, Yn = Y.shape 
    Rm, Rn = R.shape 
    assert Ym == Rm and Yn == Rn, "dim inconsistency Y vs R"
    print "> dim(Y) = dim(R) = (%d, %d)" % (Ym, Yn)
    
    print "> %s" % Y[1][:500] 
    print "> %s" % R[1][:500]
    
    
    
    
    
    