
import random
import numpy as np

HFile = 'genetic_%s.mat'


def has_dup(alist): # duplicate questions? 
    aset = set([])
    tval = False
    for _e in alist:
        if not _e in aset:
            aset.add(_e)
        else: 
            tval = True; break
    return tval

def genetic_optimize(domain,fitness,popsize=50,step=1,
                    mutprob=0.2,elite=0.2,maxiter=1000,
                    _outfile=HFile,
                    resolve=None,
                    _order='desc',
                    gsize=None,
                    _ntrial=0):
    """
    @input 
      domain: domain-specific range for each coordinate of 
               the solution vector
      fitness:  fitness function 
      popsize: number of solutions from which to select elites
      step:   increment on a particular coordinate in the solution 
              vector; used for mutation op. 
      mutprob: probability of mutatation
      elite: fraction of good solutions from population
      maxiter: max. allowable iterations
      resolve: a function that resolve (obvious) conflicts 
      _order:
      _outfile: 
      _ntrial:
      
    """
    # Mutation Operation
    def mutate(vec):
        i=random.randint(0,len(domain)-1)
        if random.random()<0.5 and vec[i]>domain[i][0]:  
            return vec[0:i]+[vec[i]-step]+vec[i+1:] 
        elif vec[i]<domain[i][1]:
            return vec[0:i]+[vec[i]+step]+vec[i+1:]
        return vec
    
    # resolve duplicates to speed up instead of incurring penalty
    def resolve_dup(vec):  # expensive, avoid it
        for i in range(0, len(vec), gsize):
            group = [vec[j] for j in range(i, i+gsize)]
            aset = set([])
            dupids = []
            for k in range(i, i+gsize):
                _e = vec[k]
                if _e in aset:  # dup!
                    dupids.append(k)
                else: 
                    aset.add(_e) 

            # replace duplicates
            if dupids: # [todo]
                others = set(range(domain[0][0],domain[0][1]))-set(group)
                nelems = random.sample(list(others), len(dupids))
                #print "> nelems: %s, dupids: %s" % (nelems, dupids)
                for k, dupid in enumerate(dupids): 
                    vec[dupid] = nelems[k]
            
            #assert not has_dup([vec[j] for j in range(i, i+gsize)]), "dup: %s" % [vec[j] for j in range(i, i+gsize)]
        return vec     
    
    # Crossover Operation
    def crossover(r1,r2):
        i=random.randint(1,len(domain)-2)
        return r1[0:i]+r2[i:]

    print "[debug] domain: %s" % domain[:2]
    
    # Build the initial population
    pop=[]
    for i in range(popsize):
        vec=[random.randint(domain[i][0],domain[i][1]) 
               for i in range(len(domain))]
        pop.append(vec)
  
    print "[debug] size(pop): %s, sample sol: %s" % (len(pop), pop[1])
    for v in pop: 
        assert type(v) == type([]), "non-list in pop? v" % str(v)
    print "[debug] fitness: %s" % [fitness(v) for i, v in enumerate(pop) if i<10]
    
    # How many winners from each generation?
    # > select a fraction of population as elites
    topelite=int(elite * popsize)
  
    # Main loop 
    score_history = []
    #group_score = []
    for i in range(maxiter):
        scores=[(fitness(v),v) for v in pop]
        if _order in ('desc'): 
            scores.sort(reverse=True)  # i.e. the higher the better
        else: 
            scores.sort()  # ascending order
        ranked=[v for (s,v) in scores]
    
        # Start with the pure winners
        pop=ranked[0:topelite]
    
        # Add mutated and bred forms of the winners
        while len(pop) < popsize:
            if random.random()< mutprob:
                # Mutation
                c=random.randint(0, topelite)# 
                #assert type(mutated_sol) == type([])
                pop.append(mutate(ranked[c]))
            else:
                # Crossover
                c1=random.randint(0,topelite)
                c2=random.randint(0,topelite)
                #assert type(cross_sol) == type([])
                pop.append(crossover(ranked[c1],ranked[c2]))
    
        # Print current best score
        print "[iter=%d] max fitness: %s" % (i, scores[0][0])
        score_history.append( np.mean([scores[i][0] for i in range(min(popsize/10, 5))]) )
        
    if _outfile:
        import scipy.io as sio
        _data = {}; _data['scores'] = score_history
        sio.savemat(_outfile % _ntrial, _data)  # [todo]
    
    return scores 
