from math import isnan
import numpy, gc
from numpy import isnan, isinf, isreal, any, dot, sqrt, concatenate

# Minimize a differentiable multivariate function. 
#
# Usage: [X, fX, i] = minimize(X, f, length, P1, P2, P3, ... )
#
# where the starting point is given by "X" (D by 1), and the function named in
# the string "f", must return a function value and a vector of partial
# derivatives of f wrt X, the "length" gives the length of the run: if it is
# positive, it gives the maximum number of line searches, if negative its
# absolute gives the maximum allowed number of function evaluations. You can
# (optionally) give "length" a second component, which will indicate the
# reduction in function value to be expected in the first line-search (defaults
# to 1.0). The parameters P1, P2, P3, ... are passed on to the function f.
#
# [Q] # of line searches vs function evaluations
#
# [note] 1. Here, f refers to gpr: 
#           [mu, S2] = gpr(loghyper, covfunc, x, y, xstar)
#
# The function returns when either its length is up, or if no further progress
# can be made (ie, we are at a (local) minimum, or so close that due to
# numerical problems, we cannot get any closer). NOTE: If the function
# terminates within a few iterations, it could be an indication that the
# function values and derivatives are not consistent (ie, there may be a bug in
# the implementation of your "f" function). The function returns the found
# solution "X", a vector of function values "fX" indicating the progress made
# and "i" the number of iterations (line searches or function evaluations,
# depending on the sign of "length") used.
#
# The Polack-Ribiere flavour of conjugate gradients is used to compute search
# directions, and a line search using quadratic and cubic polynomial
# approximations and the Wolfe-Powell stopping criteria is used together
# with
# the slope ratio method for guessing initial step sizes. Additionally a bunch
# of checks are made to make sure that exploration is taking place and that
# extrapolation will not be unboundedly large.
#
# See also: checkgrad 
#
# Copyright (C) 2001 - 2006 by Carl Edward Rasmussen (2006-09-08).


#function [X, fX, i] = minimize(X, f, length, varargin)

INT = 0.1;     #don't reevaluate within 0.1 of the limit of the current bracket
EXT = 3.0;                  # extrapolate maximum 3 times the current step-size
MAX = 20;                         # max 20 function evaluations per line search
RATIO = 100;                                       # maximum allowed slope ratio
#SIG = 0.1; RHO = SIG/2.0; 
RHO = 0.01;                            # a bunch of constants for line searches
SIG = 0.1;       # RHO and SIG are the constants in the Wolfe-Powell conditions


# SIG and RHO are the constants controlling the Wolfe-
# Powell conditions. SIG is the maximum allowed absolute ratio between
# previous and new slopes (derivatives in the search direction), thus setting
# SIG to low (positive) values forces higher precision in the line-searches.
# RHO is the minimum allowed fraction of the expected (from the slope at the
# initial point in the linesearch). Constants must satisfy 0 < RHO < SIG < 1.
# Tuning of SIG (depending on the nature of the function to be optimized) may
# speed up the minimization; it is probably not worth playing much with RHO.

# The code falls naturally into 3 parts, after the initial line search is
# started in the direction of steepest descent. 1) we first enter a while loop
# which uses point 1 (p1) and (p2) to compute an extrapolation (p3), until we
# have extrapolated far enough (Wolfe-Powell conditions). 2) if necessary, we
# enter the second loop which takes p2, p3 and p4 chooses the subinterval
# containing a (local) minimum, and interpolates it, unil an acceptable point
# is found (Wolfe-Powell conditions). Note, that points are always maintained
# in order p0 <= p1 <= p2 < p3 < p4. 3) compute a new search direction using
# conjugate gradients (Polack-Ribiere flavour), or revert to steepest if there
# was a problem in the previous line-search. Return the best value so far, if
# two consecutive line-searches fail, or whenever we run out of function
# evaluations or line-searches. During extrapolation, the "f" function may fail
# either with an error or returning Nan or Inf, and minimize should handle this
# gracefully.


def minimize(J, init_params, gradJ, length, *args): 
    """
    @input
      *init_params (or X): initial guess 

      *args: arguments passed to f  
    """
    
    # red: reduction in function value
    if len(length) == 2: 
       red, length = (length[1], length[0])
    else: 
       red, length = (1, length[0])   
    
    if length>0: 
       S='Linesearch'
    else: 
       S='Function evaluation' 

    i = 0                                     # zero the run length counter
    ls_failed = 0                             # no previous line search has failed

    # return -LL, -dLL/dlog_p with varargin: k(), x, y 
    X = init_params   # [todo]
   
    f0, df0 = (J(init_params), gradJ(init_params))     # get function value and gradient

    # [debug]
    try: 
       print "[minimizer] > initial guess: ", init_params;  
       print "[minimizer] > length, reduction: <%s, %s>" % (length, red)
       print "            > f0 =      %s" % f0
       print "            > df0 = %s" % str(df0) 
    except: 
       pass
    
    fX = []
    fX.append(f0);
    
    if length<0: i = i + 1                 # count epochs?!
    s = -df0; d0 = -dot(s, s)              # -s'*s; initial search direction (steepest) and slope
    x3 = red/(1.0-d0)                                  # initial step is red/(|s|+1)

    # [debug]
    # initial stop criterion
    nD = len(init_params)
    cnt = 0
    for e in df0: 
        if e < 1e-5: cnt +=1
    if (cnt/(nD+0.0) > 0.5) and numpy.linalg.norm(df0) <= 1e-1:
       print "[minimize] early termination ..."
       return (X, fX, 0)  

    while i < abs(length):                                 # while not finished
        
          if length>0: i = i + 1                             # count iterations?!

          X0 = X; F0 = f0; dF0 = df0                        # make a copy of current values
          
          # [debug]
#          if i in (abs(length) - 1): 
#             print "[minimizer] @iter=%d X0 = %s" % (i, str(X))                           
#             print "                     f0 = %s" % str(f0) 
#             print "                     df0= %s\n" % str(df0) 
          
          if length > 0: 
             M = MAX
          else:
             M = min(MAX, -length-i)

          while 1:                             # keep extrapolating as long as necessary
             x2 = 0; f2 = f0; d2 = d0; f3 = f0; df3 = df0
             success = 0
             while not success and M > 0:
                 try: 
                     M = M - 1
                     
                     if length < 0: i = i + 1                    # count epochs?!
        
                     #[f3 df3] = feval(f, X+x3*s, varargin{:});
                     # gp.set_params( X+x3*s ); f3, df3 = gp()
                     _params = X+x3*s
                     f3, df3 = (J( _params ), gradJ( _params ))            # change params in negative-grad direction
                     
                     if isnan(f3) or isinf(f3) or any(isnan(df3)+isinf(df3)): 
                          raise ValueError  # "Unable to compute a valid LL and gradient (NaN or Inf occurred)"
                     success = 1
                 except:                                # catch any error which occured in f
                     x3 = (x2+x3)/2.0                   # bisect and try again
              
             if f3 < F0:
                 X0 = X+x3*s; F0 = f3; dF0 = df3        # keep best values
             d3 = dot(df3, s)                           # d3 = df3'*s; new slope
             if (d3 > SIG*d0) or (f3 > f0+x3*RHO*d0) or M == 0:    # are we done extrapolating?
                break
    
             x1 = x2; f1 = f2; d1 = d2                         # move point 2 to point 1
             x2 = x3; f2 = f3; d2 = d3                         # move point 3 to point 2
             A = 6*(f1-f2)+3*(d2+d1)*(x2-x1)                 # make cubic extrapolation
             B = 3*(f2-f1)-(2*d1+d2)*(x2-x1)
             
             try: 
                x3 = x1 - (d1*(x2-x1)**2)/( B + sqrt(B*B-A*d1*(x2-x1))); # num. error possible, ok!
             except: 
                # [note] B*B-A*d1*(x2-x1) may become < 0
                #print "B: %s, A: %s" % (B, A)
                #print "d1: %s, (x1,x2)=(%s, %s)" % (d1, x1, x2)
                #print "B*B-A*d1*(x2-x1) = %s" % str(B*B-A*d1*(x2-x1))
                #import sys; sys.exit(1)
                x3 = x1 - (d1*(x2-x1)**2)/(B + 0.0)
             
             if not isreal(x3) or isnan(x3) or isinf(x3) or x3 < 0:   # num prob | wrong sign?
                x3 = x2*EXT                                           # extrapolate maximum amount
             elif x3 > x2*EXT:                                  # new point beyond extrapolation limit?
                x3 = x2*EXT                                      # extrapolate maximum amount
             elif x3 < x2+INT*(x2-x1):                        # new point too close to previous point?
                x3 = x2+INT*(x2-x1)
          # end extrapolation-while                           
                                                         # end extrapolation

          while ( (abs(d3) > -SIG*d0) or (f3 > f0+x3*RHO*d0) ) and M > 0:       # keep interpolating
             if d3 > 0 or f3 > f0+x3*RHO*d0:                         # choose subinterval
                   x4 = x3; f4 = f3; d4 = d3                      # move point 3 to point 4
             else:
                   x2 = x3; f2 = f3; d2 = d3                      # move point 3 to point 2

             if f4 > f0:           
                   x3 = x2-(0.5*d2*(x4-x2)**2)/(f4-f2-d2*(x4-x2));  # quadratic interpolation
             else:
                   A = 6*(f2-f4)/(x4-x2+0.0)+3*(d4+d2)                    # cubic interpolation
                   B = 3*(f4-f2)-(2*d2+d4)*(x4-x2)
                   x3 = x2+(sqrt(B*B-A*d2*(x4-x2)**2)-B)/A             # num. error possible, ok!
                 
             if isnan(x3) or isinf(x3):
                   x3 = (x2+x4)/2.0                     # if we had a numerical problem then bisect
                
             x3 = max(min(x3, x4-INT*(x4-x2)),x2+INT*(x4-x2))          # don't accept too close
             
             #[f3 df3] = feval(f, X+x3*s, varargin{:});
             # [debug]
             _params = X+x3*s
             try:  
                 _params = X+x3*s
                 f3, df3 = (J( _params ), gradJ( _params ))    
             except: 
                 # try bisect again
                 x3 = (x2+x4)/(2.0 + numpy.random.rand())
                 x3 = max(min(x3, x4-INT*(x4-x2)),x2+INT*(x4-x2))
                 _pp = X+x3*s
                 try: 
                    f3, df3 = (J( _pp ), gradJ( _pp ))
                 except: 
                    f3 = numpy.inf
             
             if f3 < F0:
                X0 = X+x3*s; F0 = f3; dF0 = df3                       # keep best values
             
             M = M - 1
             if length<0: i = i + 1;                                     # count epochs?!
             d3 = dot(df3,s)                                           #d3 = df3'*s; new slope
          # end interpolation-while

          if ( abs(d3) < -SIG*d0 ) and (f3 < f0+x3*RHO*d0):          # if line search succeeded
             X = X+x3*s; f0 = f3                                     # update variables
             
             #fX = [fX' f0]'; 
             fX.append(f0)
             
             #if i % 10 == 0: print "%s %6i |   Value: %4.6e" % (S, i, f0)
             
             #s = (df3'*df3-df0'*df3)/(df0'*df0)*s - df3;          # Polack-Ribiere CG direction
             s = ( (dot(df3,df3) - dot(df0, df3))/dot(df0,df0)*s ) - df3
             
             df0 = df3                                                 # swap derivatives
             d3 = d0; d0 = dot(df0,s)
             if d0 > 0:                                      # new slope must be negative
                 s = -df0; d0 = -dot(s,s)                     # otherwise use steepest direction
                 
             realmin = numpy.finfo(numpy.double).tiny      # or use 10.**-16; minimize.m uses matlab's realmin 
             x3 = x3 * min(RATIO, d3/(d0-realmin))          # slope ratio but max RATIO
             ls_failed = 0                              # this line search did not fail
          else:
             X = X0; f0 = F0; df0 = dF0                     # restore best point so far
             if ls_failed or i > abs(length):         # line search failed twice in a row
                break;                             # or we ran out of time, so we give up
             
             s = -df0; d0 = -dot(s,s)                                        # try steepest
             x3 = 1.0/(1.0-d0)                     
             ls_failed = 1                                    # this line search failed

    # end outer-while
    gc.collect() # [debug]
    
    # [debug]
    print "[minimizer] @iter=%d X        = %s" % (i, str(X))                           
    print "                     gradient = %s" % str(f0) 
    
    # [todo] another stop criterion here by comparing to the older values
    print "                     df = %s" % str(df0)
    print "               progress = %s\n" % str(fX)

    print '\n'
    return (X, fX, i, f0, df0)
