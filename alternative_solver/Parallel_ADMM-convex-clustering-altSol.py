import numpy as np
import cvxpy as cp
from sklearn.datasets.samples_generator import make_blobs
import functools
import time
import sys
from pycompss.api.task import task
from pycompss.api.api import compss_wait_on
from pycompss.api.parameter import *
from sklearn.metrics import silhouette_score
import pandas as pd
import matplotlib.pyplot as plt
import operator
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import pairwise_distances
import validclust
from sklearn.metrics import davies_bouldin_score

def admm_kmeans(filename, folder, workers=5, num_iter=100, lmbd=1, rho=1, eps=0.1, abstol = 1e-4, reltol = 1e-2):

    """ADMM kmeans represents an adapted kmeans implementation, based on the idea of SON (sum of norms) clustering, 
    but solving the problem in a distributed manner. It relies on the Alternating Direction Method
    of Multipliers (ADMM) as the solver. ADMM is renowned for being well suited to the distributed
    settings, for its guaranteed convergence and general robustness with respect to the parameters. 
    
    :param workers: The number of agents used to solve the problem
    :param num_iter: The maximum number of iterations before the algorithm stops automatically
    :param lmbd: The regularization parameter
    :param rho: The penalty parameter for constraint violation in ADMM
    """
    startRead = time.time()

    A = load_data(folder+"/"+filename)
    A = A
    print(A)
    endRead = time.time()
    readTime=endRead-startRead

    algTimeStart = time.time()
    
    N = A.shape[0]
    d = A.shape[1]

    chunk_size = N // workers
    a = [A[i*chunk_size:(i+1)*chunk_size] for i in range(workers)]

    x = [np.zeros((chunk_size, d)) for i in range(workers)]
    y = [np.zeros((d)) for i in range(workers-1)]
    y = np.asarray(y)
    lambdaVal = [np.zeros((d)) for i in range(workers-1)]

    req_iter = num_iter
    inner_iters = 20000
    
    z = [np.zeros((chunk_size, d)) for i in range(workers)]
    
    Eps = 1e-1
    mu = 2*Eps/(lmbd*chunk_size)
    L = 1 + 2*lmbd*chunk_size/mu
    
    for i in range(num_iter):
      x[0] = update_x_zero(a[0], x[0], y, lmbd, workers, rho, d, chunk_size, mu, L, inner_iters, i)
      x[1:workers] = list(map(functools.partial(update_x, rho,  lmbd, d, chunk_size, mu, L, inner_iters, i), a[1:workers], x[1:workers], y, lambdaVal))
      x[1:workers] = compss_wait_on(x[1:workers])      
      y_old=y
      y = y_update(lambdaVal, x, y, rho, workers-1, d, lmbd)
      lambdaVal = lambda_update(lambdaVal, rho, x, y, workers-1)    
     
      #adding Boyd's stopping criterion   
      primal_res = np.sqrt(np.sum([(np.linalg.norm(x[i][0]-y[i-1])**2) for i in range(1, workers)]))
      dual_res = np.linalg.norm(- rho * (y-y_old))
      eps_pri = np.sqrt(N) * abstol+reltol * max(np.linalg.norm(x), np.linalg.norm(-y))
      eps_dual = np.sqrt(N) * abstol + reltol * np.linalg.norm(lambdaVal)
      if primal_res <= eps_pri and dual_res <= eps_dual:
          req_iter=i+1
          break
      
    algTimeEnd = time.time()
    algTime = algTimeEnd - algTimeStart  
    
    mergeTimeStart = time.time() 
      
    epsilon=eps  
    xy = x.copy()
    chunk_sizes = []
    chunk_sizes.append(chunk_size)
    data_chunk_size = [chunk_size for i in range(workers)]
    for i in range(1,workers):
      xy[i]=np.concatenate((xy[i], [y[i-1]]), axis=0)
      chunk_sizes.append(chunk_size+1)
    possible_centers = list(map(functools.partial(merge_centers_locally, epsilon), xy, chunk_sizes, a, data_chunk_size))
    possible_centers = compss_wait_on(possible_centers)
    pos_centers = functools.reduce(operator.iconcat, possible_centers, [])
    centers = merge_centers_globally(pos_centers, epsilon)
    centers = np.asarray(centers) 
    
    mergeTimeEnd = time.time()
    mergeTime = mergeTimeEnd - mergeTimeStart

    allPossibleCenters = []
    for i in range(workers):
      for j in range(chunk_size):
        allPossibleCenters.append(x[i][j])
         
    for i in range(workers-1):
      allPossibleCenters.append(y[i])

    return x, y, centers, readTime, algTime, mergeTime, allPossibleCenters, req_iter, lambdaVal

@task(returns=np.array)
def merge_centers_locally(epsilon, x, chunk_size, a, data_size):
    sum=0.0
    cnt=0
    for i in range(data_size):
      for j in range(i+1, data_size):
        sum+=np.linalg.norm(a[i]-a[j])
        cnt=cnt+1
    avg_dist = sum/cnt
    eps = avg_dist/epsilon

    centers=[]
    removed_indices=[]
    for i in range(0, chunk_size):
      if i not in removed_indices:
        centers.append(x[i])
        for j in range(i+1, chunk_size):
          if np.linalg.norm(x[i] - x[j]) <= eps:
            removed_indices.append(j)
    return centers
  
def merge_centers_globally(possible_centers, epsilon):
    sum=0.0
    cnt=0
    for i in range(len(possible_centers)):
      for j in range(i+1, len(possible_centers)):
        sum+=np.linalg.norm(np.asarray(possible_centers[i]) - np.asarray(possible_centers[j]))
        cnt=cnt+1
    avg_dist = sum/cnt

    eps = avg_dist/epsilon
    print(eps)

    centers=[]
    removed_indices=[]
    for i in range(0, len(possible_centers)):
      if i not in removed_indices:
        centers.append(possible_centers[i])
        for j in range(i+1, len(possible_centers)):
          if np.linalg.norm(np.asarray(possible_centers[i]) - np.asarray(possible_centers[j])) <= eps:
            removed_indices.append(j)
    return centers

def load_data(filename): 
    f = open(filename, 'r')
    line1 = f.readline()
    dims = list(map(int, line1.split()))
    res = np.asarray(dims)
    N = res[0]
    d = res[1]
    rest = f.read()
    vecl = list(map(float, rest.split()))
    a = np.asarray(vecl)
    A = a.reshape(N, d)
    print(A)
    print(A.shape)
    return A

def loss_x1(A,x):
    return cp.sum_squares(cp.norm(A - x, p=2, axis=1))

def loss_x2(x, lmbd):
    local_n = x.shape[0]
    sumOfNorms = cp.sum([cp.norm(x[i]-x[0], p=2, axis=0) for i in range(1, local_n)])
    return lmbd * sumOfNorms

def loss_x3(x, y, lambdaVal, d):
    diff = y-x[0]
    return (cp.reshape(lambdaVal, (d,1))).T @ cp.reshape(diff, (d,1))

def loss_x4(x, y, rho):
    norm = cp.norm(y-x[0], p=2, axis=0)
    return 1/2 * rho * cp.square(norm)

def loss_x_zero(x, y, lmbd, workers):
    sumOfNorms = cp.sum([cp.norm(x[0]-y[i-1], p=2) for i in range(1, workers)])  
    return lmbd * sumOfNorms

def objective_x(a, x, lambdaVal, y, rho, lmbd, d):
    return loss_x1(a, x) + loss_x2(x, lmbd) + loss_x3(x,y,lambdaVal, d) + loss_x4(x, y, rho)

def objective_x_zero(a, x, y, lmbd, workers):
    return loss_x1(a, x) + loss_x2(x, lmbd) + loss_x_zero(x, y, lmbd, workers)

def loss_y1(lambdaVal, x, y, workers, d):
    val = cp.sum([(cp.reshape(lambdaVal[i],(d, 1))).T @ cp.reshape(y[i]-x[i+1][0], (d,1)) for i in range(0, workers)])
    return val

def loss_y2(rho, x, y, workers):
    val = 1/2 * rho * cp.sum([cp.square(cp.norm(y[i]-x[i+1][0], p=2, axis=0)) for i in range(0, workers)])
    return val
    
def loss_y3(x, y, lmbd, workers):
    normVal = cp.sum([cp.norm(x[0]-y[i-1], p=2) for i in range(1, workers)])
    return lmbd * normVal

def objective_y(lambdaVal, x, y, rho, workers, d, lmbd):
    return loss_y1(lambdaVal, x, y, workers, d) + loss_y2(rho, x, y, workers) + loss_y3(x[0], y, lmbd, workers)

def P(a):
    normValue= np.linalg.norm(a)
    if normValue>1:
        return a/normValue
    else:
        return a

@task(returns=np.array)
def update_x(rho, lmbd, d, chunk_size, mu, L, inner_iters, outer, a, x, y, lambdaVal):
    limit = 0.1/(outer+1)
    z = np.copy(x)   
    for t in range(inner_iters):
      z_old = z
      g = [np.zeros((chunk_size, d))]
      p_sum_zero = sum(P((x[0]-x[j])/mu) for j in range(1, chunk_size))
      g[0] = -lambdaVal + rho * (x[0]-y) + (x[0]-a[0]) + lmbd * p_sum_zero
      p_sum = [P((x[0]-x[j])/mu) for j in range(1, chunk_size)]  
      g[1:chunk_size] = (x[1:chunk_size] - a[1:chunk_size]) - [lmbd * p_sum[i] for i in range(chunk_size-1)]
      z = x - [1/L*g[i] for i in range(chunk_size)]
      x = z + (t+1-1)/(t+1+2)*(z-z_old)
      theNorm = np.linalg.norm(g)
      if theNorm < limit:
          break
    return x

def Prox(a, lmbd, rho):
    normA = np.linalg.norm(a)
    if normA>0:
        valA = 1 - (lmbd/rho)/normA
    else:
        valA = 0
    if valA>0:
        return valA*a
    else:
        return 0

def y_update(lambdaVal, x, y, rho, workers, dim, lmbd):  
    print("y update called")
    print(len(lambdaVal))
    delta = [Prox(-lambdaVal[i]/rho, lmbd, rho) for i in range(workers)]
    for i in range(workers):
        y[i] = x[i+1][0] + delta[i]
    return y

def lambda_update(lambdaVal, rho, x, y, workers):
    for i in range(0, workers):
        #print("CCCC",y[i])
        lambdaVal[i] = (lambdaVal[i]+ rho * (y[i]-x[i+1][0]))
    return lambdaVal
    
def update_x_zero(a, x, y, lmbd, workers, rho, d, chunk_size, mu, L, inner_iters, outer):
    limit = 0.1 / (outer+1)
    z = np.copy(x) 
    for t in range(inner_iters):
      z_old = z    
      g = np.zeros((chunk_size, d))
      p_sum_one = sum(P((x[0]-x[j])/mu) for j in range(1, chunk_size))
      p_sum_two = sum(P((x[0]-y[i-1])/mu) for i in range(1, workers))  
      g[0] = x[0]-a[0] + lmbd * p_sum_one + lmbd * p_sum_two
      p_sum = [P((x[0]-x[j])/mu) for j in range(1, chunk_size)]  
      g[1:chunk_size] = (x[1:chunk_size] - a[1:chunk_size]) - [lmbd * p_sum[i] for i in range(chunk_size-1)]
      z = x - [1/L*g[i] for i in range(chunk_size)]
      x = z + (t+1-1)/(t+1+2)*(z-z_old)
      theNorm = np.linalg.norm(g)
      if theNorm < limit and t>140:
          break
    return x
    
def main():
    workers = int(sys.argv[1])	
    filename = sys.argv[2]
    numiter = int(sys.argv[3])
    lmbda = float(sys.argv[4])
    epsilon = float(sys.argv[5])
    folder = sys.argv[6]
    start = time.time()
  
    x, y, centers, readTime, algTime, mergeTime, allPossibleCenters, reqIters, lambdaVal = admm_kmeans(filename, folder, workers+1, num_iter=numiter, lmbd=lmbda, rho=1, eps=epsilon)

    print("\nTotal elapsed time: %s" % str((time.time() - start)))
    print("\nRead time: %s" % str(readTime))
    print("\nAlgorithm time: %s" % str(algTime))
    print("\nMerge time: %s" % str(mergeTime))
    
    print("\nRequired number of iterations: %s"% str(reqIters))
    
 
if __name__ == '__main__':
    main()
