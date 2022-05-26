import numpy as np
import cvxpy as cp
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

    A = load_data(folder+"/"+filename)
    
    N = A.shape[0]
    d = A.shape[1]

    chunk_size = N // workers
    a = [A[i*chunk_size:(i+1)*chunk_size] for i in range(workers)]

    x = [np.zeros((chunk_size, d)) for i in range(workers)]
    y = [np.zeros((d)) for i in range(workers-1)]
    y = np.asarray(y)
    lambdaVal = [np.zeros((d)) for i in range(workers-1)]

    req_iter = num_iter

    for i in range(num_iter):
      x[0] = update_x_zero(a[0], x[0], y, lmbd, workers)
      x[1:workers] = list(map(functools.partial(update_x, rho,  lmbd, d), a[1:workers], x[1:workers], y, lambdaVal))
      y_old=y
      y = y_update(lambdaVal, x, y, rho, workers-1, d)
      lambdaVal = lambda_update(lambdaVal, rho, x, y, workers-1)
      
      #Boyd's stopping criterion   
      primal_res = np.sqrt(np.sum([(np.linalg.norm(x[i][0]-y[i-1])**2) for i in range(1, workers)]))
      dual_res = np.linalg.norm(- rho * (y-y_old))
      eps_pri = np.sqrt(N) * abstol+reltol * max(np.linalg.norm(x), np.linalg.norm(-y))
      eps_dual = np.sqrt(N) * abstol + reltol * np.linalg.norm(lambdaVal)
      if primal_res <= eps_pri and dual_res <= eps_dual:
          req_iter=i+1
          break
            
    epsilon=eps  
    xy = x.copy()
    chunk_sizes = []
    chunk_sizes.append(chunk_size)
    data_chunk_size = [chunk_size for i in range(workers)]
    for i in range(1,workers):
      xy[i]=np.concatenate((xy[i], [y[i-1]]), axis=0)
      chunk_sizes.append(chunk_size+1)
    possible_centers = list(map(functools.partial(merge_centers_locally, epsilon), xy, chunk_sizes, a, data_chunk_size))
    pos_centers = functools.reduce(operator.iconcat, possible_centers, [])
    centers = merge_centers_globally(pos_centers, epsilon)
    centers = np.asarray(centers) 
    
    allPossibleCenters = []
    for i in range(workers):
      for j in range(chunk_size):
        allPossibleCenters.append(x[i][j])
         
    for i in range(workers-1):
      allPossibleCenters.append(y[i])

    return x, y, centers, allPossibleCenters, req_iter

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
          if np.linalg.norm(x[i] - x[j]) <= eps:#ilon:
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
    centers=[]
    removed_indices=[]
    for i in range(0, len(possible_centers)):
      if i not in removed_indices:
        centers.append(possible_centers[i])
        for j in range(i+1, len(possible_centers)):
          if np.linalg.norm(np.asarray(possible_centers[i]) - np.asarray(possible_centers[j])) <= eps:#ilon:
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

def objective_y(lambdaVal, x, y, rho, workers, d):
    return loss_y1(lambdaVal, x, y, workers, d) + loss_y2(rho, x, y, workers) 

@task(returns=np.array)
def update_x(rho, lmbd, d, a, x, y, lambdaVal):
    sol = cp.Variable(a.shape, value=x)
    problem = cp.Problem(cp.Minimize(objective_x(a, sol, lambdaVal, y, rho, lmbd, d)))
    problem.solve()
    return sol.value

def y_update(lambdaVal, x, y, rho, workers, dim):  
    sol = cp.Variable(y.shape, value=y)
    x_zeros = [x[i+1][0] for i in range(0, workers)]
    x_zeros = np.asarray(x_zeros)
    problem = cp.Problem(cp.Minimize(objective_y(lambdaVal, x, sol, rho, workers, dim)))
    problem.solve()
    return sol.value

def lambda_update(lambdaVal, rho, x, y, workers):
    for i in range(0, workers):
    	lambdaVal[i] = (lambdaVal[i]+ rho * (y[i]-x[i+1][0]))
    return lambdaVal
    
def update_x_zero(a, x, y, lmbd, workers):
    sol=cp.Variable(a.shape, value=x)
    problem = cp.Problem(cp.Minimize(objective_x_zero(a, sol, y, lmbd, workers)))
    problem.solve()
    return sol.value
    
def recenter_the_centers(centers, A, labels):
    new_centers = []
    for i in range(len(centers)):
      sumLabel = 0.0
      count = 0  
      for j in range(A.shape[0]):
        if i == labels[j]:
          sumLabel += A[j]
          count+=1
      if count!=0:
        sumLabel/=count
      new_centers.append(sumLabel)    
    new_centers = np.asarray(new_centers)
    return new_centers
    
def main():
    workers = int(sys.argv[1])	
    filename = sys.argv[2]
    numiter = int(sys.argv[3])
    lmbda = float(sys.argv[4])
    epsilon = float(sys.argv[5])
    folder = sys.argv[6]
    start = time.time()
  
    x, y, centers, allPossibleCenters, reqIters = admm_kmeans(filename, folder, workers+1, num_iter=numiter, lmbd=lmbda, rho=1, eps=epsilon)

    print("\nTotal elapsed time: %s" % str((time.time() - start)))
    print("\nRequired number of iterations: %s"% str(reqIters))
    
    #Metrics
    
    #Load the data set
    A = load_data(folder+"/"+filename)
    
    #Define the labels    
    labels = []
    for i in range(A.shape[0]):
      min_dist = 1e10
      label = -1
      for j in range(len(centers)):
        dist = np.linalg.norm(A[i]-centers[j])
        if dist<min_dist:
          min_dist = dist
          label = j
      labels.append(label)

    #calculate silhouette score
    silScore=silhouette_score(A, labels)
    print(silScore)
    
    #calculate dunn index
    dist = pairwise_distances(A)
    labels = np.asarray(labels)
    dunn = validclust.indices.dunn(dist, labels)
    print(dunn)
    
    #calculate davis boulin score
    dbScore = davies_bouldin_score(A, labels)
    print("Davis Boulin score:")
    print(dbScore)

    #recentring the centers
    new_centers = recenter_the_centers(centers, A, labels)


if __name__ == '__main__':
    main()
