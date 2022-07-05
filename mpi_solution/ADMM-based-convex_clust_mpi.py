import numpy as np
import cvxpy as cp
import functools
import time
import sys
from sklearn.metrics import silhouette_score
import operator
from mpi4py import MPI

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
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    startRead = MPI.Wtime()
    if rank==0:
        A = load_data(folder+"/"+filename)
        A = A
    endRead = MPI.Wtime()
    readTime=endRead-startRead

    algTimeStart = MPI.Wtime()
    
    N=0
    d=0
    
    if rank==0:
        N = A.shape[0]
        d = A.shape[1]
    
    N = comm.bcast(N, root=0)
    d = comm.bcast(d, root=0)
    
    chunk_size = N // workers
    
    a = None
    
    if rank==0:
        a = [A[i*chunk_size:(i+1)*chunk_size] for i in range(workers)]
    a_local = comm.scatter(a, root=0)

    x = [np.zeros((chunk_size, d)) for i in range(workers)]
    y = [np.zeros((d)) for i in range(workers-1)]
    y = np.asarray(y)
    lambdaVal = [np.zeros((d)) for i in range(workers-1)]
    
    x_local = np.zeros((chunk_size, d))
    y_local = np.zeros((d))
    lambdaVal_local = np.zeros((d))

    req_iter = num_iter
    inner_iters = 20000
    
    z = [np.zeros((chunk_size, d)) for i in range(workers)]
    z_local = np.zeros((chunk_size, d))

    
    for i in range(num_iter):
      if rank==0:
           x_local = update_x_zero(a[0], x[0], y, lmbd, workers)
      else:
          x_local = update_x(rho, lmbd, d, a_local, x_local, y[rank-1], lambdaVal[rank-1])
      x = comm.gather(x_local, root=0) 
      breakIt = False
      if rank==0:      
          y_old=y
          y = y_update(lambdaVal, x, y, rho, workers-1, d, lmbd)
          lambdaVal = lambda_update(lambdaVal, rho, x, y, workers-1)    
     
          #adding Boyd's stopping criterion   
          primal_res = np.sqrt(np.sum([(np.linalg.norm(x[i][0]-y[i-1])**2) for i in range(1, workers)]))
          dual_res = np.linalg.norm(- rho * (y-y_old))
          eps_pri = np.sqrt(N) * abstol+reltol * max(np.linalg.norm(x), np.linalg.norm(-y))
          eps_dual = np.sqrt(N) * abstol + reltol * np.linalg.norm(lambdaVal)
          breakIt = False
          if primal_res <= eps_pri and dual_res <= eps_dual:
              req_iter=i+1
              breakIt = True
      breakIt = comm.bcast(breakIt, root=0)
      y = comm.bcast(y, root=0)
      lambdaVal = comm.bcast(lambdaVal, root=0)
      if breakIt:
          break    
      
    algTimeEnd = MPI.Wtime()
    algTime = algTimeEnd - algTimeStart  
    
    mergeTimeStart = MPI.Wtime() 
    y = comm.bcast(y, root=0)
      
    epsilon=eps  
    xy_local = x_local.copy()
    if rank>0:
        xy_local = np.concatenate((xy_local, np.asarray([y[rank-1]])), axis=0)
    theSize = chunk_size
    if rank>0:
      theSize+=1
    possible_centers_local = merge_centers_locally(epsilon, xy_local, theSize, a_local, chunk_size)
    possible_centers = comm.gather(possible_centers_local, root=0)
    centers = []
    if rank==0:
        pos_centers = functools.reduce(operator.iconcat, possible_centers, [])
        centers = merge_centers_globally(pos_centers, epsilon)
        centers = np.asarray(centers) 
    
    mergeTimeEnd = MPI.Wtime()
    mergeTime = mergeTimeEnd - mergeTimeStart
    allPossibleCenters=[]
    if rank==0:
        allPossibleCenters = []
        for i in range(workers):
          for j in range(chunk_size):
              allPossibleCenters.append(x[i][j])
         
        for i in range(workers-1):
          allPossibleCenters.append(y[i])
    return x, y, centers, readTime, algTime, mergeTime, allPossibleCenters, req_iter, lambdaVal

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


def update_x(rho, lmbd, d, a, x, y, lambdaVal):
    sol = cp.Variable(a.shape)#, value=x)
    problem = cp.Problem(cp.Minimize(objective_x(a, sol, lambdaVal, y, rho, lmbd, d)))
    problem.solve()
    return sol.value

def y_update(lambdaVal, x, y, rho, workers, dim, lmbd):  
    sol = cp.Variable(y.shape)#, value=y)
    x_zeros = [x[i+1][0] for i in range(0, workers)]
    x_zeros = np.asarray(x_zeros)
    problem = cp.Problem(cp.Minimize(objective_y(lambdaVal, x, sol, rho, workers, dim, lmbd)))
    problem.solve()
    return sol.value
    
def lambda_update(lambdaVal, rho, x, y, workers):
    for i in range(0, workers):
        lambdaVal[i] = (lambdaVal[i]+ rho * (y[i]-x[i+1][0]))
    return lambdaVal
    
def update_x_zero(a, x, y, lmbd, workers):
    sol=cp.Variable(a.shape)#, value=x)
    problem = cp.Problem(cp.Minimize(objective_x_zero(a, sol, y, lmbd, workers)))
    problem.solve()
    return sol.value
    
def main():
    workers = int(sys.argv[1])	
    filename = sys.argv[2]
    numiter = int(sys.argv[3])
    lmbda = float(sys.argv[4])
    epsilon = float(sys.argv[5])
    folder = sys.argv[6]
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    start = MPI.Wtime()
  
    x, y, centers, readTime, algTime, mergeTime, allPossibleCenters, reqIters, lambdaVal = admm_kmeans(filename, folder, workers+1, num_iter=numiter, lmbd=lmbda, rho=1, eps=epsilon)
    
    local_time = MPI.Wtime() - start
    maxTime = comm.reduce(local_time, op=MPI.MAX, root=0)
    readTimeAll = comm.reduce(readTime, op=MPI.MAX, root=0)
    algTimeAll = comm.reduce(algTime, op=MPI.MAX, root=0)
    mergeTimeAll = comm.reduce(mergeTime, op=MPI.MAX, root=0)
    
    if rank==0:
        print("\nTotal elapsed time: %s" % str((maxTime)))
        print("\nRead time: %s" % str(readTimeAll))
        print("\nAlgorithm time: %s" % str(algTimeAll))
        print("\nMerge time: %s" % str(mergeTimeAll))
    
        print("\nRequired number of iterations: %s"% str(reqIters))
    
if __name__ == '__main__':
    main()
