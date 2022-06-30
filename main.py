import time
import platform
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from joblib import Parallel, delayed
from multiprocessing import cpu_count
from copy import deepcopy
from scipy.special import expit

class ProbSAT(object):
    def __init__(self, filename, max_tries=1000, max_flips=10000):
        clauses = np.loadtxt(filename,int)[:,0:3]  # ranging from (+-) 1 to 20
        self.max_tries = max_tries
        self.max_flips = max_flips
        self.cb = 3 ** 0.8                          # section 3.2, SAT2012, pp. 15-29
        self.address = np.abs(clauses) - 1          # ranging from 0 to 19
        self.sign = np.sign(clauses)                # take value of -1 and 1
        n_variables = len(set(self.address.reshape(-1)))
        self.J, self.h = self.SAT2XOR(len(self.sign)/n_variables)
        
    def SAT2XOR(self, cost):
        """cost = [number of clauses]/[number of variables]
        """
        J = np.zeros((4*len(self.sign),4*len(self.sign)))
        h = np.zeros((4*len(self.sign),1))
        for i in range(len(self.sign)):
            J[4*i, 4*i+1] = -1                # coupling between x1 and x2
            J[4*i+1, 4*i] = -1                # coupling between x1 and x2
            J[4*i, 4*i+3] = self.sign[i,0]    # coupling between x1 and b
            J[4*i+3, 4*i] = self.sign[i,0]    # coupling between x1 and b
            J[4*i+1, 4*i+3] = self.sign[i,1]  # coupling between x2 and b
            J[4*i+3, 4*i+1] = self.sign[i,1]  # coupling between x2 and b
            J[4*i+2, 4*i+3] = -self.sign[i,2] # coupling between x3 and b
            J[4*i+3, 4*i+2] = -self.sign[i,2] # coupling between x3 and b
            h[4*i+2, 0] = self.sign[i,2]      # external field of x3
            h[4*i+3, 0] = 1                   # external field of auxiliary bits
            c = np.where(self.address==i+1)[0]
            l = np.where(self.address==i+1)[1]
            length = len(c)
            if length>1:
                idx = c*4 + l
                for j in range(length):
                    J[idx[j], idx[(j+1)%length]] = cost
                    J[idx[(j+1)%length], idx[j]] = cost
        return J, h
        
    def re_init(self, seed):
        Va = self.random_generation(seed)
        return Va
        
    def update(self, Va):    
        As = self.assignment(Va)
        Co = self.comparator(As)
        Un = self.unsat(Co)
        return Un
        
    def random_generation(self, seed):
        np.random.seed(seed)
        return 2*np.random.randint(2, size = self.address.max() + 1)-1
        
    def assignment(self, Va):
        """find the assignment for each clause"""
        return Va[self.address]    # As
        
    def comparator(self, As):
        """see whether the assignment agree with the clauses
           A  : a [n x 3] binary matrix
        return: a binary list with length n (denote as C)"""
        return (self.sign == As) + 0    # Co
    
    def unsat(self, Co):
        """returned value denote as U"""
        return np.where(Co.max(axis=1) == 0)[0]    # Un

    def breaker(self, variable, Va, Un):
        """flip one variable
           find the newly unsatisfied clauses"""
        temp = deepcopy(Va)
        temp[variable] *= -1
        _Un = self.update(temp)
        return len(set(_Un) - set(Un))
        
    
    def flipper(self, seed):
        Va = self.re_init(seed)
        Un = self.update(Va)
        for i in range(self.max_flips):
            if len(Un) == 0:
                break    
            else:
                select = np.random.randint(len(Un))
                clause = self.address[select]
                prob = np.zeros(3)
                for j in range(3):
                    variable = clause[j]
                    prob[j] = self.cb**(-self.breaker(variable, Va, Un))
                prob = prob/prob.sum() #normalize the total probability to 1
                flip = 1 - 2 * np.random.binomial(1, prob)
                for j in range(3):
                    variable = clause[j]
                    Va[variable] *= flip[j]
                    Un = self.update(Va)
        return len(Un)
    
    def solve(self):
        n_cpu = cpu_count()
        print('Solver running on {} threads'.format(n_cpu))
        Un = Parallel(n_jobs=n_cpu)(delayed(self.flipper)(i) for i in range(self.max_tries))
        return np.array(Un)
        
if __name__ =="__main__":
    start = time.process_time()
    solver = ProbSAT('example.txt')
    unsat = solver.solve()
    end = time.process_time()
    print("{} runs are satisfied, it takes {} cpu time".format((unsat==0).sum(), end - start))
