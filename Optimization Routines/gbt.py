# classic gradient descent optimization algorithm
# equipped with backtracking (satisfying Armijo conditions)
# 
# INPUT
# func: the objective function, to be defined inside this very file
#       ****IMPORTANT**** - the function here defined must return the function value and gradient at x
# x0: initial point/guess
# sigma: float in the interval (0,1) for the Armijo condition
# tol: the maximum gap, or tolerance, of the stopping criteria
# maxit: the maximum number of iterations
# 
# OUTPUT:
# x: the approximate local minimizer
# f: the function value at x
# i: the number of iterations required to get to the solution
 

import math
import numpy as np

class Gbt:
    
    def __init__(self, x0, sigma, tol, maxit):
        self.x0 = x0
        self.sigma = sigma
        self.tol = tol
        self.maxit = maxit

    def func(self, x):
        
        # DEFINE YOUR FUNCTION HERE.
        # for example purposes, we are using the Rosenbrock function
        # https://en.wikipedia.org/wiki/Rosenbrock_function
        
        f = (1 - x[0])**2 + 100*((x[1] - x[0]**2)**2)
        
        first_coord = -2*(1 - x[0]) + 200*(x[1] - x[0]**2)*(-2*x[0])
        second_coord = 200*(x[1] - x[0]**2)

        grad_f = np.array([first_coord, second_coord]).T
        
        return f, grad_f        


    def gbt(self): # the method itself

        x = self.x0      

        for i in range(0, self.maxit+1):
            (f, grad_f) = self.func(x)
            norm_grad = np.linalg.norm(grad_f)
            if norm_grad < self.tol:
                break

            t = 1
            fn = self.func(x - t*grad_f)                  

            while (fn[0] > f - t*self.sigma*(norm_grad**2)):
                t = t/2                
                xn = x - t*grad_f                             
                fn = self.func(xn)                              
            
            x = x - t*grad_f
        
        f = fn[0]     

        return x, f, i            
        




