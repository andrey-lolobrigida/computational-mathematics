# BFGS's method optimization algorithm
# equipped with backtracking (satisfying Armijo conditions).
# for benchmark purposes, we are using the Rosenbrock function as the objective function
#
# INPUT:
# x0: initial point/guess
# H0: initial approximation for the inverse Hessian
# sigma: float in the interval (0,1) for the Armijo condition
# tol: the maximum gap, or tolerance, of the stopping criteria
# maxit: the maximum number of iterations
# 
# OUTPUT:
# x: the approximate local minimizer
# f: the function value at x
# i: the number of iterations required to get to the solution
    


import numpy as np
import matplotlib.pyplot as plt

class BFGS:

    def __init__(self, x0, H0, sigma, tol, maxit):
        self.x0 = x0
        self.H0 = H0
        self.sigma = sigma
        self.tol = tol
        self.maxit = maxit


    def objective_func(self, x):
        
        # DEFINE YOUR FUNCTION HERE.
        # for benchmark purposes, we are using the Rosenbrock function
        # https://en.wikipedia.org/wiki/Rosenbrock_function
        
        f = (1 - x[0])**2 + 100*((x[1] - x[0]**2)**2)
        
        first_coord = -2*(1 - x[0]) + 200*(x[1] - x[0]**2)*(-2*x[0])
        second_coord = 200*(x[1] - x[0]**2)

        grad_f = np.array([first_coord, second_coord]).T
        
        return f, grad_f
    
    
    def stopping_criteria(self, grad_f):
        # we check if the euclidean of the 
        # gradient of f at x is smaller than the tolerance.
        # if that happens, it means we hit a critical point
        # of the function and the algorithm stops        

        norm_grad = np.linalg.norm(grad_f)
        at_critical_point = False

        if norm_grad < self.tol:
            at_critical_point = True

        return norm_grad, at_critical_point
    
    
    def set_search_dir_and_stepsize(self, x, H, f, grad_f, norm_grad):
        # setting up the search direction

        search_direction = (-1)*np.dot(H, grad_f)            

        # we find a stepsize that satisfies the Armijo condition
        # using backtracking line search

        stepsize = 1
        eval_array = self.objective_func(x + stepsize*search_direction) 

        while (eval_array[0] > f + stepsize*self.sigma*(norm_grad**2)):
            stepsize = stepsize/2                
            x_aux = x + stepsize*search_direction                             
            eval_array = self.objective_func(x_aux)

        return search_direction, stepsize, eval_array
    
    
    def compute_inverse_hessian_approx(self, x, H, grad_f, search_direction, stepsize, eval_array):

        # setting up some vectors that are used in the BFGS formula
        s = stepsize*search_direction
        y = eval_array[1] - grad_f
        x = x + s            

        # an implementation of the BFGS formula
        ro_k = 1/np.dot(y, s)
        y_reshape = y.reshape(1, -1)
        s_reshape = s.reshape(1, -1)
        rank1matrix_left = np.matmul(s_reshape.T, y_reshape)
        rank1matrix_right = np.matmul(y_reshape.T, s_reshape)
        rank1matrix_sum = np.matmul(s_reshape.T, s_reshape)           
        left_term = (np.eye(2,2)- ro_k*rank1matrix_left)
        right_term = (np.eye(2,2)- ro_k*rank1matrix_right)
        sum_term = ro_k*rank1matrix_sum
        left_mul = np.matmul(left_term, H)
            
        H = np.matmul(left_mul, right_term) + sum_term

        return x, H
    
    
    def bfgs_method(self):

        x = self.x0
        H = self.H0

        # the grad norms will be stored in this list
        grad_norm_list = []        

        for i in range(0, self.maxit):
            # we compute the functional value and the gradient at x

            (f, grad_f) = self.objective_func(x)
            norm_grad, at_critical_point = self.stopping_criteria(grad_f)
            grad_norm_list.append(norm_grad)

            if at_critical_point == True:
                break            

            search_direction, stepsize, eval_array = self.set_search_dir_and_stepsize(x, H, f, grad_f, norm_grad)

            x, H = self.compute_inverse_hessian_approx(x, H, grad_f, search_direction, stepsize, eval_array)
                       

        f = eval_array[0]        

        return x, f, i, grad_norm_list
    
    
    def bfgs_benchmark(self):

        x, f, i, grad_norm_list = self.bfgs_method()        

        grad_norm_plot = plt.plot(np.arange(0, i+1), grad_norm_list)

        print('Global minima: ' + str(x))
        print('Functional value at x: ' + str(f))
        print('Converged in ' + str(i+1) + ' iterations.')

        plt.title('BFGS')
        plt.rc('axes', titlesize=22)        
        plt.rc('legend', fontsize=14)
        plt.yscale('log')
        plt.xlabel('Iterations')
        plt.ylabel('Gradient Norm')        
        plt.grid()
        plt.show()


    
    
    






