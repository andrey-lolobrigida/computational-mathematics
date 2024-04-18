#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#


import numpy as np

class BFGS:

    def __init__(self, x0, H0, sigma, tol, maxit):
        self.x0 = x0
        self.H0 = H0
        self.sigma = sigma
        self.tol = tol
        self.maxit = maxit

    def objective_func(self, x):
        
        # DEFINE YOUR FUNCTION HERE.
        # for example purposes, we are using the Rosenbrock function
        # https://en.wikipedia.org/wiki/Rosenbrock_function
        
        f = (1 - x[0])**2 + 100*((x[1] - x[0]**2)**2)
        
        first_coord = -2*(1 - x[0]) + 200*(x[1] - x[0]**2)*(-2*x[0])
        second_coord = 200*(x[1] - x[0]**2)

        grad_f = np.array([first_coord, second_coord]).T
        
        return f, grad_f
    
    def bfgs(self):

        x = self.x0
        H = self.H0        

        for i in range(0, self.maxit):
            (f, grad_f) = self.objective_func(x)
            norm_grad = np.linalg.norm(grad_f)

            if norm_grad < self.tol:
                break

            print('=================== H f x gradf =========')
            print(H, self.H0, x, grad_f)
            search_direction = (-1)*np.dot(H, grad_f)
            print('=================== search dir =========')
            print(search_direction)

            t = 1
            fn = self.objective_func(x + t*search_direction) 

            while (fn[0] > f + t*self.sigma*(norm_grad**2)):
                t = t/2                
                xn = x + t*search_direction                             
                fn = self.objective_func(xn)

            print('===================fn t gradfn=========')
            print(fn)
            print(t)
            print(fn[1])

            s = t*search_direction
            y = fn[1] - grad_f
            x = x + s

            print('===================s y x=========')
            print(s,y,x)

            # BFGS Formula

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

        print(H)

        f = fn[0]

        return x, f, i





