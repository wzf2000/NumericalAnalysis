import numpy as np
from scipy.optimize import root

GREEN = '\033[32m'
RED = '\033[31m'
END = '\033[0m'
BOLD = '\033[1m'

def Newton(f, fprime, x0, damp=False):
    epsilon = 1e-10
    iter = 0
    x = last = x0
    print(f'Initial state: x0 = {x}, f(x0) = {f(x)}')
    while np.abs(x - last) > epsilon or np.abs(f(x)) > epsilon:
        if iter >= 100:
            print('Can not converge!')
            return None
        last, x = x, x - f(x) / fprime(x)
        iter += 1
        print(f'Iteration {iter}: x = {x:.10f}, f(x) = {f(x):.10f}')
        if damp:
            lamb = 0.9
            i = 0
            while np.abs(f(x)) > np.abs(f(last)):
                x = last - lamb * f(last) / fprime(last)
                lamb /= 2
                i += 1
                print(f'Damping iteration {i}: x = {x:.10f}, f(x) = {f(x):.10f}')
    return x

def test(f, fprime, x0, scipy_x0):
    def print_green(str):
        print(GREEN + BOLD + str + END)

    def print_red(str):
        print(RED + BOLD + str + END)
    
    print('--------------------------------------------')
    print_red('Solving with Newton method without damping:')
    x1 = Newton(f, fprime, x0)
    print_red('Solving with damping Newton method:')
    x2 = Newton(f, fprime, x0, True)
    print_red('Solving with scipy.optimize.root...')
    x3 = root(f, scipy_x0).x[0]
    print_green(f'Newton method final x = {x1}')
    print_green(f'Damping Newton method final x = {x2}')
    print_green(f'scipy final x = {x3}')
    print('--------------------------------------------')

test(lambda x: x ** 3 - 2 * x + 2, lambda x: 3 * x ** 2 - 2, 0, -1)
test(lambda x: -x ** 3 + 5 * x, lambda x: -3 * x ** 2 + 5, 1.35, 1.35)