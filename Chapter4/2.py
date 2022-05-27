import numpy as np
import matplotlib.pyplot as plt

GREEN = '\033[32m'
RED = '\033[31m'
END = '\033[0m'
BOLD = '\033[1m'

n = 100
a = 0.5
h = 1 / n

def get_Ab(epsilon):
    A = np.zeros((n - 1, n - 1))
    for i in range(n - 2):
        A[i][i + 1] = epsilon + h
    for i in range(n - 1):
        A[i][i] = -(2 * epsilon + h)
    for i in range(1, n - 1):
        A[i][i - 1] = epsilon

    b = np.ones((n - 1)) * a * h ** 2
    b[-1] -= epsilon + h
    return A, b

def y(x, epsilon):
    return (1 - a) / (1 - np.exp(-1 / epsilon)) * (1 - np.exp(-x / epsilon)) + a * x

def Jacobi(A, b):
    n, n = A.shape
    x = np.ones(n)
    iter = 0
    while True:
        y = np.copy(x)
        iter += 1
        for i in range(n):
            x[i] = b[i]
            if i > 0:
                x[i] -= A[i][i - 1] * y[i - 1]
            if i < n - 1:
                x[i] -= A[i][i + 1] * y[i + 1]
            x[i] /= A[i][i]
        if np.linalg.norm(x - y, ord=np.inf) <= 1e-3:
            print(x - y)
            return x, iter

def GS(A, b):
    n, n = A.shape
    x = np.ones(n)
    iter = 0
    while True:
        y = np.copy(x)
        iter += 1
        for i in range(n):
            x[i] = b[i]
            if i > 0:
                x[i] -= A[i][i - 1] * x[i - 1]
            if i < n - 1:
                x[i] -= A[i][i + 1] * x[i + 1]
            x[i] /= A[i][i]
        if np.linalg.norm(x - y, ord=np.inf) <= 1e-3:
            return x, iter

def SOR(A, b, w):
    n, n = A.shape
    x = np.ones(n)
    iter = 0
    while True:
        y = np.copy(x)
        z = np.copy(x)
        iter += 1
        for i in range(n):
            z[i] = b[i]
            if i > 0:
                z[i] -= A[i][i - 1] * x[i - 1]
            if i < n - 1:
                z[i] -= A[i][i + 1] * x[i + 1]
            z[i] /= A[i][i]
            x[i] = (1 - w) * x[i] + w * z[i]
        if np.linalg.norm(x - y, ord=np.inf) <= 1e-3:
            return x, iter

def work(epsilon):
    def print_green(str):
        print(GREEN + BOLD + str + END)

    def print_red(str):
        print(RED + BOLD + str + END)

    def inf_norm(x):
        return np.linalg.norm(x, np.inf)

    def second_norm(x):
        return np.linalg.norm(x, 2)
    
    A, b = get_Ab(epsilon)
    x_list = np.arange(h, 1, h)
    y_list = np.array([y(x, epsilon) for x in x_list])

    _, axes = plt.subplots()
    axes.set_xlabel(r'$x$')
    axes.set_yscale('log')
    axes.set_ylabel(r'$\epsilon(y)$')

    def print_result(method_name, y, iter):
        print_red(f'{method_name} method take {iter} iterations to convergence.')
        print_green(f'{method_name} method: inf norm = {inf_norm(y_list - y)}, second norm = {second_norm(y_list - y)}')
        plt.plot(x_list, np.abs(y_list - y), label=f'{method_name} method')

    Jacobi_y, iter = Jacobi(A, b)
    print_result('Jacobi', Jacobi_y, iter)
    GS_y, iter = GS(A, b)
    print_result('GS', GS_y, iter)
    SOR_y, iter = SOR(A, b, 1.1)
    print_result('SOR', SOR_y, iter)

    plt.legend()
    plt.savefig(f'{epsilon}.png')
    plt.show()

work(1)
work(0.1)
work(0.01)
work(0.001)