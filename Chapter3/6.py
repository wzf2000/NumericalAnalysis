import numpy as np

GREEN = '\033[32m'
RED = '\033[31m'
END = '\033[0m'
BOLD = '\033[1m'

def Hilbert(n, noise=False):
    H: np.ndarray = np.fromfunction(lambda i, j: 1 / (i + j + 1), (n, n))
    x = np.ones(n)
    b: np.ndarray = np.dot(H, x)
    if noise:
        b += np.random.normal(0, 1e-7, n)
    return H, b

def Cholesky(M: np.ndarray):
    n, n = M.shape
    L = np.zeros_like(M)

    for j in range(n):
        L[j][j] = M[j][j]
        for k in range(j):
            L[j][j] -= L[j][k] ** 2
        L[j][j] = np.sqrt(L[j][j])
        for i in range(j + 1, n):
            L[i][j] = M[i][j]
            for k in range(j):
                L[i][j] -= L[i][k] * L[j][k]
            L[i][j] /= L[j][j]
    
    return L

def solve(n, noise=False):
    H, b = Hilbert(n, noise)
    cond = np.linalg.cond(H, p=np.inf)
    print(f'Cond(H) = {cond}')
    L = Cholesky(H)
    y = np.zeros(n)

    for i in range(n):
        y[i] = b[i]
        for j in range(i):
            y[i] -= L[i][j] * y[j]
        y[i] /= L[i][i]
    
    x = np.zeros(n)

    for i in reversed(range(n)):
        x[i] = y[i]
        for j in range(i + 1, n):
            x[i] -= L[j][i] * x[j]
        x[i] /= L[i][i]

    x_ori = np.ones(n)
    r = np.max(np.abs(b - np.dot(H, x)))
    delta = np.max(np.abs(x_ori - x))
    print(f'r = {r:.20f}, delta = {delta:.20f}')
    return x

def work(n):
    def print_green(str):
        print(GREEN + BOLD + str + END)

    def print_red(str):
        print(RED + BOLD + str + END)

    print_red(f'Without noise n = {n}')
    x = solve(n)
    print_green(f'Final x = {x}')
    print_red(f'With noise n = {n}')
    x_noise = solve(n, True)
    print_green(f'Final x = {x_noise}')

work(10)
work(8)
work(12)