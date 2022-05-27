import numpy as np
from scipy.optimize import fsolve

def n_f32():
    n = 1
    sum = np.float32(0)
    new_sum = np.float32(1 + sum)
    while new_sum != sum:
        n += 1
        sum, new_sum = new_sum, np.float32(new_sum + np.float32(1 / n))

    print(f'Stop at n = {n}, sum = {sum}')
    return n

def n_f32_theory():
    n = 1
    sum = np.float64(0)
    epsilon = 6e-8
    n_rev = np.float64(1 / n)
    while n_rev > epsilon * sum / 2:
        sum += n_rev
        n += 1
        n_rev = np.float64(1 / n)

    print(f'Theory: stop at n = {n}')
    return n

def n_f32_error(limit):
    n = 1
    sum_32 = np.float32(0)
    sum_64 = np.float64(0)
    while n <= limit:
        sum_32 = np.float32(sum_32 + np.float32(1 / n))
        sum_64 = np.float64(sum_64 + np.float64(1 / n))
        n += 1

    err = np.abs(sum_64 - sum_32)
    rel_err = err / sum_64

    print(f'Absolute error = {err:.4f}, relative error = {rel_err:.4%}')

def n_f64_estimation(n):
    epsilon = 1e-16
    return 0.5 * epsilon * (np.log(n) + np.euler_gamma + 1 / n / 2) - 1 / n

limit = n_f32()
n_f32_theory()
n_f32_error(limit)
print(fsolve(n_f64_estimation, [1.0])[0])