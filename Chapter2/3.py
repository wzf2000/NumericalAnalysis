import numpy as np
from mpmath import besselj
import matplotlib.pyplot as plt

epsilon = 1e-10

def zeroin(f, a, b):
    a, b = np.float64(a), np.float64(b)
    fa, fb = f(a), f(b)
    if np.sign(fa) == np.sign(fb):
        raise Exception('Function must change sign on the interval')
    
    c, fc = a, fa
    e = d = b - c
    
    iter = 0
    
    while fb != 0:
        if np.sign(fa) == np.sign(fb):
            a, fa = c, fc
            e = d = b - c
        
        if np.abs(fa) < np.abs(fb):
            a, b = b, a
            fa, fb = fb, fa

        m = 0.5 * (a - b)
        tol = 2. * epsilon * max(np.abs(b), 1)

        if np.abs(m) <= tol or fb == 0:
            break
        
        if np.abs(e) < tol or np.abs(fc) <= np.abs(fb):
            e = d = m
        else:
            s = fb / fc
            if a == c:
                p = 2. * m * s
                q = 1. - s
            else:
                q, r = fc / fa, fb / fa
                p = s * (2. * m * q * (q - r) - (b - c) * (r - 1.))
                q = (q - 1.) * (r - 1.) * (s - 1.)

            if p > 0:
                q = -q
            else:
                p = -p

            if 2. * p < 3. * m * q - np.abs(tol * q) and p < np.abs(0.5 * e * q):
                e, d = d, p / q
            else:
                e = d = m

        iter += 1
        c, fc = b, fb

        if np.abs(d) > tol:
            b += d
        else:
            b -= np.sign(b - a) * tol
        
        fb = f(b)

    print(f'zeroin method: After {iter} iterations, the root is {b}')
    return b

J0 = lambda x: besselj(0, x)

x_list = np.arange(1, 40, 0.001)
y_list = [J0(x) for x in x_list]

_, axes = plt.subplots()
axes.set_ylabel(r'$J_0(x)$')
plt.plot(x_list, y_list)
plt.grid(True)
plt.axhline(0, color='red')
plt.show()

intervals = [
    (1, 5),
    (5, 7),
    (7, 10),
    (10, 13),
    (13, 16),
    (16, 20),
    (20, 23),
    (23, 25),
    (25, 30),
    (30, 31)
]

_, axes = plt.subplots()
axes.set_ylabel(r'$J_0(x)$')
plt.plot(x_list, y_list, zorder=1)
plt.grid(True)
plt.axhline(0, color='gray', zorder=0)

for cnt, (a, b) in enumerate(intervals):
    zero = zeroin(J0, a, b)
    plt.scatter(zero, 0, s=30, zorder=2)

plt.show()