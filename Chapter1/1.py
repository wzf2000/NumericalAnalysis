import numpy as np
import matplotlib.pyplot as plt

M = 1
epsilon = 1e-16

def trunc(h):
    return M * h / 2

def round(h):
    return epsilon * 2 / h

def total(h):
    return trunc(h) + round(h)

def actual(h):
    return np.abs(np.cos(1) - (np.sin(1 + h) - np.sin(1)) / h)

error_map = {
    '截断误差': trunc,
    '舍入误差': round,
    '总误差限': total,
    '实际总误差': actual
}

h_list = [10 ** (lgh / 10) for lgh in range(-160, 1, 1)]

errors = {
    error_type: [error_func(h) for h in h_list] for error_type, error_func in error_map.items()
}

plt.rc('font', family='Arial Unicode MS')
_, axes = plt.subplots()
axes.set_xscale('log', nonpositive='clip')
axes.set_xlim((epsilon, 1))
axes.set_xlabel(r'步长 $h$')
axes.set_yscale('log', nonpositive='clip')
axes.set_ylim((1e-17, 1))
axes.set_ylabel('误差')
for error_type, error_list in errors.items():
    plt.plot(h_list, error_list, label=error_type)
plt.legend()
plt.show()
