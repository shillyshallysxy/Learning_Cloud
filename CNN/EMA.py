from matplotlib import pyplot as plt
import numpy as np
from math import pow


def ema_algorithm(arr, gamma=0.9, method=0):
    ema_arr = []
    if method == 0:
        for i in range(len(arr)):
            ema_arr.append(ema_normal(arr, i, gamma))
    elif method == 1:
        for i in range(len(arr)):
            ema_arr.append(ema_cool(arr, i, gamma))
    return np.array(ema_arr)


def ema_normal(arr, index, gamma):
    if index == 0:
        return (1-gamma)*arr[index]
    else:
        return gamma*ema_normal(arr, index-1, gamma)+(1-gamma)*arr[index]


def ema_cool(arr, index, gamma, beta=0.9):
    if index == 0:
        return (1-gamma)*arr[index]
    else:
        return gamma*ema_normal(arr, index-1, gamma)/(1-pow(beta, index))+(1-gamma)*arr[index]


before_ema = np.array(np.random.randint(0, 50, 20))
after_ema_normal = ema_algorithm(before_ema, 0.9)
after_ema_cool = ema_algorithm(before_ema, 0.9, 1)
print(before_ema)
print(after_ema_normal)
print(after_ema_cool)
plt.plot(before_ema, label='before_ema')
plt.plot(after_ema_cool, label='after_ema_cool')
plt.plot(after_ema_normal, label='after_ema_normal')
plt.legend()
plt.show()
