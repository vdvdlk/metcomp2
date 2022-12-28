import numpy as np


def newton_rhaphson(f, f_linha, x_0: float, erro: float = 1e-15):
    x_i = x_0
    x_ii = x_i - f(x_i) / f_linha(x_i)
    
    i = 1
    while np.abs(f(x_ii) - 0) > erro:
        x_i = x_ii
        x_ii = x_i - f(x_i) / f_linha(x_i)
        i += 1
    
    return x_ii, i
