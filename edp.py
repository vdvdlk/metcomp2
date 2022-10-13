from copy import deepcopy

import numpy as np
from scipy.constants import epsilon_0

from matrizes import id_cond_contorno_2d, inf_para_zero_2d
from tensor3 import id_cond_contorno_3d, inf_para_zero_3d


def update_V_laplace_2d(posicoes, array):
    novo_array = deepcopy(array)
    delta_V = 0.0
    for elemento in posicoes:
        i, j = elemento
        novo_array[i, j] = (array[i - 1, j] + array[i + 1, j] +
                            array[i, j - 1] + array[i, j + 1]) / 4
        delta_V += np.abs(array[i, j] - novo_array[i, j])
    return novo_array, delta_V


def laplace_2d(array_inicial, erro=1e-5, printar=False):
    delta_V = 1.0
    pos = id_cond_contorno_2d(array_inicial)
    array = inf_para_zero_2d(array_inicial)
    if printar == True:
        n = 0
    while np.abs(delta_V) > erro:
        array, delta_V = update_V_laplace_2d(pos, array)
        if printar == True:
            n += 1
    if printar == True:
        print('O programa fez', n, 'iterações.')
    return array


def update_V_poisson_3d(posicoes, tensor, dx, rho):
    novo_tensor = deepcopy(tensor)
    delta_V = 0.0
    for elemento in posicoes:
        i, j, k = elemento
        novo_tensor[i, j, k] = (1 / 6) * (tensor[i + 1, j, k] + tensor[i - 1, j, k] + tensor[i, j + 1, k] +
                                          tensor[i, j - 1, k] + tensor[i, j, k + 1] + tensor[i, j, k - 1]) + rho[i, j, k] * (dx**2) / (6 * epsilon_0)
        delta_V += np.abs(tensor[i, j, k] - novo_tensor[i, j, k])
    return novo_tensor, delta_V


def poisson_3d(tensor_inicial, dx, rho, erro=1e-6, printar=False):
    delta_V = 1.0
    pos = id_cond_contorno_3d(tensor_inicial)
    tensor = inf_para_zero_3d(tensor_inicial)
    if printar == True:
        n = 0
    while abs(delta_V) > erro:
        tensor, delta_V = update_V_poisson_3d(pos, tensor, dx, rho)
        if printar == True:
            n += 1
    if printar == True:
        print('O programa fez', n, 'iterações.')
    return tensor


def propagate(y_x0, t_f, dx, c, dt=False, pe='fixa', pd='fixa'):
    if dt == False:
        dt = dx / c
    r = c * dt / dx

    i_max = np.size(y_x0)
    n_max = int(t_f / dt)

    y_xt = np.zeros((i_max, n_max))
    y_xt[:, 0] = y_x0

    for n in range(n_max - 1):
        for i in range(1, i_max - 1):
            if n == 0:
                y_xt[i, n + 1] = 2 * (1 - r**2) * y_xt[i, n] - \
                    y_x0[i] + r**2 * (y_xt[i + 1, n] + y_xt[i - 1, n])
            elif n != 0:
                y_xt[i, n + 1] = 2 * (1 - r**2) * y_xt[i, n] - \
                    y_xt[i, n - 1] + r**2 * (y_xt[i + 1, n] + y_xt[i - 1, n])

        if pe == 'fixa':
            y_xt[0, n + 1] = 0.0
        elif pe == 'solta':
            y_xt[0, n + 1] = y_xt[1, n + 1]

        if pd == 'fixa':
            y_xt[-1, n + 1] = 0.0
        elif pd == 'solta':
            y_xt[-1, n + 1] = y_xt[-2, n + 1]

    return y_xt


def difusao_2d(rho_0, t_f, dx, D, dt=False):
    if dt == False:
        dt = dx**2 / (4 * D)

    i_max, j_max = np.shape(rho_0)
    n_max = int(t_f / dt)

    rho_t = np.zeros((i_max, j_max, n_max))
    rho_t[:, :, 0] = rho_0

    for n in range(n_max - 1):
        for i in range(1, i_max - 1):
            for j in range(1, j_max - 1):
                rho_t[i, j, n + 1] = rho_t[i, j, n] + (D * dt / (dx)**2) * (rho_t[i + 1, j, n] + rho_t[i - 1, j, n] - 2 * rho_t[i, j, n] + rho_t[i, j + 1, n] + rho_t[i, j - 1, n] - 2 * rho_t[i, j, n])

    return rho_t
