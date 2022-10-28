from copy import deepcopy
from math import sqrt

import numpy as np
from tabulate import tabulate


def inicializar(dim):
    array = np.full(dim, np.inf)
    return array


def mprint(array):
    print(tabulate(array, tablefmt='plain', floatfmt='.2f'))


def metal_esq(array, constante):
    m, n = np.shape(array)
    novo_array = deepcopy(array)
    novo_array[0, :] = np.full(n, constante)
    return novo_array


def metal_dir(array, constante):
    m, n = np.shape(array)
    novo_array = deepcopy(array)
    novo_array[-1, :] = np.full(n, constante)
    return novo_array


def metal_baixo(array, constante):
    m, n = np.shape(array)
    novo_array = deepcopy(array)
    novo_array[:, 0] = np.full(m, constante)
    return novo_array


def metal_cima(array, constante):
    m, n = np.shape(array)
    novo_array = deepcopy(array)
    novo_array[:, -1] = np.full(m, constante)
    return novo_array


def metal_bordas(array, constante):
    novo_array = deepcopy(array)
    novo_array = metal_cima(novo_array, constante)
    novo_array = metal_esq(novo_array, constante)
    novo_array = metal_baixo(novo_array, constante)
    novo_array = metal_dir(novo_array, constante)
    return novo_array


def metal_int(array, constante, fracao=1 / 3):
    m, n = np.shape(array)
    i_i = int(m * fracao)
    i_f = m - (i_i + 1)
    j_i = int(n * fracao)
    j_f = m - (j_i + 1)

    novo_array = deepcopy(array)
    for i in range(i_i, i_f + 1):
        for j in range(j_i, j_f + 1):
            novo_array[i, j] = constante

    return novo_array


def metal_placas(array, i_placa):
    m, n = np.shape(array)

    novo_array = deepcopy(array)
    j_inicial = int(n / 4)
    for j in range(j_inicial, n - j_inicial):
        novo_array[i_placa, j] = 1.0
        novo_array[-(i_placa + 1), j] = - 1.0

    return novo_array


def id_cond_contorno_2d(array):
    m, n = np.shape(array)

    acumulador = []
    for i in range(1, m - 1):
        for j in range(1, n - 1):
            if array[i, j] == np.inf:
                acumulador.append([i, j])

    return acumulador


def inf_para_zero_2d(array):
    m, n = np.shape(array)
    novo_array = deepcopy(array)

    for i in range(m):
        for j in range(n):
            if array[i, j] == np.inf:
                novo_array[i, j] = 0.0
    return novo_array


def campo_eletrico_x(V, dx):
    m, n = np.shape(V)
    E_x = np.full((m, n), 0.0)
    for i in range(1, m - 1):
        for j in range(1, n - 1):
            E_x[i, j] = - (V[i + 1, j] - V[i - 1, j]) / (2 * dx)
    return E_x


def campo_eletrico_y(V, dy):
    m, n = np.shape(V)
    E_y = np.full((m, n), 0.0)
    for i in range(1, m - 1):
        for j in range(1, n - 1):
            E_y[i, j] = - (V[i, j + 1] - V[i, j - 1]) / (2 * dy)
    return E_y


def magnitude_campo_eletrico(V, dx, dy):
    E_x = campo_eletrico_x(V, dx)
    E_y = campo_eletrico_y(V, dy)
    return np.sqrt(E_x**2 + E_y**2)


def soma_primeiros_vizinhos(array, i, j):
    m, n = np.shape(array)
    if i == m - 1 and j == n - 1:
        soma = array[i, j - 1] + array[i, 0] + \
            array[i - 1, j] + array[0, j]
    elif i == m - 1 and j < n - 1:
        soma = array[i, j - 1] + array[i, j + 1] + \
            array[i - 1, j] + array[0, j]
    elif i < m - 1 and j == n - 1:
        soma = array[i, j - 1] + array[i, 0] + \
            array[i - 1, j] + array[i + 1, j]
    else:
        soma = array[i, j - 1] + array[i, j + 1] + \
            array[i - 1, j] + array[i + 1, j]
    return soma
