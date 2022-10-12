from copy import deepcopy

import numpy as np
from scipy.constants import epsilon_0


def inicializar(dim):
    tensor = np.full(dim, np.inf)
    return tensor


def metal_xi(tensor, constante):
    l, m, n = np.shape(tensor)
    novo_tensor = deepcopy(tensor)
    novo_tensor[0, :, :] = np.full((m, n), constante)
    return novo_tensor


def metal_xf(tensor, constante):
    l, m, n = np.shape(tensor)
    novo_tensor = deepcopy(tensor)
    novo_tensor[-1, :, :] = np.full((m, n), constante)
    return novo_tensor


def metal_yi(tensor, constante):
    l, m, n = np.shape(tensor)
    novo_tensor = deepcopy(tensor)
    novo_tensor[:, 0, :] = np.full((l, n), constante)
    return novo_tensor


def metal_yf(tensor, constante):
    l, m, n = np.shape(tensor)
    novo_tensor = deepcopy(tensor)
    novo_tensor[:, -1, :] = np.full((l, n), constante)
    return novo_tensor


def metal_zi(tensor, constante):
    l, m, n = np.shape(tensor)
    novo_tensor = deepcopy(tensor)
    novo_tensor[:, :, 0] = np.full((l, m), constante)
    return novo_tensor


def metal_zf(tensor, constante):
    l, m, n = np.shape(tensor)
    novo_tensor = deepcopy(tensor)
    novo_tensor[:, :, -1] = np.full((l, m), constante)
    return novo_tensor


def metal_bordas(tensor, constante):
    tensor = metal_xi(tensor, constante)
    tensor = metal_xf(tensor, constante)
    tensor = metal_yi(tensor, constante)
    tensor = metal_yf(tensor, constante)
    tensor = metal_zi(tensor, constante)
    tensor = metal_zf(tensor, constante)
    return tensor


def id_cond_contorno_3d(tensor):
    l, m, n = np.shape(tensor)

    acumulador = []
    for i in range(1, l - 1):
        for j in range(1, m - 1):
            for k in range(1, n - 1):
                if tensor[i, j, k] == np.inf:
                    acumulador.append([i, j, k])

    return acumulador


def inf_para_zero_3d(tensor):
    l, m, n = np.shape(tensor)
    novo_tensor = deepcopy(tensor)

    for i in range(l):
        for j in range(m):
            for k in range(n):
                if tensor[i, j, k] == np.inf:
                    novo_tensor[i, j, k] = 0.0
    return novo_tensor


def carga_puntiforme_central(dim, dx, q=epsilon_0):
    l, m, n = dim
    i_c = int(l / 2)
    j_c = int(m / 2)
    k_c = int(n / 2)

    tensor = np.zeros(dim)

    tensor[i_c, j_c, k_c] = q / (dx**3)

    return tensor


def carga_puntiforme(dim, indices, dx, q):
    i, j, k = indices

    tensor = np.zeros(dim)

    tensor[i, j, k] = q / (dx**3)

    return tensor


def secao_xy(tensor, k):
    matriz = tensor[:, :, k]
    return matriz
