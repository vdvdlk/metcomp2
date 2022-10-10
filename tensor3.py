from copy import deepcopy
# from math import sqrt

# from numpy import float64
from scipy.constants import epsilon_0


def inicializar(dim):
    l = dim[0]
    m = dim[1]
    n = dim[2]
    elemento = '-'
    tensor = []
    for i in range(l):
        matriz = []
        for j in range(m):
            linha = []
            for k in range(n):
                linha.append(elemento)
            matriz.append(linha)
        tensor.append(matriz)
    return tensor


def dim(tensor):
    l = len(tensor)
    m = len(tensor[0])
    n = len(tensor[0][0])
    return (l, m, n)


# def transposta(tensor):
#     l, m, n = dim(tensor)

#     novo_tensor = []

#     for i in range(n):
#         novo_linha = []
#         for j in range(m):
#             list.append(novo_linha, tensor[j][i])
#         list.append(novo_tensor, novo_linha)

#     return novo_tensor


# def tprint(tensor):
#     m, n = dim(tensor)

#     tensor_str = []
#     tamanhos = []

#     for j in range(n - 1, -1, -1):
#         linha_str = []
#         for i in range(m):
#             elemento = tensor[i][j]
#             if type(elemento) != str:
#                 elemento = round(elemento, 2)
#             linha_str.append(str(elemento))
#             tamanhos.append(len(str(elemento)))
#         list.append(tensor_str, linha_str)

#     tamanho_max = max(tamanhos)

#     m_str, n_str = dim(tensor_str)
#     for i in range(m_str):
#         for j in range(n_str):
#             diferenca = tamanho_max - len(tensor_str[i][j])
#             tensor_str[i][j] = diferenca*' ' + tensor_str[i][j]

#     for i in range(m_str):
#         for j in range(n_str):
#             print(tensor_str[i][j], end=' ')
#         print(' ')
#     print()


def metal_xi(tensor, constante):
    l, m, n = dim(tensor)

    novo_tensor = deepcopy(tensor)
    for j in range(m):
        for k in range(n):
            novo_tensor[0][j][k] = constante

    return novo_tensor


def metal_xf(tensor, constante):
    l, m, n = dim(tensor)

    novo_tensor = deepcopy(tensor)
    for j in range(m):
        for k in range(n):
            novo_tensor[-1][j][k] = constante

    return novo_tensor


def metal_yi(tensor, constante):
    l, m, n = dim(tensor)

    novo_tensor = deepcopy(tensor)
    for i in range(l):
        for k in range(n):
            novo_tensor[i][0][k] = constante

    return novo_tensor


def metal_yf(tensor, constante):
    l, m, n = dim(tensor)

    novo_tensor = deepcopy(tensor)
    for i in range(l):
        for k in range(n):
            novo_tensor[i][-1][k] = constante

    return novo_tensor


def metal_zi(tensor, constante):
    l, m, n = dim(tensor)

    novo_tensor = deepcopy(tensor)
    for i in range(l):
        for j in range(m):
            novo_tensor[i][j][0] = constante

    return novo_tensor


def metal_zf(tensor, constante):
    l, m, n = dim(tensor)

    novo_tensor = deepcopy(tensor)
    for i in range(l):
        for j in range(m):
            novo_tensor[i][j][-1] = constante

    return novo_tensor


def metal_bordas(tensor, constante):
    tensor = metal_xi(tensor, constante)
    tensor = metal_xf(tensor, constante)
    tensor = metal_yi(tensor, constante)
    tensor = metal_yf(tensor, constante)
    tensor = metal_zi(tensor, constante)
    tensor = metal_zf(tensor, constante)
    return tensor


def id_cond_contorno(tensor):
    l, m, n = dim(tensor)

    acumulador = []
    for i in range(1, l - 1):
        for j in range(1, m - 1):
            for k in range(1, n - 1):
                if tensor[i][j][k] == '-':
                    acumulador.append([i, j, k])

    return acumulador


def str_para_zero(tensor):
    novo_tensor = deepcopy(tensor)
    l, m, n = dim(novo_tensor)
    for i in range(l):
        for j in range(m):
            for k in range(n):
                if tensor[i][j][k] == '-':
                    novo_tensor[i][j][k] = 0.0
    return novo_tensor


def carga_puntiforme_central(dim, dx, q=epsilon_0):
    l, m, n = dim
    i_c = int(l / 2)
    j_c = int(m / 2)
    k_c = int(n / 2)

    tensor = inicializar(dim)
    tensor = str_para_zero(tensor)

    tensor[i_c][j_c][k_c] = q / (dx**3)

    return tensor


def carga_puntiforme(dim, indices, dx, q):
    i, j, k = indices

    tensor = inicializar(dim)
    tensor = str_para_zero(tensor)

    tensor[i][j][k] = q / (dx**3)

    return tensor


def secao_xy(tensor, k):
    l, m, n = dim(tensor)
    matriz = []
    for i in range(l):
        linha = []
        for j in range(m):
            linha.append(tensor[i][j][k])
        matriz.append(linha)
    return matriz
