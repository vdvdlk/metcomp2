from copy import deepcopy
from numpy import float64
from math import sqrt
from tabulate import tabulate


def inicializar(dim):
    m = dim[0]
    n = dim[1]
    elemento = '-'
    matriz = []
    for i in range(m):
        linha = []
        for j in range(n):
            linha.append(elemento)
        matriz.append(linha)
    return matriz


def dim(matriz):
    m = len(matriz)
    n = len(matriz[0])
    return (m, n)


def transposta(matriz):
    m, n = dim(matriz)

    nova_matriz = []

    for i in range(n):
        nova_linha = []
        for j in range(m):
            list.append(nova_linha, matriz[j][i])
        list.append(nova_matriz, nova_linha)

    return nova_matriz


def mprint(matriz):
    # m, n = dim(matriz)

    # matriz_str = []
    # tamanhos = []

    # for j in range(n - 1, -1, -1):
    #     linha_str = []
    #     for i in range(m):
    #         elemento = matriz[i][j]
    #         if type(elemento) != str:
    #             elemento = round(elemento, 2)
    #         linha_str.append(str(elemento))
    #         tamanhos.append(len(str(elemento)))
    #     list.append(matriz_str, linha_str)

    # tamanho_max = max(tamanhos)

    # m_str, n_str = dim(matriz_str)
    # for i in range(m_str):
    #     for j in range(n_str):
    #         diferenca = tamanho_max - len(matriz_str[i][j])
    #         matriz_str[i][j] = diferenca*' ' + matriz_str[i][j]

    # for i in range(m_str):
    #     for j in range(n_str):
    #         print(matriz_str[i][j], end=' ')
    #     print(' ')
    # print()

    print(tabulate(matriz, tablefmt='plain', floatfmt='.2f'))


def metal_esq(matriz, constante):
    m, n = dim(matriz)

    nova_matriz = deepcopy(matriz)
    for j in range(n):
        nova_matriz[0][j] = constante

    return nova_matriz


def metal_dir(matriz, constante):
    m, n = dim(matriz)

    nova_matriz = deepcopy(matriz)
    for j in range(n):
        nova_matriz[-1][j] = constante

    return nova_matriz


def metal_baixo(matriz, constante):
    m, n = dim(matriz)

    nova_matriz = deepcopy(matriz)
    for i in range(m):
        nova_matriz[i][0] = constante

    return nova_matriz


def metal_cima(matriz, constante):
    m, n = dim(matriz)

    nova_matriz = deepcopy(matriz)
    for i in range(m):
        nova_matriz[i][-1] = constante

    return nova_matriz


def metal_bordas(matriz, constante):
    matriz_1 = metal_cima(matriz, constante)
    matriz_2 = metal_esq(matriz_1, constante)
    matriz_3 = metal_baixo(matriz_2, constante)
    matriz_4 = metal_dir(matriz_3, constante)
    return matriz_4


def metal_int(matriz, constante, fracao=1 / 3):
    m, n = dim(matriz)
    i_i = int(m * fracao)
    i_f = m - (i_i + 1)
    j_i = int(n * fracao)
    j_f = m - (j_i + 1)

    nova_matriz = deepcopy(matriz)
    for i in range(i_i, i_f + 1):
        for j in range(j_i, j_f + 1):
            nova_matriz[i][j] = constante

    return nova_matriz


def metal_placas(matriz, i_placa):
    m, n = dim(matriz)

    nova_matriz = deepcopy(matriz)
    j_inicial = int(n / 4)
    for j in range(j_inicial, n - j_inicial):
        nova_matriz[i_placa][j] = 1.0
        nova_matriz[-(i_placa + 1)][j] = - 1.0

    return nova_matriz


# def id_cond_contorno(matriz):
#     m, n = dim(matriz)

#     acumulador = []
#     for i in range(1, m - 1):
#         for j in range(1, n - 1):
#             if matriz[i][j] != '-':
#                 acumulador.append([i, j])

#     return acumulador


def id_cond_contorno(matriz):
    m, n = dim(matriz)

    acumulador = []
    for i in range(1, m - 1):
        for j in range(1, n - 1):
            if matriz[i][j] == '-':
                acumulador.append([i, j])

    return acumulador


def str_para_zero(matriz):
    nova_matriz = deepcopy(matriz)
    m, n = dim(nova_matriz)
    for i in range(m):
        for j in range(n):
            if matriz[i][j] == '-':
                nova_matriz[i][j] = 0.0
    return nova_matriz


# def campo_eletrico(V, dx):
#     m, n = dim(V)
#     E = inicializar((m, n))
#     for i in range(1, m - 1):
#         for j in range(1, n - 1):
#             E_x = - (V[i + 1][j] - V[i - 1][j]) / (2 * dx)
#             E_y = - (V[i][j + 1] - V[i][j - 1]) / (2 * dx)
#             E[i][j] = [E_x, E_y]
    # E = []
    # for i in range(1, m - 1):
    #     linha = []
    #     for j in range(1, n - 1):
    #         elemento_x = - (V[i + 1][j] - V[i - 1][j]) / (2 * dx)
    #         elemento_y = - (V[i][j + 1] - V[i][j - 1]) / (2 * dx)
    #         linha.append([elemento_x, elemento_y])
    #     E.append(linha)
    # return E


def campo_eletrico_x(V, dx):
    m, n = dim(V)
    E_x = inicializar((m, n))
    for i in range(1, m - 1):
        for j in range(1, n - 1):
            E_x[i][j] = - (V[i + 1][j] - V[i - 1][j]) / (2 * dx)
    # E = []
    # for i in range(1, m - 1):
    #     linha = []
    #     for j in range(1, n - 1):
    #         elemento_x = - (V[i + 1][j] - V[i - 1][j]) / (2 * dx)
    #         linha.append(elemento_x)
    #     E.append(linha)
    return E_x


def campo_eletrico_y(V, dy):
    m, n = dim(V)
    E_y = inicializar((m, n))
    for i in range(1, m - 1):
        for j in range(1, n - 1):
            E_y[i][j] = - (V[i][j + 1] - V[i][j - 1]) / (2 * dy)
    # E = []
    # for i in range(1, m - 1):
    #     linha = []
    #     for j in range(1, n - 1):
    #         elemento_y = - (V[i][j + 1] - V[i][j - 1]) / (2 * dy)
    #         linha.append(elemento_y)
    #     E.append(linha)
    return E_y


# def magnitude_campo_eletrico(V, dx, dy):
#     m, n = dim(E)
#     abs_E = inicializar((m, n))
#     for i in range(1, m - 1):
#         for j in range(1, n - 1):
#             abs_E2 = 0.0
#             for elemento in E[i][j]:
#                 abs_E2 += elemento**2
#             abs_E[i][j] = sqrt(abs_E2)
#     return abs_E
