from copy import deepcopy
from matrizes import id_cond_contorno, str_para_zero


def update_V(posicoes, matriz):
    nova_matriz = deepcopy(matriz)
    delta_V = 0.0
    for elemento in posicoes:
        i, j = elemento[0], elemento[1]
        nova_matriz[i][j] = (matriz[i - 1][j] + matriz[i + 1]
                             [j] + matriz[i][j - 1] + matriz[i][j + 1]) / 4
        delta_V += abs(matriz[i][j] - nova_matriz[i][j])
    return nova_matriz, delta_V


def laplace(matriz_inicial, erro=1e-5, printar=False):
    delta_V = 1.0
    pos = id_cond_contorno(matriz_inicial)
    matriz = str_para_zero(matriz_inicial)
    if printar == True:
        n = 0
    while abs(delta_V) > erro:
        update = update_V(pos, matriz)
        matriz = update[0]
        delta_V = update[1]
        if printar == True:
            n += 1
    if printar == True:
        print('O programa fez', n, 'iterações.')
    return matriz


# def poisson(matriz_inicial, erro=1e-6, printar=False):
#     delta_V = 1.0
#     pos = id_cond_contorno(matriz_inicial)
#     matriz = str_para_zero(matriz_inicial)
#     n = 0
#     while abs(delta_V) > erro:
#         update = update_V(pos, matriz)
#         matriz = update[0]
#         delta_V = update[1]
#         n += 1
#     if printar == True:
#         print(n)
#     return matriz
