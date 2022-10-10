import numpy as np
from copy import deepcopy
from matrizes import id_cond_contorno, inf_para_zero


def update_V(posicoes, array):
    novo_array = deepcopy(array)
    delta_V = 0.0
    for elemento in posicoes:
        i, j = elemento
        novo_array[i, j] = (array[i - 1, j] + array[i + 1, j] +
                            array[i, j - 1] + array[i, j + 1]) / 4
        delta_V += np.abs(array[i, j] - novo_array[i, j])
    return novo_array, delta_V


def laplace(array_inicial, erro=1e-5, printar=False):
    delta_V = 1.0
    pos = id_cond_contorno(array_inicial)
    array = inf_para_zero(array_inicial)
    if printar == True:
        n = 0
    while np.abs(delta_V) > erro:
        array, delta_V = update_V(pos, array)
        if printar == True:
            n += 1
    if printar == True:
        print('O programa fez', n, 'iterações.')
    return array
