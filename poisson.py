from copy import deepcopy
from tensor3 import id_cond_contorno, str_para_zero
from scipy.constants import epsilon_0


def update_V(posicoes, tensor, dx, rho):
    novo_tensor = deepcopy(tensor)
    delta_V = 0.0
    for elemento in posicoes:
        i, j, k = elemento[0], elemento[1], elemento[2]
        novo_tensor[i][j][k] = (1 / 6) * (tensor[i + 1][j][k] + tensor[i - 1][j][k] + tensor[i][j + 1][k] + tensor[i]
                                          [j - 1][k] + tensor[i][j][k + 1] + tensor[i][j][k - 1]) + rho[i][j][k] * (dx**2) / (6 * epsilon_0)
        delta_V += abs(tensor[i][j][k] - novo_tensor[i][j][k])
    return novo_tensor, delta_V


def poisson(tensor_inicial, dx, rho, erro=1e-6, printar=False):
    delta_V = 1.0
    pos = id_cond_contorno(tensor_inicial)
    tensor = str_para_zero(tensor_inicial)
    n = 0
    while abs(delta_V) > erro:
        update = update_V(pos, tensor, dx, rho)
        tensor = update[0]
        delta_V = update[1]
        n += 1
    if printar == True:
        print('O programa fez', n, 'iterações.')
    return tensor
