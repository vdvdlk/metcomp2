import lmfit
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import entropy
from tqdm.notebook import trange
from uncertainties import ufloat


def rwalk(p_esq=0.5, m=500, n=100):
    x = np.zeros((n + 1, m))
    for j in range(m):
        for i in np.arange(1, n + 1):
            r = np.random.rand()
            if r < 1 - p_esq:
                x[i, j] = x[i - 1, j] + 1
            else:
                x[i, j] = x[i - 1, j] - 1
    x2ave = np.sum(x**2, axis=1) / m

    return x, x2ave


def coeficiente_D(x2ave, modelo=lmfit.models.LinearModel()):
    ajuste = modelo.fit(x2ave, x=np.arange(x2ave.size))
    slope = ufloat(ajuste.params['slope'].value, ajuste.params['slope'].stderr)
    D = (1 / 2) * slope
    return D


# def P_rwalk(x:int, n:int):
#     """Distribuição de probabilidades de estar a uma distância x após n passos para o Random Walk"""
#     if (x + n) % 2 != 0 or x < 0 or n < 0:
#         fator = 0.0
#         coef = 0
#     else:
#         fator = np.power(
#             2,
#             -n,
#             dtype=float
#         )
#         coef = comb(
#             n,
#             (x + n) / 2,
#             exact=True
#         )
#     return fator * coef


def rwalk_2d(t=100, r_0=np.array([0, 0])):
    x_u = np.array([1, 0])
    y_u = np.array([0, 1])
    r = np.zeros((t + 1, 2))
    r[0, :] = r_0
    for n in np.arange(1, t + 1):
        q = np.random.rand()
        if q < 0.25:
            r[n] = r[n - 1] + x_u
        elif q >= 0.25 and q < 0.5:
            r[n] = r[n - 1] - x_u
        elif q >= 0.5 and q < 0.75:
            r[n] = r[n - 1] + y_u
        else:
            r[n] = r[n - 1] - y_u
    return r


def cafe_com_creme(m=400, t=10000):
    x_u = np.array([1, 0])
    y_u = np.array([0, 1])

    posicoes = np.zeros((m, t + 1, 2))
    particula = np.random.randint(m, size=t + 1)
    direcao = np.random.randint(4, size=t + 1)

    for n in trange(1, t + 1, desc='Café com creme'):
        # for n in np.arange(1, t + 1):
        p = particula[n]
        q = direcao[n]

        if q == 0:
            if posicoes[p, n - 1, 0] == 5:
                posicoes[p, n:, :] = posicoes[p, n - 1, :] - x_u
            else:
                posicoes[p, n:, :] = posicoes[p, n - 1, :] + x_u
        elif q == 1:
            if posicoes[p, n - 1, 0] == -5:
                posicoes[p, n:, :] = posicoes[p, n - 1, :] + x_u
            else:
                posicoes[p, n:, :] = posicoes[p, n - 1, :] - x_u
        elif q == 2:
            if posicoes[p, n - 1, 1] == 5:
                posicoes[p, n:, :] = posicoes[p, n - 1, :] - y_u
            else:
                posicoes[p, n:, :] = posicoes[p, n - 1, :] + y_u
        else:
            if posicoes[p, n - 1, 1] == -5:
                posicoes[p, n:, :] = posicoes[p, n - 1, :] + y_u
            else:
                posicoes[p, n:, :] = posicoes[p, n - 1, :] - y_u
    return posicoes


def contagem_subvolume(pos, passo, x_inf, x_sup, y_inf, y_sup):
    cont = 0

    for elemento in pos[:, passo, :]:
        if (elemento[0] >= x_inf and elemento[0] < x_sup) and (elemento[1] >= y_inf and elemento[1] < y_sup):
            cont += 1

    return cont


def contagens(pos, grid_x=np.arange(-5, 5, 5), grid_y=np.arange(-5, 5, 5)):
    delta_x = grid_x[1] - grid_x[0]
    delta_y = grid_y[1] - grid_y[0]

    n_x = np.size(grid_x)
    n_y = np.size(grid_y)
    n_t = np.shape(pos)[1]

    cont = np.zeros((n_x, n_y, n_t))

    for passo in np.arange(n_t):
        for i in np.arange(n_x):
            for j in np.arange(n_y):
                cont[i, j, passo] = contagem_subvolume(
                    pos,
                    passo,
                    grid_x[i],
                    grid_x[i] + delta_x,
                    grid_y[j],
                    grid_y[j] + delta_y
                )

    return cont


def entropia(pos):
    m = np.shape(pos)[0]
    P = contagens(pos) / m
    S = entropy(P, axis=(0, 1))
    return S


def posicao_aleatoria(R):
    angulo = (2 * np.pi - 0.0) * np.random.ranf() + 0.0
    x = np.ceil(R * np.cos(angulo))
    y = np.ceil(R * np.sin(angulo))
    pos = np.array([x, y], dtype=int)
    return pos


# def rwalk_2d_update(r):
#     # novo_r = np.append(r, np.zeros((1, 2)), axis=0)
#     # novo_r = np.zeros(2)
#     x_u = np.array([1, 0])
#     y_u = np.array([0, 1])
#     q = np.random.rand()
#     if q < 0.25:
#         # novo_r[-1, :] = r[-1, :] + x_u
#         novo_r = r + x_u
#     elif q >= 0.25 and q < 0.5:
#         # novo_r[-1, :] = r[-1, :] - x_u
#         novo_r = r - x_u
#     elif q >= 0.5 and q < 0.75:
#         # novo_r[-1, :] = r[-1, :] + y_u
#         novo_r = r + y_u
#     else:
#         # novo_r[-1, :] = r[-1, :] - y_u
#         novo_r = r - y_u
#     return novo_r


def rwalk_2d_update(r, tamanho_passo):
    x_u = np.array([1, 0])
    y_u = np.array([0, 1])
    q = np.random.rand()
    if q < 0.25:
        novo_r = r + tamanho_passo * x_u
    elif q >= 0.25 and q < 0.5:
        novo_r = r - tamanho_passo * x_u
    elif q >= 0.5 and q < 0.75:
        novo_r = r + tamanho_passo * y_u
    else:
        novo_r = r - tamanho_passo * y_u
    return novo_r


def detectar_vizinho(r, pos_ocup):
    x_u = np.array([1, 0], dtype=int)
    y_u = np.array([0, 1], dtype=int)
    m, n = np.shape(pos_ocup)

    i = 0
    status = False
    while status == False and i < m:
        r_ocup = pos_ocup[i, :]
        bool_cima = np.array_equal(r, r_ocup + x_u)
        bool_baixo = np.array_equal(r, r_ocup - x_u)
        bool_dir = np.array_equal(r, r_ocup + y_u)
        bool_esq = np.array_equal(r, r_ocup - y_u)
        status = bool_cima or bool_baixo or bool_dir or bool_esq
        i += 1
    return status


def tamanho_max_cluster(pos_ocup):
    return np.linalg.norm(pos_ocup, axis=1).max()


def dla(n_part=740):
    pos_ocup = np.zeros((n_part, 2))

    # for i in np.arange(1, n_part):
    for i in trange(1, n_part, desc='DLA cluster'):
        abs_r_ocup_max = tamanho_max_cluster(pos_ocup)
        if abs_r_ocup_max == 0.0:
            r_inicial = posicao_aleatoria(5.0)
        elif abs_r_ocup_max > 0.0:
            r_inicial = posicao_aleatoria(5 * abs_r_ocup_max)
        else:
            print('Erro')
            break
        abs_r_inicial = np.linalg.norm(r_inicial)

        r = r_inicial
        abs_r = abs_r_inicial
        tem_vizinho = False

        while tem_vizinho == False:
            if abs_r > 1.1 * abs_r_ocup_max and abs_r_ocup_max != 0.0:
                tamanho_passo = np.int64(np.ceil(abs_r / abs_r_ocup_max))
            else:
                tamanho_passo = 1
            r = rwalk_2d_update(r, tamanho_passo)
            abs_r = np.linalg.norm(r)
            tem_vizinho = detectar_vizinho(r, pos_ocup)
            if abs_r > 1.5 * abs_r_inicial:
                r = r_inicial
                abs_r = abs_r_inicial

        pos_ocup[i, :] = r

    return pos_ocup


def massa_cluster(pos_ocup):
    abs_r_ocup_max = tamanho_max_cluster(pos_ocup)
    r_max = np.int64(np.ceil(abs_r_ocup_max))
    array = np.zeros((r_max - 1, 2))
    array[:, 0] = np.arange(1, r_max)

    i = 0
    for raio in np.arange(1, r_max):
        array[i, 1] = (pos_ocup < raio).sum()
        i += 1
    return array


def linear(x, a, b):
    return a * x + b


def dim_fractal(massa, i_inicial=0, i_final=-1, func=linear):
    x = np.log(massa[i_inicial:i_final, 0])
    y = np.log(massa[i_inicial:i_final, 1])

    popt, pcov = curve_fit(func, x, y)

    return popt
