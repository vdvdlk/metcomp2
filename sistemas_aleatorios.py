import lmfit
import numpy as np
from uncertainties import ufloat
from scipy.stats import entropy


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


def rwalk_2d(t=100):
    x_u = np.array([1, 0])
    y_u = np.array([0, 1])
    r = np.zeros((t + 1, 2))
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

    for n in np.arange(1, t + 1):
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
    # posicoes = np.cumsum(posicoes, axis=1)
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
    S = np.sum(-np.log(P**P), axis=(0, 1))
    return S
