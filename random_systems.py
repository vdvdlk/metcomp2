import random

import lmfit
import numpy
from scipy.special import comb
from uncertainties import ufloat


def rwalk(p_esq=0.5, m=500, n=100):
    x = numpy.zeros((n + 1, m))
    for j in range(m):
        for i in numpy.arange(1, n + 1):
            r = random.random()
            if r < 1 - p_esq:
                x[i, j] = x[i - 1, j] + 1
            else:
                x[i, j] = x[i - 1, j] - 1
    x2ave = numpy.sum(x**2, axis=1) / m

    modelo_1 = lmfit.models.LinearModel()
    ajuste = modelo_1.fit(x2ave, x=numpy.arange(n + 1))
    slope = ufloat(ajuste.params['slope'].value, ajuste.params['slope'].stderr)
    D = (1 / 2) * slope

    return x, x2ave, D


# def P_rwalk(x:int, n:int):
#     """Distribuição de probabilidades de estar a uma distância x após n passos para o Random Walk"""
#     if (x + n) % 2 != 0 or x < 0 or n < 0:
#         fator = 0.0
#         coef = 0
#     else:
#         fator = numpy.power(
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
