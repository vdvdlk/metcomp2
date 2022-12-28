import lmfit
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import Boltzmann
from raizes import newton_rhaphson

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    # "figure.dpi": 1000,
})


# Exercício 1 #######################

# item 1. a)  #######################


def funcao(s: float, z: int, A: float) -> float:
    return s - np.tanh((z / A) * s)


def funcao_derivada(s: float, z: int, A: float) -> float:
    return 1 - z / (np.cosh((z / A) * s)**2)


def nr_adaptado():
    newton_rhaphson()


array_A = np.linspace(
    start=0,
    stop=10,
    num=1000
)

array_S_4 = np.zeros(array_A.size)
array_it_4 = np.zeros(array_A.size)

for i in range(array_A.size):
    f = funcao(

    )
    array_S_4[i], array_it_4 = newton_rhaphson(
        f=funcao(z=4)
    )

