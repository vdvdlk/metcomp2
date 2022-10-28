import numpy as np
from copy import deepcopy
from tqdm.auto import trange
from matrizes import soma_primeiros_vizinhos
from scipy.constants import Boltzmann


def inicializar_spins(L, t):
    return np.ones((L, L, t + 1), dtype=int)


def energia_spin(spins, i, j, J, H, mu):
    termo_inter = - J * soma_primeiros_vizinhos(spins, i, j)
    termo_mag = - mu * H
    return (termo_inter + termo_mag) * spins[i, j]


def energia_flip(spins, i, j, J, H, mu):
    E_inicial = energia_spin(spins, i, j, J, H, mu)

    spins_flip = deepcopy(spins)
    spins_flip[i, j] = (- 1) * spins[i, j]
    E_final = energia_spin(spins_flip, i, j, J, H, mu)
    return E_final - E_inicial


def fator_boltzmann(E, T, si=False):
    if si==True:
        k_B = Boltzmann
    else:
        k_B = 1.0
    
    if T == 0.0:
        fator = np.exp(- np.inf)
    else:
        fator = np.exp(- E / (k_B * T))
    return fator


def ising_montecarlo(T, J=1.0, H=0.0, mu=1.0, L=10, t=1000):
    array_spins = inicializar_spins(L, t)

    for n in trange(t, desc='Modelo de Ising por Monte Carlo'):
        array_spins[:, :, n + 1] == 1 * array_spins[:, :, n]
        for i in range(L):
            for j in range(L):
                E_flip = energia_flip(array_spins[:, :, n], i, j, J, H, mu)
                if E_flip <= 0.0:
                    array_spins[i, j, n + 1] = (- 1) * array_spins[i, j, n]
                else:
                    r = np.random.random()
                    if r <= fator_boltzmann(E_flip, T):
                        array_spins[i, j, n + 1] = (- 1) * array_spins[i, j, n]

    return array_spins


print(ising_montecarlo(T=-1000.0))
