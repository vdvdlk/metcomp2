import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
from tqdm.auto import trange
from scipy.constants import Boltzmann


def fator_boltzmann(E, T, si=False):
    if si == False:
        k_B = 1.0
    else:
        k_B = Boltzmann

    if T == 0.0:
        fator = np.exp(- np.inf)
    else:
        fator = np.exp(- E / (k_B * T))
    return fator


def inicializar_spins(L, t):
    array = np.zeros(shape=(L, L, t + 1), dtype=int)
    array[:, :, 0] = np.random.choice(a=[-1, 1], size=(L, L))
    # return np.ones((L, L, t + 1), dtype=int)
    return array





def soma_dupla_primeiros_vizinhos(spins):
    soma = 0
    m, n = np.shape(spins)

    for i in range(m):
        for j in range(n):
            if i == m - 1 and j == n - 1:
                soma += spins[i, j] * \
                    (spins[i, j - 1] + spins[i, 0] +
                     spins[i - 1, j] + spins[0, j])
            elif i == m - 1 and j < n - 1:
                soma += spins[i, j] * \
                    (spins[i, j - 1] + spins[i, j + 1] +
                     spins[i - 1, j] + spins[0, j])
            elif i < m - 1 and j == n - 1:
                soma += spins[i, j] * \
                    (spins[i, j - 1] + spins[i, 0] +
                     spins[i - 1, j] + spins[i + 1, j])
            else:
                soma += spins[i, j] * \
                    (spins[i, j - 1] + spins[i, j + 1] +
                     spins[i - 1, j] + spins[i + 1, j])

    return soma



def energia_total(spins, J, H, mu):
    termo_inter = - J * soma_dupla_primeiros_vizinhos(spins)
    termo_mag = - mu * H * np.sum(spins)
    return termo_inter + termo_mag


def energia_flip(spins, i, j, J, H, mu):
    energia_inicial = energia_total(spins, J, H, mu)

    spins_final = deepcopy(spins)
    spins_final[i, j] = (- 1) * spins_final[i, j]
    energia_final = energia_total(spins_final, J, H, mu)

    return energia_final - energia_inicial





def ising_montecarlo_metropolis_geral(T, J=1.0, H=0.0, mu=0.0, L=10, t=1000):
    array_spins = inicializar_spins(L, t)

    for k in trange(t, desc='Modelo de Ising por Monte Carlo'):
        array_spins[:, :, k + 1] = deepcopy(array_spins[:, :, k])
        for i in range(L):
            for j in range(L):
                E_flip = energia_flip(array_spins[:, :, k], i, j, J, H, mu)
                if E_flip <= 0.0:
                    array_spins[i, j, k + 1] = (- 1) * array_spins[i, j, k]
                else:
                    r = np.random.random()
                    if r <= fator_boltzmann(E_flip, T):
                        array_spins[i, j, k + 1] = (- 1) * array_spins[i, j, k]

    return array_spins



def soma_primeiros_vizinhos(spins, i, j):
    soma = 0
    m, n = np.shape(spins)

    if i == m - 1 and j == n - 1:
        soma += spins[i, j] * \
            (spins[i, j - 1] + spins[i, 0] +
             spins[i - 1, j] + spins[0, j])
    elif i == m - 1 and j < n - 1:
        soma += spins[i, j] * \
            (spins[i, j - 1] + spins[i, j + 1] +
             spins[i - 1, j] + spins[0, j])
    elif i < m - 1 and j == n - 1:
        soma += spins[i, j] * \
            (spins[i, j - 1] + spins[i, 0] +
             spins[i - 1, j] + spins[i + 1, j])
    else:
        soma += spins[i, j] * \
            (spins[i, j - 1] + spins[i, j + 1] +
             spins[i - 1, j] + spins[i + 1, j])

    return soma


def energia_ij(spins, i, j, J, H, mu):
    termo_inter = - J * soma_primeiros_vizinhos(spins, i, j)
    termo_mag = - mu * H * spins[i, j]
    return termo_inter + termo_mag


def ising_montecarlo_metropolis_pv(T, J=1.0, H=0.0, mu=0.0, L=10, t=1000):
    array_spins = inicializar_spins(L, t)

    for k in trange(t, desc='Modelo de Ising por Monte Carlo'):
        array_spins[:, :, k + 1] = deepcopy(array_spins[:, :, k])
        for i in range(L):
            for j in range(L):
                E_flip = energia_flip(array_spins[:, :, k], i, j, J, H, mu)
                if E_flip <= 0.0:
                    array_spins[i, j, k + 1] = (- 1) * array_spins[i, j, k]
                else:
                    r = np.random.random()
                    if r <= fator_boltzmann(E_flip, T):
                        array_spins[i, j, k + 1] = (- 1) * array_spins[i, j, k]

    return array_spins


def array_momento_magnetico(array_spins):
    return np.sum(array_spins, axis=(0, 1))


def array_energias(array_spins, J, H, mu):
    m, n, t = np.shape(array_spins)
    energias = np.zeros(t)
    for k in range(t):
        energias[k] = energia_total(array_spins[:, :, k], J, H, mu)
    return energias


T_c = 2 / np.log(1 + np.sqrt(2))
array_teste = ising_montecarlo_metropolis_geral(T=2.25, t=1000)
mm_teste = array_momento_magnetico(array_teste)
# energias_teste = array_energias(array_teste, 1.0, 0.0, 1.0)


plt.plot(mm_teste)
plt.ylim(-200, 200)

plt.show()
