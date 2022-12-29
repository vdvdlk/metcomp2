import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import trange


def inicializar_spins(L: int, t: int) -> np.ndarray:
    formato_total = (L, L, t + 1)
    array_spins = np.zeros(
        shape=formato_total,
        dtype=int
    )

    formato = (L, L)
    array_spins[:, :, 0] = np.ones(
        shape=formato,
        dtype=int
    )

    return array_spins


def energia_rede(rede: np.ndarray, J: float = 1.0, mu: float = 1.0, H: float = 0.0, ccp: bool = True) -> float:
    termo_int = 0
    termo_int += np.sum(
        a=rede[:-1, :] * rede[1:, :]
    )
    termo_int += np.sum(
        a=rede[:, :-1] * rede[:, 1:]
    )
    L = rede.shape[0]
    if L != 2 and ccp == True:
        termo_int += np.sum(
            a=rede[-1, :] * rede[0, :]
        )
        termo_int += np.sum(
            a=rede[:, -1] * rede[:, 0]
        )

    termo_campo = rede.sum()

    E = - J * termo_int - mu * H * termo_campo
    return E


def magnetizacao(array_spins: np.ndarray) -> np.ndarray:
    L = array_spins.shape[0]
    return np.sum(array_spins, axis=(0, 1)) / L**2


# def energia(array_spins: np.ndarray, J: float = 1.0, mu: float = 1.0, H: float = 0.0, ccp: bool = True) -> float:
#     termo_int = np.zeros(shape=array_spins.shape[2], dtype=int)
#     termo_int += np.sum(
#         a=array_spins[:-1, :, :] * array_spins[1:, :, :],
#         axis=(0, 1)
#     )
#     termo_int += np.sum(
#         a=array_spins[:, :-1, :] * array_spins[:, 1:, :],
#         axis=(0, 1)
#     )
#     L = array_spins.shape[0]
#     if L != 2 and ccp == True:
#         termo_int += np.sum(
#             a=array_spins[-1, :, :] * array_spins[0, :, :],
#         axis=(0, 1)
#         )
#         termo_int += np.sum(
#             a=array_spins[:, -1, :] * array_spins[:, 0, :],
#         axis=(0, 1)
#         )

#     termo_campo = array_spins.sum(axis=(0, 1))

#     E = - J * termo_int - mu * H * termo_campo
#     return E


def ising_montecarlo(T: float, J: float = 1.0, H: float = 0.0, mu: float = 1.0, L: int = 10, t: int = 1000, ccp: bool = True):
    array_spins = inicializar_spins(L, t)
    rng = np.random.default_rng()

    for k in trange(t, desc='Modelo de Ising por Monte Carlo'):
        array_spins[:, :, k + 1] = np.copy(array_spins[:, :, k])

        indices = np.ndindex(array_spins[:, :, k + 1].shape)
        for (i, j) in indices:
            rede = array_spins[:, :, k + 1]
            rede_flip = np.copy(rede)
            rede_flip[i, j] *= - 1

            E_flip = energia_rede(rede_flip, J, mu, H, ccp) - \
                energia_rede(rede, J, mu, H, ccp)

            if E_flip <= 0.0:
                array_spins[i, j, k + 1] *= - 1
            elif E_flip > 0.0:
                r = rng.random()

                if r <= np.exp(- E_flip / T):
                    array_spins[i, j, k + 1] *= - 1

    return array_spins


T_c = 2 / np.log(1 + np.sqrt(2))
array_teste = ising_montecarlo(T=4.0, L=10, t=1000, ccp=True)
mm_teste = magnetizacao(array_teste)
# mm_teste = energia(array_teste)
# energias_teste = array_energias(array_teste, 1.0, 0.0, 1.0)


plt.plot(mm_teste)
plt.ylim(-2, 2)

plt.show()
