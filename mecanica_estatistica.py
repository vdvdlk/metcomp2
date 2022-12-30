import numpy as np
from tqdm.auto import trange

T_c = 2 / np.log(1 + np.sqrt(2))


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


def magnetizacao(array_spins: np.ndarray) -> np.ndarray:
    array_M = np.sum(array_spins, axis=(0, 1))
    return array_M


def energia_rede(rede: np.ndarray, J: float = 1.0, mu: float = 1.0, H: float = 0.0, ccp: bool = True) -> float:
    termo_int = 0
    termo_campo = 0

    if J != 0.0:
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

    if mu != 0.0 and H != 0.0:
        termo_campo = rede.sum()

    E = - J * termo_int - mu * H * termo_campo
    return E  # type: ignore


def energia(array_spins: np.ndarray, J: float = 1.0, mu: float = 1.0, H: float = 0.0, ccp: bool = True) -> np.ndarray:
    t = array_spins.shape[2]
    array_E = np.zeros(t)

    for k in range(t):
        array_E[k] = energia_rede(
            rede=array_spins[:, :, k],
            J=J,
            mu=mu,
            H=H,
            ccp=ccp
        )

    return array_E


def ising_montecarlo(T: float, J: float = 1.0, H: float = 0.0, mu: float = 1.0, L: int = 10, t: int = 1000, ccp: bool = True, barra_de_progresso: bool = True):
    array_spins = inicializar_spins(L, t)
    rng = np.random.default_rng(12345)

    if T == 0.0:
        beta = np.inf
    else:
        beta = 1 / T

    if barra_de_progresso == True:
        alcance = trange(t, desc='Modelo de Ising por Monte Carlo')
    else:
        alcance = range(t)

    for k in alcance:
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
            else:
                r = rng.random()
                if r <= np.exp(- beta * E_flip):
                    array_spins[i, j, k + 1] *= - 1

    return array_spins


def calor_especifico_diff(array_E: np.ndarray, dT: float) -> np.ndarray:
    array_C = np.gradient(array_E, dT)
    return array_C


def calor_especifico_fd(array_varE: np.ndarray, array_T: np.ndarray) -> np.ndarray:
    array_C = array_varE / array_T**2
    return array_C
