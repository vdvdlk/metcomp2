from numpy import (arange, array, identity, kron, linspace, matmul, ndarray,
                   zeros)
from numpy.linalg import eigh
from tqdm.auto import trange


def S_i_x(N: int, i: int) -> ndarray:
    S_x = array(
        object=[[0, 1], [1, 0]],
        dtype=float,
    )

    I = identity(
        n=2,
        dtype=float,
    )

    matriz = identity(
        n=1,
        dtype=float,
    )

    for j in arange(1, N + 1):
        if j == i:
            matriz = kron(
                a=matriz,
                b=S_x,
            )
        else:
            matriz = kron(
                a=matriz,
                b=I,
            )

    return matriz


def S_i_z(N: int, i: int) -> ndarray:
    S_z = array([[1, 0], [0, -1]])
    S_z = array(
        object=[[1, 0], [0, -1]],
        dtype=float,
    )

    I = identity(
        n=2,
        dtype=float,
    )

    matriz = identity(
        n=1,
        dtype=float,
    )

    for j in arange(1, N + 1):
        if j == i:
            matriz = kron(
                a=matriz,
                b=S_z,
            )
        else:
            matriz = kron(
                a=matriz,
                b=I,
            )

    return matriz


def hamiltoniana_N(N: int, lamda: float) -> ndarray:
    termo_int = zeros(
        shape=(2 ** N, 2 ** N),
        dtype=float,
    )

    termo_campo = zeros(
        shape=(2 ** N, 2 ** N),
        dtype=float,
    )

    for i in arange(1, (N - 1) + 1):
        termo_int += matmul(
            S_i_z(N=N, i=i),
            S_i_z(N=N, i=i + 1),
        )

    for i in arange(1, N + 1):
        termo_campo += S_i_x(N=N, i=i)

    return -(termo_int + lamda * termo_campo)


def diagonalizacao_N(N: int, lamda_min: float = 0.0, lamda_max: float = 5.0, num: int = 1000):
    array_lamda = linspace(
        start=lamda_min,
        stop=lamda_max,
        num=num,
    )

    autovalores = zeros(
        shape=(array_lamda.size, 2 ** N),
        dtype=float,
    )

    autovetores = zeros(
        shape=(array_lamda.size, 2 ** N, 2 ** N),
        dtype=complex,
    )

    for k in trange(array_lamda.size, desc='Diagonalização da cadeia de Ising transverso para ' + str(N) + ' sítios.'):
        H = hamiltoniana_N(
            N=N,
            lamda=array_lamda[k],
        )

        w, v = eigh(
            a=H,
        )

        autovalores[k, :] = w
        autovetores[k, :, :] = v

    return array_lamda, autovalores, autovetores