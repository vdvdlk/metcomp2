from numpy import (arange, array, identity, kron, linspace, matmul, ndarray,
                   ndindex, zeros, ones)
from numpy.linalg import eigh
from tqdm.auto import trange


def imprimir_base(N: int):
    base = ['+', '-']
    array_indices = 2 * ones(N, dtype=int)

    i = 0
    for indices in ndindex(tuple(array_indices)):
        string = ''
        for indice in indices:
            string += base[indice]
        print(i, '|' + string + '>')
        i += 1


def S_i_x(N: int, i: int) -> ndarray:
    S_x = array(
        # object=[[0, 1], [1, 0]],  # base z
        object=[[1, 0], [0, -1]],  # base x
        dtype=complex,
    )

    I = identity(
        n=2,
        dtype=complex,
    )

    matriz = identity(
        n=1,
        dtype=complex,
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
    un_im = complex(0, 1)
    S_z = array(
        # object=[[1, 0], [0, -1]],  # base z
        object=[[0, 1], [1, 0]],  # base x
        dtype=complex,
    )

    I = identity(
        n=2,
        dtype=complex,
    )

    matriz = identity(
        n=1,
        dtype=complex,
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
        dtype=complex,
    )

    termo_campo = zeros(
        shape=(2 ** N, 2 ** N),
        dtype=complex,
    )

    for i in arange(1, (N - 1) + 1):
        termo_int += matmul(
            S_i_z(N=N, i=i),
            S_i_z(N=N, i=i + 1),
        )

    for i in arange(1, N + 1):
        termo_campo += S_i_x(N=N, i=i)

    return -(termo_int + lamda * termo_campo)


def hamiltoniana_N_ccp(N: int, lamda: float) -> ndarray:
    termo_int = zeros(
        shape=(2 ** N, 2 ** N),
        dtype=complex,
    )

    termo_campo = zeros(
        shape=(2 ** N, 2 ** N),
        dtype=complex,
    )

    I_N = identity(
        n=2 ** N,
        dtype=complex
    )

    for i in arange(1, (N - 1) + 1):
        termo_int += matmul(
            S_i_z(N=N, i=i),
            S_i_z(N=N, i=i + 1),
        )
    
    if N != 2:
        termo_int += matmul(
            S_i_z(N=N, i=N),
            S_i_z(N=N, i=1)
        )

    for i in arange(1, N + 1):
        termo_campo += I_N - S_i_x(N=N, i=i)

    return termo_campo - lamda * termo_int


def diagonalizacao_N(N: int, array_lamda: ndarray):

    autovalores = zeros(
        shape=(array_lamda.size, 2 ** N),
        dtype=float,
    )

    autovetores = zeros(
        shape=(array_lamda.size, 2 ** N, 2 ** N),
        dtype=complex,
    )

    for k in trange(array_lamda.size, desc='Diagonalização da cadeia de Ising transverso para ' + str(N) + ' sítios'):
        H = hamiltoniana_N_ccp(
            N=N,
            lamda=array_lamda[k],
        )

        w, v = eigh(
            a=H,
        )

        autovalores[k, :] = w
        autovetores[k, :, :] = v

    return autovalores, autovetores
