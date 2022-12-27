from numpy import (arange, array, identity, kron, linspace, matmul, ndarray,
                   ndindex, ones, save, zeros)
from numpy.linalg import eigvalsh
from tqdm.auto import trange

N_max = 10


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
    # un_im = complex(0, 1)
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


# def hamiltoniana_N(N: int, lamda: float) -> ndarray:
#     termo_int = zeros(
#         shape=(2 ** N, 2 ** N),
#         dtype=complex,
#     )

#     termo_campo = zeros(
#         shape=(2 ** N, 2 ** N),
#         dtype=complex,
#     )

#     for i in arange(1, (N - 1) + 1):
#         termo_int += matmul(
#             S_i_z(N=N, i=i),
#             S_i_z(N=N, i=i + 1),
#         )

#     for i in arange(1, N + 1):
#         termo_campo += S_i_x(N=N, i=i)

#     return -(termo_int + lamda * termo_campo)


def hamiltoniana_N(N: int, lamda: float, ccp: bool = True) -> ndarray:
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

    if N != 2 and ccp == True:
        termo_int += matmul(
            S_i_z(N=N, i=N),
            S_i_z(N=N, i=1)
        )

    I_N = identity(
        n=2 ** N,
        dtype=complex
    )

    for i in arange(1, N + 1):
        termo_campo += I_N - S_i_x(N=N, i=i)

    return termo_campo - lamda * termo_int


def diagonalizacao_N(N: int, array_lamda: ndarray):

    autovalores = zeros(
        # shape=(array_lamda.size, 2 ** N),
        shape=(array_lamda.size, 2),
        dtype=float,
    )

    for k in trange(array_lamda.size, desc='Diagonalização da cadeia de Ising transverso para ' + str(N) + ' sítios'):
        H = hamiltoniana_N(
            N=N,
            lamda=array_lamda[k],
        )

        # Primeiro índice: lamda, Segundo índice: nível de energia
        autovalores[k, :] = eigvalsh(
            a=H,
        )[0:2]

    return autovalores


array_lamda = linspace(
    start=0.0,
    stop=10.0,
    num=1000,
)
# save(
#     file='projeto/array_lamda',
#     arr=array_lamda,
# )


def salvar_array_autoval(N: int, array_lamda: ndarray = array_lamda):
    autoval = diagonalizacao_N(
        N=N,
        array_lamda=array_lamda
    )
    save(
        file='projeto/autoval_' + str(N),
        arr=autoval,
    )


for N in arange(2, N_max + 1):
    autoval = diagonalizacao_N(
        N=N,
        array_lamda=array_lamda
    )
    save(
        file='projeto/autoval_' + str(N),
        arr=autoval[:, 0:2],
    )
