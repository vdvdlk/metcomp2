import numpy as np
import numpy.linalg as LA
from tqdm.auto import trange


def imprimir_base(N: int):
    base = ['+', '-']
    tupla_indices = N * (2,)

    i = 0
    for indices in np.ndindex(tupla_indices):
        string = ''
        for indice in indices:
            string += base[indice]
        print(i, '|' + string + '⟩')
        i += 1


def lista_base(N: int, ket: bool = True) -> list[str]:
    base = ['+', '-']
    tupla_indices = N * (2,)

    lista = []
    for indices in np.ndindex(tupla_indices):
        string = ''
        for indice in indices:
            string += base[indice]
        if ket == True:
            lista.append('|' + string + '⟩')
        else:
            lista.append(string)

    return lista


def indice_base(ket: str):
    N = ket.count('+') + ket.count('-')
    base = lista_base(N=N)
    indice = base.index(ket)
    return indice


def troca_spin_ket(ket: str, p: int):
    if ket[p] == '+':
        novo_ket = ket[0:p] + '-' + ket[p+1:]
    else:
        novo_ket = ket[0:p] + '+' + ket[p+1:]
    return novo_ket


def paridade_base(N: int) -> list[str]:
    lista = []
    for ket in lista_base(N):
        num = ket.count('-')
        if num % 2 == 0:
            lista.append('+')
        else:
            lista.append('-')

    return lista


# print(lista_base(3))
# print(paridade_base(4))


def S_i_x(N: int, i: int) -> np.ndarray:
    S_x = np.array(
        # object=[[0, 1], [1, 0]],  # base z
        object=[[1, 0], [0, -1]],  # base x
        dtype=complex,
    )

    I = np.identity(
        n=2,
        dtype=complex,
    )

    matriz = np.identity(
        n=1,
        dtype=complex,
    )

    for j in np.arange(1, N + 1):
        if j == i:
            matriz = np.kron(
                a=matriz,
                b=S_x,
            )
        else:
            matriz = np.kron(
                a=matriz,
                b=I,
            )

    return matriz


def S_i_z(N: int, i: int) -> np.ndarray:
    # un_im = complex(0, 1)
    S_z = np.array(
        # object=[[1, 0], [0, -1]],  # base z
        object=[[0, 1], [1, 0]],  # base x
        dtype=complex,
    )

    I = np.identity(
        n=2,
        dtype=complex,
    )

    matriz = np.identity(
        n=1,
        dtype=complex,
    )

    for j in np.arange(1, N + 1):
        if j == i:
            matriz = np.kron(
                a=matriz,
                b=S_z,
            )
        else:
            matriz = np.kron(
                a=matriz,
                b=I,
            )

    return matriz


def hamiltoniana_N(N: int, lamda: float, ccp: bool) -> np.ndarray:
    termo_int = np.zeros(
        shape=(2 ** N, 2 ** N),
        dtype=complex,
    )

    termo_campo = np.zeros(
        shape=(2 ** N, 2 ** N),
        dtype=complex,
    )

    for i in np.arange(1, (N - 1) + 1):
        termo_int += np.matmul(
            S_i_z(N=N, i=i),
            S_i_z(N=N, i=i + 1),
        )

    if N != 2 and ccp == True:
        termo_int += np.matmul(
            S_i_z(N=N, i=N),
            S_i_z(N=N, i=1)
        )

    I_N = np.identity(
        n=2 ** N,
        dtype=complex
    )

    for i in np.arange(1, N + 1):
        termo_campo += I_N - S_i_x(N=N, i=i)

    return termo_campo - lamda * termo_int


def diagonalizacao_N(N: int, array_lamda: np.ndarray, ccp: bool) -> np.ndarray:

    autovalores = np.zeros(
        shape=(array_lamda.size, 2),
        dtype=float,
    )

    for k in trange(array_lamda.size, desc='Diagonalização da cadeia de Ising transverso para ' + str(N) + ' sítios'):
        H = hamiltoniana_N(
            N=N,
            lamda=array_lamda[k],
            ccp=ccp
        )

        # Primeiro índice: lamda, Segundo índice: nível de energia
        autovalores[k, :] = LA.eigvalsh(
            a=H,
        )[0:2]

    return autovalores


array_lamda = np.linspace(
    start=0.0,
    stop=10.0,
    num=1000,
)


def salvar_array_autoval(N: int, ccp: bool, array_lamda: np.ndarray = array_lamda) -> None:
    autoval = diagonalizacao_N(
        N=N,
        array_lamda=array_lamda,
        ccp=ccp
    )

    string = 'projeto/autoval_'
    if ccp == True:
        string += 'ccp_'
    string += str(N)

    np.save(
        file=string,
        arr=autoval,
    )
