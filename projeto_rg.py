import numpy as np
import numpy.linalg as LA
from tqdm.auto import trange


def S_ip_x_0(b: int, i: int) -> np.ndarray:
    S_x = np.array(
        # object=[[0, 1], [1, 0]],  # base z
        object=[[1, 0], [0, -1]],  # base x
        dtype=float,
    )

    I = np.identity(
        n=2,
        dtype=float,
    )

    matriz = np.identity(
        n=1,
        dtype=float,
    )

    for j in np.arange(1, b + 1):
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


def S_ip_z_0(b: int, i: int) -> np.ndarray:
    # un_im = complex(0, 1)
    S_z = np.array(
        # object=[[1, 0], [0, -1]],  # base z
        object=[[0, 1], [1, 0]],  # base x
        dtype=float,
    )

    I = np.identity(
        n=2,
        dtype=float,
    )

    matriz = np.identity(
        n=1,
        dtype=float,
    )

    for j in np.arange(1, b + 1):
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


def H_p_0(b: int, J_0: float, Gamma_0: float):
    formato = (2**b, 2**b)
    termo_int = np.zeros(
        shape=formato,
        dtype=float
    )
    termo_campo = termo_int.copy()

    for i in range(1, (b - 1) + 1):
        termo_int += np.matmul(
            S_ip_z_0(b, i),
            S_ip_z_0(b, i + 1)
        )

    for i in range(1, b + 1):
        termo_campo += S_ip_x_0(b, i)

    H = - (J_0 * termo_int + Gamma_0 * termo_campo)

    return H


def diagonalizacao(H: np.ndarray):
    w, v = LA.eigh(
        a=H
    )
    E_0 = w[0]
    E_1 = w[1]
    ket_0 = v[:, 0]
    ket_1 = v[:, 1]

    return E_0, E_1, ket_0, ket_1


def S_p_x_n(ket_0_n: np.ndarray, ket_1_n: np.ndarray):
    b = int(np.log2(ket_0_n.size))
    formato = (ket_0_n.size, ket_1_n.size)
    matriz = np.zeros(
        shape=formato,
        dtype=float
    )
    matriz += np.matmul(
        ket_0_n.reshape(formato[0], 1),
        ket_0_n.reshape(1, formato[0])
    )
    matriz += - np.matmul(
        ket_1_n.reshape(formato[0], 1),
        ket_1_n.reshape(1, formato[0])
    )

    return matriz


def S_p_z_n(ket_0_n: np.ndarray, ket_1_n: np.ndarray):
    b = int(np.log2(ket_0_n.size))
    formato = (ket_0_n.size, ket_1_n.size)
    matriz = np.zeros(
        shape=formato,
        dtype=float
    )
    matriz += np.matmul(
        ket_1_n.reshape(formato[0], 1),
        ket_0_n.reshape(1, formato[0])
    )
    matriz += np.matmul(
        ket_0_n.reshape(formato[0], 1),
        ket_1_n.reshape(1, formato[0])
    )

    return matriz


def Gamma_nmais1(E_0_n: np.ndarray, E_1_n: np.ndarray):
    array = (1 / 2) * (E_1_n - E_0_n)
    return array


def eta_0(ket_0_1: np.ndarray, ket_1_1: np.ndarray):
    b = int(np.log2(ket_0_1.size))

    eta = LA.multi_dot([
        ket_0_1,
        S_ip_z_0(b, 1),
        ket_1_1
    ])

    return eta


# def eta_n(ket_0_nmais1: np.ndarray, ket_1_nmais1: np.ndarray, matriz_Sz):
#     b = int(np.log2(ket_0_nmais1.size))

#     eta = LA.multi_dot([
#         ket_0_nmais1,
#         S_p_z_n(),
#         ket_1_nmais1
#     ])

#     return eta


def J_nmais1(J_n: float, ket_0_n: np.ndarray, ket_1_n: np.ndarray):
    J = eta_0(ket_0_n, ket_1_n) ** 2 * J_n
    return J


def c_nmais1(b: int, c_n: float, E_0_nmais1: float, E_1_nmais1: float):
    C = b * c_n + (E_1_nmais1 + E_0_nmais1) / 2
    return C


# def H_p_n(b: int):
#     formato = (2**b, 2**b, array_J_0.size, array_Gamma_0.size)
#     array_H = np.zeros(
#         shape=formato,
#         dtype=float,
#     )

#     for (m, n) in np.ndindex(array_J_0.size, array_Gamma_0.size):
#         termo_int = np.zeros(
#             shape=(2**b, 2**b),
#             dtype=float,
#         )
#         termo_campo = termo_int.copy()

#         for i in range(1, (b - 1) + 1):
#             termo_int += np.matmul(
#                 S_i_z_0(b, i),
#                 S_i_z_0(b, i + 1)
#             )

#         for i in range(1, b + 1):
#             termo_campo += S_i_x_0(b, i)

#         array_H[:, :, m, n] = - (array_J_0[m] * termo_int +
#                                  array_Gamma_0[n] * termo_campo)

#     return array_H


array_J_0 = np.linspace(
    0,
    10,
    # num=100
)
array_Gamma_0 = np.copy(array_J_0)

