from math import log2

import matplotlib.pyplot as plt
import mpmath as mp
from tqdm.auto import tqdm

from cadeia_ising_transverso import indice_base, lista_base, troca_spin_ket

mp.mp.dps = 10**4

S_x = mp.matrix([
    [0, 1],
    [1, 0]
])
S_z = mp.matrix([
    [1, 0],
    [0, -1]
])
I = mp.eye(2)


def kron(A, B):
    m = A.rows
    n = A.cols

    p = B.rows
    q = B.cols

    matriz = mp.matrix(p*m, q*n)
    for i in range(matriz.rows):
        for j in range(matriz.cols):
            matriz[i, j] = A[i//p, j//q] * B[i % p, j % q]

    return matriz


def S_jp_x_n(n_s: int, p: int):
    matriz = mp.eye(1)

    for pp in range(1, n_s + 1):
        if pp == p:
            matriz = kron(
                A=matriz,
                B=S_x
            )
        else:
            matriz = kron(
                A=matriz,
                B=I
            )

    return matriz


def S_jp_z_n(n_s: int, p: int):
    matriz = mp.eye(1)

    for pp in range(1, n_s + 1):
        if pp == p:
            matriz = kron(
                A=matriz,
                B=S_z
            )
        else:
            matriz = kron(
                A=matriz,
                B=I
            )

    return matriz


def H_j_n(n_s: int, J_n, h_n):
    matriz_int = mp.matrix(2**n_s, 2**n_s)
    matriz_campo = matriz_int.copy()

    for p in range(1, (n_s - 1) + 1):
        matriz_int += S_jp_x_n(n_s, p) * S_jp_x_n(n_s, p + 1)

    for p in range(1, n_s + 1):
        matriz_campo += S_jp_z_n(n_s, p)

    H = - (J_n * matriz_int + h_n * matriz_campo)

    return H


def diagonalizacao(H_j_n):
    E, Q = mp.eigh(H_j_n)

    # Estado fundamental
    E_mais_nmais1 = E[0]
    ket_mais_nmais1 = Q[:, 0]

    # Primeiro estado excitado
    E_menos_nmais1 = E[1]
    ket_menos_nmais1 = Q[:, 1]

    return E_mais_nmais1, E_menos_nmais1, ket_mais_nmais1, ket_menos_nmais1


# def S_j_x_n(ket_mais_n, ket_menos_n):
#     formato = 2 * (ket_mais_n.size,)
#     # matriz = np.zeros(
#     #     shape=formato,
#     #     dtype=float
#     # )

#     # matriz = np.matmul(
#     #     ket_menos_n.reshape(formato[0], 1),
#     #     ket_mais_n.reshape(1, formato[0])
#     # )
#     matriz = np.outer(
#         a=ket_menos_n,
#         b=ket_mais_n
#     )
#     # matriz += np.matmul(
#     #     ket_mais_n.reshape(formato[0], 1),
#     #     ket_menos_n.reshape(1, formato[0])
#     # )
#     matriz += np.outer(
#         a=ket_mais_n,
#         b=ket_menos_n
#     )

#     return matriz


# def S_j_z_n(ket_mais_n, ket_menos_n):
#     formato = 2 * (ket_mais_n.size,)
#     # matriz = np.zeros(
#     #     shape=formato,
#     #     dtype=float
#     # )

#     # matriz = np.matmul(
#     #     ket_mais_n.reshape(formato[0], 1),
#     #     ket_mais_n.reshape(1, formato[0])
#     # )
#     matriz = np.outer(
#         a=ket_menos_n,
#         b=ket_menos_n
#     )
#     matriz += - np.outer(
#         a=ket_mais_n,
#         b=ket_mais_n
#     )
#     # matriz += - np.matmul(
#     #     ket_menos_n.reshape(formato[0], 1),
#     #     ket_menos_n.reshape(1, formato[0])
#     # )

#     return matriz


def csi_1_n(ket_mais_nmais1, ket_menos_nmais1):
    n_s = int(log2(ket_mais_nmais1.rows))

    csi_1 = (ket_mais_nmais1.T * S_jp_x_n(n_s, 1) * ket_menos_nmais1)[0, 0]

    return csi_1


def csi_p_n(ket_mais_nmais1, ket_menos_nmais1, p):
    n_s = int(log2(ket_mais_nmais1.rows))
    base = lista_base(
        N=n_s
    )

    csi = 0
    for ket in base:
        if ket.count('-') % 2 == 0:
            ket_trocado = troca_spin_ket(
                ket=ket,
                p=p
            )
            m = indice_base(
                ket=ket
            )
            n = indice_base(
                ket=ket_trocado
            )
            csi += ket_mais_nmais1[m] * ket_menos_nmais1[n]

    return csi


def iteracao(n_s, J_n, h_n):
    E_mais_nmais1, E_menos_nmais1, ket_mais_nmais1, ket_menos_nmais1 = diagonalizacao(
        H_j_n=H_j_n(n_s, J_n, h_n)
    )

    csi_1 = csi_1_n(
        ket_mais_nmais1,
        ket_menos_nmais1
    )
    # csi_1 = csi_p_n(
    #     ket_mais_nmais1,
    #     ket_menos_nmais1,
    #     p=1
    # )

    J_nmais1 = (csi_1 * csi_1) * J_n
    h_nmais1 = mp.mpf('0.5') * (E_menos_nmais1 - E_mais_nmais1)

    return J_nmais1, h_nmais1


def renormalizacao(n_s, num_int=50, num_pt=51):
    J_0 = mp.mpf('1.0')
    lista_h_0 = mp.linspace(0, 2, num_pt)

    lista_J_n = []
    lista_h_n = []

    alcance_array = tqdm(
        range(num_pt),
        desc='Renormalização n_s = ' + str(n_s)
    )
    alcance_int = range(1, num_int + 1)
    for r in alcance_array:
        J_n, h_n = (
            J_0,
            lista_h_0[r]
        )
        for n in alcance_int:
            J_n, h_n = iteracao(
                n_s=n_s,
                J_n=J_n,
                h_n=h_n
            )

        lista_J_n.append(J_n)
        lista_h_n.append(h_n)

    return lista_h_0, lista_J_n, lista_h_n


print(S_jp_z_n(3, 4))

# fig, axs = plt.subplots(
#     ncols=2
# )

# axs[0].set_xlabel('$h / J$')
# axs[0].set_ylabel('$J^\\infty / J$')

# axs[1].set_xlabel('$h / J$')
# axs[1].set_ylabel('$h^\\infty / J$')

# fig.set_size_inches(15, 5)

# for n_s in range(2, 3 + 1):
#     lista_h_0, lista_J_n, lista_h_n = renormalizacao(
#         n_s=n_s,
#         num_int=100,
#     )
#     axs[0].plot(
#         lista_h_0,
#         lista_J_n,
#         label='$n_s = ' + str(n_s) + '$'
#     )
#     axs[1].plot(
#         lista_h_0,
#         lista_h_n,
#         label='$n_s = ' + str(n_s) + '$'
#     )

#     axs[0].legend()
#     axs[1].legend()

#     pasta = 'projeto/renormalizacao/'
#     fig.savefig(pasta + 'teste.pdf')

# # plt.show()
