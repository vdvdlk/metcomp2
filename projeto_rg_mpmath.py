import pickle
from math import log2

import numpy as np
import matplotlib.pyplot as plt
import mpmath as mp
from tqdm.auto import tqdm

# from cadeia_ising_transverso import (indice_base, indice_base_impar,
#                                      indice_base_par, lista_base,
#                                      troca_spin_ket)

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

    H = - J_n * matriz_int - h_n * matriz_campo

    return H


def diagonalizacao(H_j_n):
    E, Q = mp.eigsy(H_j_n)

    # Estado fundamental
    E_mais_nmais1 = E[0]
    ket_mais_nmais1 = mp.chop(Q[:, 0])

    # Primeiro estado excitado
    E_menos_nmais1 = E[1]
    ket_menos_nmais1 = mp.chop(Q[:, 1])

    return E_mais_nmais1, E_menos_nmais1, ket_mais_nmais1, ket_menos_nmais1


def S_j_x_n(ket_mais_n, ket_menos_n):
    matriz = ket_menos_n * ket_mais_n.T + ket_mais_n * ket_menos_n.T
    return matriz


def S_j_z_n(ket_mais_n, ket_menos_n):
    matriz = ket_menos_n * ket_menos_n.T - ket_mais_n * ket_mais_n.T
    return matriz


def I_j_n(ket_mais_n, ket_menos_n):
    matriz = ket_menos_n * ket_menos_n.T + ket_mais_n * ket_mais_n.T
    return matriz


def csi_p_n(ket_mais_nmais1, ket_menos_nmais1, p: int):
    n_s = int(log2(ket_mais_nmais1.rows))
    csi = (ket_mais_nmais1.T * S_jp_x_n(n_s, p) * ket_menos_nmais1)[0, 0]
    return csi


# def csi_1_n(ket_mais_n, ket_menos_n):
#     n_s = int(log2(ket_mais_n.rows))
#     indices = indice_base_par(N=n_s)

#     csi = 0
#     for i in indices:
#         if i < 2**(n_s - 1):
#             i_menos = i + 2**(n_s - 1)
#         else:
#             i_menos = i - 2**(n_s - 1)

#         csi += ket_mais_n[i] * ket_menos_n[i_menos]

#     # csi = mp.fabs(csi)

#     return csi


def iteracao(n_s: int, J_n, h_n, C_n):
    E_mais_nmais1, E_menos_nmais1, ket_mais_nmais1, ket_menos_nmais1 = diagonalizacao(
        H_j_n=H_j_n(n_s, J_n, h_n)
    )

    csi_1 = csi_p_n(
        ket_mais_nmais1,
        ket_menos_nmais1,
        p=1
    )
    # csi_1 = csi_1_n(
    #     ket_mais_nmais1,
    #     ket_menos_nmais1
    # )

    J_nmais1 = csi_1**2 * J_n
    h_nmais1 = 0.5 * (E_menos_nmais1 - E_mais_nmais1)
    C_nmais1 = n_s * C_n + 0.5 * (E_mais_nmais1 + E_menos_nmais1)

    return J_nmais1, h_nmais1, C_nmais1


def renormalizacao(n_s, num_int, num_pt=200):
    J_0 = mp.mpf('1.0')
    C_0 = mp.mpf('0.0')
    lista_h_0 = mp.linspace(
        0.1,
        2 * J_0,
        num_pt
    )

    lista_J_n = []
    lista_h_n = []
    lista_C_n = []

    range_h_0 = tqdm(
        iterable=range(num_pt),
        desc='Renormalização n_s = ' + str(n_s)
    )

    for r in range_h_0:
        J_n, h_n, C_n = (
            J_0,
            lista_h_0[r],
            C_0
        )

        range_int = tqdm(
            iterable=range(1, num_int + 1),
            desc='Iteração'
        )

        for n in range_int:
            J_n, h_n, C_n = iteracao(
                n_s=n_s,
                J_n=J_n,
                h_n=h_n,
                C_n=C_n
            )

        lista_J_n.append(J_n)
        lista_h_n.append(h_n)
        lista_C_n.append(C_n)

    return lista_h_0, lista_J_n, lista_h_n, lista_C_n


def salvar_listas(n_s: int, num_int: int):
    lista_h_0, lista_J_n, lista_h_n, lista_C_n = renormalizacao(
        n_s=n_s,
        num_int=num_int,
    )

    arquivo_h_0 = 'projeto/renormalizacao/lista_h_0_' + str(n_s) + '.pickle'
    arquivo_J_n = 'projeto/renormalizacao/lista_J_n_' + str(n_s) + '.pickle'
    arquivo_h_n = 'projeto/renormalizacao/lista_h_n_' + str(n_s) + '.pickle'
    arquivo_C_n = 'projeto/renormalizacao/lista_C_n_' + str(n_s) + '.pickle'

    with open(arquivo_h_0, 'wb') as f:
        pickle.dump(lista_h_0, f)

    with open(arquivo_J_n, 'wb') as f:
        pickle.dump(lista_J_n, f)

    with open(arquivo_h_n, 'wb') as f:
        pickle.dump(lista_h_n, f)

    with open(arquivo_C_n, 'wb') as f:
        pickle.dump(lista_C_n, f)


def graficos(salvar=False, plotar=False):
    fig, axs = plt.subplots(
        nrows=2,
        ncols=2
    )

    x_0 = np.linspace(0, 2, 1000)
    y_0 = 1 - x_0**2
    axs[0][0].plot(
        x_0,
        y_0,
        color='black',
        label='Resultado exato'
    )

    x_1 = np.linspace(0, 2, 1000)
    y_1 = x_1 - 1.0
    axs[0][1].plot(
        x_1,
        y_1,
        color='black',
        label='Resultado exato'
    )

    lista_int = [15, 7, 5, 5, 5, 5]
    i = 0
    for n_s in [2, 3, 4, 5]:
        arquivo_h_0 = 'projeto/renormalizacao/lista_h_0_' + \
            str(n_s) + '.pickle'
        arquivo_J_n = 'projeto/renormalizacao/lista_J_n_' + \
            str(n_s) + '.pickle'
        arquivo_h_n = 'projeto/renormalizacao/lista_h_n_' + \
            str(n_s) + '.pickle'
        arquivo_C_n = 'projeto/renormalizacao/lista_C_n_' + \
            str(n_s) + '.pickle'

        with open(arquivo_h_0, 'rb') as f:
            lista_h_0 = pickle.load(f)
        array_h_0 = np.array(
            object=lista_h_0,
            dtype=float
        )
        dh = np.diff(array_h_0)[0]

        with open(arquivo_J_n, 'rb') as f:
            lista_J_n = pickle.load(f)

        with open(arquivo_h_n, 'rb') as f:
            lista_h_n = pickle.load(f)

        with open(arquivo_C_n, 'rb') as f:
            lista_C_n = pickle.load(f)
        array_C_n = np.array(
            object=lista_C_n,
            dtype=float
        )

        array_E_0_N = array_C_n / n_s**lista_int[i]
        array_dE_0_N = np.gradient(array_E_0_N, dh)
        array_d2E_0_N = np.gradient(array_dE_0_N, dh)

        axs[0][0].plot(
            lista_h_0,
            lista_J_n,
            label='$n_s = ' + str(n_s) + '$'
        )

        axs[0][1].plot(
            lista_h_0,
            lista_h_n,
            label='$n_s = ' + str(n_s) + '$'
        )

        axs[1][0].plot(
            lista_h_0,
            array_C_n / n_s**lista_int[i],
            label='$n_s = ' + str(n_s) + '$'
        )

        axs[1][1].plot(
            lista_h_0,
            - array_d2E_0_N,
            label='$n_s = ' + str(n_s) + '$'
        )

        i += 1

    axs[0][0].set_box_aspect(1)
    axs[0][0].set_xlabel('$h / J$')
    axs[0][0].set_ylabel('$J^\\infty / J$')
    axs[0][0].legend()
    axs[0][0].set_xlim(0, 2)
    axs[0][0].set_ylim(0, 1)

    axs[0][1].set_box_aspect(1)
    axs[0][1].set_xlabel('$h / J$')
    axs[0][1].set_ylabel('$h^\\infty / J$')
    axs[0][1].legend()
    axs[0][1].set_xlim(0, 2)
    axs[0][1].set_ylim(0, 1)

    axs[1][0].set_box_aspect(1)
    axs[1][0].set_xlabel('$h / J$')
    axs[1][0].set_ylabel('$(E_0 / N)_{N \\to \\infty}$')
    axs[1][0].legend()
    axs[1][0].set_xlim(0, 2)
    # axs[1][0].set_ylim(0, 3)

    axs[1][1].set_box_aspect(1)
    axs[1][1].set_xlabel('$h / J$')
    axs[1][1].set_ylabel('$\\chi_z$')
    axs[1][1].legend()
    axs[1][1].set_xlim(0.9, 1.3)
    axs[1][1].set_ylim(0, 3)

    fig.set_size_inches(10, 10)

    if salvar == True:
        fig.savefig(
            'projeto/apresentacao_final/renorm.pdf',
            bbox_inches='tight'
        )

    if plotar == True:
        plt.show()


if __name__ == '__main__':
    # graficos(salvar=True)

    # salvar_listas(2, 15)
    # salvar_listas(3, 7)
    # salvar_listas(4, 5)
    # salvar_listas(5, 5)
    salvar_listas(n_s=6, num_int=5)
