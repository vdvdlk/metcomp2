import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigh
from tqdm.auto import tqdm

# from cadeia_ising_transverso import indice_base, lista_base, troca_spin_ket

S_x = np.array(
    object=[
        [0, 1],
        [1, 0]
    ],
    dtype=int,
)
S_z = np.array(
    object=[
        [1, 0],
        [0, -1]
    ],
    dtype=int,
)
I = np.identity(
    n=2,
    dtype=int,
)


def S_jp_x_n(n_s: int, p: int):
    array = np.identity(
        n=1,
        dtype=int,
    )

    for pp in range(1, n_s + 1):
        if pp == p:
            array = np.kron(
                a=array,
                b=S_x,
            )
        else:
            array = np.kron(
                a=array,
                b=I,
            )

    return array


def S_jp_z_n(n_s: int, p: int):
    array = np.identity(
        n=1,
        dtype=int,
    )

    for pp in range(1, n_s + 1):
        if pp == p:
            array = np.kron(
                a=array,
                b=S_z,
            )
        else:
            array = np.kron(
                a=array,
                b=I,
            )

    return array


def H_j_n(n_s: int, J_n: float, h_n: float):
    array_int = np.zeros(
        shape=2 * (2**n_s,),
        dtype=int
    )
    array_campo = array_int.copy()

    for p in range(1, (n_s - 1) + 1):
        array_int += S_jp_x_n(n_s, p) @ S_jp_x_n(n_s, p + 1)

    for p in range(1, n_s + 1):
        array_campo += S_jp_z_n(n_s, p)

    H = - J_n * array_int - h_n * array_campo

    return H


def diagonalizacao(H_j_n, tol=1e-10):
    w, v = eigh(
        a=H_j_n,
        subset_by_index=[0, 1],
        driver='evr'
    )

    # Estado fundamental
    E_mais_nmais1 = w[0]
    ket_mais_nmais1 = v[:, 0]
    ket_mais_nmais1[np.abs(ket_mais_nmais1) < tol] = 0.0

    # Primeiro estado excitado
    E_menos_nmais1 = w[1]
    ket_menos_nmais1 = v[:, 1]
    ket_menos_nmais1[np.abs(ket_menos_nmais1) < tol] = 0.0

    return E_mais_nmais1, E_menos_nmais1, ket_mais_nmais1, ket_menos_nmais1


def S_j_x_n(ket_mais_n, ket_menos_n):
    array = np.outer(
        a=ket_menos_n,
        b=ket_mais_n
    )
    array += np.outer(
        a=ket_mais_n,
        b=ket_menos_n
    )

    return array


def S_j_z_n(ket_mais_n, ket_menos_n):
    array = np.outer(
        a=ket_menos_n,
        b=ket_menos_n
    )
    array += - np.outer(
        a=ket_mais_n,
        b=ket_mais_n
    )

    return array


def I_j_n(ket_mais_n, ket_menos_n):
    array = np.outer(
        a=ket_menos_n,
        b=ket_menos_n
    )
    array += np.outer(
        a=ket_mais_n,
        b=ket_mais_n
    )

    return array


def csi_p_n(ket_mais_nmais1, ket_menos_nmais1, p: int):
    n_s = int(np.log2(ket_mais_nmais1.size))

    csi = ket_mais_nmais1 @ S_jp_x_n(n_s, p) @ ket_menos_nmais1

    return csi


def iteracao(n_s: int, J_n: float, h_n: float, C_n: float):
    E_mais_nmais1, E_menos_nmais1, ket_mais_nmais1, ket_menos_nmais1 = diagonalizacao(
        H_j_n=H_j_n(n_s, J_n, h_n)
    )

    csi_1 = csi_p_n(
        ket_mais_nmais1,
        ket_menos_nmais1,
        p=1
    )

    J_nmais1 = csi_1**2 * J_n
    h_nmais1 = (0.5) * (E_menos_nmais1 - E_mais_nmais1)
    C_nmais1 = n_s * C_n + (0.5) * (E_mais_nmais1 + E_menos_nmais1)

    return J_nmais1, h_nmais1, C_nmais1


def renormalizacao(n_s: int, num_int: int, num_pt: int = 200):
    J_0 = 1.0
    C_0 = 0.0
    array_h_0 = np.linspace(
        0.1,
        2 * J_0,
        num_pt
    )

    array_J_n = np.zeros(
        shape=num_pt,
        dtype=float,
    )
    array_h_n = array_J_n.copy()
    array_C_n = array_J_n.copy()

    alcance_array = tqdm(
        range(num_pt),
        desc='Renormalização n_s = ' + str(n_s)
    )
    alcance_int = range(1, num_int + 1)
    for r in alcance_array:
        J_n, h_n, C_n = (
            J_0,
            array_h_0[r],
            C_0
        )
        for n in alcance_int:
            J_n, h_n, C_n = iteracao(
                n_s=n_s,
                J_n=J_n,
                h_n=h_n,
                C_n=C_n
            )

        array_J_n[r] = J_n
        array_h_n[r] = h_n
        array_C_n[r] = C_n

    return array_h_0, array_J_n, array_h_n, array_C_n


def salvar_arrays(n_s: int, num_int: int):
    array_h_0, array_J_n, array_h_n, array_C_n = renormalizacao(
        n_s=n_s,
        num_int=num_int,
    )

    arquivo_h_0 = 'projeto/renormalizacao/lista_h_0_' + str(n_s)
    arquivo_J_n = 'projeto/renormalizacao/lista_J_n_' + str(n_s)
    arquivo_h_n = 'projeto/renormalizacao/lista_h_n_' + str(n_s)
    arquivo_C_n = 'projeto/renormalizacao/lista_C_n_' + str(n_s)

    np.save(
        file=arquivo_h_0,
        arr=array_h_0
    )

    # np.save(
    #     file=arquivo_J_n,
    #     arr=array_J_n
    # )

    np.save(
        file=arquivo_h_n,
        arr=array_h_n
    )

    np.save(
        file=arquivo_C_n,
        arr=array_C_n
    )


def grafico():
    fig, axs = plt.subplots(
        nrows=2,
        ncols=2
    )

    axs[0][0].set_box_aspect(1)
    axs[0][0].set_xlabel('$h / J$')
    axs[0][0].set_ylabel('$J^\\infty / J$')
    axs[0][0].set_xlim(0, 2)
    axs[0][0].set_ylim(0, 1)

    axs[0][1].set_box_aspect(1)
    axs[0][1].set_xlabel('$h / J$')
    axs[0][1].set_ylabel('$h^\\infty / J$')
    axs[0][1].set_xlim(0, 2)
    axs[0][1].set_ylim(0, 1)

    axs[1][0].set_box_aspect(1)
    axs[1][0].set_xlabel('$h / J$')
    axs[1][0].set_ylabel('$(E_0 / N)_{N \\to \\infty}$')
    axs[1][0].set_xlim(0, 2)
    # axs[1][0].set_ylim(0, 3)

    axs[1][1].set_box_aspect(1)
    axs[1][1].set_xlabel('$h / J$')
    axs[1][1].set_ylabel('$\\chi_z$')
    axs[1][1].set_xlim(0.9, 1.3)
    axs[1][1].set_ylim(0, 3)

    fig.set_size_inches(10, 10)

    x_0 = np.linspace(0, 1, 1000)
    y_0 = 1 - x_0**2
    axs[0][0].plot(
        x_0,
        y_0,
        color='black',
        label='Resultado exato'
    )

    x_1 = np.linspace(1, 2, 1000)
    y_1 = x_1 - 1.0
    axs[0][1].plot(
        x_1,
        y_1,
        color='black',
        label='Resultado exato'
    )

    lista_int = [15, 7, 5, 5, 5, 5]
    i = 0
    for n_s in [2, 3, 4, 5, 6, 7]:
        array_h_0, array_J_n, array_h_n, array_C_n = renormalizacao(
            n_s=n_s,
            num_int=lista_int[i]
        )

        dh = np.diff(array_h_0)[0]
        array_E_0_N = array_C_n / n_s**lista_int[i]
        array_dE_0_N = np.gradient(array_E_0_N, dh)
        array_d2E_0_N = np.gradient(array_dE_0_N, dh)

        axs[0][0].plot(
            array_h_0,
            array_J_n,
            label='$n_s = ' + str(n_s) + '$'
        )
        axs[0][1].plot(
            array_h_0,
            array_h_n,
            label='$n_s = ' + str(n_s) + '$'
        )

        axs[1][0].plot(
            array_h_0,
            array_C_n / n_s**lista_int[i],
            label='$n_s = ' + str(n_s) + '$'
        )

        axs[1][1].plot(
            array_h_0,
            - array_d2E_0_N,
            label='$n_s = ' + str(n_s) + '$'
        )

        axs[0][0].legend()
        axs[0][1].legend()
        axs[1][0].legend()
        axs[1][1].legend()

        fig.savefig(
            # fname='projeto/renormalizacao/teste_numpy.pdf',
            fname='projeto/apresentacao_final/teste_numpy.pdf',
            bbox_inches='tight'
        )

        i += 1

    # plt.show()


if __name__ == '__main__':
    grafico()
