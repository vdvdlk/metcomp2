import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from cadeia_ising_transverso import lista_base, indice_base, troca_spin_ket

S_x = np.array(
    object=[[0, 1], [1, 0]],
    dtype=int,
)
S_z = np.array(
    object=[[1, 0], [0, -1]],
    dtype=int,
)
I = np.identity(
    n=2,
    dtype=int,
)


def S_jp_x_n(n_s, p):
    matriz = np.identity(
        n=1,
        dtype=int,
    )

    for pp in range(1, n_s + 1):
        if pp == p:
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


def S_jp_z_n(n_s, p):
    matriz = np.identity(
        n=1,
        dtype=int,
    )

    for pp in range(1, n_s + 1):
        if pp == p:
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


def H_j_n(n_s, J_n, h_n):
    formato = 2 * (2**n_s,)
    termo_int = np.zeros(
        shape=formato,
        dtype=int
    )
    termo_campo = termo_int.copy()

    for p in range(1, (n_s - 1) + 1):
        termo_int += np.matmul(
            S_jp_x_n(n_s, p),
            S_jp_x_n(n_s, p + 1)
        )

    for p in range(1, n_s + 1):
        termo_campo += S_jp_z_n(n_s, p)

    H_j_n = - (J_n * termo_int + h_n * termo_campo)

    return H_j_n


def diagonalizacao(H_j_n):
    w, v = eigh(
        a=H_j_n,
        subset_by_index=[0, 1],
        driver='evr'
    )

    # Estado fundamental
    E_mais_nmais1 = w[0]
    ket_mais_nmais1 = v[:, 0]

    # Primeiro estado excitado
    E_menos_nmais1 = w[1]
    ket_menos_nmais1 = v[:, 1]

    return E_mais_nmais1, E_menos_nmais1, ket_mais_nmais1, ket_menos_nmais1


def S_j_x_n(ket_mais_n, ket_menos_n):
    formato = 2 * (ket_mais_n.size,)
    # matriz = np.zeros(
    #     shape=formato,
    #     dtype=float
    # )

    # matriz = np.matmul(
    #     ket_menos_n.reshape(formato[0], 1),
    #     ket_mais_n.reshape(1, formato[0])
    # )
    matriz = np.outer(
        a=ket_menos_n,
        b=ket_mais_n
    )
    # matriz += np.matmul(
    #     ket_mais_n.reshape(formato[0], 1),
    #     ket_menos_n.reshape(1, formato[0])
    # )
    matriz += np.outer(
        a=ket_mais_n,
        b=ket_menos_n
    )

    return matriz


def S_j_z_n(ket_mais_n, ket_menos_n):
    formato = 2 * (ket_mais_n.size,)
    # matriz = np.zeros(
    #     shape=formato,
    #     dtype=float
    # )

    # matriz = np.matmul(
    #     ket_mais_n.reshape(formato[0], 1),
    #     ket_mais_n.reshape(1, formato[0])
    # )
    matriz = np.outer(
        a=ket_menos_n,
        b=ket_menos_n
    )
    matriz += - np.outer(
        a=ket_mais_n,
        b=ket_mais_n
    )
    # matriz += - np.matmul(
    #     ket_menos_n.reshape(formato[0], 1),
    #     ket_menos_n.reshape(1, formato[0])
    # )

    return matriz


def csi_1_n(ket_mais_nmais1: np.ndarray, ket_menos_nmais1: np.ndarray):
    n_s = int(np.log2(ket_mais_nmais1.size))

    csi_1 = np.linalg.multi_dot([
        ket_mais_nmais1,
        S_jp_x_n(n_s, 1),
        ket_menos_nmais1
    ])

    return csi_1


def csi_p_n(ket_mais_nmais1, ket_menos_nmais1, p):
    n_s = int(np.log2(ket_mais_nmais1.size))
    base = lista_base(
        N=n_s
    )

    csi = 0
    for ket in base:
        if ket.count('-') % 2 == 0:
            ket_trocado = troca_spin_ket(
                ket=ket,
                pp=p
            )
            m = indice_base(
                ket=ket
            )
            n = indice_base(
                ket=ket_trocado
            )
            csi += ket_mais_nmais1[m] * ket_menos_nmais1[n]

    return csi


# def iteracao(n_s: int, J_n: float, h_n: float, C_n: float):
def iteracao(n_s, J_n, h_n):
    E_mais_nmais1, E_menos_nmais1, ket_mais_nmais1, ket_menos_nmais1 = diagonalizacao(
        H_j_n=H_j_n(n_s, J_n, h_n)
    )

    # S_j_x_nmais1 = S_j_x_n(
    #     ket_mais_n=ket_mais_nmais1,
    #     ket_menos_n=ket_menos_nmais1
    # )

    # S_j_z_nmais1 = S_j_z_n(
    #     ket_mais_n=ket_mais_nmais1,
    #     ket_menos_n=ket_menos_nmais1
    # )

    # csi_1 = csi_1_n(
    #     ket_mais_nmais1,
    #     ket_menos_nmais1
    # )
    csi_1 = csi_p_n(
        ket_mais_nmais1,
        ket_menos_nmais1,
        p=1
    )

    J_nmais1 = (csi_1 * csi_1) * J_n
    h_nmais1 = (0.5) * (E_menos_nmais1 - E_mais_nmais1)
    # C_nmais1 = n_s * C_n + (1 / 2) * (E_mais_nmais1 + E_menos_nmais1)

    # return J_nmais1, h_nmais1, C_nmais1
    return J_nmais1, h_nmais1


J_0 = 1.0


def renormalizacao(n_s, num_int, array_h_0):
    # C_0 = 0.0
    num_pt = array_h_0.size

    array_J_n = np.zeros(
        shape=num_pt,
        dtype=float,
    )
    array_h_n = array_J_n.copy()
    # array_C_n = array_J_n.copy()

    alcance_array = tqdm(
        range(num_pt),
        desc='Renormalização n_s = ' + str(n_s)
    )
    alcance_int = range(1, num_int + 1)
    for r in alcance_array:
        # J_n, h_n, C_n = (
        #     J_0,
        #     array_h_0[r],
        #     C_0
        # )
        J_n, h_n = (
            J_0,
            array_h_0[r]
        )
        for n in alcance_int:
            # J_n, h_n, C_n = iteracao(
            #     n_s=n_s,
            #     J_n=J_n,
            #     h_n=h_n,
            #     C_n=C_n
            # )
            J_n, h_n = iteracao(
                n_s=n_s,
                J_n=J_n,
                h_n=h_n
            )

        array_J_n[r] = J_n
        array_h_n[r] = h_n
        # array_C_n[r] = C_n

    # return array_J_n, array_h_n, array_C_n
    return array_J_n, array_h_n


def save_ren(num_int, num_pt, n_s_max = 7):
    array_h_0 = np.linspace(
        start=0,
        stop=2 * J_0,
        num=num_pt,
        dtype=float
    )
    np.save(
        file='projeto/renormalizacao/array_h_0',
        arr=array_h_0
    )

    for n_s in range(2, n_s_max + 1):
        # array_J, array_h, array_C = renormalizacao(
        array_J, array_h = renormalizacao(
            n_s=n_s,
            num_int=num_int,
            array_h_0=array_h_0,
        )
        np.save(
            file='projeto/renormalizacao/array_J_ns' + str(n_s),
            arr=array_J
        )
        np.save(
            file='projeto/renormalizacao/array_h_ns' + str(n_s),
            arr=array_h
        )
        # np.save(
        #     file='projeto/renormalizacao/array_C_ns' + str(n_s),
        #     arr=array_C
        # )

n_s_max = 2
save_ren(
    num_int=50,
    num_pt=1000,
    n_s_max=n_s_max
)

array_h_0 = np.load(
    file='projeto/renormalizacao/array_h_0.npy'
)

fig, axs = plt.subplots(
    ncols=2
)

pasta = 'projeto/renormalizacao/'

for n_s in range(2, n_s_max + 1):
    arquivo_J = pasta + 'array_J_ns' + str(n_s) + '.npy'
    array_J = np.load(
        file=arquivo_J
    )
    axs[0].plot(
        array_h_0,
        array_J,
        label=str(n_s)
    )
    arquivo_h = pasta + 'array_h_ns' + str(n_s) + '.npy'
    array_h = np.load(
        file=arquivo_h
    )
    axs[1].plot(
        array_h_0,
        array_h,
        label=str(n_s)
    )
axs[0].legend()
axs[0].set_xlabel('$h / J$')
axs[0].set_ylabel('$J^\\infty / J$')

axs[1].legend()
axs[1].set_xlabel('$h / J$')
axs[1].set_ylabel('$h^\\infty / J$')

fig.set_size_inches(15, 5)

# plt.show()
fig.savefig(pasta + 'teste.pdf')
