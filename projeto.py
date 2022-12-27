from matplotlib.pyplot import show, subplots
from numpy import arange, argwhere, diff, linspace, load, ndarray, sign
# from lmfit

# Diagonalização exata de cadeia finita

N_max = 10

array_lamda = load(
    file='projeto/array_lamda.npy'
)


def intersec_grafico(f, g):
    return argwhere(diff(sign(f - g))).flatten()


# def grafico_autoval_cf(N: int, lamda: ndarray = array_lamda):
#     autoval = load(
#         file='projeto/autoval_' + str(N) + '.npy'
#     )

#     fig, ax = subplots()
#     ax.set_title('Autovalores em função de $\lambda$ (N = ' + str(N) + ')')
#     ax.set_xlabel('$\lambda$')
#     ax.set_ylabel('Energia')

#     for k in range(2 ** N):
#         ax.plot(lamda, autoval[:, k])

#     return fig


def gap_massa(L: int):
    autoval = load(
        file='projeto/autoval_' + str(L) + '.npy'
    )
    return autoval[:, 1] - autoval[:, 0]


def razao_gap(L: int):
    return L * gap_massa(L=L) / ((L - 1) * gap_massa(L=L - 1))


def lamda_c(L: int, array_lamda: ndarray = array_lamda):
    indice = intersec_grafico(
        f=razao_gap(L=L),
        g=1
    )
    return array_lamda[indice]


def grafico_gap(N_max: int = N_max, lamda: ndarray = array_lamda):
    fig, ax = subplots()
    ax.set_title('Gap de massa em função de $\\lambda$')

    ax.set_xlabel('$\\lambda$')
    ax.set_xlim(0, 10)

    ax.set_ylabel('$\\Delta (\\lambda)$')
    ax.set_ylim(0, 2)

    ax.grid(visible=True)

    for N in arange(2, N_max + 1):
        # autoval = load(
        #     file='projeto/autoval_' + str(N) + '.npy'
        # )
        gap = gap_massa(L=N)
        ax.plot(lamda, gap, label='L = ' + str(N))

    ax.legend()

    return fig


fig_gap = grafico_gap()


fig_razao, ax_razao = subplots()
ax_razao.plot(array_lamda, razao_gap(3))

# N = 2
# fig_2 = grafico_gap(2)
# fig_2 = grafico_autoval_cf(2)

# N = 3
# fig_3 = grafico_gap(3)
# fig_3 = grafico_autoval_cf(3)

# N = 4
# fig_4 = grafico_gap(4)
# fig_4 = grafico_autoval_cf(4)

# N = 5
# fig_5 = grafico_gap(5)
# fig_5 = grafico_autoval_cf(5)

# N = 6
# fig_6 = grafico_gap(6)
# fig_6 = grafico_autoval_cf(6)
# N = 7
# fig_7 = grafico_gap(7)
# fig_7 = grafico_autoval_cf(7)
# N = 8
# fig_8 = grafico_gap(8)
# fig_8 = grafico_autoval_cf(8)
show()
