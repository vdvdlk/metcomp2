from matplotlib.pyplot import show, subplots
from numpy import argwhere, diff, save, load, sign, linspace, arange, ndarray

from cadeia_ising_transverso import diagonalizacao_N

# Diagonalização exata de cadeia finita

# array_lamda = linspace(
#     start=0.0,
#     stop=10.0,
#     num=1000,
# )
# save(
#     file='projeto/array_lamda',
#     arr=array_lamda,
# )
array_lamda = load(
    file='projeto/array_lamda.npy'
)


# for N in arange(2, 9):
#     autoval, autovet = diagonalizacao_N(
#         N=N,
#         array_lamda=array_lamda
#     )
#     save(
#         file='projeto/autoval_' + str(N),
#         arr=autoval,
#     )
#     save(
#         file='projeto/autovet_' + str(N),
#         arr=autovet,
#     )


def intersec_grafico(f, g):
    return argwhere(diff(sign(f - g))).flatten()


def grafico_autoval_cf(N: int, lamda: ndarray = array_lamda):
    # autoval, autovet = diagonalizacao_N(
    #     N=N,
    # )

    autoval = load(
        file='projeto/autoval_' + str(N) + '.npy'
    )

    fig, ax = subplots()
    ax.set_title('Autovalores em função de $\lambda$ (N = ' + str(N) + ')')
    ax.set_xlabel('$\lambda$')
    ax.set_ylabel('Energia')

    for k in range(2 ** N):
        ax.plot(lamda, autoval[:, k])

    return fig


def grafico_gap(N: int, lamda: ndarray = array_lamda):
    # autoval, autovet = diagonalizacao_N(
    #     N=N,
    # )

    autoval = load(
        file='projeto/autoval_' + str(N) + '.npy'
    )

    fig, ax = subplots()
    ax.set_title('Autovalores em função de $\lambda$ (N = ' + str(N) + ')')
    ax.set_xlabel('$\lambda$')
    ax.set_ylabel('Energia')

    ax.plot(lamda, autoval[:, 1] - autoval[:, 0])

    return fig


# N = 2
fig_2 = grafico_gap(2)
# fig_2 = grafico_autoval_cf(2)

# N = 3
fig_3 = grafico_gap(3)
# fig_3 = grafico_autoval_cf(3)

# N = 4
fig_4 = grafico_gap(4)
# fig_4 = grafico_autoval_cf(4)

# N = 5
fig_5 = grafico_gap(5)
# fig_5 = grafico_autoval_cf(5)

# N = 6
fig_6 = grafico_gap(6)
# fig_6 = grafico_autoval_cf(6)

# N = 7
fig_7 = grafico_gap(7)
# fig_7 = grafico_autoval_cf(7)

# N = 8
fig_8 = grafico_gap(8)
# fig_8 = grafico_autoval_cf(8)

show()
