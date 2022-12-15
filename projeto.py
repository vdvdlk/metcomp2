from numpy import argwhere, diff, sign
from matplotlib.pyplot import subplots, show

from cadeia_ising_transverso import diagonalizacao_N


# Diagonalização exata de cadeia finita


def intersec_grafico(f, g):
    return argwhere(diff(sign(f - g))).flatten()


def grafico_autoval_cf(N: int):
    lamda, autoval, autovet = diagonalizacao_N(N=N)

    fig, ax = subplots()
    ax.set_title('Autovalores em função de $\lambda$ (N = ' + str(N) + ')')
    ax.set_xlabel('$\lambda$')
    ax.set_ylabel('Energia')

    for k in range(2 ** N):
        ax.plot(lamda, autoval[:, k])

    return fig


# N = 2
fig_2 = grafico_autoval_cf(2)

# N = 4

fig_4 = grafico_autoval_cf(4)

# N = 6

fig_6 = grafico_autoval_cf(6)

# N = 8  (5 min)

fig_8 = grafico_autoval_cf(8)


show()
