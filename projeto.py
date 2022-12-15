import matplotlib.pyplot as plt
import numpy as np

from cadeia_ising_transverso import diagonalizacao_N


# Diagonalização exata de cadeia finita

# N = 2

lamda_2, autoval_2, autovet_2 = diagonalizacao_N(
    N=2,
)

fig_2, ax_2 = plt.subplots()
ax_2.set_title('Autovalores em função de $\lambda$ (N = 2)')
ax_2.set_xlabel('$\lambda$')
ax_2.set_ylabel('Energia')

for k in range(2 ** 2):
    ax_2.plot(lamda_2, autoval_2[:, k])


# N = 4

lamda_4, autoval_4, autovet_4 = diagonalizacao_N(
    N=4,
)

fig_4, ax_4 = plt.subplots()
ax_4.set_title('Autovalores em função de $\lambda$ (N = 4)')
ax_4.set_xlabel('$\lambda$')
ax_4.set_ylabel('Energia')

for k in range(2 ** 4):
    ax_4.plot(lamda_4, autoval_4[:, k])


# N = 8  (5 min)

lamda_8, autoval_8, autovet_8 = diagonalizacao_N(
    N=8,
)

fig_8, ax_8 = plt.subplots()
ax_8.set_title('Autovalores em função de $\lambda$ (N = 8)')
ax_8.set_xlabel('$\lambda$')
ax_8.set_ylabel('Energia')

for k in range(2 ** 8):
    ax_8.plot(lamda_8, autoval_8[:, k])


plt.show()

