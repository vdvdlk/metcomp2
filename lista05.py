import lmfit
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import newton
from tqdm.auto import trange

import mecanica_estatistica as me

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    # "figure.dpi": 1000,
})


# Exercício 1 #######################

# item 1. a)  #######################


def funcao(s: float, z: int, T: float) -> float:
    return s - np.tanh((z / T) * s)


def funcao_derivada(s: float, z: int, T: float) -> float:
    return 1 - (z / T) / np.cosh((z / T) * s)**2


array_T_1a = np.linspace(
    start=0.1,
    stop=10,
    num=1000
)

array_S_4 = np.zeros_like(array_T_1a)
array_it_4 = np.zeros_like(array_T_1a)


for i in range(array_T_1a.size):
    resultado_a = newton(
        func=funcao,
        fprime=funcao_derivada,
        x0=5.0,
        full_output=True,
        args=(4, array_T_1a[i],),
        tol=1e-8
    )
    array_S_4[i] = resultado_a[0]
    array_it_4[i] = resultado_a[1].iterations


fig_1a, ax_1a = plt.subplots()

ax_1a.set_title('Magnetização em função da temperatura')
ax_1a.set_xlabel('Temperatura $T$ ($J / k_B$)')
ax_1a.set_ylabel('Magnetização $<s>$')
ax_1a.grid(visible=True)

ax_1a.plot(array_T_1a, array_S_4)


# item 1. b)  #######################

fig_1b, ax_1b = plt.subplots()

ax_1b.set_title('Número de iterações em função da temperatura')
ax_1b.set_xlabel('Temperatura $T$ ($J / k_B$)')
ax_1b.set_ylabel('Número de iterações')
ax_1b.grid(visible=True)

ax_1b.plot(array_T_1a, array_it_4)


# item 1. c)  #######################

def magnetizacao(z: int, T):
    return np.sqrt(3 * T ** 2 / z ** 3) * np.sqrt(z - T)


fig_1c, ax_1c = plt.subplots()

ax_1c.set_title('Magnetização em função da temperatura')
ax_1c.set_xlabel('Temperatura $T$ ($J / k_B$)')
ax_1c.set_ylabel('Magnetização $<s>$')
ax_1c.grid(visible=True)

ax_1c.plot(
    array_T_1a,
    array_S_4,
    label='Solução numérica',
)

array_T_1c = np.linspace(
    start=0.0,
    stop=4.0,
    num=1000
)
ax_1c.plot(
    array_T_1c,
    magnetizacao(z=4, T=array_T_1c),
    label='Solução analítica para $<s>$ pequeno'
)

ax_1c.legend()


# item 1. d)  #######################

array_S_6 = np.zeros(array_T_1a.size)
array_it_6 = np.zeros(array_T_1a.size)


for i in range(array_T_1a.size):
    resultado_d = newton(
        func=funcao,
        fprime=funcao_derivada,
        x0=5.0,
        full_output=True,
        args=(6, array_T_1a[i],),
        tol=1e-8
    )
    array_S_6[i] = resultado_d[0]
    array_it_6[i] = resultado_d[1].iterations

fig_1d, axs_1d = plt.subplots(
    ncols=2
)

axs_1d[0].set_title('Magnetização em função da temperatura')
axs_1d[0].set_xlabel('Temperatura $T$ ($J / k_B$)')
axs_1d[0].set_ylabel('Magnetização $<s>$')
axs_1d[0].grid(visible=True)

axs_1d[0].plot(
    array_T_1a,
    array_S_6,
    label='Solução numérica',
)

array_T_1d = np.linspace(
    start=0.0,
    stop=6.0,
    num=1000
)
axs_1d[0].plot(
    array_T_1d,
    magnetizacao(z=6, T=array_T_1d),
    label='Solução analítica para $<s>$ pequeno'
)

axs_1d[1].set_title('Número de iterações em função da temperatura')
axs_1d[1].set_xlabel('Temperatura $T$ ($J / k_B$)')
axs_1d[1].set_ylabel('Número de iterações')
axs_1d[1].grid(visible=True)

axs_1d[1].plot(array_T_1a, array_it_6)

fig_1d.set_size_inches(w=2 * 6.4, h=4.8)


# Exercício 2 #######################

# item 2. a)  #######################


array_T_2a = np.linspace(
    start=0,
    stop=10,
    num=100
)

array_M_2a = np.zeros_like(array_T_2a)
array_M_2a = np.load(
    file='lista05/array_M_2a.npy'
)

# for i in trange(array_T_2a.size, desc='Exercício 2. a)'):
#     array_spins = me.ising_montecarlo(
#         T=array_T_2a[i]
#     )
#     array_M_2a[i] = np.mean(
#         a=me.magnetizacao(array_spins=array_spins)
#     )
# np.save(
#     file='lista05/array_M_2a',
#     arr=array_M_2a,
# )

fig_2a, ax_2a = plt.subplots()

ax_2a.set_title('Magnetização em função da temperatura')
ax_2a.set_xlabel('Temperatura $T$ ($1 / k_B$)')
ax_2a.set_ylabel('Magnetização por spin $M$')
ax_2a.grid(visible=True)

ax_2a.plot(
    array_T_2a,
    array_M_2a,
    'o',
    markersize=2
)

# Salvar figuras ####################

# fig_1a.savefig(
#     fname='lista05/fig_1a.pdf'
# )
# fig_1b.savefig(
#     fname='lista05/fig_1b.pdf'
# )
# fig_1c.savefig(
#     fname='lista05/fig_1c.pdf'
# )
# fig_1d.savefig(
#     fname='lista05/fig_1d.pdf'
# )

fig_2a.savefig(
    fname='lista05/fig_2a.pdf'
)

# plt.show()
