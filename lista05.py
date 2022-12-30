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


def funcao_1a(s: float, z: int, T: float) -> float:
    return s - np.tanh((z / T) * s)


def funcao_derivada_1a(s: float, z: int, T: float) -> float:
    return 1 - (z / T) / np.cosh((z / T) * s)**2


array_T_1a = np.linspace(
    start=0.1,
    stop=10,
    num=1000
)

array_S_4_1a = np.zeros_like(array_T_1a)
array_it_4_1a = np.zeros_like(array_T_1a)


for i in range(array_T_1a.size):
    resultado_a = newton(
        func=funcao_1a,
        fprime=funcao_derivada_1a,
        x0=5.0,
        full_output=True,
        args=(4, array_T_1a[i],),
        tol=1e-8
    )
    array_S_4_1a[i] = resultado_a[0]
    array_it_4_1a[i] = resultado_a[1].iterations


fig_1a, ax_1a = plt.subplots()

ax_1a.set_title('Magnetização em função da temperatura')
ax_1a.set_xlabel('Temperatura $T$ ($J / k_B$)')
ax_1a.set_ylabel('Magnetização $<s>$')
ax_1a.grid(visible=True)

ax_1a.plot(array_T_1a, array_S_4_1a)


# item 1. b)  #######################

fig_1b, ax_1b = plt.subplots()

ax_1b.set_title('Número de iterações em função da temperatura')
ax_1b.set_xlabel('Temperatura $T$ ($J / k_B$)')
ax_1b.set_ylabel('Número de iterações')
ax_1b.grid(visible=True)

ax_1b.plot(array_T_1a, array_it_4_1a)


# item 1. c)  #######################

def magnetizacao_1c(z: int, T):
    return np.sqrt(3 * T ** 2 / z ** 3) * np.sqrt(z - T)


fig_1c, ax_1c = plt.subplots()

ax_1c.set_title('Magnetização em função da temperatura')
ax_1c.set_xlabel('Temperatura $T$ ($J / k_B$)')
ax_1c.set_ylabel('Magnetização $<s>$')
ax_1c.grid(visible=True)

ax_1c.plot(
    array_T_1a,
    array_S_4_1a,
    label='Solução numérica',
)

array_T_1c = np.linspace(
    start=0.0,
    stop=4.0,
    num=1000
)
ax_1c.plot(
    array_T_1c,
    magnetizacao_1c(z=4, T=array_T_1c),
    label='Solução analítica para $<s>$ pequeno'
)

ax_1c.legend()


# item 1. d)  #######################

array_S_6_1d = np.zeros(array_T_1a.size)
array_it_6_1d = np.zeros(array_T_1a.size)


for i in range(array_T_1a.size):
    resultado_d = newton(
        func=funcao_1a,
        fprime=funcao_derivada_1a,
        x0=5.0,
        full_output=True,
        args=(6, array_T_1a[i],),
        tol=1e-8
    )
    array_S_6_1d[i] = resultado_d[0]
    array_it_6_1d[i] = resultado_d[1].iterations

fig_1d, axs_1d = plt.subplots(
    ncols=2
)

axs_1d[0].set_title('Magnetização em função da temperatura')
axs_1d[0].set_xlabel('Temperatura $T$ ($J / k_B$)')
axs_1d[0].set_ylabel('Magnetização $<s>$')
axs_1d[0].grid(visible=True)

axs_1d[0].plot(
    array_T_1a,
    array_S_6_1d,
    label='Solução numérica',
)

array_T_1d = np.linspace(
    start=0.0,
    stop=6.0,
    num=1000
)
axs_1d[0].plot(
    array_T_1d,
    magnetizacao_1c(z=6, T=array_T_1d),
    label='Solução analítica para $<s>$ pequeno'
)

axs_1d[1].set_title('Número de iterações em função da temperatura')
axs_1d[1].set_xlabel('Temperatura $T$ ($J / k_B$)')
axs_1d[1].set_ylabel('Número de iterações')
axs_1d[1].grid(visible=True)

axs_1d[1].plot(array_T_1a, array_it_6_1d)

fig_1d.set_size_inches(w=2 * 6.4, h=4.8)


# Exercício 2 #######################

# item 2. a)  #######################

L_2a = 10
N_2a = L_2a**2

array_T_2a = np.linspace(
    start=0,
    stop=10,
    num=100
)

# array_M_2a = np.zeros_like(array_T_2a)
# array_stdM_2d = np.zeros_like(array_T_2a)
# for i in trange(array_T_2a.size, desc='Exercício 2. a)'):
#     array_spins = me.ising_montecarlo(
#         T=array_T_2a[i],
#         L=L_2a
#     )
#     array_M_2a[i] = np.mean(
#         a=me.magnetizacao(array_spins=array_spins)
#     )
#     array_stdM_2d[i] = np.std(
#         a=me.magnetizacao(array_spins=array_spins)
#     )
# np.save(
#     file='lista05/array_M_2a',
#     arr=array_M_2a,
# )
# np.save(
#     file='lista05/array_stdM_2d',
#     arr=array_stdM_2d,
# )

array_M_2a = np.load(
    file='lista05/array_M_2a.npy'
)

fig_2a, ax_2a = plt.subplots()

ax_2a.set_title('Magnetização em função da temperatura')
ax_2a.set_xlabel('Temperatura $T$ ($1 / k_B$)')
ax_2a.set_ylabel('Magnetização por spin $M$')
ax_2a.grid(visible=True)

ax_2a.plot(
    array_T_2a,
    array_M_2a / N_2a,
    'o',
    markersize=2
)


# item 2. b)  #######################

array_T_2b = np.linspace(
    start=0,
    stop=10,
    num=100
)
dT_2b = np.diff(array_T_2b)[0]

array_L_2b = np.array(
    object=[4, 6, 8, 10],
    dtype=int
)
array_N_2b = array_L_2b**2

# array_E_2b = np.zeros((array_L_2b.size, array_T_2b.size))
# array_varE_2b = np.zeros((array_L_2b.size, array_T_2b.size))
# for l in trange(array_L_2b.size, desc='Exercício 2. b)'):
#     for i in range(array_T_2b.size):
#         array_spins = me.ising_montecarlo(
#             T=array_T_2b[i],
#             L=array_L_2b[l],
#             barra_de_progresso=False
#         )
#         array_E_2b[l, i] = np.mean(
#             a=me.energia(array_spins=array_spins)
#         )
#         array_varE_2b[l, i] = np.var(
#             a=me.energia(array_spins=array_spins)
#         )
# np.save(
#     file='lista05/array_E_2b',
#     arr=array_E_2b,
# )
# np.save(
#     file='lista05/array_varE_2b',
#     arr=array_varE_2b,
# )

array_E_2b = np.load(
    file='lista05/array_E_2b.npy'
)
array_varE_2b = np.load(
    file='lista05/array_varE_2b.npy'
)

fig_2b, axs_2b = plt.subplots(
    ncols=3
)
fig_2b.set_size_inches(w=3 * 6.4, h=4.8)

axs_2b[0].set_title('Energia em função da temperatura')
axs_2b[0].set_xlabel('Temperatura $T$ ($1 / k_B$)')
axs_2b[0].set_ylabel('Energia por spin')
axs_2b[0].grid(visible=True)

for l in range(array_L_2b.size):
    axs_2b[0].plot(
        array_T_2b,
        array_E_2b[l, :] / array_N_2b[l],
        # 'o',
        # markersize=2,
        label='L = ' + str(array_L_2b[l])
    )
axs_2b[0].legend()


axs_2b[1].set_title('Calor específico em função da temperatura\n(Diferenciação numérica)')
axs_2b[1].set_xlabel('Temperatura $T$ ($1 / k_B$)')
axs_2b[1].set_ylabel('Calor específico por spin ($k_B$)')
axs_2b[1].grid(visible=True)

for l in range(array_L_2b.size):
    axs_2b[1].plot(
        array_T_2b,
        me.calor_especifico_diff(array_E_2b[l, :], dT_2b) / array_N_2b[l],
        # 'o',
        # markersize=2,
        label='L = ' + str(array_L_2b[l])
    )
axs_2b[1].legend()


axs_2b[2].set_title('Calor específico em função da temperatura\n(Teorema flutuação-dissipação)')
axs_2b[2].set_xlabel('Temperatura $T$ ($1 / k_B$)')
axs_2b[2].set_ylabel('Calor específico por spin ($k_B$)')
# axs_2b[2].set_ylim(0, 2)
axs_2b[2].grid(visible=True)

for l in range(array_L_2b.size):
    array_C = me.calor_especifico_fd(array_varE_2b[l, :], array_T_2b) / array_N_2b[l]

    axs_2b[2].plot(
        array_T_2b,
        array_C,
        # 'o',
        # markersize=2,
        label='L = ' + str(array_L_2b[l])
    )
axs_2b[2].legend()


# item 2. c)  #######################

array_Tc_2c = np.zeros_like(array_L_2b)
for l in range(array_L_2b.size):
    array_C = me.calor_especifico_fd(array_varE_2b[l, :], array_T_2b) / array_N_2b[l]
    indice = np.nanargmax(array_C)
    print(
        array_L_2b[l],
        # indice,
        # array_C[indice],
        array_T_2b[indice]
    )

axs_2b[2].legend()

# print(array_Tc_2c)


# item 2. d)  #######################

array_stdM_2d = np.load(
    file='lista05/array_stdM_2d.npy'
)

fig_2d, ax_2d = plt.subplots()
ax_2d.set_title('Desvio padrão da magnetização em função da temperatura')
ax_2d.set_xlabel('Temperatura $T$ ($1 / k_B$)')
ax_2d.set_ylabel('$\\Delta M$')
ax_2d.grid(visible=True)

ax_2d.plot(
    array_T_2a,
    array_stdM_2d,
    # 'o',
    # markersize=2,
)

indice_2d = np.nanargmax(array_stdM_2d)
print(
    10,
    indice_2d,
    array_stdM_2d[indice_2d],
    array_T_2a[indice_2d]
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

# fig_2a.savefig(
#     fname='lista05/fig_2a.pdf'
# )
# fig_2b.savefig(
#     fname='lista05/fig_2b.pdf'
# )
fig_2d.savefig(
    fname='lista05/fig_2d.pdf'
)

# plt.show()
