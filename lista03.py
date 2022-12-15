#!/usr/bin/env python
# coding: utf-8

# # Imports

# In[1]:


import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from mpl_toolkits.mplot3d import axes3d

from edp import difusao_2d
from matrizes import mprint
from sistemas_aleatorios import (cafe_com_creme, coeficiente_D, dim_fractal,
                                 dla, entropia, massa_cluster, pos_ocup_2,
                                 rwalk, rwalk_2d)

tamanho_mark = 1
tamanho_plot = (10, 10)


# https://mathworld.wolfram.com/RandomWalk1-Dimensional.html

# # Exercício 1

# In[2]:


m_1 = 100000
n_1 = 100
x_1, x2ave_1 = rwalk(m=m_1, n=n_1)


# In[3]:


D_1 = coeficiente_D(x2ave_1)
print(D_1)


# In[4]:


fig_1, ax_1 = plt.subplots()
ax_1.set_title('Random Walk, ' + str(m_1) + ' andarilhos')
ax_1.set_xlabel('Número de passos')
ax_1.set_ylabel('<x²>')
ax_1.set_aspect('equal')
ax_1.grid()
ax_1.set_xlim(0, n_1)
ax_1.plot(x2ave_1)


# # Exercício 2

# In[5]:


m_2 = 100000
n_2 = 100
p_2 = 0.25
x_2, x2ave_2 = rwalk(p_esq=p_2, m=m_2, n=n_2)


# In[6]:


fig_2, ax_2 = plt.subplots()
ax_2.set_title('Random Walk, ' +
               '$p_{esq} = 0.25$, ' + str(m_2) + ' andarilhos')
ax_2.set_xlabel('Número de passos')
ax_2.set_ylabel('<x²>')
# ax_2.set_aspect('equal')
ax_2.grid()
ax_2.set_xlim(0, n_2)
ax_2.plot(x2ave_2)


# # Exercício 3

# In[7]:


posicoes = cafe_com_creme(m=1000, t=20000)
S_4 = entropia(posicoes)


# In[8]:


fig_3a, axs_3a = plt.subplots(2, 2)
fig_3a.suptitle('Café com creme')
fig_3a.set_size_inches(8, 8)

axs_3a[0, 0].set_title('$t = 0$')
# axs_3a[0, 0].set_xlabel('$x$')
axs_3a[0, 0].set_ylabel('$y$')
axs_3a[0, 0].set_xlim(-5, 5)
axs_3a[0, 0].set_ylim(-5, 5)
axs_3a[0, 0].set_aspect('equal')
axs_3a[0, 0].grid()

axs_3a[0, 1].set_title('$t = 10^2$')
# axs_3a[0, 1].set_xlabel('$x$')
axs_3a[0, 1].set_ylabel('$y$')
axs_3a[0, 1].set_xlim(-5, 5)
axs_3a[0, 1].set_ylim(-5, 5)
axs_3a[0, 1].set_aspect('equal')
axs_3a[0, 1].grid()

axs_3a[1, 0].set_title('$t = 10^3$')
axs_3a[1, 0].set_xlabel('$x$')
axs_3a[1, 0].set_ylabel('$y$')
axs_3a[1, 0].set_xlim(-5, 5)
axs_3a[1, 0].set_ylim(-5, 5)
axs_3a[1, 0].set_aspect('equal')
axs_3a[1, 0].grid()

axs_3a[1, 1].set_title('$t = 10^4$')
axs_3a[1, 1].set_xlabel('$x$')
axs_3a[1, 1].set_ylabel('$y$')
axs_3a[1, 1].set_xlim(-5, 5)
axs_3a[1, 1].set_ylim(-5, 5)
axs_3a[1, 1].set_aspect('equal')
axs_3a[1, 1].grid()

for particula in range(np.shape(posicoes)[0]):
    axs_3a[0, 0].plot(posicoes[particula, 0, 0],
                      posicoes[particula, 0, 1], '.', color='black')
    axs_3a[0, 1].plot(posicoes[particula, 100, 0],
                      posicoes[particula, 100, 1], '.', color='black')
    axs_3a[1, 0].plot(posicoes[particula, 1000, 0],
                      posicoes[particula, 1000, 1], '.', color='black')
    axs_3a[1, 1].plot(posicoes[particula, 10000, 0],
                      posicoes[particula, 10000, 1], '.', color='black')

fig_3b, ax_3b = plt.subplots()
ax_3b.set_title('Entropia da xícara em função do tempo')
ax_3b.set_xlabel('$t$')
ax_3b.set_ylabel('$S$')
ax_3b.plot(S_4)


# # Exercício 4

# In[45]:


dim_4 = np.array([15, 15])
meio_4 = dim_4 // 2
L_4 = 1.0
D_4 = 1.0
dx_4 = 0.01
dt_4a = (dx_4)**2 / (4 * D_4)
t_f_4 = 10.0

X_4 = np.linspace(-L_4 / 2, L_4 / 2, dim_4[0])
Y_4 = np.linspace(-L_4 / 2, L_4 / 2, dim_4[1])
Xm_4, Ym_4 = np.meshgrid(X_4, Y_4)
Xm_4 = Xm_4.transpose()
Ym_4 = Ym_4.transpose()


# ## 4. a)

# In[46]:


rho_0_4a = np.zeros(dim_4)
rho_0_4a[meio_4[0] - 2:meio_4[0] + 2 + 1,
         meio_4[1] - 2:meio_4[1] + 2 + 1] = 1.0

# mprint(rho_0_4a)


# In[47]:


rho_4a = difusao_2d(
    rho_0=rho_0_4a,
    t_f=t_f_4,
    dx=dx_4,
    D=D_4
)


# In[103]:


fig_4a, axes_4a = plt.subplots(
    nrows=2, ncols=2, subplot_kw={"projection": "3d"})
fig_4a.suptitle('Densidade de massa em função de $x$ e $y$')
fig_4a.set_size_inches(tamanho_plot)

tempos = np.array([0, 2, 10, 100], dtype=int)
i = 0
for ax in axes_4a.flat:
    tempo = tempos[i]
    ax.set_title('$t = $' + str(tempo) + ' s')
    ax.set_xlabel('$x$ (m)')
    ax.set_ylabel('$y$ (m)')
    ax.set_zlabel('$\\rho$ (m⁻²)')
    ax.set_xlim(- L_4 / 2, L_4 / 2)
    ax.set_ylim(- L_4 / 2, L_4 / 2)
    ax.set_zlim(0, 1)
    ax.plot_surface(Xm_4, Ym_4, rho_4a[:, :, tempo], cmap=cm.inferno)
    i += 1


# ## 4. b)

# In[99]:


rho_0_4b = np.zeros(dim_4)
rho_0_4b[:, meio_4[1]] = 1.0

# mprint(rho_0_4b)


# In[100]:


rho_4b = difusao_2d(
    rho_0=rho_0_4b,
    t_f=t_f_4,
    dx=dx_4,
    D=D_4
)


# In[102]:


fig_4b, axes_4b = plt.subplots(
    nrows=2, ncols=2, subplot_kw={"projection": "3d"})
fig_4b.suptitle('Densidade de massa em função de $x$ e $y$')
fig_4b.set_size_inches(tamanho_plot)

tempos = np.array([0, 2, 10, 100], dtype=int)
i = 0
for ax in axes_4b.flat:
    tempo = tempos[i]
    ax.set_title('$t = $' + str(tempo) + ' s')
    ax.set_xlabel('$x$ (m)')
    ax.set_ylabel('$y$ (m)')
    ax.set_zlabel('$\\rho$ (m⁻²)')
    ax.set_xlim(- L_4 / 2, L_4 / 2)
    ax.set_ylim(- L_4 / 2, L_4 / 2)
    ax.set_zlim(0, 1)
    ax.plot_surface(Xm_4, Ym_4, rho_4b[:, :, tempo], cmap=cm.inferno)
    i += 1


# ## 4. c)

# In[112]:


rho_0_4c = np.cos(np.pi / 2 * Xm_4 / 0.5) * np.cos(np.pi / 2 * Ym_4 / 0.5)
# rho_0_4c[:, meio_4[1]] = 1.0

mprint(rho_0_4c)


# In[113]:


rho_4c = difusao_2d(
    rho_0=rho_0_4c,
    t_f=t_f_4,
    dx=dx_4,
    D=D_4
)


# In[114]:


fig_4c, axes_4c = plt.subplots(
    nrows=2, ncols=2, subplot_kw={"projection": "3d"})
fig_4c.suptitle('Densidade de massa em função de $x$ e $y$')
fig_4c.set_size_inches(tamanho_plot)

tempos = np.array([0, 2, 10, 100], dtype=int)
i = 0
for ax in axes_4c.flat:
    tempo = tempos[i]
    ax.set_title('$t = $' + str(tempo) + ' s')
    ax.set_xlabel('$x$ (m)')
    ax.set_ylabel('$y$ (m)')
    ax.set_zlabel('$\\rho$ (m⁻²)')
    ax.set_xlim(- L_4 / 2, L_4 / 2)
    ax.set_ylim(- L_4 / 2, L_4 / 2)
    ax.set_zlim(0, 1)
    ax.plot_surface(Xm_4, Ym_4, rho_4c[:, :, tempo], cmap=cm.inferno)
    i += 1


# # Exercício 5

# In[2]:


# pos_ocup = dla(n_part=500)  # 30 part

# np.savetxt(
#     fname='lista03/' + 'dla.txt',
#     X=pos_ocup,
# )

# np.savetxt(
#     fname='lista03/' + 'dla_t.txt',
#     X=pos_ocup,
#     fmt='%d'
# )

pos_ocup = 1 * pos_ocup_2


# In[3]:


massa = massa_cluster(pos_ocup=pos_ocup)


# In[4]:


m, n = pos_ocup.shape

fig_5a, ax_5a = plt.subplots()

ax_5a.set_title('Cluster DLA, ' + str(m) + ' partículas')
ax_5a.set_xlabel('$x$')
ax_5a.set_ylabel('$y$')
ax_5a.set_aspect('equal')

ax_5a.plot(pos_ocup[0, 0], pos_ocup[0, 1], 's', color='red')
for i in np.arange(1, m):
    ax_5a.plot(pos_ocup[i, 0], pos_ocup[i, 1], 's', color='black')


# In[5]:


d_f, intercept = dim_fractal(massa, 1, -17)

fig_5b, ax_5b = plt.subplots()

ax_5b.set_title('Cluster DLA, ' + str(m) + ' partículas')
ax_5b.set_xlabel('Raio')
ax_5b.set_xscale("log")
ax_5b.set_ylabel('Massa')
ax_5b.set_yscale("log")
ax_5b.grid(visible=True, which='both')

ax_5b.plot(massa[:, 0], massa[:, 1], '.')

x_ajuste = np.linspace(massa[:, 0].min(), massa[:, 0].max())
y_ajuste = np.exp(d_f * np.log(x_ajuste) + intercept)
ax_5b.plot(x_ajuste, y_ajuste)


# In[6]:


print(d_f)


# # Salvar imagens

# In[7]:


# fig_1.savefig(fname='lista03/' + 'fig_1.pdf')

# fig_2.savefig(fname='lista03/' + 'fig_2.pdf')

# fig_3a.savefig(fname='lista03/' + 'fig_3a.pdf')
# fig_3b.savefig(fname='lista03/' + 'fig_3b.pdf')

# fig_4a.savefig(fname='lista03/' + 'fig_4a.pdf')
# fig_4b.savefig(fname='lista03/' + 'fig_4b.pdf')
# fig_4c.savefig(fname='lista03/' + 'fig_4c.pdf')

# fig_5a.savefig(fname='lista03/' + 'fig_5a.pdf')
# fig_5b.savefig(fname='lista03/' + 'fig_5b.pdf')
