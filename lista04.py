import matplotlib.pyplot as plt
import numpy as np
import lmfit

from sistemas_aleatorios import regra_90, massa_vec, leath, ensemble_leath


# Exercício 1 #######################

sierpinski = regra_90()
# print(sierpinski)

fig1, ax1 = plt.subplots()
# fig1.set_size_inches((20, 20))
ax1.set_title('Triângulo de Sierpinski')
ax1.set_aspect('equal')
ax1.set_xlabel('Posição $i$')
ax1.set_ylabel('Passo $t$')

ax1.imshow(X=sierpinski)

# fig1.savefig(
#     fname='lista04/fig_1.pdf',
#     dpi=1000
# )


# Exercício 2 #######################

array_L = 2 ** np.arange(1, 6)
array_r = array_L / 70
array_massa = massa_vec(L=array_L)

x2 = np.log(1 / array_r)
y2 = np.log(array_massa)

fig2, ax2 = plt.subplots()
ax2.set_title('Dimensão fractal')
ax2.grid(visible=True)

ax2.set_xlabel('$\log (1 / r_n)$')
ax2.set_xlim(0, 4)

ax2.set_ylabel('$\log N(r_n)$')
ax2.set_ylim(0, 7)

ax2.plot(x2, y2, 'o')

modelo2 = lmfit.models.LinearModel()
fit2 = modelo2.fit(y2, x=x2)


d_f = fit2.params['slope'].value
coef_lin = fit2.params['intercept'].value

x2_fit = np.linspace(0, 4)
y2_fit = d_f * x2_fit + coef_lin
ax2.plot(x2_fit, y2_fit)


# Exercício 3 #######################

# item 3. a)  #######################

cluster_a = leath(100)

fig3a, ax3a = plt.subplots()

ax3a.set_title('Cluster por algoritmo de Leath')

ax3a.set_xlabel('$x$')
ax3a.set_ylabel('$y$')

ax3a.imshow(cluster_a[:, :, 0])

# item 3. b)  #######################

array_L_3 = 2 ** np.arange(2, 7)
array_r_3 = array_L_3 / 100
array_massa_3 = massa_vec(array_L_3)

x3 = np.log(1 / array_r_3)
y3 = np.log(array_massa_3)

fig3b, ax3b = plt.subplots()
ax3b.set_title('Dimensão fractal - Cluster de Leath')
ax3b.grid(visible=True)

ax3b.set_xlabel('$\log (1 / r_n)$')
ax3b.set_xlim(0, 4)

ax3b.set_ylabel('$\log N(r_n)$')
ax3b.set_ylim(0, 6)

ax3b.plot(x3, y3, 'o')

modelo = lmfit.models.LinearModel()
fit3 = modelo.fit(y3, x=x3)

d_f_3 = fit3.params['slope'].value
coef_lin_3 = fit3.params['intercept'].value

x3_fit = np.linspace(0, 4)
y3_fit = d_f_3 * x3_fit + coef_lin_3
ax3b.plot(x3_fit, y3_fit)

# item 3. c)  #######################

# ensemble_10 = ensemble_leath(L=10)
ensemble_100 = ensemble_leath(L=100)
# ensemble_200 = ensemble_leath(L=200)

def main():
    plt.show()

    # fig1.savefig(
    #     fname='lista04/fig_1.pdf',
    #     dpi=1000
    # )

    print(fit2.fit_report(), '\n')

    # fig2.savefig(
    #     fname='lista04/fig_2.pdf',
    #     dpi=1000
    # )

    # fig3a.savefig(
    #     fname='lista04/fig_3a.pdf',
    #     dpi=1000
    # )

    print(fit3.fit_report(), '\n')

    # fig3b.savefig(
    #     fname='lista04/fig_3b.pdf',
    #     dpi=1000
    # )

    exit()


main()
