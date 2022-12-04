import matplotlib.pyplot as plt
import numpy as np
import lmfit

from sistemas_aleatorios import regra_90, massa_vec, leath, ensemble_leath, p_infty, n_s

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    # "figure.dpi": 1000,
})


# Exercício 1 #######################

sierpinski = regra_90()

fig_1, ax_1 = plt.subplots()
# fig_1.set_size_inches((20, 20))
ax_1.set_title('Triângulo de Sierpinski')
ax_1.set_aspect('equal')
ax_1.set_xlabel('Posição $i$')
ax_1.set_ylabel('Passo $t$')

ax_1.imshow(
    X=sierpinski,
    cmap='Greys'
)


# Exercício 2 #######################

array_L = 2 ** np.arange(1, 6)
array_r = array_L / 70
array_massa = massa_vec(L=array_L)

x_2 = np.log(1 / array_r)
y_2 = np.log(array_massa)

fig_2, ax_2 = plt.subplots()
ax_2.set_title('Dimensão fractal')
ax_2.grid(visible=True)

ax_2.set_xlabel('$\log (1 / r_n)$')
ax_2.set_xlim(0, 4)

ax_2.set_ylabel('$\log N(r_n)$')
ax_2.set_ylim(0, 7)

ax_2.plot(x_2, y_2, 'o')

modelo_2 = lmfit.models.LinearModel()
fit_2 = modelo_2.fit(y_2, x=x_2)


d_f = fit_2.params['slope'].value
intercept = fit_2.params['intercept'].value

x2_fit = np.linspace(0, 4)
y2_fit = d_f * x2_fit + intercept
ax_2.plot(x2_fit, y2_fit)


# Exercício 3 #######################

# item 3. a)  #######################

cluster_3a = leath(100)

fig_3a, ax_3a = plt.subplots()

ax_3a.set_title('Cluster por algoritmo de Leath')

ax_3a.set_xlabel('$x$')
ax_3a.set_ylabel('$y$')

ax_3a.imshow(
    X=cluster_3a,
    cmap='Greys'
)


# item 3. b)  #######################

array_L_3 = 2 ** np.arange(2, 7)
array_r_3 = array_L_3 / 100
array_massa_3 = massa_vec(array_L_3)

x_3b = np.log(1 / array_r_3)
y_3b = np.log(array_massa_3)

fig_3b, ax_3b = plt.subplots()
ax_3b.set_title('Dimensão fractal - Cluster de Leath')
ax_3b.grid(visible=True)

ax_3b.set_xlabel('$\log (1 / r_n)$')
ax_3b.set_xlim(0, 4)

ax_3b.set_ylabel('$\log N(r_n)$')
ax_3b.set_ylim(0, 6)

ax_3b.plot(x_3b, y_3b, 'o')

modelo_3b = lmfit.models.LinearModel()
fit_3b = modelo_3b.fit(y_3b, x=x_3b)

d_f_3b = fit_3b.params['slope'].value
intercept_3b = fit_3b.params['intercept'].value

x_3b_fit = np.linspace(0, 4)
y_3b_fit = d_f_3b * x_3b_fit + intercept_3b
ax_3b.plot(x_3b_fit, y_3b_fit)


# item 3. c)  #######################

# ensemble_10 = ensemble_leath(L=10)
# np.save(
#     file='lista04/ensemble_10',
#     arr=ensemble_10,
# )
ensemble_10 = np.load('lista04/ensemble_10.npy')
p_infty_10 = p_infty(ensemble_10)

# ensemble_20 = ensemble_leath(L=20)
# np.save(
#     file='lista04/ensemble_20',
#     arr=ensemble_20,
# )
ensemble_20 = np.load('lista04/ensemble_20.npy')
p_infty_20 = p_infty(ensemble_20)

# ensemble_50 = ensemble_leath(L=50)
# np.save(
#     file='lista04/ensemble_50',
#     arr=ensemble_50,
# )
ensemble_50 = np.load('lista04/ensemble_50.npy')
p_infty_50 = p_infty(ensemble_50)

# ensemble_100 = ensemble_leath(L=100)
# np.save(
#     file='lista04/ensemble_100',
#     arr=ensemble_100,
# )
ensemble_100 = np.load('lista04/ensemble_100.npy')
p_infty_100 = p_infty(ensemble_100)

# ensemble_200 = ensemble_leath(L=200)
# np.save(
#     file='lista04/ensemble_200',
#     arr=ensemble_200,
# )
ensemble_200 = np.load('lista04/ensemble_200.npy')
p_infty_200 = p_infty(ensemble_200)

x_3c = np.array([10, 20, 50, 100, 200], dtype=int)
y_3c = np.array(
    [p_infty_10, p_infty_20, p_infty_50, p_infty_100, p_infty_200]
)

fig_3c, ax_3c = plt.subplots()
ax_3c.set_title('Gráfico $P_\infty (p_c, L) \\times L$')
ax_3c.grid(visible=True)

ax_3c.set_xlabel('$L$')
ax_3c.set_xlim(0, 210)

ax_3c.set_ylabel('$P_\infty (p_c, L)$')
ax_3c.set_ylim(0, 0.4)

ax_3c.plot(x_3c, y_3c, 'o')

modelo_3c = lmfit.models.PowerLawModel()
fit_3c = modelo_3c.fit(y_3c, x=x_3c)
A_3c = fit_3c.params['amplitude'].value
k_3c = fit_3c.params['exponent'].value

x_3c_fit = np.linspace(
    start=0.01,
    stop=210,
    num=500
)
y_3c_fit = A_3c * x_3c_fit ** k_3c
ax_3c.plot(x_3c_fit, y_3c_fit)


# item 3. d)  #######################

dist_3d = n_s(ensemble_200)


def main():
    # Exercício 1

    # fig_1.savefig(
    #     fname='lista04/fig_1.pdf',
    # )

    # Exercício 2

    # with open('lista04/fit_2.txt', 'w') as f:
    #     f.write(fit_2.fit_report())

    # fig_2.savefig(
    #     fname='lista04/fig_2.pdf',
    # )

    # Exercício 3.a)

    # fig_3a.savefig(
    #     fname='lista04/fig_3a.pdf',
    # )

    # Exercício 3.b)

    # with open('lista04/fit_3b.txt', 'w') as f:
    #     f.write(fit_3b.fit_report())

    # fig_3b.savefig(
    #     fname='lista04/fig_3b.pdf',
    # )

    # Exercício 3.c)

    # fig_3c.savefig(
    #     fname='lista04/fig_3c.pdf',
    # )

    # with open('lista04/fit_3c.txt', 'w') as f:
    #     f.write(fit_3c.fit_report())

    # Exercício 3.d)

    np.savetxt(
        fname='lista04/n_s_3d.txt',
        X=dist_3d
    )

    # plt.show()
    exit()


main()
