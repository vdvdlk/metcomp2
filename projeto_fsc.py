import matplotlib.pyplot as plt
import numpy as np

from lmfit.models import ConstantModel, PowerLawModel


N_max = 10


def gap_massa(L: int, ccp: bool) -> np.ndarray:
    string = 'projeto/autoval_'
    if ccp == True:
        string += 'ccp_'
    string += str(L) + '.npy'

    autoval = np.load(
        file=string
    )

    return autoval[:, 1] - autoval[:, 0]


def diff_gap_massa(L: int, ccp: bool) -> np.ndarray:
    array_lamda = np.load(
        file='projeto/array_lamda.npy'
    )
    dgap = np.diff(a=gap_massa(L=L, ccp=ccp))
    dlamda = np.diff(a=array_lamda)

    return dgap / dlamda


def razao_gap(L: int, ccp: bool) -> np.ndarray:
    return L * gap_massa(L=L, ccp=ccp) / ((L - 1) * gap_massa(L=L - 1, ccp=ccp))


def funcao_beta(L: int, ccp: bool) -> np.ndarray:
    x = np.load(
        file='projeto/array_lamda.npy'
    )[:-1]
    F = gap_massa(L=L, ccp=ccp)[:-1]
    F_linha = diff_gap_massa(L=L, ccp=ccp)

    return F / (F - 2 * x * F_linha)


def rqppp(x_1: float, y_1: float, x_2: float, y_2: float):
    'Reta que passa pelos pontos 1 e 2'
    det = x_1 - x_2
    a = (y_1 - y_2) / det
    b = (x_1 * y_2 - x_2 * y_1) / det

    return a, b


def iercc(a_1: float, b_1: float, a_2: float, b_2: float):
    'Interseção entre as retas com coeficientes 1 e 2'
    det = a_2 - a_1
    x = (b_1 - b_2) / det
    y = (a_2 * b_1 - a_1 * b_2) / det

    return x, y


def lamda_c(L: int, ccp: bool):
    array_lamda = np.load(
        file='projeto/array_lamda.npy'
    )
    razao = razao_gap(L=L, ccp=ccp)
    indice = int(np.argwhere(razao > 1)[-1][0])

    a, b = rqppp(
        x_1=array_lamda[indice],
        y_1=razao[indice],
        x_2=array_lamda[indice + 1],
        y_2=razao[indice + 1]
    )

    x, y = iercc(
        a_1=a,
        b_1=b,
        a_2=0.0,
        b_2=1.0
    )
    return x


def estimativa_B(L: int, ccp: bool):
    x = np.load(
        file='projeto/array_lamda.npy'
    )[:-1]

    B = funcao_beta(L=L, ccp=ccp)

    indice = int(np.argwhere(x < 1)[-1][0])

    a, b = rqppp(
        x_1=x[indice],
        y_1=B[indice],
        x_2=x[indice + 1],
        y_2=B[indice + 1]
    )

    return a + b


def estimativa_S(L: int, ccp: bool):
    numerador = np.log(estimativa_B(L=L, ccp=ccp)) - \
        np.log(estimativa_B(L=L - 1, ccp=ccp))

    denominador = np.log(L) - np.log(L - 1)

    return numerador / denominador


def relacao_escala(ccp: bool, N_max: int = N_max):
    array_L = np.arange(4, N_max + 1)
    array_lamda_c = np.zeros(array_L.size)

    i = 0
    for L in array_L:
        array_lamda_c[i] = lamda_c(L=L, ccp=ccp)
        i += 1

    return array_L, array_lamda_c


def grafico_gap(ccp: bool, N_max: int = N_max):
    fig, ax = plt.subplots()
    titulo = 'Gap de massa em função de $\\lambda$'
    if ccp == True:
        titulo += ' (Condições de contorno periódicas)'
    else:
        titulo += ' (Cadeia aberta)'
    ax.set_title(titulo)

    ax.set_xlabel('$\\lambda$')
    ax.set_xlim(0, 10)

    ax.set_ylabel('$\\Delta (\\lambda)$')
    ax.set_ylim(0, 2)

    ax.grid(visible=True)

    array_lamda = np.load(
        file='projeto/array_lamda.npy'
    )

    for N in np.arange(2, N_max + 1):
        gap = gap_massa(L=N, ccp=ccp)
        ax.plot(array_lamda, gap, label='L = ' + str(N))

    ax.legend()

    return fig


def grafico_correl(ccp: bool, N_max: int = N_max):
    fig, ax = plt.subplots()
    titulo = 'Gap de massa em função de $\\lambda$'
    if ccp == True:
        titulo += ' (Condições de contorno periódicas)'
    else:
        titulo += ' (Cadeia aberta)'
    ax.set_title(titulo)

    ax.set_xlabel('$\\lambda$')
    ax.set_xlim(0, 10)

    ax.set_ylabel('$\\Delta (\\lambda)$')
    ax.set_ylim(0, 2)

    ax.grid(visible=True)

    array_lamda = np.load(
        file='projeto/array_lamda.npy'
    )

    for N in np.arange(2, N_max + 1):
        gap = gap_massa(L=N, ccp=ccp)
        ax.plot(array_lamda, 1 / gap, label='L = ' + str(N))

    ax.legend()

    return fig


def ajuste_rel_escala(ccp: bool):
    modelo = ConstantModel() + PowerLawModel()
    x, y = relacao_escala(ccp=ccp)
    resultado = modelo.fit(
        y,
        x=x,
        c=1.0,
        exponent=-1.0,
        amplitude=-1.0
    )

    return resultado


def grafico_rel_escala(ccp: bool, N_max: int = N_max):
    fig, ax = plt.subplots()
    titulo = 'Relação de escala'
    if ccp == True:
        titulo += ' (Condições de contorno periódicas)'
    else:
        titulo += ' (Cadeia aberta)'
    ax.set_title(titulo)

    ax.set_xlabel('$L$')
    ax.set_xlim(2, 11)

    ax.set_ylabel('$\\lambda_c (L)$')
    ax.set_ylim(0, 1.25)

    ax.grid(visible=True)

    x, y = relacao_escala(N_max=N_max, ccp=ccp)
    ax.plot(x, y, 'o')

    resultado = ajuste_rel_escala(ccp=ccp)
    c = resultado.params['c'].value
    A = resultado.params['amplitude'].value
    k = resultado.params['exponent'].value
    x_fit = np.linspace(
        start=1,
        stop=20,
        num=1000
    )
    y_fit = c + A * x_fit ** k
    ax.plot(x_fit, y_fit)

    # ax.plot(x, resultado.init_fit)
    # ax.plot(x, resultado.best_fit)

    return fig


grafico_gap(ccp=False).savefig(
    fname='projeto/gap.pdf'
)
grafico_correl(ccp=False)

grafico_rel_escala(ccp=False)

print(ajuste_rel_escala(ccp=False).fit_report())

grafico_gap(ccp=True).savefig(
    fname='projeto/gap.pdf'
)
grafico_correl(ccp=True)

grafico_rel_escala(ccp=True)

print(ajuste_rel_escala(ccp=True).fit_report())

# array_lamda = np.load(
#     file='projeto/array_lamda.npy'
# )

# plt.plot(
#     array_lamda[:-1],
#     funcao_beta(L=4, ccp=True)
# )


# print(estimativa_S(L=10, ccp=True))


# print(lamda_c(10, ccp=True))

plt.show()
