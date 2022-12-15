import lmfit
import numpy as np
from scipy.ndimage import label
from scipy.optimize import curve_fit
from scipy.special import comb
from scipy.stats import entropy
from tqdm.auto import trange
from uncertainties import ufloat


def rwalk(p_esq=0.5, m=500, n=100):
    x = np.zeros((n + 1, m))
    for j in range(m):
        for i in np.arange(1, n + 1):
            r = np.random.rand()
            if r < 1 - p_esq:
                x[i, j] = x[i - 1, j] + 1
            else:
                x[i, j] = x[i - 1, j] - 1
    x2ave = np.sum(x**2, axis=1) / m

    return x, x2ave


def coeficiente_D(x2ave, modelo=lmfit.models.LinearModel()):
    ajuste = modelo.fit(x2ave, x=np.arange(x2ave.size))
    slope = ufloat(ajuste.params['slope'].value, ajuste.params['slope'].stderr)
    D = (1 / 2) * slope
    return D


def P_rwalk(x: int, n: int):
    """Distribuição de probabilidades de estar a uma distância x após n passos para o Random Walk"""
    if (x + n) % 2 != 0 or x < 0 or n < 0:
        fator = 0.0
        coef = 0
    else:
        fator = np.power(
            2,
            -n,
            dtype=float
        )
        coef = comb(
            n,
            (x + n) / 2,
            exact=True
        )
    return fator * coef


def rwalk_2d(t=100, r_0=np.array([0, 0])):
    x_u = np.array([1, 0])
    y_u = np.array([0, 1])
    r = np.zeros((t + 1, 2))
    r[0, :] = r_0
    for n in np.arange(1, t + 1):
        q = np.random.rand()
        if q < 0.25:
            r[n] = r[n - 1] + x_u
        elif q >= 0.25 and q < 0.5:
            r[n] = r[n - 1] - x_u
        elif q >= 0.5 and q < 0.75:
            r[n] = r[n - 1] + y_u
        else:
            r[n] = r[n - 1] - y_u
    return r


def cafe_com_creme(m=400, t=10000):
    x_u = np.array([1, 0])
    y_u = np.array([0, 1])

    posicoes = np.zeros((m, t + 1, 2))
    particula = np.random.randint(m, size=t + 1)
    direcao = np.random.randint(4, size=t + 1)

    for n in trange(1, t + 1, desc='Café com creme'):
        # for n in np.arange(1, t + 1):
        p = particula[n]
        q = direcao[n]

        if q == 0:
            if posicoes[p, n - 1, 0] == 5:
                posicoes[p, n:, :] = posicoes[p, n - 1, :] - x_u
            else:
                posicoes[p, n:, :] = posicoes[p, n - 1, :] + x_u
        elif q == 1:
            if posicoes[p, n - 1, 0] == -5:
                posicoes[p, n:, :] = posicoes[p, n - 1, :] + x_u
            else:
                posicoes[p, n:, :] = posicoes[p, n - 1, :] - x_u
        elif q == 2:
            if posicoes[p, n - 1, 1] == 5:
                posicoes[p, n:, :] = posicoes[p, n - 1, :] - y_u
            else:
                posicoes[p, n:, :] = posicoes[p, n - 1, :] + y_u
        else:
            if posicoes[p, n - 1, 1] == -5:
                posicoes[p, n:, :] = posicoes[p, n - 1, :] + y_u
            else:
                posicoes[p, n:, :] = posicoes[p, n - 1, :] - y_u
    return posicoes


def contagem_subvolume(pos, passo, x_inf, x_sup, y_inf, y_sup):
    cont = 0

    for elemento in pos[:, passo, :]:
        if (elemento[0] >= x_inf and elemento[0] < x_sup) and (elemento[1] >= y_inf and elemento[1] < y_sup):
            cont += 1

    return cont


def contagens(pos, grid_x=np.arange(-5, 5, 5), grid_y=np.arange(-5, 5, 5)):
    delta_x = grid_x[1] - grid_x[0]
    delta_y = grid_y[1] - grid_y[0]

    n_x = np.size(grid_x)
    n_y = np.size(grid_y)
    n_t = np.shape(pos)[1]

    cont = np.zeros((n_x, n_y, n_t))

    for passo in np.arange(n_t):
        for i in np.arange(n_x):
            for j in np.arange(n_y):
                cont[i, j, passo] = contagem_subvolume(
                    pos,
                    passo,
                    grid_x[i],
                    grid_x[i] + delta_x,
                    grid_y[j],
                    grid_y[j] + delta_y
                )

    return cont


def entropia(pos):
    m = np.shape(pos)[0]
    P = contagens(pos) / m
    S = entropy(P, axis=(0, 1))
    return S


pos_ocup_1 = np.loadtxt(
    fname='lista03/' + 'dla.txt',
    dtype=int
)

pos_ocup_2 = np.loadtxt(
    fname='lista03/' + 'dla_t.txt',
    dtype=int
)


def posicao_aleatoria(raio):
    angulo = (2 * np.pi - 0.0) * np.random.ranf() + 0.0
    x = np.ceil(raio * np.cos(angulo))
    y = np.ceil(raio * np.sin(angulo))
    pos = np.array([x, y], dtype=int)
    return pos


def rwalk_2d_update(r, tamanho_passo):
    x_u = np.array([1, 0], dtype=int)
    y_u = np.array([0, 1], dtype=int)
    q = np.random.rand()
    if q < 0.25:
        novo_r = r + tamanho_passo * x_u
    elif q >= 0.25 and q < 0.5:
        novo_r = r - tamanho_passo * x_u
    elif q >= 0.5 and q < 0.75:
        novo_r = r + tamanho_passo * y_u
    else:
        novo_r = r - tamanho_passo * y_u
    return novo_r


def detectar_vizinho(r, pos_ocup):
    x_u = np.array([1, 0], dtype=int)
    y_u = np.array([0, 1], dtype=int)
    m, n = np.shape(pos_ocup)

    i = 0
    status = False
    while status == False and i < m:
        r_ocup = pos_ocup[i, :]
        bool_cima = np.array_equal(r + x_u, r_ocup)
        bool_baixo = np.array_equal(r - x_u, r_ocup)
        bool_dir = np.array_equal(r + y_u, r_ocup)
        bool_esq = np.array_equal(r - y_u, r_ocup)
        status = bool_cima or bool_baixo or bool_dir or bool_esq
        i += 1
    return status


def tamanho_max_cluster(pos_ocup):
    return np.linalg.norm(pos_ocup, axis=1).max()


def dla(n_part=500):
    pos_ocup = np.zeros((n_part, 2))

    # for i in np.arange(1, n_part):
    for i in trange(1, n_part, desc='DLA cluster'):
        abs_r_ocup_max = tamanho_max_cluster(pos_ocup)
        if abs_r_ocup_max == 0.0:
            r_inicial = posicao_aleatoria(5.0)
        elif abs_r_ocup_max > 0.0:
            r_inicial = posicao_aleatoria(5 * abs_r_ocup_max)
        else:
            print('Erro')
            break
        abs_r_inicial = np.linalg.norm(r_inicial)

        r = 1 * r_inicial
        abs_r = 1.0 * abs_r_inicial
        tem_vizinho = False

        while tem_vizinho == False:
            if abs_r_ocup_max == 0.0:
                tamanho_passo = 1
            elif abs_r > 1.5 * abs_r_ocup_max:
                # tamanho_passo = np.int64(np.ceil(abs_r / abs_r_ocup_max))
                tamanho_passo = 2
            else:
                tamanho_passo = 1

            r = 1 * rwalk_2d_update(r, tamanho_passo)
            abs_r = 1.0 * np.linalg.norm(r)
            tem_vizinho = detectar_vizinho(r, pos_ocup)

            if abs_r > 1.5 * abs_r_inicial:
                r = 1 * r_inicial
                abs_r = 1.0 * abs_r_inicial

        pos_ocup[i, :] = 1 * r

    return pos_ocup


def massa_cluster(pos_ocup):
    abs_pos_ocup = np.linalg.norm(pos_ocup, axis=1)
    raio_max = np.int64(np.ceil(abs_pos_ocup.max()))
    array = np.zeros((raio_max - 1, 2))
    array[:, 0] = np.arange(1, raio_max)

    i = 0
    for raio in trange(1, raio_max, desc='Massa do Cluster'):
        array[i, 1] = (abs_pos_ocup < raio).sum()
        i += 1
    return array


def linear(x, a, b):
    return a * x + b


def dim_fractal(massa, i_inicial=0, i_final=-1, func=linear):
    x = np.log(massa[i_inicial:i_final, 0])
    y = np.log(massa[i_inicial:i_final, 1])

    popt, pcov = curve_fit(func, x, y)

    return popt


# Lista 4 ##########

def xor(bool1: bool, bool2: bool) -> bool:
    return bool1 != bool2


def regra_90(N: int = 141, t: int = 70) -> np.ndarray:
    matriz = np.zeros((t, N), dtype=bool)
    matriz[0, N // 2] = True

    for tt in range(t - 1):
        for i in range(1, N - 1):
            matriz[tt + 1, i] = xor(matriz[tt, i - 1], matriz[tt, i + 1])

    return matriz.astype(int)


def massa(L: int, array: np.ndarray = regra_90()) -> np.ndarray:
    t, N = np.shape(array)
    n_t, n_N = t // L, N // L

    novo_array = np.zeros(
        shape=(n_t, n_N),
        dtype=int
    )

    for i in np.arange(n_t):
        for j in np.arange(n_N):
            novo_array[i, j] = int(
                1 in array[i * L:(i + 1) * L, j * L:(j + 1) * L])

    M = np.sum(novo_array)

    return M


massa_vec = np.vectorize(massa)


def array_inicial(L: int) -> np.ndarray:
    # if L % 2 == 0:
    #     L += 1

    formato = (L, L, 3)
    coord_central = L // 2

    array = np.zeros(shape=formato, dtype=int)
    array[coord_central, coord_central, 0] = 1
    array[coord_central, coord_central, 2] = 1

    return array


def primeiros_vizinhos(array: np.ndarray) -> np.ndarray:
    L_x, L_y = np.shape(array[:, :, 0])
    posicoes_ocupadas = np.argwhere(array[:, :, 0])
    novo_array = np.copy(array)

    for posicao in posicoes_ocupadas:
        i, j = posicao
        if i == 0 and j == 0:
            if novo_array[i + 1, j, 2] == 0:
                novo_array[i + 1, j, 1] = 1
            if novo_array[i, j + 1, 2] == 0:
                novo_array[i, j + 1, 1] = 1
        elif i == 0 and j == L_y - 1:
            if novo_array[i + 1, j, 2] == 0:
                novo_array[i + 1, j, 1] = 1
            if novo_array[i, j - 1, 2] == 0:
                novo_array[i, j - 1, 1] = 1
        elif i == L_x - 1 and j == 0:
            if novo_array[i - 1, j, 2] == 0:
                novo_array[i - 1, j, 1] = 1
            if novo_array[i, j + 1, 2] == 0:
                novo_array[i, j + 1, 1] = 1
        elif i == L_x - 1 and j == L_y - 1:
            if novo_array[i - 1, j, 2] == 0:
                novo_array[i - 1, j, 1] = 1
            if novo_array[i, j - 1, 2] == 0:
                novo_array[i, j - 1, 1] = 1
        elif i == 0:
            if novo_array[i + 1, j, 2] == 0:
                novo_array[i + 1, j, 1] = 1
            if novo_array[i, j - 1, 2] == 0:
                novo_array[i, j - 1, 1] = 1
            if novo_array[i, j + 1, 2] == 0:
                novo_array[i, j + 1, 1] = 1
        elif j == 0:
            if novo_array[i, j + 1, 2] == 0:
                novo_array[i, j + 1, 1] = 1
            if novo_array[i - 1, j, 2] == 0:
                novo_array[i - 1, j, 1] = 1
            if novo_array[i + 1, j, 2] == 0:
                novo_array[i + 1, j, 1] = 1
        elif i == L_x - 1:
            if novo_array[i - 1, j, 2] == 0:
                novo_array[i - 1, j, 1] = 1
            if novo_array[i, j - 1, 2] == 0:
                novo_array[i, j - 1, 1] = 1
            if novo_array[i, j + 1, 2] == 0:
                novo_array[i, j + 1, 1] = 1
        elif j == L_y - 1:
            if novo_array[i, j - 1, 2] == 0:
                novo_array[i, j - 1, 1] = 1
            if novo_array[i - 1, j, 2] == 0:
                novo_array[i - 1, j, 1] = 1
            if novo_array[i + 1, j, 2] == 0:
                novo_array[i + 1, j, 1] = 1
        else:
            if novo_array[i, j - 1, 2] == 0:
                novo_array[i, j - 1, 1] = 1
            if novo_array[i, j + 1, 2] == 0:
                novo_array[i, j + 1, 1] = 1
            if novo_array[i - 1, j, 2] == 0:
                novo_array[i - 1, j, 1] = 1
            if novo_array[i + 1, j, 2] == 0:
                novo_array[i + 1, j, 1] = 1

    return novo_array


# def primeiros_vizinhos(array: np.ndarray) -> np.ndarray:
#     L_x, L_y = np.shape(array[:, :, 0])
#     posicoes_ocupadas = np.argwhere(array[:, :, 0])
#     novo_array = np.copy(array)

#     for posicao in posicoes_ocupadas:
#         i, j = posicao

#         if i == L_x - 1:
#             i = -1
#         if j == L_y - 1:
#             j = -1

#         if novo_array[i, j - 1, 2] == 0:
#             novo_array[i, j - 1, 1] = 1
#         if novo_array[i, j + 1, 2] == 0:
#             novo_array[i, j + 1, 1] = 1
#         if novo_array[i - 1, j, 2] == 0:
#             novo_array[i - 1, j, 1] = 1
#         if novo_array[i + 1, j, 2] == 0:
#             novo_array[i + 1, j, 1] = 1

#     return novo_array


def iteracao_leath(array: np.ndarray, p: float = 0.5927) -> np.ndarray:
    posicoes_perimetro = np.argwhere(array[:, :, 1])
    novo_array = np.copy(array)

    for posicao in posicoes_perimetro:
        i, j = posicao
        r = np.random.random()
        if r <= p:
            novo_array[i, j, 0] = 1
        novo_array[i, j, 1] = 0
        novo_array[i, j, 2] = 1

    return novo_array


def leath(L: int, p: float = 0.5927) -> np.ndarray:
    array = array_inicial(L)
    array = primeiros_vizinhos(array)

    formato_final = np.shape(array[:, :, 1])
    array_final = np.zeros(shape=formato_final)

    while np.array_equal(array[:, :, 1], array_final) == False:
        array = iteracao_leath(array, p)
        array = primeiros_vizinhos(array)

    return array[:, :, 0]


def ensemble_leath(L: int, N: int = 10000) -> np.ndarray:
    formato = (L, L, N)
    array = np.zeros(shape=formato, dtype=int)

    for n in trange(N, desc='Ensemble de Cluster Leath para L = ' + str(L)):
        array[:, :, n] = leath(L)

    return array


def p_infty(ensemble: np.ndarray) -> float:
    P_N = np.mean(ensemble, axis=(0, 1))
    P = np.mean(P_N)

    return P


def verificar_percolacao(array: np.ndarray) -> bool:
    return ((1 in array[0, :]) and (1 in array[-1, :]) and (1 in array[:, 0]) and ((1 in array[:, -1])))


def cont_s(ensemble: np.ndarray) -> float:
    formato = np.shape(ensemble)
    N = formato[2]

    cont = np.zeros(N, dtype=int)

    for i in trange(N, desc='Distribuição de clusters n_s'):
        array = ensemble[:, :, i]
        if verificar_percolacao(array) == False:
            cont[i] = np.sum(array)
        else:
            cont[i] = 0

    return cont
