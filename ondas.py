import numpy as np


# def propagate(y_0, t_f, dx, c, dt=False, pe='fixa', pd='fixa'):
#     if dt == False:
#         dt = dx / c
#     r = c * dt / dx

#     matriz_y = []
#     matriz_y.append(y_0)
#     len_y = len(y_0)

#     n = 0
#     while n * dt < t_f:
#         linha_y = []

#         for i in range(1, len_y - 1):
#             elemento_y = 0.0
#             if n == 0:
#                 elemento_y += 2 * \
#                     (1 - r**2) * matriz_y[n][i] - y_0[i] + r**2 * \
#                     (matriz_y[n][i + 1] + matriz_y[n][i - 1])
#             elif n != 0:
#                 elemento_y += 2 * (1 - r**2) * matriz_y[n][i] - matriz_y[n - 1][i] + r**2 * (
#                     matriz_y[n][i + 1] + matriz_y[n][i - 1])
#             linha_y.append(elemento_y)

#         if pe == 'fixa':
#             inicio = [0.0]
#         elif pe == 'solta':
#             inicio = [linha_y[0]]

#         if pd == 'fixa':
#             fim = [0.0]
#         elif pd == 'solta':
#             fim = [linha_y[-1]]

#         matriz_y.append(inicio + linha_y + fim)

#         n += 1

#     return matriz_y


def propagate(y_x0, t_f, dx, c, dt=False, pe='fixa', pd='fixa'):
    if dt == False:
        dt = dx / c
    r = c * dt / dx

    i_max = np.size(y_x0)
    n_max = int(t_f / dt)

    y_xt = np.zeros((i_max, n_max + 1))
    y_xt[:, 0] = y_x0

    for n in range(n_max):
        for i in range(1, i_max - 1):
            if n == 0:
                y_xt[i, n + 1] = 2 * (1 - r**2) * y_xt[i, n] - y_x0[i] + r**2 * (y_xt[i + 1, n] + y_xt[i - 1, n])
            elif n != 0:
                y_xt[i, n + 1] = 2 * (1 - r**2) * y_xt[i, n] - y_xt[i, n - 1] + r**2 * (y_xt[i + 1, n] + y_xt[i - 1, n])
            # y_xt[:, n] = y_in

        if pe == 'fixa':
            y_xt[0, n + 1] = 0.0
        elif pe == 'solta':
            y_xt[0, n + 1] = y_xt[1, n + 1]

        if pd == 'fixa':
            y_xt[-1, n + 1] = 0.0
        elif pd == 'solta':
            y_xt[-1, n + 1] = y_xt[-2, n + 1]

    return y_xt
