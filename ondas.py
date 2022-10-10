import numpy as np


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
                y_xt[i, n + 1] = 2 * (1 - r**2) * y_xt[i, n] - \
                    y_x0[i] + r**2 * (y_xt[i + 1, n] + y_xt[i - 1, n])
            elif n != 0:
                y_xt[i, n + 1] = 2 * (1 - r**2) * y_xt[i, n] - y_xt[i,
                                                                    n - 1] + r**2 * (y_xt[i + 1, n] + y_xt[i - 1, n])
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
