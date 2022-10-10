import numpy as np


def euler(y_0, dt, t_f, funcional):
    """Realiza o cálculo númerico pelo método de Euler."""
    n_max = int(t_f / dt)

    y = np.zeros(n_max)
    y[0] = y_0

    t = np.zeros(n_max)
    t[0] = 0.0

    for i in range(n_max - 1):
        y[i + 1] = y[i] + funcional(y[i], t[i]) * dt
        t[i + 1] = t[i] + dt
    return y, t


def rungekutta2(y_0, dt, t_f, funcional):
    """Realiza o cálculo númerico pelo método de Runge-Kutta de segunda ordem."""
    n_max = int(t_f / dt)

    y = np.zeros(n_max)
    y[0] = y_0

    t = np.zeros(n_max)
    t[0] = 0.0

    for i in range(n_max - 1):
        y_l = y[i] + (1 / 2) * funcional(y[i], t[i]) * dt
        t_l = t[i] + (1 / 2) * dt
        y[i + 1] = y[i] + funcional(y_l, t_l) * dt
        t[i + 1] = t[i] + dt
    return y, t


def rungekutta4(y_0, dt, t_f, funcional):
    """Realiza o cálculo númerico pelo método de Runge-Kutta de segunda ordem."""
    n_max = int(t_f / dt)

    y = np.zeros(n_max)
    y[0] = y_0

    t = np.zeros(n_max)
    t[0] = 0.0

    for i in range(n_max - 1):
        y_l_1 = y[i]
        t_l_1 = t[i]

        y_l_2 = y[i] + (1 / 2) * funcional(y_l_1, t_l_1) * dt
        t_l_2 = t[i] + (1 / 2) * dt

        y_l_3 = y[i] + (1 / 2) * funcional(y_l_2, t_l_2) * dt
        t_l_3 = t[i] + (1 / 2) * dt

        y_l_4 = y[i] + funcional(y_l_3, t_l_3) * dt
        t_l_4 = t[i] + dt

        y[i + 1] = y[i] + (1 / 6) * (funcional(y_l_1, t_l_1) + 2 * funcional(
            y_l_2, t_l_2) + 2 * funcional(y_l_3, t_l_3) + funcional(y_l_4, t_l_4)) * dt
        t[i + 1] = t[i] + dt
    return y, t
