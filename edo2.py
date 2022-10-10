import numpy as np


def euler(y_0, v_0, dt, t_f, funcional):
    """Realiza o cálculo númerico pelo método de Euler."""
    n_max = int(t_f / dt)

    v = np.zeros(n_max)
    v[0] = v_0

    y = np.zeros(n_max)
    y[0] = y_0

    t = np.zeros(n_max)
    t[0] = 0.0

    for i in range(n_max - 1):
        v[i + 1] = v[i] + funcional(y[i], v[i], t[i]) * dt
        y[i + 1] = y[i] + v[i] * dt
        t[i + 1] = t[i] + dt
    return y, v, t


def eulercromer(y_0, v_0, dt, t_f, funcional):
    """Realiza o cálculo númerico pelo método de Euler-Cromer."""
    n_max = int(t_f / dt)

    v = np.zeros(n_max)
    v[0] = v_0

    y = np.zeros(n_max)
    y[0] = y_0

    t = np.zeros(n_max)
    t[0] = 0.0

    for i in range(n_max - 1):
        v[i + 1] = v[i] + funcional(y[i], v[i], t[i]) * dt
        y[i + 1] = y[i] + v[i + 1] * dt
        t[i + 1] = t[i] + dt
    return y, v, t


def rungekutta(y_0, v_0, dt, t_f, funcional):
    """Realiza o cálculo númerico pelo método de Runge-Kutta."""
    n_max = int(t_f / dt)

    v = np.zeros(n_max)
    v[0] = v_0

    y = np.zeros(n_max)
    y[0] = y_0

    t = np.zeros(n_max)
    t[0] = 0.0

    for i in range(n_max - 1):
        y_l = y[i] + (1 / 2) * v[i] * dt
        v_l = v[i] + (1 / 2) * funcional(y[i], v[i], t[i]) * dt
        t_l = t[i] + (1 / 2) * dt
        y[i + 1] = y[i] + v_l * dt
        v[i + 1] = v[i] + funcional(y_l, v_l, t_l) * dt
        t[i + 1] = t[i] + dt
    return y, v, t


def verlet(y_0, v_0, dt, t_f, funcional):
    """Realiza o cálculo númerico pelo método de Verlet."""
    n_max = int(t_f / dt)

    v = np.zeros(n_max)
    v[0] = v_0

    y = np.zeros(n_max)
    y[0] = y_0

    t = np.zeros(n_max)
    t[0] = 0.0

    y[1] = y[0] + v[0] * dt
    t[1] = t[0] + dt
    for i in range(1, n_max - 1):
        y[i + 1] = 2 * y[i] - y[i - 1] + funcional(y[i], v[i], t[i]) * (dt)**2
        v[i] = (1 / 2) * (y[i + 1] - y[i - 1]) / dt
        t[i + 1] = t[i] + dt
    return y, v, t


def leapfrog(y_0, v_0, dt, t_f, funcional):
    """Realiza o cálculo númerico pelo método Leapfrog."""
    n_max = int(t_f / dt)

    v = np.zeros(n_max)
    v[0] = v_0

    y = np.zeros(n_max)
    y[0] = y_0

    t = np.zeros(n_max)
    t[0] = 0.0

    v[1] = v[0] + funcional(y[0], v[0], t[0]) * dt
    y[1] = y[0] + v[0] * dt
    t[1] = t[0] + dt
    for i in range(1, n_max - 1):
        v[i + 1] = v[i - 1] + 2 * funcional(y[i], v[i], t[i])
        y[i + 2] = y[i] + 2 * v[i + 1] * dt
        t[i + 1] = t[i] + dt
    return y, v, t
