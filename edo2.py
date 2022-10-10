import numpy as np


def euler(y_0, v_0, dt, t_f, funcional):
    """Realiza o cálculo númerico pelo método de Euler."""
    n = int(t_f / dt)
    lista_v = [v_0]
    lista_y = [y_0]
    lista_t = [0]
    for i in range(n):
        lista_v.append(lista_v[i] + funcional(lista_y[i],
                                              lista_v[i], lista_t[i]) * dt)
        lista_y.append(lista_y[i] + lista_v[i] * dt)
        lista_t.append(lista_t[i] + dt)
    array_v = np.array(lista_v)
    array_y = np.array(lista_y)
    array_t = np.array(lista_t)
    return array_y, array_v, array_t


def eulercromer(y_0, v_0, dt, t_f, funcional):
    """Realiza o cálculo númerico pelo método de Euler-Cromer."""
    n = int(t_f / dt)
    lista_v = [v_0]
    lista_y = [y_0]
    lista_t = [0]
    for i in range(n):
        lista_v.append(lista_v[i] + funcional(lista_y[i],
                                              lista_v[i], lista_t[i]) * dt)
        lista_y.append(lista_y[i] + lista_v[i + 1] * dt)
        lista_t.append(lista_t[i] + dt)
    array_v = np.array(lista_v)
    array_y = np.array(lista_y)
    array_t = np.array(lista_t)
    return array_y, array_v, array_t


def rungekutta(y_0, v_0, dt, t_f, funcional):
    """Realiza o cálculo númerico pelo método de Runge-Kutta."""
    n = int(t_f / dt)
    lista_v = [v_0]
    lista_y = [y_0]
    lista_t = [0]
    for i in range(n):
        y_l = lista_y[i] + (1 / 2) * lista_v[i] * dt
        v_l = lista_v[i] + (1 / 2) * funcional(lista_y[i],
                                               lista_v[i], lista_t[i]) * dt
        t_l = lista_t[i] + (1 / 2) * dt
        lista_y.append(lista_y[i] + v_l * dt)
        lista_v.append(lista_v[i] + funcional(y_l, v_l, t_l) * dt)
        lista_t.append(lista_t[i] + dt)
    array_v = np.array(lista_v)
    array_y = np.array(lista_y)
    array_t = np.array(lista_t)
    return array_y, array_v, array_t


def verlet(y_0, v_0, dt, t_f, funcional):
    """Realiza o cálculo númerico pelo método de Verlet."""
    n = int(t_f / dt)
    lista_y = [y_0]
    lista_v = [v_0]
    lista_t = [0]
    lista_y.append(lista_y[0] + lista_v[0] * dt)
    lista_t.append(lista_t[0] + dt)
    for i in range(n):
        lista_y.append(2 * lista_y[i + 1] - lista_y[i] + funcional(
            lista_y[i + 1], lista_v[i + 1], lista_t[i + 1]) * (dt)**2)
        lista_v.append((1 / 2) * (lista_y[i + 1] - lista_y[i - 1]) / dt)
        lista_t.append(lista_t[i] + dt)
    array_y = np.array(lista_y)
    array_v = np.array(lista_v)
    array_t = np.array(lista_t)
    return array_y, array_v, array_t


def leapfrog(y_0, v_0, dt, t_f, funcional):
    """Realiza o cálculo númerico pelo método Leapfrog."""
    n = int(t_f / dt)
    lista_v = [v_0]
    lista_y = [y_0]
    lista_t = [0]
    lista_v.append(lista_v[0] + funcional(lista_y[0],
                                          lista_v[0], lista_t[0]) * dt)
    lista_y.append(lista_y[0] + lista_v[0] * dt)
    lista_t.append(lista_t[0] + dt)
    for i in range(1, n):
        lista_v.append(lista_v[i - 1] + 2 *
                       funcional(lista_y[i], lista_v[i], lista_t[i]))
        lista_y.append(lista_y[i - 1] + 2 * lista_v[i] * dt)
        lista_t.append(lista_t[i] + dt)
    array_v = np.array(lista_v)
    array_y = np.array(lista_y)
    array_t = np.array(lista_t)
    return array_y, array_v, array_t
