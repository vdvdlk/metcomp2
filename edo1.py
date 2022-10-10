import numpy as np


def euler(y_0, dt, t_f, funcional):
    """Realiza o cálculo númerico pelo método de Euler."""
    n = int(t_f / dt)
    lista_y = [y_0]
    lista_t = [0]
    for i in range(n):
        lista_y.append(lista_y[i] + funcional(lista_y[i], lista_t[i]) * dt)
        lista_t.append(lista_t[i] + dt)
    array_y = np.array(lista_y)
    array_t = np.array(lista_t)
    return array_y, array_t


def rungekutta2(y_0, dt, t_f, funcional):
    """Realiza o cálculo númerico pelo método de Runge-Kutta de segunda ordem."""
    n = int(t_f / dt)
    lista_y = [y_0]
    lista_t = [0]
    for i in range(n):
        y_l = lista_y[i] + (1 / 2) * funcional(lista_y[i], lista_t[i]) * dt
        t_l = lista_t[i] + (1 / 2) * dt
        lista_y.append(lista_y[i] + funcional(y_l, t_l) * dt)
        lista_t.append(lista_t[i] + dt)
    array_y = np.array(lista_y)
    array_t = np.array(lista_t)
    return array_y, array_t


def rungekutta4(y_0, dt, t_f, funcional):
    """Realiza o cálculo númerico pelo método de Runge-Kutta de segunda ordem."""
    n = int(t_f / dt)
    lista_y = [y_0]
    lista_t = [0]
    for i in range(n):
        y_l_1 = lista_y[i]
        t_l_1 = lista_t[i]

        y_l_2 = lista_y[i] + (1 / 2) * funcional(y_l_1, t_l_1) * dt
        t_l_2 = lista_t[i] + (1 / 2) * dt

        y_l_3 = lista_y[i] + (1 / 2) * funcional(y_l_2, t_l_2) * dt
        t_l_3 = lista_t[i] + (1 / 2) * dt

        y_l_4 = lista_y[i] + funcional(y_l_3, t_l_3) * dt
        t_l_4 = lista_t[i] + dt

        lista_y.append(lista_y[i] + (1 / 6) * (funcional(y_l_1, t_l_1) + 2 * funcional(
            y_l_2, t_l_2) + 2 * funcional(y_l_3, t_l_3) + funcional(y_l_4, t_l_4)) * dt)
        lista_t.append(lista_t[i] + dt)
    array_y = np.array(lista_y)
    array_t = np.array(lista_t)
    return array_y, array_t
