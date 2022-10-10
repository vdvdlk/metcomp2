import numpy as np


def dominio_rad(angulo):
    """Restringe o domínio dos ângulos em radianos para o intervalo (pi, pi]"""
    return np.arctan2(np.sin(angulo), np.cos(angulo))


def poincare(theta, omega, dt, T):
    """Função que produz os gráficos das seções de Poincaré"""
    omega_p = []
    theta_p = []
    n = 0
    for i in range(len(theta)):
        if abs(i * dt - n * T) < (dt / 2):
            omega_p.append(omega[i])
            theta_p.append(theta[i])
            n += 1
    return theta_p, omega_p
