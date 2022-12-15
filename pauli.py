import numpy as np

def matriz_pauli(j: int):
    i = complex(0, 1)

    delta_k = np.identity(
        n=3,
        dtype=int
    )

    matriz = np.zeros(
        shape=(2, 2),
        dtype=complex,
    )

    matriz[0, 0] = delta_k[j - 1, 2]
    matriz[0, 1] = delta_k[j - 1, 0] - i * delta_k[j - 1, 1]
    matriz[1, 0] = delta_k[j - 1, 0] + i * delta_k[j - 1, 1]
    matriz[1, 1] = - delta_k[j - 1, 2]

    return matriz
