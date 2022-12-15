from numpy import ndarray, identity, zeros


def matriz_pauli(j: int) -> ndarray:
    i = complex(0, 1)

    delta_k = identity(
        n=3,
        dtype=int
    )

    matriz = zeros(
        shape=(2, 2),
        dtype=complex,
    )

    matriz[0, 0] = delta_k[j - 1, 2]
    matriz[0, 1] = delta_k[j - 1, 0] - i * delta_k[j - 1, 1]
    matriz[1, 0] = delta_k[j - 1, 0] + i * delta_k[j - 1, 1]
    matriz[1, 1] = - delta_k[j - 1, 2]

    return matriz
