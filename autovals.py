import mpmath as mp


def determinante(lamda, matriz):
    n = matriz.rows
    I = mp.eye(n)
    matriz_eq = lamda * I - matriz
    det = mp.det(matriz_eq)
    return det

# def pol_carac(matriz):
