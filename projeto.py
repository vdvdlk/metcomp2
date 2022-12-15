import numpy as np
import matplotlib.pyplot as plt

from pauli import matriz_pauli






# def hamiltoniana_2(lamda):
#     H = np.zeros(
#         shape=(4, 4)
#     )

#     H[0, 0] = - 1
#     H[0, 1] = - lamda
#     H[0, 2] = - lamda

#     H[1, 0] = - lamda
#     H[1, 1] = 1
#     H[1, 3] = - lamda

#     H[2, 0] = - lamda
#     H[2, 2] = 1
#     H[2, 3] = - lamda

#     H[3, 1] = - lamda
#     H[3, 2] = - lamda
#     H[3, 3] = -1

#     return H


# lamda = np.linspace(
#     start=0,
#     stop=10,
#     num=100,
# )

# autovalores = np.zeros(
#     shape=lamda.shape
# )

# for L in lamda:


# print(np.linalg.eigh(
#     a=hamiltoniana_2(0),
#     UPLO='U'
# ))

# def
