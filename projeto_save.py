import numpy as np

from cadeia_ising_transverso import array_lamda, salvar_array_autoval

N_min = 2
N_max = 10

np.save(
    file='projeto/array_lamda',
    arr=array_lamda,
)

for N in np.arange(N_min, N_max + 1):
    salvar_array_autoval(
        N=N,
        ccp=False
    )

    salvar_array_autoval(
        N=N,
        ccp=True
    )
