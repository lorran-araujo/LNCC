import numpy as np
import sympy
import matplotlib.pyplot as plt
import matplotlib

from sympy.abc import x

matplotlib.use('TkAgg')

# Definir a função para calcular I_n usando a fórmula de recorrência
def calculate_in(n_max):
    e = np.exp(1)
    I = np.zeros(n_max + 1)
    I[0] = 1 - 1/e
    for n in range(1, n_max + 1):
        I[n] = 1 - n * I[n-1]
    return I

# Definir a integral de I_n para obter a solução exata
def exact_integral(n):
    expression = (sympy.E**(x-1))*(x**n)
    return sympy.integrate(expression, (x, 0, 1))
    

# Calcular I_n para n = 1, 2, ..., 30
n_max = 30
I_values = calculate_in(n_max)

# Calcular a solução exata para n = 1, 2, ..., 30
exact_values = [exact_integral(n) for n in range(n_max + 1)]

# Plotar os valores calculados pela fórmula de recorrência e os valores exatos
n_values = np.arange(n_max + 1)

plt.figure(figsize=(12, 6))
plt.plot(n_values, I_values, label="I_n (recorrência)", marker='o')
plt.plot(n_values, exact_values, label="I_n (exato)", marker='x')
plt.yscale('log')
plt.xlabel('n')
plt.ylabel('I_n')
plt.title('Comparação entre I_n calculado pela fórmula de recorrência e a solução exata')
plt.legend()
plt.grid(True, which="both", ls="--")
plt.show()