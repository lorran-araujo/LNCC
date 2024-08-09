import random as r
import math as m 

min = m.exp(20)

for i in range(1, 10000000):
    y = r.uniform(0, 2)

    v = m.exp(pow(y, 2)-(1/4)) - 2

    if (v >= 0 and v <= min) or (v<0 and v > min):
        x = y
        f = v
        if f >= 0: 
            min = f
        else:
            min = -f

print(f"Raiz encontrada: x = {x}, com f(x) = {f}.")