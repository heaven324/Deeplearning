import numpy as np

x = np.array([[1.0, -0.5], [-2.0, 3.0]])
print(x)

mask = (x <= 0)
print(mask)
out = x.copy()
print(out)
out[mask] = 0
print(out)