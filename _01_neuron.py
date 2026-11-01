import numpy as np
x = np.array([5, 2])
w = np.array([2, 0.8])
b = 5.0
z = np.dot(w, x) + b
a = max(0, z)
print("z =", z)
print("activation =", a)