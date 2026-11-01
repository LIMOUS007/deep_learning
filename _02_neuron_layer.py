import numpy as np
x = np.array([4.0, 0.9])
w = np.array([
    [0.7, 0.4],
    [0.2, 0.8],
    [-0.5, 1]
])
b = np.array([-3.0, -1.0, 2])
z = w @ x + b
a = np.maximum(0, z)
print("z =", z)
print("activation =", a)