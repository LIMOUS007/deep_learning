import numpy as np

x = np.array([np.random.randn() for x in range(3)])

w1 = np.array([[np.random.randn() for x in range(3)],
              [np.random.randn() for x in range(3)],
              [np.random.randn() for x in range(3)],
              [np.random.randn() for x in range(3)]])

b1 = np.array([np.random.randn() for x in range(4)])

z1 = x @ w1.T + b1

a1 = np.maximum(0, z1)

w2 = np.array([[np.random.randn() for x in range(4)],
              [np.random.randn() for x in range(4)],
              [np.random.randn() for x in range(4)]])

b2 = np.array([np.random.randn() for x in range(3)])

z2 = a1 @ w2.T + b2

a2 = 1 / (1 + np.exp(-z2))

w3 = np.array([[np.random.randn() for x in range(3)],
              [np.random.randn() for x in range(3)],
              [np.random.randn() for x in range(3)]])

b3= np.array([np.random.randn() for x in range(3)])

z3 = a2 @ w3.T + b3

a3 = np.exp(z3) / np.sum(np.exp(z3))

print("z3 =", z3)
print("activation =", a3)
