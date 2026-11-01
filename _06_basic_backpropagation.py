import numpy as np
def silu(x):
    return x / (1.0 + np.exp(-x))
x = np.array([1., 2., 3., 4., 5., 6., 7., 8., 9., 10.])
y = np.array([2., 4., 6., 8., 10., 12., 14., 16., 18., 20.])
w = 0.0
b = 1
lr = 5e-3
for epoch in range(1000):
    z = x * w + b
    a = silu(z)
    loss = np.mean((y - a) ** 2)
    gradient_w = np.mean(-2 * x * (y - a))
    gradient_b = np.mean(-2 * (y - a))
    w = w - lr * gradient_w
    b = b - lr * gradient_b
    print(f"epoch {epoch + 1}, "
          f"w={w:.4f}, "
          f"b={b:.4f}, "
          f"prediction: {a}, "
          f"loss: {loss:.4f}, "
          f"gradient_w: {gradient_w:.4f}, "
          f"gradient_b: {gradient_b:.4f}")
    