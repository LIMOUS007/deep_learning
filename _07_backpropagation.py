import numpy as np
np.random.seed(42)
def leaky_relu(x, alpha=0.01):
    x = np.asarray(x, dtype=float)
    return np.where(x > 0, x, alpha * x)

def sigmoid(x):
    # Stable form: exp() only ever sees a non-positive argument,
    # so large |x| can never overflow.
    x = np.asarray(x, dtype=float)
    z = np.exp(-np.abs(x))
    return np.where(x >= 0, 1 / (1 + z), z / (1 + z))

def silu(x):
    x = np.asarray(x, dtype=float)
    return x * sigmoid(x)
    
def silu_grad(x):
    s = sigmoid(x)
    return s + x * s * (1 - s)

x = np.array([
    [1., 2.],
    [2., 3.],
    [3., 4.],
    [4., 5.],
    [5., 6.],
    [6., 7.],
    [7., 8.],
    [8., 9.],
    [9., 10.],
    [10., 11.]
])

y = np.array([
    4.,
    7.,
    10.,
    13.,
    16.,
    19.,
    22.,
    25.,
    28.,
    31.
])


w1 = np.random.randn(4, 2) * 0.1
b1 = np.zeros(4)

w2 = np.random.randn(4, 4) * 0.1
b2 = np.zeros(4)

w3 = np.random.randn(1, 4) * 0.1
b3 = np.zeros(1)
lr = 0.01
x_train = x[:8]
y_train = y[:8]
x_val = x[8:]
y_val = y[8:]
n = len(x_train)
for epoch in range(10000):
    # Training forward pass
    z1 = x_train @ w1.T + b1
    z1 = np.asarray(z1, dtype=float)
    a1 = leaky_relu(z1)
    z2 = a1 @ w2.T + b2
    z2 = np.asarray(z2, dtype=float)
    a2 = leaky_relu(z2)
    z3 = a2 @ w3.T + b3
    y_pred = z3[:, 0]
    
    # Validation forward pass
    z1_val = x_val @ w1.T + b1
    z1_val = np.asarray(z1_val, dtype=float)
    a1_val = leaky_relu(z1_val)
    z2_val = a1_val @ w2.T + b2
    z2_val = np.asarray(z2_val, dtype=float)
    a2_val = leaky_relu(z2_val)
    z3_val = a2_val @ w3.T + b3
    val_pred = z3_val[:, 0]
    val_loss = np.mean((y_val - val_pred) ** 2)
    loss = np.mean((y_train - y_pred) ** 2)
    dL_dy_pred = 2 * (y_pred - y_train) / n
    dL_dz3 = dL_dy_pred[:, None]
    dw3 = dL_dz3.T @ a2
    db3 = np.sum(dL_dz3, axis=0)
    da2 = dL_dz3 @ w3
    leaky_relu2_gradient = (z2 > 0).astype(float) + 0.01 * (z2 <= 0).astype(float)
    dL_dz2 = da2 * leaky_relu2_gradient
    dw2 = dL_dz2.T @ a1
    db2 = np.sum(dL_dz2, axis=0)
    da1 = dL_dz2 @ w2
    leaky_relu1_gradient = (z1 > 0).astype(float) + 0.01 * (z1 <= 0).astype(float)
    dL_dz1 = da1 * leaky_relu1_gradient
    dw1 = dL_dz1.T @ x_train
    db1 = np.sum(dL_dz1, axis=0)
    dx = dL_dz1 @ w1
    w1 -= lr * dw1
    b1 -= lr * db1
    w2 -= lr * dw2
    b2 -= lr * db2 
    w3 -= lr * dw3
    b3 -= lr * db3
    if epoch % 1000 == 0:
        print(
            f"Epoch {epoch:3d} | "
            f"Loss: {loss:.16f} | "
            f"Prediction: {y_pred[0]:.4f}"
        )
        print("\nValidation predictions:", val_pred)
        print("Actual values:", y_val)
        print("Validation loss:", val_loss)