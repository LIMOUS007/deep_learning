import numpy as np

def binary_step(x):
    x = np.asarray(x, dtype=float)
    return np.where(x >= 0, 1, 0)

def sigmoid(x):
    # Stable form: exp() only ever sees a non-positive argument,
    # so large |x| can never overflow.
    x = np.asarray(x, dtype=float)
    z = np.exp(-np.abs(x))
    return np.where(x >= 0, 1 / (1 + z), z / (1 + z))

def tanh(x):
    return np.tanh(x)

def relu(x):
    x = np.asarray(x, dtype=float)
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.01):
    x = np.asarray(x, dtype=float)
    return np.where(x > 0, x, alpha * x)

def prelu(x, alpha=0.25):
    """
    Same shape as Leaky ReLU, but alpha is a *learned* parameter rather
    than a fixed constant, so it may be an array broadcastable to x
    (e.g. one slope per channel).
    """
    x = np.asarray(x, dtype=float)
    alpha = np.asarray(alpha, dtype=float)
    return np.where(x > 0, x, alpha * x)

def elu(x, alpha=1.0):
    # np.where evaluates both branches, so clamp the exp() input at 0
    # to keep the unused positive branch from overflowing.
    x = np.asarray(x, dtype=float)
    return np.where(x > 0, x, alpha * (np.exp(np.minimum(x, 0)) - 1))

def selu(x):
    alpha = 1.6732632423543772
    scale = 1.0507009873554805

    x = np.asarray(x, dtype=float)
    return scale * np.where(
        x > 0,
        x,
        alpha * (np.exp(np.minimum(x, 0)) - 1)
    )

def softplus(x):
    # log(1 + exp(x)), computed as logaddexp(0, x) so that large x
    # returns ~x instead of overflowing inside exp().
    return np.logaddexp(0, np.asarray(x, dtype=float))

def gelu(x):
    x = np.asarray(x, dtype=float)
    return 0.5 * x * (
        1 + np.tanh(
            np.sqrt(2 / np.pi) *
            (x + 0.044715 * x**3)
        )
    )

def silu(x):
    x = np.asarray(x, dtype=float)
    return x * sigmoid(x)

def swish(x, beta=1.0):
    x = np.asarray(x, dtype=float)
    return x * sigmoid(beta * x)

def mish(x):
    x = np.asarray(x, dtype=float)
    return x * np.tanh(softplus(x))

def linear(x):
    return np.asarray(x, dtype=float)

def softmax(x):
    x = np.asarray(x, dtype=float)
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))

    return exp_x / np.sum(
        exp_x,
        axis=-1,
        keepdims=True
    )

def log_softmax(x):
    x = np.asarray(x, dtype=float)
    shifted = x - np.max(
        x,
        axis=-1,
        keepdims=True
    )
    return shifted - np.log(
        np.sum(np.exp(shifted), axis=-1, keepdims=True)
    )

def glu(A, B):
    return np.asarray(A, dtype=float) * sigmoid(B)

def reglu(A, B):
    return np.asarray(A, dtype=float) * relu(B)

def geglu(A, B):
    return np.asarray(A, dtype=float) * gelu(B)

def swiglu(A, B):
    return np.asarray(A, dtype=float) * silu(B)

def maxout(x, weights, biases):
    """
    x       : input, shape (..., d_in)
    weights : collection of weight vectors/matrices, each (d_in,) or (d_in, d_out)
    biases  : corresponding biases, one per weight

    Each weight/bias pair creates a linear function.
    Maxout returns the largest one, elementwise.
    """
    if len(weights) != len(biases):
        raise ValueError(
            f"got {len(weights)} weights but {len(biases)} biases"
        )
    if len(weights) == 0:
        raise ValueError("maxout needs at least one weight/bias pair")

    x = np.asarray(x, dtype=float)

    outputs = []

    for w, b in zip(weights, biases):
        outputs.append(x @ np.asarray(w, dtype=float) + b)

    return np.max(np.stack(outputs), axis=0)

if __name__ == "__main__":
    x = np.array([-3., -2., -1., 0., 1., 2., 3.])
    A = np.array([-2., -1., 0., 1., 2.])
    B = np.array([-2., -1., 0., 1., 2.])

    # Maxout is a linear layer, so its input needs a feature dimension
    # matching the weight vectors: 3 samples x 2 features.
    x_maxout = np.array([
        [-1., 2.],
        [0.5, -0.5],
        [3., 1.]
    ])
    weights = [
        np.array([1., 2.]),
        np.array([2., 1.]),
        np.array([-1., 1.])
    ]
    biases = [0., 1., 2.]

    print("Binary Step :", binary_step(x))
    print("Sigmoid     :", sigmoid(x))
    print("Tanh        :", tanh(x))
    print("ReLU        :", relu(x))
    print("Leaky ReLU  :", leaky_relu(x))
    print("PReLU       :", prelu(x))
    print("ELU         :", elu(x))
    print("SELU        :", selu(x))
    print("Softplus    :", softplus(x))
    print("GELU        :", gelu(x))
    print("SiLU        :", silu(x))
    print("Swish       :", swish(x))
    print("Mish        :", mish(x))
    print("Linear      :", linear(x))
    print("Softmax     :", softmax(x))
    print("LogSoftmax  :", log_softmax(x))
    print("GLU     :", glu(A, B))
    print("ReGLU   :", reglu(A, B))
    print("GeGLU   :", geglu(A, B))
    print("SwiGLU  :", swiglu(A, B))
    print("Maxout  :", maxout(x_maxout, weights, biases))