import numpy as np

def gradient_descent(param, grad, lr=0.01):
    return param - lr * grad

def momentum(param, grad, velocity, lr=0.01, beta=0.9):
    param = np.asarray(param, dtype=float)
    velocity = beta * velocity + (1 - beta) * grad
    param = param - lr * velocity
    return param, velocity

def adagrad(param, grad, accumulated_grad, lr=0.01, epsilon=1e-8):
    param = np.asarray(param, dtype=float)
    accumulated_grad = accumulated_grad + grad ** 2
    param = param - lr * grad / (np.sqrt(accumulated_grad) + epsilon)
    return param, accumulated_grad

def rmsprop(param, grad, squared_avg, lr=0.001, beta=0.9, epsilon=1e-8):
    param = np.asarray(param, dtype=float)
    squared_avg = beta * squared_avg + (1 - beta) * grad ** 2
    param = param - lr * grad / (np.sqrt(squared_avg) + epsilon)
    return param, squared_avg

def adam(param, grad, m, v, t, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
    param = np.asarray(param, dtype=float)
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * grad ** 2
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)
    param = param - lr * m_hat / (np.sqrt(v_hat) + epsilon)
    return param, m, v

def adamw(param, grad, m, v, t, lr=0.001, beta1=0.9, beta2=0.999, weight_decay=0.01, epsilon=1e-8):
    param = np.asarray(param, dtype=float)
    param = param - lr * weight_decay * param
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * grad ** 2
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)
    param = param - lr * m_hat / (np.sqrt(v_hat) + epsilon)
    return param, m, v

if __name__ == "__main__":
    target = 3.0
    steps = 50

    def grad_of(w):
        return 2 * (w - target)

    w = 0.0
    for _ in range(steps):
        w = gradient_descent(w, grad_of(w), lr=0.1)
    print("Gradient Descent :", w)

    w, velocity = 0.0, 0.0
    for _ in range(steps):
        w, velocity = momentum(w, grad_of(w), velocity, lr=0.1)
    print("Momentum         :", w)

    w, accumulated_grad = 0.0, 0.0
    for _ in range(steps):
        w, accumulated_grad = adagrad(w, grad_of(w), accumulated_grad, lr=0.5)
    print("Adagrad          :", w)

    w, squared_avg = 0.0, 0.0
    for _ in range(steps):
        w, squared_avg = rmsprop(w, grad_of(w), squared_avg, lr=0.05)
    print("RMSProp          :", w)

    w, m, v = 0.0, 0.0, 0.0
    for t in range(1, steps + 1):
        w, m, v = adam(w, grad_of(w), m, v, t, lr=0.05)
    print("Adam             :", w)

    w, m, v = 0.0, 0.0, 0.0
    for t in range(1, steps + 1):
        w, m, v = adamw(w, grad_of(w), m, v, t, lr=0.05)
    print("AdamW            :", w)
