import numpy as np

# Every optimizer works on NumPy arrays and follows the same pattern:
#   new_param, *new_state = optimizer(param, grad, *state, **hyperparameters)
# State (velocities, moment estimates, ...) starts at 0.0 / None and is
# handed back on the next call. Optimizers that need to evaluate the
# gradient or loss themselves take a function (grad_fn / loss_fn) instead.


# ---------------------------------------------------------------------------
# Gradient descent
# ---------------------------------------------------------------------------

def gradient_descent(param, grad, lr=0.01):
    return param - lr * grad

def batch_gradient_descent(param, grad_fn, X, y, lr=0.01):
    # Gradient over the entire dataset.
    return param - lr * grad_fn(param, X, y)

def sgd(param, grad_fn, X, y, lr=0.01, rng=None):
    # Gradient from one randomly chosen example.
    rng = np.random.default_rng() if rng is None else rng
    i = rng.integers(len(X))
    return param - lr * grad_fn(param, X[i:i + 1], y[i:i + 1])

def minibatch_sgd(param, grad_fn, X, y, lr=0.01, batch_size=32, rng=None):
    # Gradient from a small random batch.
    rng = np.random.default_rng() if rng is None else rng
    idx = rng.choice(len(X), size=min(batch_size, len(X)), replace=False)
    return param - lr * grad_fn(param, X[idx], y[idx])


# ---------------------------------------------------------------------------
# Momentum-based
# ---------------------------------------------------------------------------

def momentum(param, grad, velocity, lr=0.01, beta=0.9):
    param = np.asarray(param, dtype=float)
    velocity = beta * velocity + (1 - beta) * grad
    param = param - lr * velocity
    return param, velocity

def nesterov_momentum(param, grad_fn, velocity, lr=0.01, beta=0.9):
    # Gradient is taken at the look-ahead point param + beta * velocity.
    param = np.asarray(param, dtype=float)
    grad = grad_fn(param + beta * velocity)
    velocity = beta * velocity - lr * grad
    param = param + velocity
    return param, velocity


# ---------------------------------------------------------------------------
# Adaptive learning rate
# ---------------------------------------------------------------------------

def adagrad(param, grad, accumulated_grad, lr=0.01, epsilon=1e-8):
    param = np.asarray(param, dtype=float)
    accumulated_grad = accumulated_grad + grad ** 2
    param = param - lr * grad / (np.sqrt(accumulated_grad) + epsilon)
    return param, accumulated_grad

def adadelta(param, grad, squared_avg, delta_avg, lr=1.0, rho=0.9, epsilon=1e-6):
    param = np.asarray(param, dtype=float)
    squared_avg = rho * squared_avg + (1 - rho) * grad ** 2
    update = np.sqrt(delta_avg + epsilon) / np.sqrt(squared_avg + epsilon) * grad
    delta_avg = rho * delta_avg + (1 - rho) * update ** 2
    param = param - lr * update
    return param, squared_avg, delta_avg

def rmsprop(param, grad, squared_avg, lr=0.001, beta=0.9, epsilon=1e-8):
    param = np.asarray(param, dtype=float)
    squared_avg = beta * squared_avg + (1 - beta) * grad ** 2
    param = param - lr * grad / (np.sqrt(squared_avg) + epsilon)
    return param, squared_avg


# ---------------------------------------------------------------------------
# Adaptive moment
# ---------------------------------------------------------------------------

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

def adamax(param, grad, m, u, t, lr=0.002, beta1=0.9, beta2=0.999, epsilon=1e-8):
    # Infinity norm replaces the second moment.
    param = np.asarray(param, dtype=float)
    m = beta1 * m + (1 - beta1) * grad
    u = np.maximum(beta2 * u, np.abs(grad))
    param = param - (lr / (1 - beta1 ** t)) * m / (u + epsilon)
    return param, m, u

def nadam(param, grad, m, v, t, lr=0.002, beta1=0.9, beta2=0.999, epsilon=1e-8):
    param = np.asarray(param, dtype=float)
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * grad ** 2
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)
    # Nesterov look-ahead: mix the bias-corrected momentum with the current gradient.
    m_bar = beta1 * m_hat + (1 - beta1) * grad / (1 - beta1 ** t)
    param = param - lr * m_bar / (np.sqrt(v_hat) + epsilon)
    return param, m, v

def radam(param, grad, m, v, t, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
    param = np.asarray(param, dtype=float)
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * grad ** 2
    m_hat = m / (1 - beta1 ** t)
    rho_inf = 2 / (1 - beta2) - 1
    rho_t = rho_inf - 2 * t * beta2 ** t / (1 - beta2 ** t)
    if rho_t > 5:
        # Variance of the adaptive rate is tractable: rectify it.
        v_hat = v / (1 - beta2 ** t)
        r = np.sqrt((rho_t - 4) * (rho_t - 2) * rho_inf / ((rho_inf - 4) * (rho_inf - 2) * rho_t))
        param = param - lr * r * m_hat / (np.sqrt(v_hat) + epsilon)
    else:
        # Early steps: plain momentum (no adaptive scaling).
        param = param - lr * m_hat
    return param, m, v

def amsgrad(param, grad, m, v, v_max, t, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
    param = np.asarray(param, dtype=float)
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * grad ** 2
    v_max = np.maximum(v_max, v)
    m_hat = m / (1 - beta1 ** t)
    v_hat = v_max / (1 - beta2 ** t)
    param = param - lr * m_hat / (np.sqrt(v_hat) + epsilon)
    return param, m, v, v_max


# ---------------------------------------------------------------------------
# Adaptive variants
# ---------------------------------------------------------------------------

def adabelief(param, grad, m, s, t, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
    # s tracks how far the gradient deviates from its own moving average.
    param = np.asarray(param, dtype=float)
    m = beta1 * m + (1 - beta1) * grad
    s = beta2 * s + (1 - beta2) * (grad - m) ** 2 + epsilon
    m_hat = m / (1 - beta1 ** t)
    s_hat = s / (1 - beta2 ** t)
    param = param - lr * m_hat / (np.sqrt(s_hat) + epsilon)
    return param, m, s

def adabound(param, grad, m, v, t, lr=0.001, final_lr=0.1, gamma=1e-3,
             beta1=0.9, beta2=0.999, epsilon=1e-8):
    param = np.asarray(param, dtype=float)
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * grad ** 2
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)
    # Bounds start wide (Adam-like) and close in on final_lr (SGD-like).
    lower = final_lr * (1 - 1 / (gamma * t + 1))
    upper = final_lr * (1 + 1 / (gamma * t))
    step_size = np.clip(lr / (np.sqrt(v_hat) + epsilon), lower, upper)
    param = param - step_size * m_hat
    return param, m, v

def adan(param, grad, m, v, n, prev_grad, t, lr=0.001, beta1=0.98, beta2=0.92,
         beta3=0.99, weight_decay=0.0, epsilon=1e-8):
    param = np.asarray(param, dtype=float)
    grad_diff = np.zeros_like(param) if prev_grad is None else grad - prev_grad
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * grad_diff
    n = beta3 * n + (1 - beta3) * (grad + beta2 * grad_diff) ** 2
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)
    n_hat = n / (1 - beta3 ** t)
    update = (m_hat + beta2 * v_hat) / (np.sqrt(n_hat) + epsilon)
    param = (param - lr * update) / (1 + lr * weight_decay)
    return param, m, v, n, grad


# ---------------------------------------------------------------------------
# Averaged SGD
# ---------------------------------------------------------------------------

def asgd(param, grad, avg_param, t, lr=0.01):
    # Takes a normal SGD step, and keeps a running average of all iterates.
    # Use avg_param as the final model.
    param = np.asarray(param, dtype=float)
    param = param - lr * grad
    avg_param = avg_param + (param - avg_param) / t
    return param, avg_param


# ---------------------------------------------------------------------------
# Sign-based
# ---------------------------------------------------------------------------

def rprop(param, grad, prev_grad, step_size, eta_plus=1.2, eta_minus=0.5,
          step_min=1e-6, step_max=50.0):
    param = np.asarray(param, dtype=float)
    sign_change = grad * prev_grad
    # Same sign -> grow the step, opposite sign -> shrink it.
    step_size = np.where(sign_change > 0, np.minimum(step_size * eta_plus, step_max),
                np.where(sign_change < 0, np.maximum(step_size * eta_minus, step_min),
                         step_size))
    # After a sign flip, skip this step (classic Rprop- behaviour).
    grad = np.where(sign_change < 0, 0.0, grad)
    param = param - np.sign(grad) * step_size
    return param, grad, step_size

def lion(param, grad, m, lr=1e-4, beta1=0.9, beta2=0.99, weight_decay=0.0):
    param = np.asarray(param, dtype=float)
    update = np.sign(beta1 * m + (1 - beta1) * grad)
    param = param - lr * (update + weight_decay * param)
    m = beta2 * m + (1 - beta2) * grad
    return param, m


# ---------------------------------------------------------------------------
# Layer-wise adaptive (call once per layer's weight array)
# ---------------------------------------------------------------------------

def lars(param, grad, velocity, lr=0.1, beta=0.9, weight_decay=0.0,
         trust_coefficient=0.001, epsilon=1e-8):
    param = np.asarray(param, dtype=float)
    w_norm = np.linalg.norm(param)
    g_norm = np.linalg.norm(grad)
    if w_norm > 0 and g_norm > 0:
        local_lr = trust_coefficient * w_norm / (g_norm + weight_decay * w_norm + epsilon)
    else:
        local_lr = 1.0
    velocity = beta * velocity + lr * local_lr * (grad + weight_decay * param)
    param = param - velocity
    return param, velocity

def lamb(param, grad, m, v, t, lr=0.001, beta1=0.9, beta2=0.999,
         weight_decay=0.01, epsilon=1e-6):
    param = np.asarray(param, dtype=float)
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * grad ** 2
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)
    update = m_hat / (np.sqrt(v_hat) + epsilon) + weight_decay * param
    w_norm = np.linalg.norm(param)
    u_norm = np.linalg.norm(update)
    trust_ratio = w_norm / u_norm if w_norm > 0 and u_norm > 0 else 1.0
    param = param - lr * trust_ratio * update
    return param, m, v


# ---------------------------------------------------------------------------
# Memory-efficient
# ---------------------------------------------------------------------------

def adafactor(param, grad, row_avg, col_avg, t, lr=0.01, decay_rate=-0.8,
              epsilon=1e-30, clip_threshold=1.0):
    # For a 2-D param, only a row vector and a column vector of squared-gradient
    # statistics are stored instead of a full matrix. 1-D params keep the
    # full vector in row_avg (col_avg is unused).
    param = np.asarray(param, dtype=float)
    beta2 = 1 - t ** decay_rate
    sq = grad ** 2 + epsilon
    if param.ndim >= 2:
        row_avg = beta2 * row_avg + (1 - beta2) * sq.mean(axis=1)
        col_avg = beta2 * col_avg + (1 - beta2) * sq.mean(axis=0)
        v_hat = np.outer(row_avg, col_avg) / row_avg.mean()
    else:
        row_avg = beta2 * row_avg + (1 - beta2) * sq
        v_hat = row_avg
    update = grad / np.sqrt(v_hat)
    rms = np.sqrt(np.mean(update ** 2))
    update = update / max(1.0, rms / clip_threshold)
    param = param - lr * update
    return param, row_avg, col_avg

def _quantize(x, signed):
    # Per-tensor absmax quantization to 8 bits.
    levels = 127 if signed else 255
    scale = np.max(np.abs(x)) / levels
    if scale == 0:
        scale = 1.0
    q = np.round(x / scale).astype(np.int8 if signed else np.uint8)
    return q, scale

def _dequantize(state, shape):
    if state is None:
        return np.zeros(shape)
    q, scale = state
    return q.astype(float) * scale

def adam_8bit(param, grad, m_state, v_state, t, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
    # m is stored as int8, sqrt(v) as uint8 (sqrt keeps more dynamic range).
    # Real 8-bit Adam (bitsandbytes) uses block-wise dynamic quantization;
    # this is a simplified per-tensor version. States start as None.
    param = np.asarray(param, dtype=float)
    m = _dequantize(m_state, param.shape)
    v = _dequantize(v_state, param.shape) ** 2
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * grad ** 2
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)
    param = param - lr * m_hat / (np.sqrt(v_hat) + epsilon)
    return param, _quantize(m, signed=True), _quantize(np.sqrt(v), signed=False)

def galore(param, grad, projection, m, v, t, lr=0.01, rank=1, update_proj_gap=200,
           scale=1.0, beta1=0.9, beta2=0.999, epsilon=1e-8):
    # 2-D params only. Adam runs in a rank-r subspace found by SVD of the gradient,
    # so m and v have shape (rank, cols) instead of (rows, cols).
    param = np.asarray(param, dtype=float)
    if projection is None or (t - 1) % update_proj_gap == 0:
        U, _, _ = np.linalg.svd(grad, full_matrices=False)
        projection = U[:, :rank]
    low_rank_grad = projection.T @ grad
    if m is None:
        m = np.zeros_like(low_rank_grad)
        v = np.zeros_like(low_rank_grad)
    m = beta1 * m + (1 - beta1) * low_rank_grad
    v = beta2 * v + (1 - beta2) * low_rank_grad ** 2
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)
    update = projection @ (m_hat / (np.sqrt(v_hat) + epsilon))
    param = param - lr * scale * update
    return param, projection, m, v

def apollo(param, grad, projection, m, v, t, lr=0.01, rank=1, update_proj_gap=200,
           scale=1.0, beta1=0.9, beta2=0.999, epsilon=1e-8, rng=None):
    # 2-D params only. Adam moments are tracked on a random low-rank projection;
    # they only decide a per-column scaling factor applied to the raw gradient.
    rng = np.random.default_rng() if rng is None else rng
    param = np.asarray(param, dtype=float)
    if projection is None or (t - 1) % update_proj_gap == 0:
        projection = rng.standard_normal((rank, param.shape[0])) / np.sqrt(rank)
    low_rank_grad = projection @ grad
    if m is None:
        m = np.zeros_like(low_rank_grad)
        v = np.zeros_like(low_rank_grad)
    m = beta1 * m + (1 - beta1) * low_rank_grad
    v = beta2 * v + (1 - beta2) * low_rank_grad ** 2
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)
    adapted = m_hat / (np.sqrt(v_hat) + epsilon)
    column_scale = np.linalg.norm(adapted, axis=0) / (np.linalg.norm(low_rank_grad, axis=0) + epsilon)
    param = param - lr * scale * grad * column_scale
    return param, projection, m, v


# ---------------------------------------------------------------------------
# Sparse / online
# ---------------------------------------------------------------------------

def sparse_adam(param, grad, m, v, t, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
    # Only entries with a non-zero gradient (e.g. embedding rows that were used)
    # have their moments and values updated.
    param = np.array(param, dtype=float)
    m = np.array(m, dtype=float) * np.ones_like(param)
    v = np.array(v, dtype=float) * np.ones_like(param)
    mask = grad != 0
    m[mask] = beta1 * m[mask] + (1 - beta1) * grad[mask]
    v[mask] = beta2 * v[mask] + (1 - beta2) * grad[mask] ** 2
    m_hat = m[mask] / (1 - beta1 ** t)
    v_hat = v[mask] / (1 - beta2 ** t)
    param[mask] = param[mask] - lr * m_hat / (np.sqrt(v_hat) + epsilon)
    return param, m, v

def ftrl(param, grad, z, n, alpha=0.1, beta=1.0, l1=0.0, l2=0.0):
    # FTRL-Proximal. The weights are recomputed from z and n each step,
    # and l1 drives small weights to exactly zero.
    param = np.asarray(param, dtype=float)
    sigma = (np.sqrt(n + grad ** 2) - np.sqrt(n)) / alpha
    z = z + grad - sigma * param
    n = n + grad ** 2
    param = np.where(np.abs(z) <= l1, 0.0,
                     -(z - np.sign(z) * l1) / ((beta + np.sqrt(n)) / alpha + l2))
    return param, z, n


# ---------------------------------------------------------------------------
# Matrix preconditioning (2-D params)
# ---------------------------------------------------------------------------

def _matrix_power(matrix, power, epsilon=1e-12):
    # Power of a symmetric positive semi-definite matrix via eigendecomposition.
    eigvals, eigvecs = np.linalg.eigh(matrix)
    eigvals = np.maximum(eigvals, epsilon)
    return (eigvecs * eigvals ** power) @ eigvecs.T

def shampoo(param, grad, left, right, lr=0.1, epsilon=1e-4):
    # left / right start as None. Update = L^(-1/4) G R^(-1/4).
    param = np.asarray(param, dtype=float)
    if left is None:
        left = epsilon * np.eye(param.shape[0])
        right = epsilon * np.eye(param.shape[1])
    left = left + grad @ grad.T
    right = right + grad.T @ grad
    update = _matrix_power(left, -0.25) @ grad @ _matrix_power(right, -0.25)
    param = param - lr * update
    return param, left, right

def soap(param, grad, m, v, left, right, q_left, q_right, t, lr=0.003, beta1=0.95,
         beta2=0.95, shampoo_beta=0.95, precondition_frequency=10, epsilon=1e-8):
    # Adam running in the eigenbasis of Shampoo's preconditioners.
    # left / right / q_left / q_right start as None.
    # (The full method also rotates v when the basis changes; skipped here.)
    param = np.asarray(param, dtype=float)
    if left is None:
        left = np.zeros((param.shape[0], param.shape[0]))
        right = np.zeros((param.shape[1], param.shape[1]))
    left = shampoo_beta * left + (1 - shampoo_beta) * grad @ grad.T
    right = shampoo_beta * right + (1 - shampoo_beta) * grad.T @ grad
    if q_left is None or (t - 1) % precondition_frequency == 0:
        q_left = np.linalg.eigh(left)[1]
        q_right = np.linalg.eigh(right)[1]
    m = beta1 * m + (1 - beta1) * grad
    grad_rot = q_left.T @ grad @ q_right
    m_rot = q_left.T @ m @ q_right
    v = beta2 * v + (1 - beta2) * grad_rot ** 2
    m_hat = m_rot / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)
    update = q_left @ (m_hat / (np.sqrt(v_hat) + epsilon)) @ q_right.T
    param = param - lr * update
    return param, m, v, left, right, q_left, q_right

def _newton_schulz(matrix, steps=5, epsilon=1e-7):
    # Approximately maps a matrix to the nearest orthogonal matrix (U V^T).
    a, b, c = 3.4445, -4.7750, 2.0315
    x = matrix / (np.linalg.norm(matrix) + epsilon)
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T
    for _ in range(steps):
        A = x @ x.T
        B = b * A + c * A @ A
        x = a * x + B @ x
    return x.T if transposed else x

def muon(param, grad, momentum_buffer, lr=0.02, beta=0.95, nesterov=True, ns_steps=5):
    param = np.asarray(param, dtype=float)
    momentum_buffer = beta * momentum_buffer + grad
    direction = grad + beta * momentum_buffer if nesterov else momentum_buffer
    update = _newton_schulz(direction, ns_steps)
    update = update * np.sqrt(max(1.0, param.shape[0] / param.shape[1]))
    param = param - lr * update
    return param, momentum_buffer


# ---------------------------------------------------------------------------
# Natural / second-order
# ---------------------------------------------------------------------------

def newton_method(param, grad, hessian, lr=1.0):
    # hessian is the (n x n) matrix of second derivatives for the flattened param.
    param = np.asarray(param, dtype=float)
    step = np.linalg.solve(np.atleast_2d(hessian), np.ravel(grad))
    return param - lr * step.reshape(param.shape)

def bfgs(param, grad_fn, inv_hessian, lr=1.0):
    # inv_hessian starts as None (identity). Fixed step size, no line search.
    param = np.asarray(param, dtype=float)
    shape = param.shape
    x = param.ravel()
    g = np.ravel(grad_fn(param))
    if inv_hessian is None:
        inv_hessian = np.eye(x.size)
    x_new = x - lr * inv_hessian @ g
    g_new = np.ravel(grad_fn(x_new.reshape(shape)))
    s = x_new - x
    y = g_new - g
    sy = s @ y
    if sy > 1e-12:
        rho = 1.0 / sy
        I = np.eye(x.size)
        inv_hessian = (I - rho * np.outer(s, y)) @ inv_hessian @ (I - rho * np.outer(y, s)) \
            + rho * np.outer(s, s)
    return x_new.reshape(shape), inv_hessian

def lbfgs(param, grad_fn, s_history, y_history, lr=1.0, history_size=10):
    # Keeps only the last history_size (s, y) pairs. Histories start as [].
    param = np.asarray(param, dtype=float)
    shape = param.shape
    x = param.ravel()
    g = np.ravel(grad_fn(param))
    # Two-loop recursion computes inv_hessian @ g without storing the matrix.
    q = g.copy()
    alphas = []
    for s, y in reversed(list(zip(s_history, y_history))):
        a = (s @ q) / (y @ s)
        alphas.append(a)
        q = q - a * y
    if s_history:
        q = q * (s_history[-1] @ y_history[-1]) / (y_history[-1] @ y_history[-1])
    for (s, y), a in zip(zip(s_history, y_history), reversed(alphas)):
        b = (y @ q) / (y @ s)
        q = q + s * (a - b)
    x_new = x - lr * q
    s = x_new - x
    y = np.ravel(grad_fn(x_new.reshape(shape))) - g
    if s @ y > 1e-12:
        s_history = (s_history + [s])[-history_size:]
        y_history = (y_history + [y])[-history_size:]
    return x_new.reshape(shape), s_history, y_history

def kfac(param, grad, a_factor, g_factor, activations, output_grads, lr=0.01,
         decay=0.95, damping=1e-3):
    # For a dense layer W (out x in): activations are the layer inputs (batch x in),
    # output_grads are dLoss/dOutput (batch x out). The Fisher is approximated as
    # A (x) G, so its inverse is applied as G^-1 @ grad @ A^-1. Factors start as None.
    param = np.asarray(param, dtype=float)
    batch = activations.shape[0]
    a_cov = activations.T @ activations / batch
    g_cov = output_grads.T @ output_grads / batch
    a_factor = a_cov if a_factor is None else decay * a_factor + (1 - decay) * a_cov
    g_factor = g_cov if g_factor is None else decay * g_factor + (1 - decay) * g_cov
    a_inv = np.linalg.inv(a_factor + damping * np.eye(a_factor.shape[0]))
    g_inv = np.linalg.inv(g_factor + damping * np.eye(g_factor.shape[0]))
    param = param - lr * g_inv @ grad @ a_inv
    return param, a_factor, g_factor

def hutchinson_hessian_diag(hvp_fn, param, rng=None):
    # Unbiased estimate of the Hessian diagonal: u * (H u) with random +-1 u.
    # hvp_fn(param, u) must return the Hessian-vector product H @ u.
    rng = np.random.default_rng() if rng is None else rng
    u = rng.choice([-1.0, 1.0], size=np.shape(param))
    return u * hvp_fn(param, u)

def sophia(param, grad, m, h, hessian_estimate=None, lr=1e-4, beta1=0.965, beta2=0.99,
           rho=0.04, weight_decay=0.1, epsilon=1e-12):
    # Pass a fresh hessian_estimate (e.g. hutchinson_hessian_diag) every k steps,
    # and None in between.
    param = np.asarray(param, dtype=float)
    m = beta1 * m + (1 - beta1) * grad
    if hessian_estimate is not None:
        h = beta2 * h + (1 - beta2) * hessian_estimate
    param = param - lr * weight_decay * param
    param = param - lr * np.clip(m / np.maximum(rho * h, epsilon), -1, 1)
    return param, m, h


# ---------------------------------------------------------------------------
# Optimization wrappers / methods
# ---------------------------------------------------------------------------

def sam(param, grad_fn, lr=0.01, rho=0.05, epsilon=1e-12):
    # 1) climb to the worst nearby point, 2) take the gradient there,
    # 3) apply it from the original point (plain SGD as the base step).
    param = np.asarray(param, dtype=float)
    grad = grad_fn(param)
    perturbation = rho * grad / (np.linalg.norm(grad) + epsilon)
    sharp_grad = grad_fn(param + perturbation)
    return param - lr * sharp_grad

def lookahead(fast_param, slow_param, step, k=5, alpha=0.5):
    # Call after the inner optimizer updates fast_param. Every k steps the
    # slow weights move part-way toward the fast ones, and fast resets to slow.
    fast_param = np.asarray(fast_param, dtype=float)
    if step % k == 0:
        slow_param = slow_param + alpha * (fast_param - slow_param)
        fast_param = slow_param.copy()
    return fast_param, slow_param


# ---------------------------------------------------------------------------
# Evolutionary / derivative-free (loss_fn only, no gradients)
# ---------------------------------------------------------------------------

def genetic_algorithm(population, loss_fn, elite_fraction=0.2, mutation_std=0.1, rng=None):
    # population: (size x dims). Keep the best, breed the rest by crossover + mutation.
    rng = np.random.default_rng() if rng is None else rng
    losses = np.array([loss_fn(p) for p in population])
    ranked = population[np.argsort(losses)]
    n_elite = max(2, int(len(population) * elite_fraction))
    elites = ranked[:n_elite]
    parents_a = elites[rng.integers(n_elite, size=len(population) - n_elite)]
    parents_b = elites[rng.integers(n_elite, size=len(population) - n_elite)]
    mask = rng.random(parents_a.shape) < 0.5
    children = np.where(mask, parents_a, parents_b)
    children = children + mutation_std * rng.standard_normal(children.shape)
    return np.vstack([elites, children])

def cma_es_init(mean, sigma=0.5):
    mean = np.asarray(mean, dtype=float)
    n = mean.size
    pop_size = 4 + int(3 * np.log(n))
    mu = pop_size // 2
    weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    weights = weights / weights.sum()
    mu_eff = 1 / np.sum(weights ** 2)
    return {
        "mean": mean, "sigma": sigma, "C": np.eye(n),
        "p_sigma": np.zeros(n), "p_c": np.zeros(n), "t": 0,
        "pop_size": pop_size, "mu": mu, "weights": weights, "mu_eff": mu_eff,
        "c_c": (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n),
        "c_sigma": (mu_eff + 2) / (n + mu_eff + 5),
        "c_1": 2 / ((n + 1.3) ** 2 + mu_eff),
        "c_mu": min(1 - 2 / ((n + 1.3) ** 2 + mu_eff),
                    2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff)),
        "chi_n": np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2)),
    }

def cma_es(state, loss_fn, rng=None):
    # One generation of CMA-ES: sample, rank, then adapt mean, step size and covariance.
    rng = np.random.default_rng() if rng is None else rng
    s = state
    n = s["mean"].size
    s["t"] += 1
    eigvals, B = np.linalg.eigh(s["C"])
    D = np.sqrt(np.maximum(eigvals, 1e-20))
    z = rng.standard_normal((s["pop_size"], n))
    y = z @ (B * D).T
    x = s["mean"] + s["sigma"] * y
    order = np.argsort([loss_fn(xi) for xi in x])
    y_best = y[order[:s["mu"]]]
    y_w = s["weights"] @ y_best
    s["mean"] = s["mean"] + s["sigma"] * y_w

    c_s, c_c, c_1, c_mu, mu_eff = s["c_sigma"], s["c_c"], s["c_1"], s["c_mu"], s["mu_eff"]
    damps = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_s
    C_inv_sqrt = (B / D) @ B.T
    s["p_sigma"] = (1 - c_s) * s["p_sigma"] + np.sqrt(c_s * (2 - c_s) * mu_eff) * C_inv_sqrt @ y_w
    ps_norm = np.linalg.norm(s["p_sigma"])
    h_sigma = ps_norm / np.sqrt(1 - (1 - c_s) ** (2 * s["t"])) / s["chi_n"] < 1.4 + 2 / (n + 1)
    s["p_c"] = (1 - c_c) * s["p_c"] + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * y_w
    rank_mu = (y_best.T * s["weights"]) @ y_best
    s["C"] = ((1 - c_1 - c_mu) * s["C"]
              + c_1 * (np.outer(s["p_c"], s["p_c"]) + (1 - h_sigma) * c_c * (2 - c_c) * s["C"])
              + c_mu * rank_mu)
    s["sigma"] = s["sigma"] * np.exp(c_s / damps * (ps_norm / s["chi_n"] - 1))
    return s

def pso_init(positions, loss_fn):
    positions = np.asarray(positions, dtype=float)
    losses = np.array([loss_fn(p) for p in positions])
    return {
        "positions": positions, "velocities": np.zeros_like(positions),
        "personal_best": positions.copy(), "personal_best_loss": losses,
        "global_best": positions[np.argmin(losses)].copy(),
    }

def pso(state, loss_fn, inertia=0.7, cognitive=1.5, social=1.5, rng=None):
    # Particles are pulled toward their own best point and the swarm's best point.
    rng = np.random.default_rng() if rng is None else rng
    s = state
    r1 = rng.random(s["positions"].shape)
    r2 = rng.random(s["positions"].shape)
    s["velocities"] = (inertia * s["velocities"]
                       + cognitive * r1 * (s["personal_best"] - s["positions"])
                       + social * r2 * (s["global_best"] - s["positions"]))
    s["positions"] = s["positions"] + s["velocities"]
    losses = np.array([loss_fn(p) for p in s["positions"]])
    improved = losses < s["personal_best_loss"]
    s["personal_best"][improved] = s["positions"][improved]
    s["personal_best_loss"][improved] = losses[improved]
    s["global_best"] = s["personal_best"][np.argmin(s["personal_best_loss"])].copy()
    return s

def nelder_mead(simplex, loss_fn, alpha=1.0, gamma=2.0, rho=0.5, sigma=0.5):
    # simplex: (dims + 1) x dims. One reflect / expand / contract / shrink step.
    simplex = np.asarray(simplex, dtype=float)
    losses = np.array([loss_fn(p) for p in simplex])
    order = np.argsort(losses)
    simplex, losses = simplex[order], losses[order]
    centroid = simplex[:-1].mean(axis=0)
    worst = simplex[-1]

    reflected = centroid + alpha * (centroid - worst)
    f_r = loss_fn(reflected)
    if losses[0] <= f_r < losses[-2]:
        simplex[-1] = reflected
    elif f_r < losses[0]:
        expanded = centroid + gamma * (reflected - centroid)
        simplex[-1] = expanded if loss_fn(expanded) < f_r else reflected
    else:
        contracted = centroid + rho * (worst - centroid)
        if loss_fn(contracted) < losses[-1]:
            simplex[-1] = contracted
        else:
            simplex[1:] = simplex[0] + sigma * (simplex[1:] - simplex[0])
    return simplex

def _line_minimize(loss_fn, x, direction, span=10.0, iterations=60):
    # Golden-section search for the best step along direction in [-span, span].
    golden = (np.sqrt(5) - 1) / 2
    lo, hi = -span, span
    for _ in range(iterations):
        a = hi - golden * (hi - lo)
        b = lo + golden * (hi - lo)
        if loss_fn(x + a * direction) < loss_fn(x + b * direction):
            hi = b
        else:
            lo = a
    return x + (lo + hi) / 2 * direction

def powell(param, loss_fn, directions=None):
    # One iteration: line-search along each direction, then swap the direction
    # that gave the biggest drop for the overall displacement.
    x = np.asarray(param, dtype=float).copy()
    if directions is None:
        directions = np.eye(x.size)
    start = x.copy()
    drops = []
    for d in directions:
        before = loss_fn(x)
        x = _line_minimize(loss_fn, x, d)
        drops.append(before - loss_fn(x))
    displacement = x - start
    if np.linalg.norm(displacement) > 1e-12:
        directions = directions.copy()
        directions[int(np.argmax(drops))] = displacement / np.linalg.norm(displacement)
        x = _line_minimize(loss_fn, x, directions[int(np.argmax(drops))])
    return x, directions


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    steps = 200

    # Most optimizers minimise loss(W) = sum((W - T) ** 2) for a 3 x 2 matrix W.
    T = np.array([[3.0, -1.0], [0.5, 2.0], [-2.0, 1.0]])

    def loss_of(W):
        return float(np.sum((W - T) ** 2))

    def grad_of(W):
        return 2 * (W - T)

    def report(name, W):
        print(f"{name:<24}: loss = {loss_of(W):.6f}")

    # --- Gradient descent on a linear-regression dataset ---------------------
    X = rng.standard_normal((200, 3))
    true_w = np.array([1.5, -2.0, 0.5])
    y = X @ true_w
    def data_grad(w, Xb, yb):
        return 2 * Xb.T @ (Xb @ w - yb) / len(Xb)
    for name, fn, kwargs in [("Batch GD", batch_gradient_descent, {}),
                             ("SGD", sgd, {"rng": rng}),
                             ("Mini-batch SGD", minibatch_sgd, {"rng": rng})]:
        w = np.zeros(3)
        for _ in range(steps):
            w = fn(w, data_grad, X, y, lr=0.05, **kwargs)
        print(f"{name:<24}: weight error = {np.linalg.norm(w - true_w):.6f}")

    W = np.zeros_like(T)
    for _ in range(steps):
        W = gradient_descent(W, grad_of(W), lr=0.1)
    report("Gradient Descent", W)

    # --- Momentum -------------------------------------------------------------
    W, vel = np.zeros_like(T), 0.0
    for _ in range(steps):
        W, vel = momentum(W, grad_of(W), vel, lr=0.1)
    report("Momentum", W)

    W, vel = np.zeros_like(T), 0.0
    for _ in range(steps):
        W, vel = nesterov_momentum(W, grad_of, vel, lr=0.05)
    report("Nesterov Momentum", W)

    # --- Adaptive learning rate ----------------------------------------------
    W, acc = np.zeros_like(T), 0.0
    for _ in range(steps):
        W, acc = adagrad(W, grad_of(W), acc, lr=0.5)
    report("AdaGrad", W)

    W, sq, dl = np.zeros_like(T), 0.0, 0.0
    for _ in range(steps * 10):
        W, sq, dl = adadelta(W, grad_of(W), sq, dl)
    report("AdaDelta (2000 steps)", W)

    W, sq = np.zeros_like(T), 0.0
    for _ in range(steps):
        W, sq = rmsprop(W, grad_of(W), sq, lr=0.05)
    report("RMSProp", W)

    # --- Adam family ------------------------------------------------------------
    for name, fn, lr in [("Adam", adam, 0.1), ("AdamW", adamw, 0.1), ("Adamax", adamax, 0.1),
                         ("NAdam", nadam, 0.1), ("RAdam", radam, 0.1),
                         ("AdaBelief", adabelief, 0.1), ("AdaBound", adabound, 0.1),
                         ("LAMB", lamb, 0.05)]:
        W, m, v = np.zeros_like(T), 0.0, 0.0
        if name == "LAMB":
            W = np.full_like(T, 0.1)   # trust ratio needs non-zero weights
        for t in range(1, steps + 1):
            W, m, v = fn(W, grad_of(W), m, v, t, lr=lr * (1 - t / (steps + 1)))
        report(name, W)

    W, m, v, vmax = np.zeros_like(T), 0.0, 0.0, 0.0
    for t in range(1, steps + 1):
        W, m, v, vmax = amsgrad(W, grad_of(W), m, v, vmax, t, lr=0.1)
    report("AMSGrad", W)

    W, m, v, n, prev = np.zeros_like(T), 0.0, 0.0, 0.0, None
    for t in range(1, steps + 1):
        W, m, v, n, prev = adan(W, grad_of(W), m, v, n, prev, t, lr=0.1 * (1 - t / (steps + 1)))
    report("Adan", W)

    # --- Averaged / sign-based -----------------------------------------------
    W, avg = np.zeros_like(T), np.zeros_like(T)
    for t in range(1, steps + 1):
        W, avg = asgd(W, grad_of(W), avg, t, lr=0.1)
    report("ASGD (averaged)", avg)

    W, prev, step = np.zeros_like(T), np.zeros_like(T), np.full_like(T, 0.1)
    for _ in range(steps):
        W, prev, step = rprop(W, grad_of(W), prev, step)
    report("Rprop", W)

    W, m = np.zeros_like(T), 0.0
    for t in range(steps):
        W, m = lion(W, grad_of(W), m, lr=0.05 * (1 - t / steps))
    report("Lion", W)

    # --- Layer-wise ---------------------------------------------------------------
    W, vel = np.full_like(T, 0.1), 0.0
    for _ in range(steps * 5):
        W, vel = lars(W, grad_of(W), vel, lr=1.0, trust_coefficient=0.02)
    report("LARS (1000 steps)", W)

    # --- Memory-efficient ---------------------------------------------------------
    W, row, col = np.zeros_like(T), 0.0, 0.0
    for t in range(1, steps + 1):
        W, row, col = adafactor(W, grad_of(W), row, col, t, lr=0.05)
    report("Adafactor", W)

    W, ms, vs = np.zeros_like(T), None, None
    for t in range(1, steps + 1):
        W, ms, vs = adam_8bit(W, grad_of(W), ms, vs, t, lr=0.1)
    report("8-bit Adam", W)

    W, P, m, v = np.zeros_like(T), None, None, None
    for t in range(1, steps + 1):
        W, P, m, v = galore(W, grad_of(W), P, m, v, t, lr=0.1, rank=2, update_proj_gap=20)
    report("GaLore", W)

    W, P, m, v = np.zeros_like(T), None, None, None
    for t in range(1, steps + 1):
        W, P, m, v = apollo(W, grad_of(W), P, m, v, t, lr=0.1, rank=2,
                            update_proj_gap=20, rng=rng)
    report("APOLLO", W)

    # --- Sparse / online ----------------------------------------------------------
    W, m, v = np.zeros_like(T), 0.0, 0.0
    for t in range(1, steps + 1):
        g = grad_of(W) * (rng.random(T.shape) < 0.5)   # only some entries get gradient
        W, m, v = sparse_adam(W, g, m, v, t, lr=0.1)
    report("SparseAdam", W)

    W, z, n = np.zeros_like(T), 0.0, 0.0
    for _ in range(steps):
        W, z, n = ftrl(W, grad_of(W), z, n, alpha=1.0)
    report("FTRL", W)

    # --- Matrix preconditioning ---------------------------------------------------
    W, L, R = np.zeros_like(T), None, None
    for _ in range(steps):
        W, L, R = shampoo(W, grad_of(W), L, R, lr=0.5)
    report("Shampoo", W)

    W, m, v, L, R, QL, QR = np.zeros_like(T), 0.0, 0.0, None, None, None, None
    for t in range(1, steps + 1):
        W, m, v, L, R, QL, QR = soap(W, grad_of(W), m, v, L, R, QL, QR, t, lr=0.05)
    report("SOAP", W)

    W, buf = np.zeros_like(T), 0.0
    for t in range(steps):
        W, buf = muon(W, grad_of(W), buf, lr=0.1 * (1 - t / steps))
    report("Muon", W)

    # --- Second-order -------------------------------------------------------------
    W = newton_method(np.zeros_like(T), grad_of(np.zeros_like(T)), 2 * np.eye(T.size))
    report("Newton (1 step)", W)

    W, H = np.zeros_like(T), None
    for _ in range(20):
        W, H = bfgs(W, grad_of, H, lr=0.25)
    report("BFGS (20 steps)", W)

    W, sh, yh = np.zeros_like(T), [], []
    for _ in range(20):
        W, sh, yh = lbfgs(W, grad_of, sh, yh, lr=0.25)
    report("L-BFGS (20 steps)", W)

    # K-FAC on a real dense layer: learn W so that inputs @ W.T matches targets.
    inputs = rng.standard_normal((64, 2))
    targets = inputs @ T.T
    W, A, G = np.zeros_like(T), None, None
    for _ in range(50):
        layer_grad = 2 * (inputs @ W.T - targets).T @ inputs / len(inputs)
        # K-FAC needs output gradients sampled from the model's own distribution
        # (true Fisher), not the training-loss gradients (empirical Fisher).
        sampled_out_grads = np.sqrt(2) * rng.standard_normal((len(inputs), T.shape[0]))
        W, A, G = kfac(W, layer_grad, A, G, inputs, sampled_out_grads, lr=0.5)
    print(f"{'K-FAC (50 steps)':<24}: weight error = {np.linalg.norm(W - T):.6f}")

    W, m, h = np.zeros_like(T), 0.0, 0.0
    hvp = lambda _param, u: 2 * u   # Hessian of the loss is 2I
    for t in range(steps):
        est = hutchinson_hessian_diag(hvp, W, rng) if t % 10 == 0 else None
        W, m, h = sophia(W, grad_of(W), m, h, est, lr=0.1 * (1 - t / steps),
                         weight_decay=0.0)
    report("Sophia", W)

    # --- Wrappers -----------------------------------------------------------------
    W = np.zeros_like(T)
    for _ in range(steps):
        W = sam(W, grad_of, lr=0.1)
    report("SAM", W)

    fast, slow = np.zeros_like(T), np.zeros_like(T)
    for t in range(1, steps + 1):
        fast = gradient_descent(fast, grad_of(fast), lr=0.1)
        fast, slow = lookahead(fast, slow, t)
    report("Lookahead (+GD)", slow)

    # --- Derivative-free (on the flattened 6-dim problem) -------------------------
    flat_loss = lambda x: loss_of(x.reshape(T.shape))

    pop = rng.standard_normal((30, T.size))
    for _ in range(steps):
        pop = genetic_algorithm(pop, flat_loss, mutation_std=0.05, rng=rng)
    report("Genetic Algorithm", min(pop, key=flat_loss).reshape(T.shape))

    state = cma_es_init(np.zeros(T.size))
    for _ in range(steps):
        state = cma_es(state, flat_loss, rng=rng)
    report("CMA-ES", state["mean"].reshape(T.shape))

    state = pso_init(rng.standard_normal((30, T.size)) * 3, flat_loss)
    for _ in range(steps):
        state = pso(state, flat_loss, rng=rng)
    report("PSO", state["global_best"].reshape(T.shape))

    simplex = np.vstack([np.zeros(T.size), np.eye(T.size)])
    for _ in range(steps * 3):
        simplex = nelder_mead(simplex, flat_loss)
    report("Nelder-Mead (600 steps)", min(simplex, key=flat_loss).reshape(T.shape))

    x, dirs = np.zeros(T.size), None
    for _ in range(3):
        x, dirs = powell(x, flat_loss, dirs)
    report("Powell (3 iterations)", x.reshape(T.shape))
