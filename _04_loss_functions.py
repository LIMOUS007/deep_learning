import numpy as np

def _log_softmax(logits):
    logits = np.asarray(logits, dtype=float)
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    return shifted - np.log(
        np.sum(np.exp(shifted), axis=-1, keepdims=True)
    )

def mse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return np.mean((y_true - y_pred) ** 2)

def mae(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return np.mean(np.abs(y_true - y_pred))

def huber(y_true, y_pred, delta=1.0):
    error = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    abs_error = np.abs(error)
    return np.mean(np.where(
        abs_error <= delta,
        0.5 * error ** 2,
        delta * (abs_error - 0.5 * delta)
    ))

def binary_cross_entropy(y_true, y_pred):
    eps = 1e-15
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.clip(np.asarray(y_pred, dtype=float), eps, 1 - eps)
    return -np.mean(
        y_true * np.log(y_pred)
        + (1 - y_true) * np.log(1 - y_pred)
    )

def binary_cross_entropy_with_logits(y_true, logits):
    y_true = np.asarray(y_true, dtype=float)
    z = np.asarray(logits, dtype=float)
    return np.mean(
        np.maximum(z, 0) - z * y_true + np.log1p(np.exp(-np.abs(z)))
    )

def categorical_cross_entropy(y_true, y_pred):
    """
    y_true : one-hot targets, shape (..., C)
    y_pred : predicted probabilities, same shape
    """
    eps = 1e-15
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.clip(np.asarray(y_pred, dtype=float), eps, 1.0)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=-1))

def sparse_categorical_cross_entropy(y_true, y_pred):
    """
    y_true : integer class indices, shape (N,)
    y_pred : predicted probabilities, shape (N, C)
    """
    eps = 1e-15
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.clip(np.asarray(y_pred, dtype=float), eps, 1.0)
    picked = y_pred[np.arange(len(y_true)), y_true]
    return -np.mean(np.log(picked))

def cross_entropy_with_logits(y_true, logits):
    """
    logits : raw scores, shape (N, C)
    y_true : one-hot (N, C) or integer class indices (N,)
    """
    logits = np.asarray(logits, dtype=float)
    y_true = np.asarray(y_true)
    log_p = _log_softmax(logits)

    if y_true.shape == logits.shape:
        return -np.mean(np.sum(y_true.astype(float) * log_p, axis=-1))

    picked = log_p[np.arange(len(y_true)), y_true.astype(int)]
    return -np.mean(picked)

def focal_loss(y_true, y_pred, gamma=2.0, alpha=0.25):
    eps = 1e-15
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.clip(np.asarray(y_pred, dtype=float), eps, 1 - eps)

    p_t = np.where(y_true == 1, y_pred, 1 - y_pred)
    alpha_t = np.where(y_true == 1, alpha, 1 - alpha)
    return -np.mean(alpha_t * (1 - p_t) ** gamma * np.log(p_t))

def kl_divergence(p, q):
    eps = 1e-15
    p = np.clip(np.asarray(p, dtype=float), eps, 1.0)
    q = np.clip(np.asarray(q, dtype=float), eps, 1.0)
    return np.mean(np.sum(p * np.log(p / q), axis=-1))

def contrastive_loss(y_true, distance, margin=1.0):
    """
    y_true   : 1 for a similar pair, 0 for a dissimilar one
    distance : embedding distance for each pair
    """
    y_true = np.asarray(y_true, dtype=float)
    d = np.asarray(distance, dtype=float)
    return 0.5 * np.mean(
        y_true * d ** 2
        + (1 - y_true) * np.maximum(0.0, margin - d) ** 2
    )

def triplet_loss(anchor, positive, negative, margin=1.0):
    anchor = np.asarray(anchor, dtype=float)
    positive = np.asarray(positive, dtype=float)
    negative = np.asarray(negative, dtype=float)

    d_pos = np.sum((anchor - positive) ** 2, axis=-1)
    d_neg = np.sum((anchor - negative) ** 2, axis=-1)
    return np.mean(np.maximum(0.0, d_pos - d_neg + margin))

def dice_loss(y_true, y_pred, smooth=1.0):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    intersection = np.sum(y_true * y_pred, axis=-1)
    total = np.sum(y_true, axis=-1) + np.sum(y_pred, axis=-1)
    return np.mean(1.0 - (2.0 * intersection + smooth) / (total + smooth))

def tversky_loss(y_true, y_pred, alpha=0.3, beta=0.7, smooth=1.0):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    tp = np.sum(y_true * y_pred, axis=-1)
    fp = np.sum((1 - y_true) * y_pred, axis=-1)
    fn = np.sum(y_true * (1 - y_pred), axis=-1)
    return np.mean(
        1.0 - (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
    )

def ctc_loss(log_probs, targets, blank=0):
    """
    log_probs : (T, C) log-probabilities, one row per timestep
    targets   : class indices of the label sequence, blanks excluded
    blank     : index of the blank symbol
    """
    log_probs = np.asarray(log_probs, dtype=float)
    targets = np.asarray(targets, dtype=int)
    T = log_probs.shape[0]

    extended = np.full(2 * len(targets) + 1, blank, dtype=int)
    extended[1::2] = targets
    S = len(extended)

    repeats = int(np.sum(targets[1:] == targets[:-1])) if len(targets) > 1 else 0
    if T < len(targets) + repeats:
        raise ValueError(
            f"{T} timesteps cannot cover a target needing at least "
            f"{len(targets) + repeats}"
        )

    alpha = np.full(S, -np.inf)
    alpha[0] = log_probs[0, extended[0]]
    if S > 1:
        alpha[1] = log_probs[0, extended[1]]

    for t in range(1, T):
        prev = alpha
        alpha = np.full(S, -np.inf)

        for s in range(S):
            total = prev[s]
            if s >= 1:
                total = np.logaddexp(total, prev[s - 1])

            if s >= 2 and extended[s] != blank and extended[s] != extended[s - 2]:
                total = np.logaddexp(total, prev[s - 2])

            alpha[s] = total + log_probs[t, extended[s]]

    if S > 1:
        return -np.logaddexp(alpha[-1], alpha[-2])
    return -alpha[-1]

def vae_loss(x, x_reconstructed, mu, logvar):
    """
    x, x_reconstructed : (N, D), reconstruction as Bernoulli probabilities
    mu, logvar         : (N, D_latent), the encoder's Gaussian parameters
    """
    eps = 1e-15
    x = np.asarray(x, dtype=float)
    x_reconstructed = np.clip(
        np.asarray(x_reconstructed, dtype=float), eps, 1 - eps
    )
    mu = np.asarray(mu, dtype=float)
    logvar = np.asarray(logvar, dtype=float)

    reconstruction = -np.sum(
        x * np.log(x_reconstructed)
        + (1 - x) * np.log(1 - x_reconstructed),
        axis=-1
    )
    kl = -0.5 * np.sum(1 + logvar - mu ** 2 - np.exp(logvar), axis=-1)
    return np.mean(reconstruction + kl)

if __name__ == "__main__":
    y_true_reg = np.array([3.0, -0.5, 2.0, 7.0])
    y_pred_reg = np.array([2.5, 0.0, 2.0, 8.0])

    y_true_bin = np.array([1., 0., 1., 1., 0.])
    y_prob_bin = np.array([0.9, 0.1, 0.8, 0.6, 0.3])
    logits_bin = np.log(y_prob_bin / (1 - y_prob_bin))

    y_idx = np.array([0, 2, 1])
    y_onehot = np.eye(4)[y_idx]
    probs = np.array([
        [0.7, 0.1, 0.1, 0.1],
        [0.1, 0.2, 0.6, 0.1],
        [0.2, 0.5, 0.2, 0.1]
    ])
    logits_multi = np.log(probs)

    p = np.array([[0.1, 0.4, 0.5]])
    q = np.array([[0.2, 0.3, 0.5]])

    pair_labels = np.array([1., 0., 1.])
    pair_dist = np.array([0.2, 0.9, 1.4])
    anchor = np.array([[0.0, 0.0], [1.0, 1.0]])
    positive = np.array([[0.1, 0.1], [1.2, 0.9]])
    negative = np.array([[0.3, 0.2], [0.9, 1.3]])

    mask_true = np.array([[1., 1., 0., 0., 1.]])
    mask_pred = np.array([[0.9, 0.6, 0.1, 0.2, 0.7]])

    rng = np.random.default_rng(0)
    ctc_log_probs = _log_softmax(rng.normal(size=(6, 3)))
    ctc_targets = np.array([1, 2])

    x = np.array([[1., 0., 1., 1.]])
    x_hat = np.array([[0.8, 0.2, 0.7, 0.9]])
    mu = np.array([[0.1, -0.2]])
    logvar = np.array([[-0.5, 0.3]])

    print("MSE                   :", mse(y_true_reg, y_pred_reg))
    print("MAE                   :", mae(y_true_reg, y_pred_reg))
    print("Huber                 :", huber(y_true_reg, y_pred_reg))
    print("BCE                   :", binary_cross_entropy(y_true_bin, y_prob_bin))
    print("BCE with Logits       :", binary_cross_entropy_with_logits(y_true_bin, logits_bin))
    print("Categorical CE        :", categorical_cross_entropy(y_onehot, probs))
    print("Sparse Categorical CE :", sparse_categorical_cross_entropy(y_idx, probs))
    print("CE with Logits        :", cross_entropy_with_logits(y_idx, logits_multi))
    print("Focal                 :", focal_loss(y_true_bin, y_prob_bin))
    print("KL Divergence         :", kl_divergence(p, q))
    print("Contrastive           :", contrastive_loss(pair_labels, pair_dist))
    print("Triplet               :", triplet_loss(anchor, positive, negative))
    print("Dice                  :", dice_loss(mask_true, mask_pred))
    print("Tversky               :", tversky_loss(mask_true, mask_pred))
    print("CTC                   :", ctc_loss(ctc_log_probs, ctc_targets))
    print("VAE / ELBO            :", vae_loss(x, x_hat, mu, logvar))
