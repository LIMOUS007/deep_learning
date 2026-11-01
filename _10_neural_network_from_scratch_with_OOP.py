import numpy as np
import matplotlib.pyplot as plt

from _04_loss_functions import mse
from _05_activation_functions import leaky_relu
from _08_optimizers import (
    adagrad,
    adam,
    adamw,
    gradient_descent,
    momentum,
    rmsprop,
)

def linear_forward(x, w, b):
    return x @ w.T + b


def linear_backward(dz, x, w):
    dx = dz @ w
    dw = dz.T @ x
    db = np.sum(dz, axis=0)
    return dx, dw, db


def leaky_relu_grad(x, alpha=0.01):
    return np.where(x > 0, 1.0, alpha)


def mse_backward(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_true.size


def initialize_parameters(n_input, n_hidden1, n_hidden2, n_output, seed=42, scale=0.1):
    np.random.seed(seed)
    w1 = np.random.randn(n_hidden1, n_input) * scale
    b1 = np.zeros(n_hidden1)
    w2 = np.random.randn(n_hidden2, n_hidden1) * scale
    b2 = np.zeros(n_hidden2)
    w3 = np.random.randn(n_output, n_hidden2) * scale
    b3 = np.zeros(n_output)
    return [w1, b1, w2, b2, w3, b3]

def init_optimizer_state(params, optimizer):
    keys = OPTIMIZER_STATE_KEYS[optimizer]
    return [{k: np.zeros_like(p) for k in keys} for p in params]

class NeuralNetwork:
    def __init__(self, n_input, n_hidden1, n_hidden2, n_output, optimizer="adam", lr=None, seed = 42):
        self.params = initialize_parameters(n_input, n_hidden1, n_hidden2, n_output, seed = seed)
        self.optimizer = optimizer
        if lr is None:
            lr = DEFAULT_LR[optimizer]
        self.lr = lr
        self.state = init_optimizer_state(self.params, optimizer)
        self.cache = None
        self.t = 0
        
    def forward(self, x):
        w1, b1, w2, b2, w3, b3 = self.params
        z1 = linear_forward(x, w1, b1)
        a1 = leaky_relu(z1)
        z2 = linear_forward(a1, w2, b2)
        a2 = leaky_relu(z2)
        z3 = linear_forward(a2, w3, b3)
        y_pred = z3[:, 0]
        self.cache = {"x": x, "z1": z1, "a1": a1, "z2": z2, "a2": a2}
        return y_pred

    def predict(self, x):
        return self.forward(x)
    
    def count_parameters(self):
        return sum(p.size for p in self.params)
    
    def backward(self, y_true, y_pred):
        w1, _, w2, _, w3, _ = self.params
        dy_pred = mse_backward(y_true, y_pred)
        dz3 = dy_pred[:, None]
        da2, dw3, db3 = linear_backward(dz3, self.cache["a2"], w3)
        dz2 = da2 * leaky_relu_grad(self.cache["z2"])
        da1, dw2, db2 = linear_backward(dz2, self.cache["a1"], w2)
        dz1 = da1 * leaky_relu_grad(self.cache["z1"])
        _, dw1, db1 = linear_backward(dz1, self.cache["x"], w1)
        return [dw1, db1, dw2, db2, dw3, db3]
    
    def update_parameters(self, grads):
        self.t += 1
        for i, (p, g, s) in enumerate(zip(self.params, grads, self.state)):
            if self.optimizer == "sgd":
                self.params[i] = gradient_descent(p, g, self.lr)
            elif self.optimizer == "momentum":
                self.params[i], s["velocity"] = momentum(p, g, s["velocity"], self.lr)
            elif self.optimizer == "adagrad":
                self.params[i], s["accumulated_grad"] = adagrad(p, g, s["accumulated_grad"], self.lr)
            elif self.optimizer == "rmsprop":
                self.params[i], s["squared_avg"] = rmsprop(p, g, s["squared_avg"], self.lr)
            elif self.optimizer == "adam":
                self.params[i], s["m"], s["v"] = adam(p, g, s["m"], s["v"], self.t, self.lr)
            elif self.optimizer == "adamw":
                self.params[i], s["m"], s["v"] = adamw(p, g, s["m"], s["v"], self.t, self.lr)
            else:
                raise ValueError(f"unknown optimizer: {self.optimizer}")
    
    def train(self, x, y, x_val, y_val, epochs=2000, batch_size=None, seed=0):
        if batch_size is None:
            batch_size = len(x)
        rng = np.random.default_rng(seed)
        history = {"train": [], "val": []}
        for _ in range(epochs):
            order = rng.permutation(len(x))
            epoch_loss = 0.0
            for start in range(0, len(x), batch_size):
                batch = order[start:start + batch_size]
                x_batch = x[batch]
                y_batch = y[batch]
                y_pred = self.forward(x_batch)
                epoch_loss += mse(y_batch, y_pred) * len(batch)
                grads = self.backward(y_batch, y_pred)
                self.update_parameters(grads)
            history["train"].append(epoch_loss / len(x))
            history["val"].append(mse(y_val, self.predict(x_val)))
        return history
    
    def gradient_check(self, x, y_true, epsilon=1e-4):
        y_pred = self.forward(x)
        grads = self.backward(y_true, y_pred)
        errors = []
        for p, g in zip(self.params, grads):
            numerical = np.zeros_like(p)
            for i in range(p.size):
                original = p.flat[i]
                p.flat[i] = original + epsilon
                loss_plus = mse(y_true, self.predict(x))
                p.flat[i] = original - epsilon
                loss_minus = mse(y_true, self.predict(x))
                p.flat[i] = original
                numerical.flat[i] = (loss_plus - loss_minus) / (2 * epsilon)
            scale = np.maximum(np.abs(g) + np.abs(numerical), 1e-12)
            errors.append(np.max(np.abs(g - numerical) / scale))
        
        return errors

    
    
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
    4., 7., 10., 13., 16.,
    19., 22., 25., 28., 31.
])

N_TRAIN = 8

x_train = x[:N_TRAIN]
y_train = y[:N_TRAIN]
x_val = x[N_TRAIN:]
y_val = y[N_TRAIN:]

PARAM_NAMES = ["w1", "b1", "w2", "b2", "w3", "b3"]

OPTIMIZER_STATE_KEYS = {
    "sgd": [],
    "momentum": ["velocity"],
    "adagrad": ["accumulated_grad"],
    "rmsprop": ["squared_avg"],
    "adam": ["m", "v"],
    "adamw": ["m", "v"],
}

DEFAULT_LR = {
    "sgd": 0.05,
    "momentum": 0.2,
    "adagrad": 0.5,
    "rmsprop": 0.001,
    "adam": 0.1,
    "adamw": 0.05,
}


def plot_history(history, path="loss_history.png", floor=1e-16, show=False):
    if not show:
        plt.switch_backend("Agg")
    train = np.maximum(history["train"], floor)
    val = np.maximum(history["val"], floor)
    plt.figure(figsize=(8, 5))
    plt.plot(train, label="Train")
    plt.plot(val, label="Validation")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel(f"MSE loss (clipped at {floor:g})")
    plt.title("Training and validation loss")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    if show:
        plt.show()
    plt.close()
    return path


if __name__ == "__main__":
    print("NETWORK  2 -> 4 -> 4 -> 1")
    model = NeuralNetwork(2, 4, 4, 1, optimizer="adam")
    print(f"  {model.count_parameters()} parameters " f"({', '.join( f'{n}={p.size}' for n, p in zip(PARAM_NAMES, model.params))})")
    print()
    print("GRADIENT CHECK")
    for name, err in zip(PARAM_NAMES, model.gradient_check(x_train, y_train)):
        status = "PASS" if err < 1e-4 else "FAIL"
        print(f"  d{name:3s}: " f"max relative error {err:.2e}  " f"{status}")
    print()
    print("TRAINING")
    history = model.train(x_train, y_train, x_val, y_val, batch_size=4)
    print()
    print("RESULT")
    print("  train pred :", np.round(model.predict(x_train), 3))
    print("  train true :",y_train)
    print("  val pred   :", np.round(model.predict(x_val), 3))
    print("  val true   :", y_val)
    print()
    print("  loss curve saved to", plot_history(history))