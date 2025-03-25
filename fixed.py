import numpy as np
from typing import Callable, Tuple

# === Constants

Vector = np.ndarray
Learning = Callable[[int], float]

SYSTEM_EPS = np.sqrt(np.finfo(float).eps)

# === Function wrapper

class BiFunc:
    Type = Callable[[float, float], float]
    func: Type
    count: int

    def __init__(self, func: Type):
        self.func = func
        self.count = 0

    def __call__(self, x: Vector) -> float:
        return self.func(x[0], x[1])

    def gradient(self, x: Vector, ε: float = SYSTEM_EPS) -> Vector:
        self.count += 1
        gradient = np.zeros_like(x)
        size = len(x)
        for i in range(size):
            dx = np.zeros(size)
            h = max(ε, ε * abs(x[i]))
            dx[i] = h
            gradient[i] = (self(x + dx) - self(x - dx)) / (2 * h)
        return gradient

# === Gradient descent

def gradient_descent(
    func: BiFunc,
    start: Vector,
    learning: Learning = lambda _: 1,
    search: Callable[[BiFunc, Vector, Vector], float] = lambda *_: 1,
    limit: float = 1e3,
    eps: float = 1e-6
) -> Tuple[Vector, int]:
    x = start.copy()
    k = 0
    while True:
        gradient = func.gradient(x)
        u = -gradient
        alpha = search(func, x, u)
        x += alpha * learning(k) * u
        if np.linalg.norm(gradient) ** 2 < eps or k > limit:
            break
        k += 1
    return x, func.count

# === Learnings

def h(k: int) -> float:
    return 1 / (k + 1) ** 0.5

def constant(λ: float) -> Learning:
    return lambda k: λ

def geometric() -> Learning:
    return lambda k: h(k) / 2 ** k

def exponential_decay(λ: float) -> Learning:
    return lambda k: h(k) * np.exp(-λ * k)

def polynomial_decay(α: float, β: float) -> Learning:
    return lambda k: h(k) * (β * k + 1) ** -α

# === Rules

def armijo_rule(
    func: BiFunc,
    x: Vector,
    direction: Vector
) -> float:
    α: float = 0.5
    q: float = 0.5
    c: float = 0.5
    while True:
        if func(x + α * direction) <= func(x) + c * α * np.linalg.norm(direction):
            return α
        α *= q

def wolfe_rule(
    func: BiFunc,
    x: Vector,
    direction: Vector
) -> float:
    α: float = 1.0
    c1: float = 1e-4
    c2: float = 0.9
    max_iter: int = 100
    for _ in range(max_iter):
        if func(x + α * direction) > func(x) + c1 * α * np.dot(func.gradient(x), direction):
            α *= 0.5
        elif np.dot(func.gradient(x + α * direction), direction) < c2 * np.dot(func.gradient(x), direction):
            α *= 1.5
        else:
            return α
    return α

# === Launcher

if __name__ == "__main__":
    circle = BiFunc(lambda x, y: x ** 2 + y ** 2)
    start = np.array([-42.0, 31.0])

    print("Constant", gradient_descent(circle, start, learning=constant(0.05)))
    print("Exponential decay", gradient_descent(circle, start, learning=exponential_decay(0.01)))
    print("Polynomial decay", gradient_descent(circle, start, learning=polynomial_decay(0.5, 1)))

    print("Armijo rule", gradient_descent(circle, start, search=armijo_rule))
    print("Wolfe rule", gradient_descent(circle, start, search=wolfe_rule))

# Constant (array([-0.00035008,  0.00025839]), 111)
# Exponential decay (array([ 7.76383337e-05, -5.73045017e-05]), 120)
# Polynomial decay (array([0., 0.]), 123)
# Armijo rule (array([0., 0.]), 125)
# Wolfe rule (array([0., 0.]), 134)
