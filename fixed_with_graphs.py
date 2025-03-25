import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Импорт для 3D-графиков
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


# === Функция для отрисовки траектории градиентного спуска (2D)

def plot_trajectory(func: BiFunc, trajectory: list) -> None:
    xs = [point[0] for point in trajectory]
    ys = [point[1] for point in trajectory]

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    x_range = x_max - x_min
    y_range = y_max - y_min
    pad_x = x_range * 0.1 if x_range != 0 else 1
    pad_y = y_range * 0.1 if y_range != 0 else 1
    x_min, x_max = x_min - pad_x, x_max + pad_x
    y_min, y_max = y_min - pad_y, y_max + pad_y

    x_grid = np.linspace(x_min, x_max, 100)
    y_grid = np.linspace(y_min, y_max, 100)
    X, Y = np.meshgrid(x_grid, y_grid)
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = func(np.array([X[i, j], Y[i, j]]))

    plt.figure()
    cp = plt.contour(X, Y, Z, levels=30)
    plt.clabel(cp, inline=True, fontsize=8)
    plt.plot(xs, ys, marker='o', label='Траектория')
    plt.plot(xs[0], ys[0], marker='s', markersize=8, label='Начало')
    plt.plot(xs[-1], ys[-1], marker='*', markersize=12, label='Конец')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Траектория градиентного спуска')
    plt.legend()
    plt.show()


# === Функция для 3D‑визуализации функции BiFunc

def plot_3d_func(func: BiFunc,
                 x_range: Tuple[float, float],
                 y_range: Tuple[float, float],
                 steps: int = 50,
                 trajectory: list = None) -> None:
    """
    Строит 3D-поверхность функции на заданном диапазоне.
    Если передан список trajectory (координаты точек градиентного спуска),
    то накладывает на поверхность траекторию.
    """
    x_min, x_max = x_range
    y_min, y_max = y_range
    x_grid = np.linspace(x_min, x_max, steps)
    y_grid = np.linspace(y_min, y_max, steps)
    X, Y = np.meshgrid(x_grid, y_grid)
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = func(np.array([X[i, j], Y[i, j]]))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # Строим поверхность
    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)

    # Если передана траектория, добавляем её на график
    if trajectory is not None:
        trajectory = np.array(trajectory)
        xs = trajectory[:, 0]
        ys = trajectory[:, 1]
        zs = np.array([func(np.array([x, y])) for x, y in zip(xs, ys)])
        ax.plot(xs, ys, zs, marker='o', color='r', label='Траектория')
        ax.scatter(xs[0], ys[0], zs[0], color='g', marker='s', s=50, label='Начало')
        ax.scatter(xs[-1], ys[-1], zs[-1], color='b', marker='*', s=70, label='Конец')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('f(x,y)')
    ax.set_title('3D график BiFunc')
    ax.legend()
    plt.show()


# === Gradient descent

def gradient_descent(
        func: BiFunc,
        start: Vector,
        learning: Learning = lambda _: 1,
        search: Callable[[BiFunc, Vector, Vector], float] = lambda *_: 1,
        limit: float = 1e3,
        eps: float = 1e-6,
        draw: bool = False,  # Рисование 2D-траектории
        draw3d: bool = False  # Рисование 3D-графика функции с траекторией
) -> Tuple[Vector, int]:
    x = start.copy()
    trajectory = [x.copy()] if (draw or draw3d) else None
    k = 0
    while True:
        gradient = func.gradient(x)
        u = -gradient
        alpha = search(func, x, u)
        x += alpha * learning(k) * u
        if trajectory is not None:
            trajectory.append(x.copy())
        if np.linalg.norm(gradient) ** 2 < eps or k > limit:
            break
        k += 1
    if draw and trajectory is not None:
        plot_trajectory(func, trajectory)
    if draw3d and trajectory is not None:
        # Определяем диапазон для 3D-графика на основе траектории
        xs = [pt[0] for pt in trajectory]
        ys = [pt[1] for pt in trajectory]
        pad = 5
        x_range = (min(xs) - pad, max(xs) + pad)
        y_range = (min(ys) - pad, max(ys) + pad)
        plot_3d_func(func, x_range, y_range, steps=50, trajectory=trajectory)
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
    # Пример функции: окружность
    circle = BiFunc(lambda x, y: x ** 2 + y ** 2)
    start = np.array([-42.0, 31.0])

    print("Constant", gradient_descent(circle, start, learning=constant(0.05),draw3d=True))
    print("Exponential decay", gradient_descent(circle, start, learning=exponential_decay(0.01),draw3d=True))
    print("Polynomial decay", gradient_descent(circle, start, learning=polynomial_decay(0.5, 1),draw3d=True))
    print("Armijo rule", gradient_descent(circle, start, search=armijo_rule,draw3d=True))
    print("Wolfe rule", gradient_descent(circle, start, search=wolfe_rule,draw3d=True))

    # Пример запуска градиентного спуска с отрисовкой 2D-траектории
    print("Градиентный спуск с отрисовкой 2D-траектории:")
    result, count = gradient_descent(circle, start, learning=constant(0.05), draw=True)
    print("Результат:", result, "Вызовов функции:", count)

    # Пример запуска градиентного спуска с отрисовкой 3D-графика (с возможностью вращения)
    print("Градиентный спуск с отрисовкой 3D-графика:")
    result, count = gradient_descent(circle, start, learning=constant(0.05), draw3d=True)
    print("Результат:", result, "Вызовов функции:", count)
