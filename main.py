import numpy as np
from scipy.optimize._linesearch import line_search_wolfe1, scalar_search_armijo
from prettytable import PrettyTable
from typing import Callable, Tuple
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# === Constants

Vector = np.ndarray
Scheduling = Callable[[int], float]

SYSTEM_EPS = np.sqrt(np.finfo(float).eps)

# === Function wrapper


class Func:
    Type = Callable[..., float]
    func: Type
    count: int

    def __init__(self, func: Type):
        self.func = func
        self.count = 0

    def __call__(self, x: Vector) -> float:
        return self.func(*x)

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


def plot_trajectory(func: Callable[[np.ndarray], float], trajectory) -> None:
    trajectory = np.array(trajectory)
    xs = trajectory.flatten()
    fs = np.array([func(np.array([x])) for x in xs])

    plt.figure(figsize=(10, 6))
    x_min, x_max = xs.min(), xs.max()
    pad = max(1.0, (x_max - x_min) * 0.2)

    x_grid = np.linspace(x_min - pad, x_max + pad, 200)
    f_grid = np.array([func(np.array([x])) for x in x_grid])

    plt.plot(x_grid, f_grid, "b-", label="Функция")
    plt.plot(xs, fs, "ro-", markersize=4, linewidth=1.5, label="Траектория")
    plt.scatter(xs[0], fs[0], c="green", marker="s", s=100, label="Начало")
    plt.scatter(xs[-1], fs[-1], c="blue", marker="*", s=150, label="Конец")

    plt.xlabel("x", fontsize=12)
    plt.ylabel("f(x)", fontsize=12)
    plt.title("Траектория градиентного спуска (1D)", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("gd_1d.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_3d_func(
    func: Func,
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    steps: int = 50,
    trajectory: list = None,
) -> None:
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
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z, cmap="viridis", alpha=0.7)

    if trajectory is not None:
        trajectory = np.array(trajectory)
        xs = trajectory[:, 0]
        ys = trajectory[:, 1]
        zs = np.array([func(np.array([x, y])) for x, y in zip(xs, ys)])

        ax.plot(xs, ys, zs, marker="o", color="r", label="Траектория")
        ax.scatter(xs[0], ys[0], zs[0], color="g", marker="s", s=50, label="Начало")
        ax.scatter(xs[-1], ys[-1], zs[-1], color="b", marker="*", s=70, label="Конец")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("f(x,y)")
    ax.set_title("3D график Func")

    ax.legend()
    plt.savefig("gd.png", dpi=150, bbox_inches="tight")


def try_draw_gradient(
    func: Callable[[np.ndarray], float], draw2d: bool, draw3d: bool, trajectory
) -> None:
    if trajectory is None:
        return

    if draw2d:
        plot_trajectory(func, trajectory)

    if draw3d:
        xs = [pt[0] for pt in trajectory]
        ys = [pt[1] for pt in trajectory]
        pad = max(1.0, max(np.ptp(xs), np.ptp(ys))) * 0.2
        x_range = (min(xs) - pad, max(xs) + pad)
        y_range = (min(ys) - pad, max(ys) + pad)
        plot_3d_func(func, x_range, y_range, steps=50, trajectory=trajectory)


# === Gradient descent


def isScheduling(algorithm):
    return hasattr(algorithm, "__code__") and algorithm.__code__.co_argcount == 1


def gradient_descent(
    func: Func,
    start: Vector,
    learning,
    limit: float = 1e3,
    eps: float = 1e-6,
    on_error=0.1,
    draw2d: bool = False,
    draw3d: bool = False,
) -> Tuple[Vector, int]:
    x = start.copy()
    trajectory = [x.copy()] if (draw3d or draw2d) else None
    k = 0

    while True:
        gradient = func.gradient(x)
        u = -gradient
        alpha = learning(k) if isScheduling(learning) else learning(func, x, u)
        alpha = on_error if alpha is None else alpha  # scipy algorithms can give Nones

        x += alpha * u
        if trajectory is not None:
            trajectory.append(x.copy())

        if np.linalg.norm(gradient) ** 2 < eps or k > limit:
            break
        k += 1

    try_draw_gradient(func, draw2d, draw3d, trajectory)
    return x, func.count


# === Learnings


def h(k: int) -> float:
    return 1 / (k + 1) ** 0.5


def constant(λ: float) -> Scheduling:
    return lambda k: λ


def geometric() -> Scheduling:
    return lambda k: h(k) / 2**k


def exponential_decay(λ: float) -> Scheduling:
    return lambda k: h(k) * np.exp(-λ * k)


def polynomial_decay(α: float, β: float) -> Scheduling:
    return lambda k: h(k) * (β * k + 1) ** -α


# === Rules


def armijo_rule(func: Func, x: Vector, direction: Vector) -> float:
    α: float = 0.5
    q: float = 0.5
    c: float = 0.4
    max_iter: int = 80
    for i in range(max_iter):
        if func(x + α * direction) <= func(x) + c * α * np.linalg.norm(direction):
            return α
        α *= q
    return None

def wolfe_rule(func: Func, x: Vector, direction: Vector) -> float:
    α: float = 0.5
    c1: float = 1e-4
    c2: float = 0.3
    max_iter: int = 80
    for _ in range(max_iter):
        if func(x + α * direction) > func(x) + c1 * α * np.dot(
            func.gradient(x), direction
        ):
            α *= 0.5
        elif np.dot(func.gradient(x + α * direction), direction) < c2 * np.dot(
            func.gradient(x), direction
        ):
            α *= 1.5
        else:
            return α
    return None


# === Scipy


def scipy_wolfe(func: Func, x: Vector, direction: Vector) -> float:
    return line_search_wolfe1(func, func.gradient, x, direction)[0]


def scipy_armijo(func: Func, x: Vector, direction: Vector) -> float:
    return scalar_search_armijo(
        phi=lambda a: func(x + a * direction),
        phi0=func(x),
        derphi0=np.dot(func.gradient(x), direction),
        c1=0.4,
        alpha0=0.5,
    )[0]


# === Data container


class AlgoData:
    def __init__(self, name, algo):
        self.name = name
        self.algorithm = algo

    def get_data(self, f, start):
        gd = gradient_descent(f, start, self.algorithm)
        return [self.name] + list(gd[0]) + [gd[1]]


# === Output


def print_algorithms(algos, f, start):
    table = PrettyTable()
    table.field_names = (
        ["Method"] + ["Coordinate " + str(i + 1) for i in range(len(start))] + ["Steps"]
    )
    table.add_rows(
        sorted(
            [alg.get_data(f, start) for alg in algos],
            key=lambda x: (
                f(x[1:-1]),
                x[-1],
            ),  # sort by efficiency of minimizing, then by amount of steps
        )
    )
    print(table)


# === Launcher

if __name__ == "__main__":
    testing_func = Func(lambda x, y: x**2 + y**2)
    start_point = np.array([420.0, 100])

    testing_algorithms = [
        AlgoData("Constant", constant(0.05)),
        AlgoData("Exponential Decay", exponential_decay(0.01)),
        AlgoData("Polynomial Decay", polynomial_decay(0.5, 1)),
        AlgoData("Armijo Rule", armijo_rule),
        AlgoData("Wolfe Rule", wolfe_rule),
        AlgoData("SciPy Armijo", scipy_armijo),
        AlgoData("SciPy Wolfe", scipy_wolfe),
    ]
    print_algorithms(testing_algorithms, testing_func, start_point)

    # draw example
    #gradient_descent(testing_func, start_point, wolfe_rule, draw2d = False, draw3d=True)
