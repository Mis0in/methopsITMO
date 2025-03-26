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


# === 3D-visualization of BiFunc with path trajectory if given

def plot_3d_func(
    func: BiFunc,
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
    ax.set_title("3D график BiFunc")

    ax.legend()
    
    plt.savefig("gd.png")
    #plt.show()

def try_draw_gradient(func, draw3d, trajectory):
    #if draw and trajectory is not None:
        #plot_trajectory(func, trajectory)

    if draw3d and trajectory is not None:
        xs = [pt[0] for pt in trajectory]
        ys = [pt[1] for pt in trajectory]
        pad = 5
        x_range = (min(xs) - pad, max(xs) + pad)
        y_range = (min(ys) - pad, max(ys) + pad)
        plot_3d_func(func, x_range, y_range, steps=50, trajectory=trajectory) 
          
# === Gradient descent

def isScheduling(algorithm):
    return hasattr(algorithm, "__code__") and algorithm.__code__.co_argcount == 1

def gradient_descent(
    func: BiFunc,
    start: Vector,
    learning,
    limit: float = 1e3,
    eps: float = 1e-6,
    on_error = 0.1,
    draw3d: bool = False, 
) -> Tuple[Vector, int]:
    x = start.copy()
    trajectory = [x.copy()] if draw3d else None
    k = 0

    while True:
        gradient = func.gradient(x)
        u = -gradient
        alpha = learning(k) if isScheduling(learning) else learning(func, x, u)
        alpha = on_error if alpha is None else alpha #scipy algorithms can give Nones
        
        x += alpha * u
        if trajectory is not None:
            trajectory.append(x.copy())

        if np.linalg.norm(gradient) ** 2 < eps or k > limit:
            break
        k += 1
    try_draw_gradient(func, draw3d, trajectory)
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

def armijo_rule(func: BiFunc, x: Vector, direction: Vector) -> float:
    α: float = 0.5
    q: float = 0.5
    c: float = 0.4
    while True:
        if func(x + α * direction) <= func(x) + c * α * np.linalg.norm(direction):
            return α
        α *= q

def wolfe_rule(func: BiFunc, x: Vector, direction: Vector) -> float:
    α: float = 0.5
    c1: float = 1e-4
    c2: float = 0.3
    max_iter: int = 100
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
    return α


# === Scipy

def scipy_wolfe(func: BiFunc, x: Vector, direction: Vector) -> float:
    return line_search_wolfe1(func, func.gradient, x, direction)[0]

def scipy_armijo(func: BiFunc, x: Vector, direction: Vector) -> float:
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
    table.field_names = ["Method", "Coordinate X", "Coordinate Y", "Steps"]
    table.add_rows(
        sorted(
        [alg.get_data(f, start) for alg in algos],
        key=lambda x: (f([x[1],x[2]]), x[-1]) #sort by efficiency of minimizing, then by amount of steps
        ))
    print(table)


# === Launcher

if __name__ == "__main__":
    testing_func = BiFunc(lambda x, y: 2*x**2 + 3*y**2 + np.arctan(x))
    start_point = np.array([-42.0, 31.0])

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

    #draw example
    #gradient_descent(testing_func, start_point, armijo_rule, draw3d=True)
