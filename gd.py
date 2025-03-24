import numpy as np
import math

# ---differentation and gradient---

def get_dx(dim, index, eps=1e-8):
    dx = np.zeros(dim)
    dx[index] = eps
    return dx
    
def approximate_deriv(f, x, dx, eps=1e-8):
    return (f(x + dx) - f(x)) / eps

def symmetric_deriv(f, x, dx, eps=1e-8):
    return (f(x + dx) - f(x - dx)) / (2 * eps)

def approximate_gradient(f, x, eps=1e-8, dif = symmetric_deriv):
    grad = np.zeros_like(x)
    for i in range(len(x)):
        
        dx = get_dx(len(x), i, eps)
        grad[i] = dif(f, x, dx, eps=eps)
    return grad

# ---gradient descent algorithm---

def identity_one(*args):
    return 1

def gradient_descent(f, start, lrate = identity_one, search = identity_one, iter_limit=1000, e=1e-6):
    x = start.copy()
    k = 0

    while True:
        k += 1
        grad = approximate_gradient(f, x)
        u = -grad #direction is antigradient

        alpha = search(f, x, u)  
        x += alpha * lrate(k) * u 

        if np.linalg.norm(grad)**2 < e or k > iter_limit:
            break
    return x

# ---strategies of choosing learning rate---

# -scheduling-

def fixed_step(alpha):
    return lambda k: alpha

def h0(k):
    return 1 / math.sqrt(k + 1)

def exponential_decay(lamb):
    return lambda k: h0(k) * np.exp(-lamb* k)

def polynomial_decay(alpha, beta):
    return lambda k: h0(k) * (beta * k + 1)**(-alpha)

# -one-dimensional searches-

def armijo_rule(f, x, direction, alpha=0.5, q=0.5, c=0.5):
    while True:
        if f(x + alpha * direction) <= f(x) + c * alpha * np.linalg.norm(direction):
            return alpha
        else:
            alpha *= q

# ---testing, examples---

def f(x):
    return x[0]**2 + x[1]**2

start = np.array([-42.0, 31.0])

result_fixed = gradient_descent(
    f, start,
    lrate=fixed_step(0.05),
)

result_exp_decay = gradient_descent(
    f, start,
    lrate=exponential_decay(0.01),  
)

result_pol_decay = gradient_descent(
    f, start,
    lrate=polynomial_decay(0.5, 1),
)

result_armijo = gradient_descent(
    f, start,
    search=lambda f, x, direction: armijo_rule(f, x, direction)
)

print("fixed step:", result_fixed)
print("exponential decay:", result_exp_decay)
print("polynomial decay:", result_pol_decay)
print("Armijo rule:", result_armijo)