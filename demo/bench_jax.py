import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_SKIP_CUDA_CONSTRAINTS_CHECK", "1")

from time import perf_counter

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax.scipy.linalg import expm
import magnus.jax as magnus_jax
import numpy as np
from scipy.integrate import solve_ivp


Z = np.array([[1, 0], [0, -1]])
X = np.array([[0, 1], [1, 0]])


def Y(t):
    return t * X + (1 - t) * Z


def solve_matrix_ivp_at(coef, t, *, rtol=1e-10, atol=1e-12):
    y0 = np.asarray(coef(0.0), dtype=np.float64)
    dim = y0.shape[0]

    if t == 0:
        return np.eye(dim, dtype=np.float64)

    def rhs(t_curr, flat_state):
        state = flat_state.reshape(dim, dim)
        return (np.asarray(coef(t_curr), dtype=np.float64) @ state).reshape(-1)

    sol = solve_ivp(
        rhs,
        (0.0, t),
        np.eye(dim, dtype=np.float64).reshape(-1),
        t_eval=[t],
        rtol=rtol,
        atol=atol,
    )

    if not sol.success:
        raise RuntimeError(sol.message)

    return sol.y[:, -1].reshape(dim, dim)


def block_until_ready(result):
    return jax.block_until_ready(result)


def median_runtime(fn, repeats=7):
    block_until_ready(fn())
    times = []
    for _ in range(repeats):
        t_start = perf_counter()
        result = fn()
        block_until_ready(result)
        times.append(perf_counter() - t_start)
    return float(np.median(times)), result


T = 1.0
n_samples = 101
t_samples = np.linspace(0, T, n_samples)
a_samples_np = np.array([Y(t) for t in t_samples], dtype=np.float64)
a_samples = jnp.asarray(a_samples_np)

ivp_time, ivp_res = median_runtime(
    lambda: solve_matrix_ivp_at(Y, T, rtol=1e-12, atol=1e-14)
)

jax_expm = jax.jit(expm)

print(f"Samples: {n_samples}")
print(f"SciPy solve_ivp runtime: {ivp_time:.6f}s")
print("order | jax_magnus | jax_expm  | total     | speedup vs ivp | max abs err")

for n_order in [6, 8, 10, 12, 14, 16, 18, 20, 22]:
    jax_magnus_sum = jax.jit(
        lambda data, n_order=n_order: magnus_jax.sum_s(
            n_order,
            data,
            0.0,
            T,
            matrix_backend="Fixed2",
            integrator="Boole",
        )
    )

    magnus_time, omega = median_runtime(lambda: jax_magnus_sum(a_samples))
    expm_time, exp_res = median_runtime(lambda: jax_expm(omega))

    exp_res_np = np.asarray(exp_res)
    total_time = magnus_time + expm_time
    diff = exp_res_np - ivp_res
    max_err = np.max(np.abs(diff))
    speedup = ivp_time / total_time

    print(
        f"{n_order:5d} | "
        f"{magnus_time:10.6f}s | "
        f"{expm_time:9.6f}s | "
        f"{total_time:9.6f}s | "
        f"{speedup:12.3f}x | "
        f"{max_err:.6e}"
    )
