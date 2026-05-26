import magnus
from tqdm import tqdm
import numpy as np
from time import perf_counter
from scipy.linalg import expm
from scipy.integrate import solve_ivp

X = np.array([[0, 1], [1, 0]])
Z = np.array([[1, 0], [0, -1]])

def Y(t):
    return t * X + (1.0 - t) * Z


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


def median_runtime(fn, repeats=7):
    fn()
    times = []
    for _ in range(repeats):
        t_start = perf_counter()
        result = fn()
        times.append(perf_counter() - t_start)
    return float(np.median(times)), result


def omega2_bruteforce(A, t0, tf, sample_len, dtype=np.float64):
    dt = (tf - t0) / sample_len
    ts = t0 + dt * np.arange(sample_len)

    mats = [np.asarray(A(t), dtype=dtype) for t in ts]
    d = mats[0].shape[0]

    omega = np.zeros((d, d), dtype=dtype)

    for i in tqdm(range(sample_len)):
        A1 = mats[i]
        for j in range(i):
            A2 = mats[j]
            omega += A1 @ A2 - A2 @ A1

    return 0.5 * (dt ** 2) * omega

T = 1

n_samples = 257
t_samples = np.linspace(0, T, n_samples)
a_samples = np.array([Y(t) for t in t_samples])

ivp_time, ivp_res = median_runtime(
    lambda: solve_matrix_ivp_at(Y, T, rtol=1e-12, atol=1e-14)
)

print(f"Samples: {n_samples}")
print(f"SciPy solve_ivp runtime: {ivp_time:.6f}s")
print("order | magnus_sum | expm      | total     | speedup vs ivp | max abs err")

for n_order in [8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]:
    magnus_time, omega = median_runtime(
        lambda: magnus.sum(n=n_order, data=a_samples, t0=0, tf=T)
    )

    expm_time, exp_res = median_runtime(lambda: expm(omega))

    total_time = magnus_time + expm_time
    diff = exp_res - ivp_res
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
