import magnus
from tqdm import tqdm
import numpy as np

X = np.array([[0, 1], [1, 0]])
Z = np.array([[1, 0], [0, -1]])

def A(t):
    return t * X + (1.0 - t) * Z

def omega2_bruteforce(A, t0, tf, sample_len, dtype=np.float64):
    """
    Compute Omega_2 = 1/2 ∫_{t0}^{tf} dt1 ∫_{t0}^{t1} dt2 [A(t1), A(t2)]
    using a left-Riemann discretization.

    A: callable t -> (d, d) ndarray
    """
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

n_samples = 4096
t_samples = np.linspace(0, T, n_samples)
a_samples = np.array([A(t) for t in t_samples])

print( magnus.magnus_f64(n=2, data=a_samples, t0=0, tf=T) )
print( omega2_bruteforce(A, 0, T, n_samples) )