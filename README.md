# Efficient Numerical Magnus Integration

This python software package solves linear first-order differential equations using the **Magnus Expansion**:

$$\frac{dY}{dt} = A(t) Y(t), \quad Y(t_0) = Y_0$$

The solution is expressed as $Y(t) = \exp(\Omega(t, t_0)) Y_0$, where $\Omega(t, t_0) = \sum_{k=1}^{\infty} \Omega_k(t, t_0)$ is the Magnus expansion.

This library is designed for numerical integration of these terms, supporting multiple computational backends, integrators, and reverse-mode automatic differentiation via Vector-Jacobian Products (VJP) in both raw Python/NumPy and JAX.

---

## Requirements & Building

### Requirements
- **CMake** (version 3.16+)
- **C++ Compiler** supporting C++23 (e.g., GCC 11+, Clang 13+)
- **BLAS** & **TBB** libraries (for multi-threaded matrix operations)
- **Python 3** with development headers
- **NumPy**
- **JAX** (Optional, for JAX CPU FFI functionality)

### Installation
Run the build script to compile the C++ core and construct the local Python package:
```bash
chmod +x build.sh
./build.sh
```
This builds the package and generates typing stubs inside the `magnus` directory.

---

## Python / NumPy API Reference

### Core Computation Functions

#### `compute`
Computes the Magnus expansion terms for matrix-valued functions.
```python
def compute(
    n: int,
    f: Callable[[np.ndarray], np.ndarray],
    t0: float,
    tf: float,
    samples: int,
    *,
    op: Literal["one", "many", "sum"] = "sum",
    dtype: Any = None,
    vectorized: bool = True,
    matrix_backend: str = "Auto",
    integrator: str = "Auto",
    record_vjp: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
```
- **`n`**: Maximum Magnus order to compute.
- **`f`**: Callable sampling the matrix-valued function $A(t)$. If `vectorized=True`, it must accept a 1D time-array of shape `(S,)` and return a `(S, d, d)` array. If `vectorized=False`, it is called sequentially for each scalar time.
- **`t0`, `tf`**: Time interval bounds.
- **`samples`**: Number of time samples. The interval count (`samples - 1`) must satisfy the divisibility constraints of the integrator.
- **`op`**:
  - `"one"`: Returns only the $n$-th order Magnus term $\Omega_n(t_f, t_0)$ with shape `(d, d)`.
  - `"many"`: Returns all terms individually $\Omega_1, \dots, \Omega_n$ with shape `(n, d, d)`.
  - `"sum"`: Returns the cumulative sum $\sum_{i=1}^n \Omega_i(t_f, t_0)$ with shape `(d, d)`.
- **`matrix_backend`**: `"Auto"`, `"Fixed2"`, `"Manual"`, `"Blas"`, or `"SpaceCurve"`.
- **`integrator`**: `"Auto"`, `"Boole"`, `"Simpson"`, `"Trapezoid"`, or `"Riemann"`.
- **`record_vjp`**: If `True`, returns `(output, vjp_data)`. `vjp_data` can be supplied to `compute_vjp` to speed up backpropagation.

#### `compute_sc`
Computes the Magnus expansion for 3-vector functions (representing rotation generator trajectories in $\mathfrak{su}(2)$ / SpaceCurve).
```python
def compute_sc(
    n: int,
    f: Callable[[np.ndarray], np.ndarray],
    t0: float,
    tf: float,
    samples: int,
    *,
    op: Literal["one", "many", "sum"] = "sum",
    dtype: Any = None,
    vectorized: bool = True,
    integrator: str = "Auto",
    record_vjp: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
```
*Signature maps to `compute`, but the input function `f` must return `(S, 3)` arrays and output is either `(3,)` or `(n, 3)`.*

---

### VJP / Gradient Functions

#### `compute_vjp`
Computes the Vector-Jacobian Product with respect to the sampled matrix input data.
```python
def compute_vjp(
    n: int,
    f: Callable[[np.ndarray], np.ndarray],
    cotangent: np.ndarray,
    t0: float,
    tf: float,
    samples: int,
    *,
    op: Literal["one", "many", "sum"] = "sum",
    dtype: Any = None,
    vectorized: bool = True,
    matrix_backend: str = "Auto",
    integrator: str = "Auto",
    vjp_data: np.ndarray | None = None,
) -> np.ndarray:
```
- **`cotangent`**: Cotangent matching the shape of the forward `compute` output.
- **`vjp_data`**: Recorded forward carry data from `compute(..., record_vjp=True)`. If omitted, the forward sweep is re-run internally.

#### `compute_sc_vjp`
Computes the VJP with respect to the sampled SpaceCurve input data.
```python
def compute_sc_vjp(
    n: int,
    f: Callable[[np.ndarray], np.ndarray],
    cotangent: np.ndarray,
    t0: float,
    tf: float,
    samples: int,
    *,
    op: Literal["one", "many", "sum"] = "sum",
    dtype: Any = None,
    vectorized: bool = True,
    integrator: str = "Auto",
    vjp_data: np.ndarray | None = None,
) -> np.ndarray:
```

---

## JAX API Reference

The `magnus.jax` module mirrors the core library but is compatible with JAX's autograd engine.

```python
import magnus.jax as magnus_jax
```

### JAX Compute Functions

- **`magnus_jax.compute(n, f, t0, tf, samples, ...)`**
- **`magnus_jax.compute_sc(n, f, t0, tf, samples, ...)`**

*These functions register custom VJPs with JAX automatically. You can compute gradients, VJPs, and JIT-compile code involving Magnus integrations.*
---

## Code Examples

### 1. NumPy Forward & VJP Integration
```python
import numpy as np
import magnus

# Define a time-dependent matrix function A(t)
def matrix_func(t):
    # Accepts shape (S,) and returns (S, 2, 2)
    t = np.asarray(t)
    out = np.zeros((t.size, 2, 2))
    out[:, 0, 0] = np.sin(t)
    out[:, 0, 1] = t
    out[:, 1, 0] = -t
    out[:, 1, 1] = np.cos(t)
    return out

# Integrate over [0.0, 1.0] with 9 samples using Boole's rule
n_order = 4
t0, tf = 0.0, 1.0
samples = 9  # (9-1)=8 is divisible by 4 (Boole's requirement)

# Compute with forward carry logging enabled
out, vjp_data = magnus.compute(
    n_order,
    matrix_func,
    t0,
    tf,
    samples,
    op="sum",
    integrator="Boole",
    record_vjp=True
)
print("Magnus Sum:\n", out)
```

### 2. JAX JIT and Gradient Optimization
```python
import jax
import jax.numpy as jnp
import magnus.jax as magnus_jax

# Enable 64-bit precision for JAX
jax.config.update("jax_enable_x64", True)

# Define vector representation of a trajectory
def loss_fn(params):
    # params: parameters of the trajectory
    t0, tf = 0.0, 1.0
    samples = 17
    
    # 3-vector function mapping t -> R^3
    def trajectory(t):
        return jnp.stack([
            params[0] * jnp.sin(t),
            params[1] * jnp.cos(t),
            params[2] * t**2
        ], axis=-1)

    # Compute SpaceCurve Magnus integration
    out = magnus_jax.compute_sc(
        n=4,
        f=trajectory,
        t0=t0,
        tf=tf,
        samples=samples,
        integrator="Boole"
    )
    # Return L2 norm of the resulting Lie algebra sum
    return jnp.sum(out ** 2)

# Compute gradients and JIT compile
params = jnp.array([1.2, -0.5, 0.8])
grad_fn = jax.jit(jax.grad(loss_fn))

gradient = grad_fn(params)
print("Loss gradient w.r.t parameters:", gradient)
```
