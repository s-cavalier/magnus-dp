#include "matrix.hpp"
#include <iostream>
#include <cmath>
#include <ranges>
#include <random>
#include <algorithm>
#include <span>

std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<float> dist(-10, 10);

struct RandomGenerator {
    float operator()() const {
        return dist(gen);
    }
};

int main() {
    Magnus::DynMatrix<float> x(2);
    x.fill(RandomGenerator{});

    Magnus::DynMatrix<float> y(2);
    y.fill(RandomGenerator{});
    
    for (auto row : x) {
        for ( auto f : row ) std::cout << f << ' ';
        std::cout << '\n';
    }
    std::cout << '\n';

    for (auto row : y) {
        for ( auto f : row ) std::cout << f << ' ';
        std::cout << '\n';
    }
    std::cout << '\n';

    auto c = Magnus::matmul(x, y);
    
    for (auto row : c) {
        for ( auto f : row ) std::cout << f << ' ';
        std::cout << '\n';
    }
    std::cout << '\n';

    return 0;
}

//def magnus(self, n: int, samples = 4096):
    //if n < 1:
        //raise valueerror("n must be at least 1")
    //if samples < 2:
        //raise valueerror("samples must be at least 2")
    //if n == 1:
        //return self.position(self.tf) - self.position(self.t0)

    //time = jnp.linspace(self.t0, self.tf, samples)
    //t = jax.vmap(self.velocity)(time)
    //dt = (self.tf - self.t0) / (samples - 1)

    //complex_dtype = jnp.result_type(t.dtype, jnp.complex64)
    //t = t.astype(complex_dtype)

    //x_order = (n + 1) // 2
    //x_nodes_np, x_weights_np = _gauss_legendre_01(x_order)
    //x_nodes = jnp.asarray(x_nodes_np, dtype=t.real.dtype)
    //x_weights = jnp.asarray(x_weights_np, dtype=t.real.dtype)
    //shift = x_nodes - 1.0

    //n_x = x_nodes.shape[0]
    //y_a = jnp.zeros((samples, n_x), dtype=complex_dtype)
    //y_b = jnp.broadcast_to(t[:, none, :], (samples, n_x, 3))

    //for _ in range(1, n):
        //prefix_a = _cumulative_trapezoid(y_a, dt)
        //prefix_b = _cumulative_trapezoid(y_b, dt)

        //total_a = prefix_a[-1]
        //total_b = prefix_b[-1]

        //s_a = prefix_a + shift[none, :] * total_a[none, :]
        //s_b = prefix_b + shift[none, :, none] * total_b[none, :, :]

        //y_a = jnp.einsum("td,txd->tx", t, s_b)
        //y_b = s_a[:, :, none] * t[:, none, :] + 1j * jnp.cross(t[:, none, :], s_b, axis=-1)

    //final_b = _cumulative_trapezoid(y_b, dt)[-1]
    //value = jnp.sum(x_weights[:, none] * final_b, axis=0)

    //return jnp.real((1.0j ** (1 - n)) * value)
