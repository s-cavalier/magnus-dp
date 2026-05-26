#include "gaussLegendre.hpp"
#include "integrate.hpp"
#include "magnus.hpp"
#include "matrix.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

extern "C" std::byte _binary_gl_nodes_bin_start[];

namespace {

using NumT = double;
using MatrixPolicyT = Magnus::FixedDimPolicy<NumT, 2>;
using IntegratorT = Magnus::BooleIntegrator<NumT, MatrixPolicyT>;
using Clock = std::chrono::steady_clock;

struct Options {
    std::string mode = "sum";
    size_t order = 16;
    size_t samples = 4097;
    size_t loops = 1000;
    size_t warmup = 10;
    double tf = 1.0;
};

void usage(const char* program) {
    std::cerr
        << "Usage: " << program << " [sum|one|many|all] [options]\n"
        << "Options:\n"
        << "  --order N      Magnus order, default 16\n"
        << "  --samples N    Number of sampled matrices, default 4097\n"
        << "  --loops N      Timed iterations, default 1000\n"
        << "  --warmup N     Warmup iterations, default 10\n"
        << "  --tf T         Final time, default 1.0\n";
}

size_t parse_size(const char* text, const char* name) {
    char* end = nullptr;
    unsigned long long value = std::strtoull(text, &end, 10);
    if (*text == '\0' || *end != '\0') {
        throw std::runtime_error(std::string("invalid ") + name + ": " + text);
    }
    return static_cast<size_t>(value);
}

double parse_double(const char* text, const char* name) {
    char* end = nullptr;
    double value = std::strtod(text, &end);
    if (*text == '\0' || *end != '\0') {
        throw std::runtime_error(std::string("invalid ") + name + ": " + text);
    }
    return value;
}

Options parse_options(int argc, char** argv) {
    Options opts;

    int i = 1;
    if (i < argc && argv[i][0] != '-') {
        opts.mode = argv[i++];
    }

    while (i < argc) {
        std::string_view arg = argv[i++];
        auto require_value = [&](const char* name) -> const char* {
            if (i >= argc) throw std::runtime_error(std::string("missing value for ") + name);
            return argv[i++];
        };

        if (arg == "--order") {
            opts.order = parse_size(require_value("--order"), "--order");
        } else if (arg == "--samples") {
            opts.samples = parse_size(require_value("--samples"), "--samples");
        } else if (arg == "--loops") {
            opts.loops = parse_size(require_value("--loops"), "--loops");
        } else if (arg == "--warmup") {
            opts.warmup = parse_size(require_value("--warmup"), "--warmup");
        } else if (arg == "--tf") {
            opts.tf = parse_double(require_value("--tf"), "--tf");
        } else if (arg == "--help" || arg == "-h") {
            usage(argv[0]);
            std::exit(0);
        } else {
            throw std::runtime_error("unknown option: " + std::string(arg));
        }
    }

    if (opts.mode != "sum" && opts.mode != "one" && opts.mode != "many" && opts.mode != "all") {
        throw std::runtime_error("mode must be one of: sum, one, many, all");
    }
    if (opts.order == 0) throw std::runtime_error("--order must be at least 1");
    if (opts.samples < 5) throw std::runtime_error("--samples must be at least 5 for Boole integration");
    if ((opts.samples - 1) % 4 != 0) {
        throw std::runtime_error("--samples must satisfy (samples - 1) % 4 == 0 for Boole integration");
    }

    return opts;
}

std::vector<NumT> make_samples(size_t samples, double tf) {
    std::vector<NumT> data(samples * 4);

    for (size_t i = 0; i < samples; ++i) {
        double t = tf * static_cast<double>(i) / static_cast<double>(samples - 1);
        double z = 1.0 - t;

        data[4 * i + 0] = z;
        data[4 * i + 1] = t;
        data[4 * i + 2] = t;
        data[4 * i + 3] = -z;
    }

    return data;
}

double checksum(const std::vector<NumT>& data) {
    double total = 0.0;
    for (double value : data) total += value;
    return total;
}

void do_not_optimize(double value) {
    asm volatile("" : : "g"(value) : "memory");
}

template <class Fn>
void run_bench(std::string_view name, size_t warmup, size_t loops, Fn&& fn) {
    double guard = 0.0;

    for (size_t i = 0; i < warmup; ++i) {
        guard += fn();
    }

    auto start = Clock::now();
    for (size_t i = 0; i < loops; ++i) {
        guard += fn();
    }
    auto stop = Clock::now();

    do_not_optimize(guard);

    double seconds = std::chrono::duration<double>(stop - start).count();
    double per_loop = seconds / static_cast<double>(std::max<size_t>(loops, 1));

    std::cout
        << name
        << " loops=" << loops
        << " seconds=" << seconds
        << " per_loop=" << per_loop
        << " checksum=" << guard
        << '\n';
}

void bench_one(const Options& opts, std::vector<NumT>& samples) {
    std::vector<NumT> out(4);
    IntegratorT::matrix_span_t input(samples.data(), 2, opts.samples);
    IntegratorT::matrix_t output(out.data(), 2);

    run_bench("one", opts.warmup, opts.loops, [&] {
        Magnus::one<IntegratorT>(output, opts.order, input, 0.0, opts.tf);
        return checksum(out);
    });
}

void bench_sum(const Options& opts, std::vector<NumT>& samples) {
    std::vector<NumT> out(4);
    IntegratorT::matrix_span_t input(samples.data(), 2, opts.samples);
    IntegratorT::matrix_t output(out.data(), 2);

    run_bench("sum", opts.warmup, opts.loops, [&] {
        Magnus::sum<IntegratorT>(output, opts.order, input, 0.0, opts.tf);
        return checksum(out);
    });
}

void bench_many(const Options& opts, std::vector<NumT>& samples) {
    std::vector<NumT> out(opts.order * 4);
    IntegratorT::matrix_span_t input(samples.data(), 2, opts.samples);
    IntegratorT::matrix_span_t output(out.data(), 2, opts.order);

    run_bench("many", opts.warmup, opts.loops, [&] {
        Magnus::many<IntegratorT>(output, input, 0.0, opts.tf);
        return checksum(out);
    });
}

} // namespace

int main(int argc, char** argv) {
    try {
        Magnus::GLTable::update(std::make_shared<Magnus::StaticTable>(_binary_gl_nodes_bin_start));

        Options opts = parse_options(argc, argv);
        std::vector<NumT> samples = make_samples(opts.samples, opts.tf);

        if (opts.mode == "one" || opts.mode == "all") bench_one(opts, samples);
        if (opts.mode == "sum" || opts.mode == "all") bench_sum(opts, samples);
        if (opts.mode == "many" || opts.mode == "all") bench_many(opts, samples);

        return 0;
    } catch (const std::exception& err) {
        std::cerr << "error: " << err.what() << '\n';
        usage(argv[0]);
        return 1;
    }
}
