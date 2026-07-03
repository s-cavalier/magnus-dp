#include "backendcomposer.hpp"

#include <cstddef>
#include <memory>
#include <mutex>
#include <stdexcept>

extern "C" std::byte _binary_gl_nodes_bin_start[];

static std::once_flag default_gl_table_once;

void Magnus::initialize_default_gl_table() {
    std::call_once(default_gl_table_once, [] {
        try {
            (void)GLTable::get();
        } catch (const std::runtime_error&) {
            GLTable::update(std::make_shared<StaticTable>(_binary_gl_nodes_bin_start));
        }
    });
}

std::unique_ptr<Magnus::KernelPlan> Magnus::make_plan(Params& p, size_t num_idx, size_t mat_idx, size_t int_idx) {
    initialize_default_gl_table();

    if ( p.n > 1 ) {
        size_t required_order = (p.n + 3) / 2;
        if ( required_order > GLTable::get()->max_order() ) throw std::invalid_argument("requested magnus order exceeds GL table. Consider providing a larger table.");
    }
    
    return NumBackends::dispatch( num_idx, p, [&p, &mat_idx, &int_idx]<Dispatchable NumSpec>() -> std::unique_ptr<KernelPlan> {
        using NumT = typename NumSpec::type;

        return MatrixBackends::dispatch( mat_idx, p, [&p, &int_idx]<Dispatchable MatSpec> () -> std::unique_ptr<KernelPlan> {
            using MatPolicy = typename MatSpec::template type<NumT>;
            static_assert(MatrixPolicy<MatPolicy>);

            return IntegratorBackends::dispatch( int_idx, p, [&p]<Dispatchable IntSpec> () -> std::unique_ptr<KernelPlan> {
                using Int = typename IntSpec::template type<NumT, MatPolicy>;
                static_assert( Integrator<Int> );

                if ( (p.samples - 1) % Int::divisibility_requirement() != 0 ) throw std::invalid_argument("sample interval count violates integrator divisibility (a power of 2 is recommneded; i.e., total samples is 2^k + 1 for some k > 4)");

                return std::make_unique< TypedKernelPlan<Int> >( std::move(p) );
            } );

        });

    });

}
