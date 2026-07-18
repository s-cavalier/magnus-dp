#ifndef __GAUSS_LEGENDRE_HPP__
#define __GAUSS_LEGENDRE_HPP__
#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>
#include <memory>
#include <atomic>
#include <concepts>
#include <algorithm>
#include <functional>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace Magnus {

    /*
    An abstract handle onto some GaussLegendre table.
    We use a global PIMPL approach so that way the details of "where" the data is stored is abstracted away.
    It could be a statically linked GL table, mmapped table, heap table, or even just a file.

    Assumes doubles. Maybe extend later for other purposes.
    */
    class GLTable {
        static std::atomic<std::shared_ptr<const GLTable>> global_table;

    protected:
        size_t m_max_order;
        GLTable(size_t max_order);

    public:
        class DataView {
            const double* m_w;
            const double* m_x;
            size_t n;

        public:
            DataView(const double* weights, const double* nodes, size_t order);

            size_t order() const;
            std::pair<double, double> operator[](size_t i) const;
            std::pair<double, double> at(size_t i) const;

            const double* weights() const;
            const double* nodes() const;
        };

        virtual ~GLTable();
        virtual DataView get_order(size_t n) const = 0;

        size_t max_order() const;

        static std::shared_ptr<const GLTable> get();
        static void update(std::shared_ptr<const GLTable> new_table);

    };

    class StaticTable : public GLTable {
        struct PyGLTableHeader {
            char magic[4];
            uint32_t max_n;

            size_t offsets_offset;
            size_t nodes_offset;
            size_t weights_offset;
        };

        size_t* offsets;
        double* weights;
        double* nodes;

    public:
        StaticTable(std::byte* table);

        DataView get_order(size_t n) const override;

    };

    class OwningTable final : public GLTable {
        std::vector<size_t> offsets;
        std::vector<double> weights;
        std::vector<double> nodes;

    public:
        OwningTable(
            size_t max_order,
            std::vector<size_t> offsets,
            std::vector<double> weights,
            std::vector<double> nodes
        );

        DataView get_order(size_t n) const override;
    };

    struct GL_forloop {
        static constexpr size_t lane_count([[maybe_unused]]size_t) { return 1; }

        static void invoke(size_t order, auto&& fn) {
            for (size_t q = 0; q < order; ++q) std::invoke(fn, q, 0);
        }

    };

    struct GL_openmp {
        static size_t lane_count(size_t order) {
#ifdef _OPENMP
            return std::min(order, size_t(omp_get_max_threads()));
#else
            return std::min(order, size_t{1});
#endif
        }

        static void invoke(size_t order, auto&& fn) {
#ifdef _OPENMP
            int lanes = static_cast<int>(lane_count(order));
            #pragma omp parallel for schedule(static) num_threads(lanes)
            for (size_t q = 0; q < order; ++q) {
                std::invoke(fn, q, omp_get_thread_num());
            }
#else
            for (size_t q = 0; q < order; ++q) std::invoke(fn, q, 0);
#endif
        }

    };

}


#endif
