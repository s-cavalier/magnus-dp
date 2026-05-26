#ifndef __INTEGRATE_HPP__
#define __INTEGRATE_HPP__
#include <span>
#include <utility>
#include <memory>
#include <atomic>
#include <concepts>
#include <algorithm>
#include "matrix.hpp"

/*
An abstract handle onto some GaussLegendre table.
We use a global PIMPL approach so that way the details of "where" the data is stored is abstracted away.
It could be a statically linked GL table, mmapped table, heap table, or even just a file.

Assumes doubles. Maybe extend later for other purposes.
*/
class GLTable {
    inline static std::atomic<std::shared_ptr<const GLTable>> global_table{nullptr};

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

    virtual ~GLTable() = default;
    virtual DataView get_order(size_t n) const = 0;

    size_t max_order() const;

    static std::shared_ptr<const GLTable> get() {
        auto ptr = global_table.load(std::memory_order_acquire);

        if (!ptr) throw std::runtime_error("Global Gauss-Legendre table is null");

        return ptr;
    }

    static void update(std::shared_ptr<const GLTable> new_table) {
        if (!new_table) throw std::invalid_argument("Tried to initialize table with nullptr");
        global_table.store( std::move(new_table), std::memory_order_release );
    }

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
    StaticTable(std::byte* table) : GLTable(0) {
        PyGLTableHeader* header = (PyGLTableHeader*)table;

        if ( header->magic[0] != 'L' || header->magic[1] != 'G' || header->magic[2] != '0' || header->magic[3] != '1' )
            throw std::runtime_error("Invalid magic number; StaticTable(std::byte*)");

        this->m_max_order = header->max_n;

        offsets = (size_t*)(table + header->offsets_offset);
        weights = (double*)(table + header->weights_offset);
        nodes = (double*)(table + header->nodes_offset);
    }

    DataView get_order(size_t n) const override {
        size_t offset = offsets[n];

        return DataView{
            weights + offset,
            nodes + offset,
            n
        };
    }

};



template <class T>
concept Integrator = requires(
    T& integrator, 
    Magnus::MatrixSpan<typename T::numeric_t> data, 
    Magnus::MatrixView<typename T::numeric_t>& out, 
    const typename T::allocator_t& alloc
) {
    typename T::allocator_t;
    typename T::numeric_t;
    { T( size_t{}, alloc ) };
    { integrator.prefix( data, typename T::numeric_t{} ) } -> std::same_as<void>;
    { integrator.sum( data, out, typename T::numeric_t{} ) } -> std::same_as<void>;
    { integrator.borrow_scratch() } -> std::same_as< Magnus::MatrixView<typename T::numeric_t> >;
};

template <class NumT, class AllocatorT = std::allocator<NumT>>
class DefaultIntegrator {
    Magnus::DynMatrix<NumT, AllocatorT> tmp;

public:
    using numeric_t = NumT;
    using allocator_t = AllocatorT;

    DefaultIntegrator( size_t dim, const AllocatorT& alloc = AllocatorT() ) : tmp(dim, alloc) {}

    void prefix(Magnus::MatrixSpan<NumT>& A, NumT dt) { 
        A[0].scale(dt);
        for (size_t i = 1; i < A.length(); ++i) A[i].scale(dt).add(A[i - 1]);
    }

    void sum(Magnus::MatrixSpan<NumT>& A, Magnus::MatrixView<NumT>& out, NumT dt) {
        for (size_t i = 0; i < A.length(); ++i) out.add_scaled( A[i], dt );
    }

    Magnus::MatrixView<NumT> borrow_scratch() {
        return tmp.asView();
    }
    
};

template <class NumT, class AllocatorT = std::allocator<NumT>>
class TrapezoidalIntegrator {
    Magnus::DynMatrix<NumT, AllocatorT> tmp;

public:
    using numeric_t = NumT;
    using allocator_t = AllocatorT;

    TrapezoidalIntegrator(size_t dim, const AllocatorT& alloc = AllocatorT()) : tmp(dim, alloc) {}

    void prefix(Magnus::MatrixSpan<NumT>& A, NumT dt) {
        size_t len = A.length();

        tmp.copy_from(A[0]);
        A[0].zero();

        for (size_t i = 1; i < len; ++i) {
            A[i].scale(dt / NumT(2)).add_scaled(tmp, dt / NumT(2)).add(A[i - 1]);
            tmp.scale(NumT(-1)).add_scaled(A[i], NumT(2) / dt).add_scaled(A[i - 1], NumT(-2) / dt);
        }
    }

    void sum(Magnus::MatrixSpan<NumT>& A, Magnus::MatrixView<NumT>& out, NumT dt) {
        size_t len = A.length();

        out.add_scaled(A[0], dt / NumT(2));
        for (size_t i = 1; i + 1 < len; ++i) out.add_scaled(A[i], dt);
        out.add_scaled(A[len - 1], dt / NumT(2));
    }

    Magnus::MatrixView<NumT> borrow_scratch() {
        return tmp.asView();
    }
};

template <class NumT, class AllocatorT = std::allocator<NumT>>
class SimpsonIntegrator {
    Magnus::DynMatrixSpan<NumT, AllocatorT> scratch;

public:
    using numeric_t = NumT;
    using allocator_t = AllocatorT;

    SimpsonIntegrator(size_t dim, const AllocatorT& alloc = AllocatorT()) :
        scratch(dim, 3, alloc) {}

    void prefix(Magnus::MatrixSpan<NumT>& A, NumT dt) {
        size_t len = A.length();
        Magnus::MatrixView<NumT> tmp = scratch[0];
        Magnus::MatrixView<NumT> prev2 = scratch[1];
        Magnus::MatrixView<NumT> prev1 = scratch[2];

        NumT half_dt = dt / NumT(2);
        NumT third_dt = dt / NumT(3);

        prev2.copy_from(A[0]);
        A[0].zero();

        prev1.copy_from(A[1]);
        A[1].scale(half_dt).add_scaled(prev2, half_dt);

        for (size_t i = 2; i < len; ++i) {
            tmp.copy_from(A[i]);

            if (i % 2 == 0) {
                A[i]
                    .scale(third_dt)
                    .add_scaled(prev2, third_dt)
                    .add_scaled(prev1, NumT(4) * third_dt)
                    .add(A[i - 2]);
            } else {
                A[i]
                    .scale(half_dt)
                    .add_scaled(prev1, half_dt)
                    .add(A[i - 1]);
            }

            prev2.copy_from(prev1);
            prev1.copy_from(tmp);
        }
    }

    void sum(Magnus::MatrixSpan<NumT>& A, Magnus::MatrixView<NumT>& out, NumT dt) {
        size_t len = A.length();

        size_t simpson_last = (len % 2 == 1) ? len - 1 : len - 2;

        out.add_scaled(A[0], dt / NumT(3));
        for (size_t i = 1; i < simpson_last; ++i) {
            out.add_scaled(A[i], (i % 2 == 1) ? NumT(4) * dt / NumT(3) : NumT(2) * dt / NumT(3));
        }
        out.add_scaled(A[simpson_last], dt / NumT(3));

        if (simpson_last + 1 < len) {
            out.add_scaled(A[simpson_last], dt / NumT(2));
            out.add_scaled(A[simpson_last + 1], dt / NumT(2));
        }
    }

    Magnus::MatrixView<NumT> borrow_scratch() {
        return scratch[0];
    }
};

template <class NumT, class AllocatorT = std::allocator<NumT>>
class BooleIntegrator {
    Magnus::DynMatrixSpan<NumT, AllocatorT> scratch;

public:
    using numeric_t = NumT;
    using allocator_t = AllocatorT;

    BooleIntegrator(size_t dim, const AllocatorT& alloc = AllocatorT()) :
        scratch(dim, 5, alloc) {}

    void prefix(Magnus::MatrixSpan<NumT>& A, NumT dt) {
        size_t len = A.length();

        Magnus::MatrixView<NumT> f0 = scratch[0];
        Magnus::MatrixView<NumT> f1 = scratch[1];
        Magnus::MatrixView<NumT> f2 = scratch[2];
        Magnus::MatrixView<NumT> f3 = scratch[3];
        Magnus::MatrixView<NumT> f4 = scratch[4];

        f0.copy_from(A[0]);
        A[0].zero();

        size_t start = 0;
        while (start + 1 < len) {
            size_t block_len = std::min<size_t>(4, len - start - 1);
            Magnus::MatrixView<NumT> base = A[start];

            f1.copy_from(A[start + 1]);
            A[start + 1]
                .scale(dt / NumT(2))
                .add_scaled(f0, dt / NumT(2))
                .add(base);

            f2.copy_from(A[start + 2]);
            A[start + 2]
                .scale(dt / NumT(3))
                .add_scaled(f0, dt / NumT(3))
                .add_scaled(f1, NumT(4) * dt / NumT(3))
                .add(base);

            f3.copy_from(A[start + 3]);
            A[start + 3]
                .scale(NumT(3) * dt / NumT(8))
                .add_scaled(f0, NumT(3) * dt / NumT(8))
                .add_scaled(f1, NumT(9) * dt / NumT(8))
                .add_scaled(f2, NumT(9) * dt / NumT(8))
                .add(base);

            f4.copy_from(A[start + 4]);
            A[start + 4]
                .scale(NumT(14) * dt / NumT(45))
                .add_scaled(f0, NumT(14) * dt / NumT(45))
                .add_scaled(f1, NumT(64) * dt / NumT(45))
                .add_scaled(f2, NumT(24) * dt / NumT(45))
                .add_scaled(f3, NumT(64) * dt / NumT(45))
                .add(base);

            f0.copy_from(f4);
            start += 4;
        }
    }

    void sum(Magnus::MatrixSpan<NumT>& A, Magnus::MatrixView<NumT>& out, NumT dt) {
        size_t len = A.length();
        if (len < 2) return;

        size_t intervals = len - 1;
        size_t boole_intervals = intervals - intervals % 4;

        for (size_t i = 0; i < boole_intervals; i += 4) {
            out.add_scaled(A[i], NumT(14) * dt / NumT(45));
            out.add_scaled(A[i + 1], NumT(64) * dt / NumT(45));
            out.add_scaled(A[i + 2], NumT(24) * dt / NumT(45));
            out.add_scaled(A[i + 3], NumT(64) * dt / NumT(45));
            out.add_scaled(A[i + 4], NumT(14) * dt / NumT(45));
        }

        size_t tail = intervals - boole_intervals;
        size_t start = boole_intervals;

        if (tail == 1) {
            out.add_scaled(A[start], dt / NumT(2));
            out.add_scaled(A[start + 1], dt / NumT(2));
        } else if (tail == 2) {
            out.add_scaled(A[start], dt / NumT(3));
            out.add_scaled(A[start + 1], NumT(4) * dt / NumT(3));
            out.add_scaled(A[start + 2], dt / NumT(3));
        } else if (tail == 3) {
            out.add_scaled(A[start], NumT(3) * dt / NumT(8));
            out.add_scaled(A[start + 1], NumT(9) * dt / NumT(8));
            out.add_scaled(A[start + 2], NumT(9) * dt / NumT(8));
            out.add_scaled(A[start + 3], NumT(3) * dt / NumT(8));
        }
    }

    Magnus::MatrixView<NumT> borrow_scratch() {
        return scratch[0];
    }
};

#endif
