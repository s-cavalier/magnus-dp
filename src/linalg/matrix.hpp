#ifndef __MATRIX_HPP__
#define __MATRIX_HPP__
#include <span>
#include <complex>
#include <memory>
#include <utility>
#include <algorithm>
#include <stdexcept>
#include <string>
#include <concepts>

#ifndef MAGNUS_RESTRICT
#if defined(__GNUC__) || defined(__clang__)
#define MAGNUS_RESTRICT __restrict__
#else
#define MAGNUS_RESTRICT
#endif
#endif
namespace Magnus {

    template <class T>
    concept MatrixPolicy = requires(
        const typename T::numeric_t* cm,
        typename T::numeric_t* m
    ) {
        typename T::numeric_t;

        { T::matmul( size_t{}, cm, cm, m ) } -> std::same_as<void>;
        { T::matmul_vjp( size_t{}, m, cm, cm, cm, m ) } -> std::same_as<void>;
        { T::matadd( size_t{}, m, cm, double{} ) } -> std::same_as<void>;
        { T::matscale( size_t{}, m, double{} ) } -> std::same_as<void>;
        { T::matcopy( size_t{}, cm, m ) } -> std::same_as<void>;
        { T::matwcopy( size_t{}, cm, m ) } -> std::same_as<void>;
        { T::matzero( size_t{}, m ) } -> std::same_as<void>;
        { T::matwzero( size_t{}, m ) } -> std::same_as<void>;
        { T::matwadd( size_t{}, m, cm ) } -> std::same_as<void>;
        { T::sample_update( size_t{}, size_t{}, cm, m, cm, double{}, m ) } -> std::same_as<void>;
        { T::sample_update_vjp( size_t{}, size_t{}, m, cm, m, cm, double{}, m ) } -> std::same_as<void>;
    };

    template <class NumT>
    using MatMulKernelT = void(&)(size_t, const NumT* MAGNUS_RESTRICT, const NumT* MAGNUS_RESTRICT, NumT* MAGNUS_RESTRICT);

    // Accumulates dA and overwrites dB. B and dB may alias.
    template <class NumT>
    using MatMulVJPKernelT = void(&)(size_t, NumT* MAGNUS_RESTRICT, const NumT* MAGNUS_RESTRICT, const NumT* MAGNUS_RESTRICT, const NumT*, NumT*);

    template <class NumT>
    using MatAddKernelT = void(&)(size_t, NumT* MAGNUS_RESTRICT, const NumT* MAGNUS_RESTRICT, double);

    template <class NumT>
    using MatScaleKernelT = void(&)(size_t, NumT* MAGNUS_RESTRICT, double);

    template <class NumT>
    using MatCopyKernelT = void(&)(size_t, const NumT* MAGNUS_RESTRICT, NumT* MAGNUS_RESTRICT);

    template <class NumT>
    using MatZeroKernelT = void(&)(size_t, NumT* MAGNUS_RESTRICT);

    template <class NumT>
    using MatWideAddKernelT = void(&)(size_t, NumT* MAGNUS_RESTRICT, const NumT* MAGNUS_RESTRICT);

    template <class NumT>
    using MatWideZeroKernelT = void(&)(size_t, NumT* MAGNUS_RESTRICT);

    template <class NumT>
    using SampleUpdateKernelT = void(&)( size_t, size_t, const NumT* MAGNUS_RESTRICT, NumT* MAGNUS_RESTRICT, const NumT* MAGNUS_RESTRICT, double, NumT* MAGNUS_RESTRICT );

    template <class NumT>
    using SampleUpdateVJPKernelT = void(&)( size_t, size_t, NumT* MAGNUS_RESTRICT, const NumT* MAGNUS_RESTRICT, NumT* MAGNUS_RESTRICT, const NumT* MAGNUS_RESTRICT, double, NumT* MAGNUS_RESTRICT );

    template <class NumT, MatrixPolicy MatPolicyT>
    class MatrixView {
    protected:
        NumT* m_data;
        size_t m_dim;

    public:
        using matrix_policy_t = MatPolicyT;

        MatrixView(NumT* data, size_t dimension) : m_data(data), m_dim(dimension) {}

        size_t dim() const { return m_dim; }
        size_t size() const { return m_dim * m_dim; }
        size_t rows() const { return m_dim; }
        size_t cols() const { return m_dim; }
        bool empty() const { return m_dim == 0; }

        NumT* data() { return m_data; }
        const NumT* data() const { return m_data; }

        class iterator {
            NumT* m_data;
            size_t m_dim;
            iterator(NumT* data, size_t dim) : m_data(data), m_dim(dim) {}
            friend class MatrixView;

        public:
            std::span<NumT> operator*() const { return {m_data, m_dim}; }

            iterator& operator++() {
                m_data += m_dim;
                return *this;
            }

            bool operator!=(const iterator& other) const { return m_data != other.m_data; }
            bool operator==(const iterator& other) const { return m_data == other.m_data; }
        };

        class const_iterator {
            const NumT* m_data;
            size_t m_dim;
            const_iterator(const NumT* data, size_t dim) : m_data(data), m_dim(dim) {}
            friend class MatrixView;

        public:
            std::span<const NumT> operator*() const { return {m_data, m_dim}; }

            const_iterator& operator++() {
                m_data += m_dim;
                return *this;
            }

            bool operator!=(const const_iterator& other) const { return m_data != other.m_data; }
            bool operator==(const const_iterator& other) const { return m_data == other.m_data; }
        };

        iterator begin() { return {m_data, m_dim}; }
        iterator end() { return {m_data + size(), m_dim}; }
        const_iterator begin() const { return {m_data, m_dim}; }
        const_iterator end() const { return {m_data + size(), m_dim}; }

        std::span<NumT> flat_view() { return {m_data, m_dim * m_dim}; }
        std::span<const NumT> flat_view() const { return {m_data, m_dim * m_dim}; }

        std::span<NumT> operator[](size_t i) {
            return { m_data + i * m_dim, m_dim };
        }

        std::span<const NumT> operator[](size_t i) const {
            return { m_data + i * m_dim, m_dim };
        }

        NumT& operator[](size_t r, size_t c) {
            return m_data[ r * m_dim + c ];
        }

        const NumT& operator[](size_t r, size_t c) const {
            return m_data[ r * m_dim + c ];
        }

        // Does NOT check for proper sizes.
        void copy_from(const MatrixView& other) {
            matrix_policy_t::matcopy( m_dim, other.m_data, m_data );
        };

        std::span<NumT> at(size_t i) {
            if ( i >= m_dim ) {
                std::string err_msg = "Out of bounds access at row ";
                err_msg += std::to_string(i);
                err_msg += " with bound ";
                err_msg += std::to_string( m_dim );
                err_msg += " in method at(size_t) ";
                throw std::out_of_range(err_msg);
            }

            return { m_data + i * m_dim, m_dim };
        }

        std::span<const NumT> at(size_t i) const {
            if ( i >= m_dim ) {
                std::string err_msg = "Out of bounds access at row ";
                err_msg += std::to_string(i);
                err_msg += " with bound ";
                err_msg += std::to_string( m_dim );
                err_msg += " in method at(size_t) const ";
                throw std::out_of_range(err_msg);
            }
            return { m_data + i * m_dim, m_dim };
        }

        NumT& at(size_t r, size_t c) {
            if ( r >= m_dim || c >= m_dim ) {
                std::string err_msg = "Out of bounds access at (";
                err_msg += std::to_string(r);
                err_msg += ", ";
                err_msg += std::to_string(c);
                err_msg += ") with dimension ";
                err_msg += std::to_string(m_dim);
                throw std::out_of_range(err_msg);
            }
            return m_data[ r * m_dim + c ];
        }

        const NumT& at(size_t r, size_t c) const {
            if ( r >= m_dim || c >= m_dim ) {
                std::string err_msg = "Out of bounds access at (";
                err_msg += std::to_string(r);
                err_msg += ", ";
                err_msg += std::to_string(c);
                err_msg += ") with dimension ";
                err_msg += std::to_string(m_dim);
                throw std::out_of_range(err_msg);
            }
            return m_data[ r * m_dim + c ];
        }


        template <class Callable>
        void fill( Callable&& callable ) {
            size_t s = size();
            for (size_t i = 0; i < s; ++i) m_data[i] = callable();
        }

        void zero() {
            matrix_policy_t::matzero( m_dim, m_data );
        }

        MatrixView& add( const MatrixView& other, double scalar = 1.0 ) {
            matrix_policy_t::matadd( m_dim, m_data, other.data(), scalar );
            return *this;
        }

        MatrixView& scale( double scalar ) {
            matrix_policy_t::matscale( m_dim, m_data, scalar );
            return *this;
        }

        // Note; saves the result of A @ B
        MatrixView& save_matmul( const MatrixView& a, const MatrixView& b ) {
            matrix_policy_t::matmul( m_dim, a.data(), b.data(), m_data );
            return *this;
        }

    };

        namespace detail {
            template <class Alloc>
            struct alloc_holder {
                [[no_unique_address]] Alloc alloc;

                const Alloc& get_allocator() const noexcept { return alloc; }
            };

        }

    template <class NumT, MatrixPolicy MatPolicyT, class AllocatorT = std::allocator<NumT>>
    class DynMatrix : public MatrixView<NumT, MatPolicyT>, private detail::alloc_holder<AllocatorT> {
        using traits_t = std::allocator_traits<AllocatorT>;

        void allocate_for_dim(size_t dim) {
            this->m_dim = dim;
            this->m_data = dim == 0 ? nullptr : traits_t::allocate(this->alloc, dim * dim);
        }

        void reset() {
            if (this->m_data != nullptr) traits_t::deallocate(this->alloc, this->m_data, this->size());
            this->m_data = nullptr;
            this->m_dim = 0;
        }

        DynMatrix() {};

    public:
        using Base = MatrixView<NumT, MatPolicyT>;
        using AllocBase = detail::alloc_holder<AllocatorT>;

        DynMatrix(size_t dim, const AllocatorT& alloc = AllocatorT()) : Base(nullptr, dim), AllocBase(alloc) {
            this->m_data = dim == 0 ? nullptr : traits_t::allocate(this->alloc, dim * dim);
        }

        DynMatrix(const DynMatrix& other) :
            Base(nullptr, other.m_dim),
            AllocBase(traits_t::select_on_container_copy_construction(other.alloc)) {
            size_t other_size = other.size();
            this->m_data = other_size == 0 ? nullptr : traits_t::allocate(this->alloc, other_size);
            if (other_size != 0) std::copy_n(other.m_data, other_size, this->m_data);
        }

        DynMatrix(DynMatrix&& other) noexcept : Base(other.m_data, other.m_dim), AllocBase( std::move( other.alloc ) ) {
            other.m_data = nullptr;
            other.m_dim = 0;
        }

        DynMatrix& operator=(const DynMatrix& other) {
            if ( this == &other ) return *this;
            size_t other_size = other.size();

            if constexpr (traits_t::propagate_on_container_copy_assignment::value) {
                if (!(this->alloc == other.alloc)) {
                    reset();
                    this->alloc = other.alloc;
                }
            }

            if ( this->m_dim != other.m_dim ) {
                reset();
                allocate_for_dim(other.m_dim);
            }

            if (other_size != 0) std::copy_n(other.m_data, other_size, this->m_data);

            return *this;
        }

        DynMatrix& operator=(DynMatrix&& other) noexcept {
            if (this == &other) return *this;

            if constexpr (traits_t::propagate_on_container_move_assignment::value) {
                reset();
                this->alloc = std::move(other.alloc);
                this->m_data = other.m_data;
                this->m_dim = other.m_dim;
                other.m_data = nullptr;
                other.m_dim = 0;
                return *this;
            }

            if (this->alloc == other.alloc) {
                reset();
                this->m_data = other.m_data;
                this->m_dim = other.m_dim;
                other.m_data = nullptr;
                other.m_dim = 0;
                return *this;
            }

            if (this->m_dim != other.m_dim) {
                reset();
                allocate_for_dim(other.m_dim);
            }

            if (other.size() != 0) std::move(other.m_data, other.m_data + other.size(), this->m_data);
            other.reset();
            return *this;
        }

        ~DynMatrix() {
            reset();
        }

        Base& asView() { return static_cast<Base&>(*this); }
        const Base& asView() const { return static_cast<const Base&>(*this); }

    };

    template <class NumT, MatrixPolicy MatPolicyT>
    class MatrixSpan {
    protected:
        NumT* m_data;
        size_t m_dim;
        size_t m_size;
        size_t m_len;

    public:
        using matrix_policy_t = MatPolicyT;
        using matrix_t = MatrixView<NumT, MatPolicyT>;

        MatrixSpan( NumT* data, size_t dim, size_t length ) : m_data(data), m_dim(dim), m_size(dim * dim), m_len(length) {}

        size_t mat_dim() const { return m_dim; }
        size_t mat_size() const { return m_size; }
        size_t length() const { return m_len; }
        size_t size() const { return m_size * m_len; }
        bool empty() const { return m_len == 0; }

        matrix_t operator[](size_t i) { return matrix_t( m_data + i * m_size, m_dim ); }

        NumT* data() { return m_data; }
        const NumT* data() const { return m_data; }

        void copy_from( const MatrixSpan& other ) {
            matrix_policy_t::matwcopy( m_size * m_len, other.m_data, m_data );
        }

        void zero() {
            matrix_policy_t::matwzero( m_size * m_len, m_data );
        }

        void add( const MatrixSpan& other ) {
            matrix_policy_t::matwadd( m_size * m_len, m_data, other.m_data );
        }

        void sample_update(
            const MatrixSpan& A,
            const matrix_t& total,
            double shift,
            matrix_t& temp
        ) {
            matrix_policy_t::sample_update(
                m_dim,
                m_len,
                A.data(),
                m_data,
                total.data(),
                shift,
                temp.data()
            );
        }

        void sample_update_vjp(
            MatrixSpan& dA,
            const MatrixSpan& A,
            const MatrixSpan& prefix,
            double shift,
            matrix_t& temp
        ) {
            matrix_policy_t::sample_update_vjp(
                m_dim,
                m_len,
                dA.data(),
                A.data(),
                m_data,
                prefix.data(),
                shift,
                temp.data()
            );
        }

    };

    template <class NumT, MatrixPolicy MatPolicyT, class AllocatorT = std::allocator<NumT>>
    class DynMatrixSpan : public MatrixSpan<NumT, MatPolicyT>, private detail::alloc_holder<AllocatorT> {
        using traits_t = std::allocator_traits<AllocatorT>;

        void allocate_for_shape(size_t dim, size_t length) {
            this->m_dim = dim;
            this->m_size = dim * dim;
            this->m_len = length;
            this->m_data = this->size() == 0 ? nullptr : traits_t::allocate(this->alloc, this->size());
        }

        void reset() {
            if (this->m_data != nullptr) traits_t::deallocate(this->alloc, this->m_data, this->size());

            this->m_data = nullptr;
            this->m_dim = 0;
            this->m_size = 0;
            this->m_len = 0;
        }

    public:
        using Base = MatrixSpan<NumT, MatPolicyT>;
        using AllocBase = detail::alloc_holder<AllocatorT>;

        DynMatrixSpan(size_t dim, size_t length, const AllocatorT& alloc = AllocatorT()) : Base(nullptr, dim, length), AllocBase(alloc) {
            this->m_data = this->size() == 0 ? nullptr : traits_t::allocate(this->alloc, this->size());
        }

        DynMatrixSpan(const DynMatrixSpan& other) :
            Base(nullptr, other.m_dim, other.m_len),
            AllocBase(traits_t::select_on_container_copy_construction(other.alloc)) {
            this->m_data = this->size() == 0 ? nullptr : traits_t::allocate(this->alloc, this->size());
            if (this->size() != 0) std::copy_n(other.m_data, this->size(), this->m_data);
        }

        DynMatrixSpan(DynMatrixSpan&& other) noexcept : Base(other.m_data, other.m_dim, other.m_len), AllocBase(std::move(other.alloc)) {
            other.m_data = nullptr;
            other.m_dim = 0;
            other.m_size = 0;
            other.m_len = 0;
        }

        DynMatrixSpan& operator=(const DynMatrixSpan& other) {
            if (this == &other) return *this;

            size_t other_size = other.size();

            if constexpr (traits_t::propagate_on_container_copy_assignment::value) {
                if (!(this->alloc == other.alloc)) {
                    reset();
                    this->alloc = other.alloc;
                }
            }

            if (this->m_dim != other.m_dim || this->m_len != other.m_len) {
                reset();
                allocate_for_shape(other.m_dim, other.m_len);
            }

            if (other_size != 0) std::copy_n(other.m_data, other_size, this->m_data);

            return *this;
        }

        DynMatrixSpan& operator=(DynMatrixSpan&& other) noexcept {
            if (this == &other) return *this;

            if constexpr (traits_t::propagate_on_container_move_assignment::value) {
                reset();
                this->alloc = std::move(other.alloc);
                this->m_data = other.m_data;
                this->m_dim = other.m_dim;
                this->m_size = other.m_size;
                this->m_len = other.m_len;

                other.m_data = nullptr;
                other.m_dim = 0;
                other.m_size = 0;
                other.m_len = 0;
                return *this;
            }

            if (this->alloc == other.alloc) {
                reset();
                this->m_data = other.m_data;
                this->m_dim = other.m_dim;
                this->m_size = other.m_size;
                this->m_len = other.m_len;

                other.m_data = nullptr;
                other.m_dim = 0;
                other.m_size = 0;
                other.m_len = 0;
                return *this;
            }

            if (this->m_dim != other.m_dim || this->m_len != other.m_len) {
                reset();
                allocate_for_shape(other.m_dim, other.m_len);
            }

            if (other.size() != 0) std::move(other.m_data, other.m_data + other.size(), this->m_data);
            other.reset();

            return *this;
        }

        ~DynMatrixSpan() {
            reset();
        }

        Base& asSpan() { return static_cast<Base&>(*this); }
        const Base& asSpan() const { return static_cast<const Base&>(*this); }

    };

}

#endif
