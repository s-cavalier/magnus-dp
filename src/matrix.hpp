#ifndef __MATRIX_HPP__
#define __MATRIX_HPP__
#include <span>
#include <complex>
#include <memory>
#include <utility>
#include <algorithm>
#include <cblas.h>
#include <stdexcept>
#include <string>
#include <concepts>

namespace Magnus {

    template <class NumT>
    class MatrixView {
    protected:
        NumT* m_data;
        size_t m_dim;

    public:
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
            std::copy_n( other.m_data, size(), m_data );
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

        // blas convention:
        // s = real float
        // c = complex float
        // d = real double
        // z = complex double

        /* **NOTE:** operator+ does NOT check for equal dimension. Should be optionally verified beforehand. */
        MatrixView& add( const MatrixView& other ) {
            if constexpr ( std::is_same_v<NumT, float> ) {
                cblas_saxpy( size(), 1, other.m_data, 1, m_data, 1);
            } 
            else if constexpr ( std::is_same_v<NumT, double> ) {
                cblas_daxpy( size(), 1, other.m_data, 1, m_data, 1);
            }
            else if constexpr ( std::is_same_v<NumT, std::complex<float>> ) {
                std::complex<float> alpha(1, 0);
                cblas_caxpy( size(), (const void*)&alpha, other.m_data, 1, m_data, 1);
            }
            else if constexpr ( std::is_same_v<NumT, std::complex<double>> ) {
                std::complex<double> alpha(1, 0);
                cblas_zaxpy( size(), (const void*)&alpha, other.m_data, 1, m_data, 1);
            } else {
                for (size_t i = 0; i < size(); ++i) m_data[i] += other.m_data[i];
            }

            return *this;
        }

        /* **NOTE:** operator+ does NOT check for equal dimension. Should be optionally verified beforehand. */
        template <class ScalarT>
        MatrixView& add_scaled( const MatrixView& other, const ScalarT& scalar ) {
            if constexpr ( std::is_same_v<NumT, float> && std::is_convertible_v<ScalarT, float> ) {
                cblas_saxpy( size(), scalar, other.m_data, 1, m_data, 1);
            } 
            else if constexpr ( std::is_same_v<NumT, double> && std::is_convertible_v<ScalarT, double> ) {
                cblas_daxpy( size(), scalar, other.m_data, 1, m_data, 1);
            }
            else if constexpr ( std::is_same_v<NumT, std::complex<float>> && std::is_convertible_v<ScalarT, std::complex<float>> ) {
                cblas_caxpy( size(), (const void*)&scalar, other.m_data, 1, m_data, 1);
            }
            else if constexpr ( std::is_same_v<NumT, std::complex<double>> && std::is_convertible_v<ScalarT, std::complex<double>> ) {
                cblas_zaxpy( size(), (const void*)&scalar, other.m_data, 1, m_data, 1);
            } else {
                for (size_t i = 0; i < size(); ++i) m_data[i] += other.m_data[i] * scalar;
            }

            return *this;
        }

        /* **NOTE:** operator+ does NOT check for equal dimension. Should be optionally verified beforehand. */
        MatrixView& sub( const MatrixView& other ) {
            if constexpr ( std::is_same_v<NumT, float> ) {
                cblas_saxpy( size(), -1, other.m_data, 1, m_data, 1);
            } 
            else if constexpr ( std::is_same_v<NumT, double> ) {
                cblas_daxpy( size(), -1, other.m_data, 1, m_data, 1);
            }
            else if constexpr ( std::is_same_v<NumT, std::complex<float>> ) {
                std::complex<float> alpha(-1, 0);
                cblas_caxpy( size(), (const void*)&alpha, other.m_data, 1, m_data, 1);
            }
            else if constexpr ( std::is_same_v<NumT, std::complex<double>> ) {
                std::complex<double> alpha(-1, 0);
                cblas_zaxpy( size(), (const void*)&alpha, other.m_data, 1, m_data, 1);
            } else {
                for (size_t i = 0; i < size(); ++i) m_data[i] += other.m_data[i];
            }

            return *this;
        }

        template <class ScalarT>
        MatrixView& scale(const ScalarT& num) {
            if constexpr ( std::is_same_v<NumT, float> ) {
                cblas_sscal( size(), num, m_data, 1 );
            } 
            else if constexpr ( std::is_same_v<NumT, double> ) {
                cblas_dscal( size(), num, m_data, 1 );
            }
            else if constexpr ( std::is_same_v<NumT, std::complex<float>> && std::is_same_v<ScalarT, std::complex<float>> ) {
                cblas_cscal( size(), (const void*)&num, m_data, 1 );
            }
            else if constexpr ( std::is_same_v<NumT, std::complex<float>> && std::is_convertible_v<ScalarT, float> ) {
                cblas_csscal( size(), num, m_data, 1 );
            }
            else if constexpr ( std::is_same_v<NumT, std::complex<double>> && std::is_same_v<ScalarT, std::complex<double>> ) {
                cblas_zdscal( size(), (const void*)&num, m_data, 1 );
            } 
            else if constexpr ( std::is_same_v<NumT, std::complex<double>> && std::is_convertible_v<ScalarT, double> ) {
                cblas_zdscal( size(), num, m_data, 1 );
            } 
            else {
                for (size_t i = 0; i < size(); ++i) m_data[i] *= num;
            }

            return *this;
        }

        void zero() {
            std::fill_n( m_data, size(), 0 );
        }

    };

        namespace detail {

            template <class Alloc>
            struct alloc_holder {
                [[no_unique_address]] Alloc alloc;

                const Alloc& get_allocator() const noexcept { return alloc; }
            };

        }

    template <class NumT, class AllocatorT = std::allocator<NumT>>
    class DynMatrix : public MatrixView<NumT>, private detail::alloc_holder<AllocatorT> {

        void reset() {
            if (this->m_dim > 0) this->alloc.deallocate(this->m_data, this->size());
            this->m_dim = 0;
        }

        DynMatrix() {};

    public:
        using Base = MatrixView<NumT>;
        using AllocBase = detail::alloc_holder<AllocatorT>;

        DynMatrix(size_t dim, const AllocatorT& alloc = AllocatorT()) : Base(nullptr, dim), AllocBase(alloc) {
            this->m_data = this->alloc.allocate(dim * dim);
        }

        DynMatrix(const DynMatrix& other) : Base(nullptr, other.m_dim), AllocBase(other.alloc) {
            size_t other_size = other.size();
            this->m_data = this->alloc.allocate(other_size);
            std::copy_n( other.m_data, other_size, this->m_data );
        }

        DynMatrix(DynMatrix&& other) noexcept : Base(other.m_data, other.m_dim), AllocBase( std::move( other.alloc ) ) {
            other.m_dim = 0;
        }

        DynMatrix& operator=(const DynMatrix& other) {
            if ( this == &other ) return *this;
            size_t other_size = other.size();

            if ( this->m_dim == other.m_dim ) {
                std::copy_n( other.m_data, other_size, this->m_data );
                return *this;
            }

            reset();
            this->m_data = this->alloc.allocate(other_size);
            this->m_dim = other.m_dim;
            std::copy_n( other.m_data, other_size, this->m_data );

            return *this;
        }

        DynMatrix& operator=(DynMatrix&& other) noexcept {
            if (this == &other) return *this;
            reset();
            this->m_data = other.m_data;
            this->m_dim = other.m_dim;
            other.m_dim = 0;
            return *this;
        }

        ~DynMatrix() {
            reset();
        }

        Base& asView() { return static_cast<Base&>(*this); }
        const Base& asView() const { return static_cast<const Base&>(*this); }

    };

    template <class NumT>
    void blas_matmul( const MatrixView<NumT>& a, const MatrixView<NumT>& b, MatrixView<NumT>& out ) {
        size_t dim = a.dim();

        if constexpr ( std::is_same_v<NumT, float> ) {
            cblas_sgemm(
                CblasRowMajor, 
                CblasNoTrans, 
                CblasNoTrans, 
                dim, dim, dim, 
                1, 
                a.data(), 
                dim,
                b.data(),
                dim,
                0.0f,
                out.data(),
                dim
            );

        } 
        else if constexpr ( std::is_same_v<NumT, double> ) {
            cblas_dgemm(
                CblasRowMajor, 
                CblasNoTrans, 
                CblasNoTrans, 
                dim, dim, dim, 
                1, 
                a.data(), 
                dim,
                b.data(),
                dim,
                0.0f,
                out.data(),
                dim
            );

        }
        else if constexpr ( std::is_same_v<NumT, std::complex<float>> ) {
            std::complex<float> alpha(1, 0);
            std::complex<float> beta(0, 0);
            cblas_cgemm(
                CblasRowMajor, 
                CblasNoTrans, 
                CblasNoTrans, 
                dim, dim, dim, 
                (const void*)&alpha, 
                a.data(), 
                dim,
                b.data(),
                dim,
                (const void*)&beta,
                out.data(),
                dim
            );

        }
        else if constexpr ( std::is_same_v<NumT, std::complex<double>> ) {
            std::complex<double> alpha(1, 0);
            std::complex<double> beta(0, 0);
            cblas_cgemm(
                CblasRowMajor, 
                CblasNoTrans, 
                CblasNoTrans, 
                dim, dim, dim, 
                (const void*)&alpha, 
                a.data(), 
                dim,
                b.data(),
                dim,
                (const void*)&beta,
                out.data(),
                dim
            );

        } 
    }

    template <class NumT>
    void manual_matmul(const MatrixView<NumT>& a, const MatrixView<NumT>& b, MatrixView<NumT>& out) {
        size_t dim = a.dim();

        for (size_t i = 0; i < dim; ++i) {
            for (size_t j = 0; j < dim; ++j) {
                out[i, j] = 0;

                for (size_t k = 0; k < dim; ++k) out[i, j] += a[i, k] * b[k, j];
            }
        }

    }
    
    template <class NumT>
    void dim2_matmul(const MatrixView<NumT>& a, const MatrixView<NumT>& b, MatrixView<NumT>& out) {
        out[0, 0] = a[0, 0] * b[0, 0] + a[0, 1] * b[1, 0];
        out[0, 1] = a[0, 0] * b[0, 1] + a[0, 1] * b[1, 1];
        out[1, 0] = a[1, 0] * b[0, 0] + a[1, 1] * b[1, 0];
        out[1, 1] = a[1, 0] * b[0, 1] + a[1, 1] * b[1, 1];
    }

    template <class NumT>
    class MatrixSpan {
    protected:
        NumT* m_data;
        size_t m_dim;
        size_t m_size;
        size_t m_len;

    public:
        MatrixSpan( NumT* data, size_t dim, size_t length ) : m_data(data), m_dim(dim), m_size(dim * dim), m_len(length) {}
        
        size_t mat_dim() const { return m_dim; }
        size_t mat_size() const { return m_size; }
        size_t length() const { return m_len; }
        size_t size() const { return m_size * m_len; }
        bool empty() const { return m_len == 0; }

        MatrixView<NumT> operator[](size_t i) { return MatrixView<NumT>( m_data + i * m_size, m_dim ); }
        MatrixView<const NumT> operator[](size_t i) const { return MatrixView<const NumT>( m_data + i * m_size, m_dim ); }

        NumT* data() { return m_data; }
        const NumT* data() const { return m_data; }

        void copy_from( const MatrixSpan& other ) {
            std::copy_n( other.m_data, m_len * m_size, m_data );
        }

    };

    template <class NumT, class AllocatorT = std::allocator<NumT>>
    class DynMatrixSpan : public MatrixSpan<NumT>, private detail::alloc_holder<AllocatorT> {
        void reset() {
            if (this->m_data != nullptr) this->alloc.deallocate(this->m_data, this->size());

            this->m_data = nullptr;
            this->m_dim = 0;
            this->m_size = 0;
            this->m_len = 0;
        }

    public:
        using Base = MatrixSpan<NumT>;
        using AllocBase = detail::alloc_holder<AllocatorT>;

        DynMatrixSpan(size_t dim, size_t length, const AllocatorT& alloc = AllocatorT()) : Base(nullptr, dim, length), AllocBase(alloc) {
            this->m_data = this->alloc.allocate(this->size());
        }

        DynMatrixSpan(const DynMatrixSpan& other) : Base(nullptr, other.m_dim, other.m_len), AllocBase(other.alloc) {
            this->m_data = this->alloc.allocate(this->size());
            std::copy_n(other.m_data, this->size(), this->m_data);
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

            if (this->m_dim == other.m_dim && this->m_len == other.m_len) {
                std::copy_n(other.m_data, other_size, this->m_data);
                return *this;
            }

            reset();
            this->m_dim = other.m_dim;
            this->m_size = other.m_size;
            this->m_len = other.m_len;

            this->m_data = this->alloc.allocate(other_size);
            std::copy_n(other.m_data, other_size, this->m_data);

            return *this;
        }

        DynMatrixSpan& operator=(DynMatrixSpan&& other) noexcept {
            if (this == &other) return *this;

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

        ~DynMatrixSpan() {
            reset();
        }

        Base& asSpan() { return static_cast<Base&>(*this); }
        const Base& asSpan() const { return static_cast<const Base&>(*this); }

    };

}

#endif
