#ifndef __GRAD_DATA_HPP__
#define __GRAD_DATA_HPP__
#include "matrix.hpp"
#include "util.hpp"

namespace Magnus::VJP {

    template <class T, class SpanT>
    concept DataRecorder = requires ( T& recorder, const SpanT& span ) {
        { recorder.record_prefix( size_t{}, size_t{}, span ) } -> std::same_as<void>;
    };

    struct NoData {
        template <class SpanT>
        void record_prefix(size_t, size_t, const SpanT&) const {}
    };

    template <class NumT>
    class Data {
        NumT* m_data;
        size_t m_q_count;
        size_t m_depth_count;
        size_t m_sample_count;
        size_t m_dim;

        size_t depth_offset(size_t q, size_t k) const {
            return (q * m_depth_count + (k - 2)) * span_size();
        }

        size_t sample_offset(size_t q, size_t k, size_t sample) const {
            return depth_offset(q, k) + sample * matrix_size();
        }

    public:
        using numeric_t = NumT;

        static constexpr size_t rank() { return 4; }

        Data(NumT* data, size_t n, size_t samples, size_t dim) :
            Data(data, gl_max_n(n), total_orders(n), samples, dim) {}

        Data(
            NumT* data,
            size_t q_count,
            size_t depth_count,
            size_t sample_count,
            size_t dim
        ) :
            m_data(data),
            m_q_count(q_count),
            m_depth_count(depth_count),
            m_sample_count(sample_count),
            m_dim(dim) {}

        NumT* data() const { return m_data; }

        size_t q_count() const { return m_q_count; }
        size_t depth_count() const { return m_depth_count; }
        size_t samples() const { return m_sample_count; }
        size_t dim() const { return m_dim; }
        size_t rows() const { return m_dim; }
        size_t cols() const { return m_dim; }
        size_t matrix_size() const { return m_dim * m_dim; }
        size_t span_size() const { return samples() * matrix_size(); }

        size_t scalar_count() const { return q_count() * depth_count() * span_size(); }

        bool empty() const { return scalar_count() == 0; }

        NumT* prefix_data(size_t q, size_t k) const {
            return m_data + depth_offset(q, k);
        }

        NumT* prefix_data(size_t q, size_t k, size_t sample) const {
            return m_data + sample_offset(q, k, sample);
        }

        template <MatrixPolicy MatPolicyT>
        MatrixSpan<NumT, MatPolicyT> prefix(size_t q, size_t k) const {
            return MatrixSpan<NumT, MatPolicyT>(
                prefix_data(q, k),
                dim(),
                samples()
            );
        }

        template <MatrixPolicy MatPolicyT>
        MatrixView<NumT, MatPolicyT> prefix(size_t q, size_t k, size_t sample) const {
            return MatrixView<NumT, MatPolicyT>(
                prefix_data(q, k, sample),
                dim()
            );
        }

        template <MatrixPolicy MatPolicyT>
        MatrixView<NumT, MatPolicyT> total(size_t q, size_t k) const {
            return prefix<MatPolicyT>(q, k, samples() - 1);
        }

        template <class SpanT>
        void record_prefix(size_t q, size_t k, const SpanT& value) const {
            prefix<typename SpanT::matrix_policy_t>(q, k).copy_from(value);
        }

    };

}

#endif
