#ifndef __DISPATCH_HPP__
#define __DISPATCH_HPP__
#include "grad.hpp"
#include "graddata.hpp"
#include "magnus.hpp"
#include "util/memorybuffer.hpp"
#include <optional>
#include <string_view>
#include <variant>

namespace Magnus {
    
    using f32 = float;
    using f64 = double;
    using c32 = std::complex<float>;
    using c64 = std::complex<double>;

    template <class T>
    concept Numeric = 
    std::same_as<T, f32> ||
    std::same_as<T, f64> ||
    std::same_as<T, c32> ||
    std::same_as<T, c64>;

    struct ParamsBase {
        size_t n, dim, samples;
        double t0, tf;

        ParamsBase(size_t n, size_t dim, size_t samples, double t0, double tf) :
            n(n), dim(dim), samples(samples), t0(t0), tf(tf) {}
    };

    template <Numeric NumT>
    struct DefaultData {
        NumT* in;
        NumT* out;
        NumT* vjp_data;

        DefaultData(NumT* in, NumT* out, NumT* vjp_data = nullptr) :
            in(in), out(out), vjp_data(vjp_data) {}
    };

    template <Numeric NumT>
    struct VJPData {
        NumT* in;
        NumT* cotangent;
        NumT* out;
        NumT* carry;

        VJPData(NumT* in, NumT* cotangent, NumT* out, NumT* carry = nullptr) :
            in(in), cotangent(cotangent), out(out), carry(carry) {}
    };

    struct Params : public ParamsBase {
        using data_t = std::variant<DefaultData<f32>, DefaultData<f64>, DefaultData<c32>, DefaultData<c64>>;
        data_t data;

        Params(data_t data, size_t n, size_t dim, size_t samples, double t0, double tf) :
            ParamsBase(n, dim, samples, t0, tf), data(std::move(data)) {}
    };

    struct VJPParams : public ParamsBase {
        using data_t = std::variant<VJPData<f32>, VJPData<f64>, VJPData<c32>, VJPData<c64>>;
        data_t data;

        VJPParams(data_t data, size_t n, size_t dim, size_t samples, double t0, double tf) :
            ParamsBase(n, dim, samples, t0, tf), data(std::move(data)) {}

        template <Numeric NumT>
        std::optional<VJP::Data<NumT>> make_vjp_carry() {
            auto& buffers = std::get<VJPData<NumT>>(data);
            if (buffers.carry == nullptr) return std::nullopt;

            return VJP::Data<NumT>(buffers.carry, n, samples, dim);
        }
    };

    template <class>
    inline constexpr bool always_false_v = false;

    template <class T>
    concept Dispatchable = requires ( const ParamsBase& p ) {
        { T::valid(p) } -> std::same_as<bool>;
        { T::dispatch_name() } -> std::convertible_to<std::string_view>;
    };

    template <class T>
    concept Selectable = requires (const ParamsBase& p) {
        { T::use(p) } -> std::same_as<bool>;
    };

    template <class T>
    concept AutoDispatchable = Dispatchable<T> && Selectable<T>;

    /** 
     * Type list with a bunch of convience ops. Almost like a "type array" class, where static functions serve as "type methods".
     * Some functionality (auto selection) is only enabled if each type is Selectable as well as dispatchable.
     * Do note that, during auto selection, priority is first-to-last. You should specify the order of your types according to the order of auto selection, if all types ::use() return true.
     * Unique names for each type is required.
     * 
    */
    template <Dispatchable... Ts> 
    struct type_list {
        static constexpr bool AutoEnabled = ( Selectable<Ts> && ... );

        static constexpr std::array<std::string_view, sizeof...(Ts)> dispatch_names{ Ts::dispatch_name() ... };

        static consteval size_t size() { return sizeof...(Ts); }

        static consteval size_t auto_sentinel() requires AutoEnabled {
            return std::numeric_limits<size_t>::max();
        }

        static constexpr size_t resolve( std::string_view name ) {
            if constexpr ( AutoEnabled ) {
                if ( name == "Auto" ) return auto_sentinel();
            }

            for (size_t i = 0; i < dispatch_names.size(); ++i) {
                if ( name == dispatch_names[i] ) return i;
            }

            throw std::invalid_argument( "Could not resolve dispatch name option" );
        }

        static constexpr std::string_view from_idx( size_t idx ) {
            if constexpr ( AutoEnabled ) {
                if ( idx == auto_sentinel() ) return "Auto";
            }

            return dispatch_names.at(idx);
        }

        template <class F>
        static decltype(auto) dispatch( size_t idx, const ParamsBase& p, F&& f ) {
            if constexpr ( AutoEnabled ) {
                if ( idx == auto_sentinel() ) idx = select_impl<0, Ts...>( p );
            }

            return dispatch_impl<0, Ts...>(idx, p, std::forward<F>(f));
        }

    private:
        static consteval bool unique_names() {
            for (size_t i = 0; i < dispatch_names.size(); ++i)
                for (size_t j = i + 1; j < dispatch_names.size(); ++j)
                    if (dispatch_names[i] == dispatch_names[j]) return false;

            return true;
        }

        static_assert( unique_names(), "found duplicate dispatch names in type_list" );
    
        template <size_t I, Dispatchable First, Dispatchable... Rest, class F>
        static decltype(auto) dispatch_impl(size_t index, const ParamsBase& p, F&& f) {
            if (index == I) {
                if ( !First::valid(p) ) throw std::invalid_argument( "invalid dispatch option (based on params)" );

                return std::forward<F>(f).template operator()<First>();
            }

            if constexpr (sizeof...(Rest) > 0) {
                return dispatch_impl<I + 1, Rest...>(index, p, std::forward<F>(f));
            } else {
                throw std::out_of_range("dispatch index out of range");
            }
        }

        template <size_t I, AutoDispatchable First, AutoDispatchable... Rest> requires (AutoEnabled)
        static size_t select_impl(const ParamsBase& params) {
            if (First::valid(params) && First::use(params)) return I;

            if constexpr (sizeof...(Rest) > 0) {
                return select_impl<I + 1, Rest...>(params);
            } else {
                throw std::invalid_argument("no dispatch option matched");
            }
        }

    };

    // just for some convenient templated naming
    template <size_t N>
    consteval size_t digit_count() {
        size_t x = N;
        size_t n = 1;
        while (x >= 10) {
            x /= 10;
            ++n;
        }
        return n;
    }

    template <size_t N, size_t PrefixLen, size_t SuffixLen>
    consteval auto make_integral_name(
        const char (&prefix)[PrefixLen],
        const char (&suffix)[SuffixLen]
    ) {
        constexpr size_t digits = digit_count<N>();

        // -1 because string literals include the null terminator
        constexpr size_t prefix_len = PrefixLen - 1;
        constexpr size_t suffix_len = SuffixLen - 1;

        std::array<char, prefix_len + digits + suffix_len> out{};

        for (size_t i = 0; i < prefix_len; ++i)
            out[i] = prefix[i];

        size_t x = N;
        for (size_t i = 0; i < digits; ++i) {
            out[prefix_len + digits - 1 - i] = char('0' + x % 10);
            x /= 10;
        }

        for (size_t i = 0; i < suffix_len; ++i) out[prefix_len + digits + i] = suffix[i];

        return out;
    }

    // numeric dispatch
    template <typename T>
    struct type_name;

    template <typename T>
    constexpr std::string_view type_name_v = type_name<T>::value;

    #define REGISTER_TYPE_NAME(T)           \
        template <>                         \
        struct type_name<T> {               \
            static constexpr std::string_view value = #T; \
        }

    REGISTER_TYPE_NAME(float);
    REGISTER_TYPE_NAME(double);
    REGISTER_TYPE_NAME(std::complex<float>);
    REGISTER_TYPE_NAME(std::complex<double>);
    
    #undef REGISTER_TYPE_NAME

    template <Numeric NumT>
    struct NumBackend {
        using type = NumT;

        static consteval std::string_view dispatch_name() { return type_name_v<NumT>; }

        static bool valid( [[maybe_unused]] const ParamsBase& ) { return true; }
    };

    using NumBackends = type_list<
        NumBackend<f32>,
        NumBackend<f64>,
        NumBackend<c32>,
        NumBackend<c64>
    >;

    // similarly, for VJP data recorders

    template <bool RecordVJPData, Integrator Int>
    VJP::recorder_t<RecordVJPData, Int> make_vjp_recorder(Params& p) {
        using NumT = typename Int::numeric_t;

        if constexpr (RecordVJPData) {
            auto& buffers = std::get<DefaultData<NumT>>(p.data);

            return VJP::Data<NumT>(
                buffers.vjp_data,
                p.n,
                p.samples,
                p.dim
            );
        } else {
            return VJP::NoData{};
        }
    }

    // similarly, for the kernels

namespace Dispatch {
    enum class KernelOp : size_t {
        ONE, MANY, SUM
    };

    inline constexpr std::array<std::string_view, 3> kernel_op_names{
        "one",
        "many",
        "sum"
    };

    constexpr KernelOp op_from_str( std::string_view str ) {
        if ( str == "one" || str == "ONE" ) return KernelOp::ONE;
        if ( str == "many" || str == "MANY" ) return KernelOp::MANY;
        if ( str == "sum" || str == "SUM" ) return KernelOp::SUM;

        throw std::invalid_argument( "could not deduce kernel op from str in Magnus::Dispatch::op_from_str(std::string_view)" );
    }

    template <Integrator Int, bool RecordVJP>
    void one( Params& p, const typename Int::allocator_t& alloc = typename Int::allocator_t() ) {
        using numeric_t = typename Int::numeric_t;
        using matrix_t = typename Int::matrix_t;
        using matrix_span_t = typename Int::matrix_span_t;

        DefaultData<numeric_t>& data = std::get<DefaultData<numeric_t>>(p.data);
        matrix_t out( data.out, p.dim );
        matrix_span_t in( data.in, p.dim, p.samples );

        auto recorder = make_vjp_recorder<RecordVJP, Int>(p);
        Magnus::one<Int>(out, p.n, in, p.t0, p.tf, recorder, alloc);
    }

    template <Integrator Int, bool RecordVJP>
    void many( Params& p, const typename Int::allocator_t& alloc = typename Int::allocator_t() ) {
        using numeric_t = typename Int::numeric_t;
        using matrix_t = typename Int::matrix_t;
        using matrix_span_t = typename Int::matrix_span_t;

        DefaultData<numeric_t>& data = std::get<DefaultData<numeric_t>>(p.data);
        matrix_span_t out( data.out, p.dim, p.n );
        matrix_span_t in( data.in, p.dim, p.samples );

        auto recorder = make_vjp_recorder<RecordVJP, Int>(p);
        Magnus::many<Int>(out, in, p.t0, p.tf, recorder, alloc);
    }

    template <Integrator Int, bool RecordVJP>
    void sum( Params& p, const typename Int::allocator_t& alloc = typename Int::allocator_t()  ) {
        using numeric_t = typename Int::numeric_t;
        using matrix_t = typename Int::matrix_t;
        using matrix_span_t = typename Int::matrix_span_t;

        DefaultData<numeric_t>& data = std::get<DefaultData<numeric_t>>(p.data);
        matrix_t out( data.out, p.dim );
        matrix_span_t in( data.in, p.dim, p.samples );

        auto recorder = make_vjp_recorder<RecordVJP, Int>(p);
        Magnus::sum<Int>(out, p.n, in, p.t0, p.tf, recorder, alloc);
    }
}
    template <class Allocator>
    using kernel_dispatch_t = void(*)(Params&, const Allocator&);

    template <Integrator Int, bool RecordVJP>
    inline constexpr std::array< kernel_dispatch_t<typename Int::allocator_t>, 3> kernels{ 
        &Dispatch::one<Int, RecordVJP>, 
        &Dispatch::many<Int, RecordVJP>, 
        &Dispatch::sum<Int, RecordVJP> 
    };

namespace Dispatch {
    template <Integrator Int>
    void one_vjp(VJPParams& p, const typename Int::allocator_t& alloc = typename Int::allocator_t()) {
        using NumT = typename Int::numeric_t;
        using MatrixT = typename Int::matrix_t;
        using SpanT = typename Int::matrix_span_t;

        auto& buffers = std::get<VJPData<NumT>>(p.data);
        SpanT dA(buffers.out, p.dim, p.samples);
        MatrixT dOut(buffers.cotangent, p.dim);
        SpanT A(buffers.in, p.dim, p.samples);
        auto carry = p.template make_vjp_carry<NumT>();

        VJP::one<Int>(dA, dOut, p.n, A, p.t0, p.tf, carry ? &*carry : nullptr, alloc);
    }

    template <Integrator Int>
    void many_vjp(VJPParams& p, const typename Int::allocator_t& alloc = typename Int::allocator_t()) {
        using NumT = typename Int::numeric_t;
        using SpanT = typename Int::matrix_span_t;

        auto& buffers = std::get<VJPData<NumT>>(p.data);
        SpanT dA(buffers.out, p.dim, p.samples);
        SpanT dOut(buffers.cotangent, p.dim, p.n);
        SpanT A(buffers.in, p.dim, p.samples);
        auto carry = p.template make_vjp_carry<NumT>();

        VJP::many<Int>(dA, dOut, A, p.t0, p.tf, carry ? &*carry : nullptr, alloc);
    }

    template <Integrator Int>
    void sum_vjp(VJPParams& p, const typename Int::allocator_t& alloc = typename Int::allocator_t()) {
        using NumT = typename Int::numeric_t;
        using MatrixT = typename Int::matrix_t;
        using SpanT = typename Int::matrix_span_t;

        auto& buffers = std::get<VJPData<NumT>>(p.data);
        SpanT dA(buffers.out, p.dim, p.samples);
        MatrixT dOut(buffers.cotangent, p.dim);
        SpanT A(buffers.in, p.dim, p.samples);
        auto carry = p.template make_vjp_carry<NumT>();

        VJP::sum<Int>(dA, dOut, p.n, A, p.t0, p.tf, carry ? &*carry : nullptr, alloc);
    }
}

    template <class Allocator>
    using vjp_kernel_dispatch_t = void(*)(VJPParams&, const Allocator&);

    template <Integrator Int>
    inline constexpr std::array<vjp_kernel_dispatch_t<typename Int::allocator_t>, 3> vjp_kernels{
        &Dispatch::one_vjp<Int>,
        &Dispatch::many_vjp<Int>,
        &Dispatch::sum_vjp<Int>
    };

    struct KernelPlan {
        virtual ~KernelPlan() = default;
        virtual size_t divisibility_requirement() const = 0;
        virtual void run() = 0;
    };

    template <Integrator Int>
    class TypedKernelPlan final : public KernelPlan {
        Params p;
        kernel_dispatch_t<typename Int::allocator_t> kernel_ptr;
    
    public:
        explicit TypedKernelPlan(
            Params params, 
            kernel_dispatch_t<typename Int::allocator_t> kernel_ptr
        ) : p( std::move(params) ), kernel_ptr(kernel_ptr) {}

        size_t divisibility_requirement() const override {
            return Int::divisibility_requirement();
        }

        static size_t required_bytes(const Params& p) {
            size_t matrix_size = p.dim * p.dim;
            size_t scalar_count = 0;
            scalar_count += Int::memory_requirement() * matrix_size;

            // Internal Y buffer plus a copied Y[last] buffer allocated by Magnus::one/many/sum only when n > 1.
            if (p.n > 1) scalar_count += (p.samples + 1) * matrix_size;

            static constexpr size_t SCRATCH = 512;
            return scalar_count * sizeof(typename Int::numeric_t) + SCRATCH;
        }

        void run() override {
            using NumT = typename Int::numeric_t;

            MemoryBuffer memory( required_bytes(p) );
            auto alloc = memory.get_allocator<NumT>();

            kernel_ptr(p, alloc);
        }
    };

    template <Integrator Int>
    class TypedVJPKernelPlan final : public KernelPlan {
        VJPParams p;
        vjp_kernel_dispatch_t<typename Int::allocator_t> kernel_ptr;

    public:
        explicit TypedVJPKernelPlan(
            VJPParams params,
            vjp_kernel_dispatch_t<typename Int::allocator_t> kernel_ptr
        ) : p(std::move(params)), kernel_ptr(kernel_ptr) {}

        size_t divisibility_requirement() const override {
            return Int::divisibility_requirement();
        }

        static size_t required_bytes(const VJPParams& p) {
            using NumT = typename Int::numeric_t;

            size_t matrix_size = p.dim * p.dim;
            size_t scalar_count = Int::memory_requirement() * matrix_size;

            if (p.n > 1) {
                scalar_count += p.samples * matrix_size;

                const auto& buffers = std::get<VJPData<NumT>>(p.data);
                if (buffers.carry == nullptr) {
                    scalar_count += Int::memory_requirement() * matrix_size;
                    scalar_count += gl_max_n(p.n) * total_orders(p.n) * p.samples * matrix_size;
                    scalar_count += (p.samples + 1) * matrix_size;
                }
            }

            static constexpr size_t SCRATCH = 512;
            return scalar_count * sizeof(NumT) + SCRATCH;
        }

        void run() override {
            using NumT = typename Int::numeric_t;

            MemoryBuffer memory(required_bytes(p));
            auto alloc = memory.get_allocator<NumT>();

            kernel_ptr(p, alloc);
        }
    };

}

#endif
