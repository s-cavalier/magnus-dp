#ifndef __DISPATCH_HPP__
#define __DISPATCH_HPP__
#include <cstddef>
#include <type_traits>
#include <concepts>
#include <tuple>
#include <stdexcept>
#include "memorybuffer.hpp"
#include "integrate.hpp"
#include "magnus.hpp"

namespace Magnus {

    namespace Dispatch {

        enum class Type : unsigned char {
            R32,
            R64,
            C32,
            C64
        };        

        struct Parameters {
            void* in;             
            void* out; 
            Type dtype;
            size_t n;
            size_t dim;
            size_t samples;
            double t0; double tf;
            MemoryBuffer::Allocator<std::byte, 8> alloc; 
            // set to std::byte so it can be freely swapped for float/double/c32/c64/ 
        };

        template <Integrator Int>
        void type_helper_one( Parameters& param ) {
            switch ( param.dtype ) {
                case Type::R32: 
                    one<Int>(
                        typename Int::matrix_t( (float*)param.out, param.dim ),
                        param.n,
                        typename Int::matrix_span_t( (float*)param.in, param.dim, param.samples ),
                        param.t0, param.tf,
                        MemoryBuffer::Allocator<float, 8>(param.alloc)
                    );
                    break;
                
                case Type::R64: 
                    one<Int>(
                        typename Int::matrix_t( (double*)param.out, param.dim ),
                        param.n,
                        typename Int::matrix_span_t( (double*)param.in, param.dim, param.samples ),
                        param.t0, param.tf,
                        MemoryBuffer::Allocator<double, 8>(param.alloc)
                    );
                    break;
                case Type::C32: 
                    one<Int>(
                        typename Int::matrix_t( (std::complex<float>*)param.out, param.dim ),
                        param.n,
                        typename Int::matrix_span_t( (std::complex<float>*)param.in, param.dim, param.samples ),
                        param.t0, param.tf,
                        MemoryBuffer::Allocator<std::complex<float>, 8>(param.alloc)
                    );
                    break;
                case Type::C64: 
                    one<Int>(
                        typename Int::matrix_t( (std::complex<double>*)param.out, param.dim ),
                        param.n,
                        typename Int::matrix_span_t( (std::complex<double>*)param.in, param.dim, param.samples ),
                        param.t0, param.tf,
                        MemoryBuffer::Allocator<std::complex<double>, 8>(param.alloc)
                    );
                    break;

                default: throw std::runtime_error("magnus.one only supports dtypes float32, float64, complex64, complex128.");
            }

        }

        template <Integrator Int>
        void type_helper_many( Parameters& param ) {
            switch ( param.dtype ) {
                case Type::R32: 
                    many<Int>(
                        typename Int::matrix_span_t( (float*)param.out, param.dim, param.n ),
                        typename Int::matrix_span_t( (float*)param.in, param.dim, param.samples ),
                        param.t0, param.tf,
                        MemoryBuffer::Allocator<float, 8>(param.alloc)
                    );
                    break;
                
                case Type::R64: 
                    many<Int>(
                        typename Int::matrix_span_t( (double*)param.out, param.dim, param.n ),
                        typename Int::matrix_span_t( (double*)param.in, param.dim, param.samples ),
                        param.t0, param.tf,
                        MemoryBuffer::Allocator<double, 8>(param.alloc)
                    );
                    break;
                case Type::C32: 
                    many<Int>(
                        typename Int::matrix_span_t( (std::complex<float>*)param.out, param.dim, param.n ),
                        typename Int::matrix_span_t( (std::complex<float>*)param.in, param.dim, param.samples ),
                        param.t0, param.tf,
                        MemoryBuffer::Allocator<std::complex<float>, 8>(param.alloc)
                    );
                    break;
                case Type::C64: 
                    many<Int>(
                        typename Int::matrix_span_t( (std::complex<double>*)param.out, param.dim, param.n ),
                        typename Int::matrix_span_t( (std::complex<double>*)param.in, param.dim, param.samples ),
                        param.t0, param.tf,
                        MemoryBuffer::Allocator<std::complex<double>, 8>(param.alloc)
                    );
                    break;

                default: throw std::runtime_error("magnus.many only supports dtypes float32, float64, complex64, complex128.");
            }

        }
        
        template <Integrator Int>
        void type_helper_sum( Parameters& param ) {
            switch ( param.dtype ) {
                case Type::R32: 
                    sum<Int>(
                        typename Int::matrix_t( (float*)param.out, param.dim ),
                        param.n,
                        typename Int::matrix_span_t( (float*)param.in, param.dim, param.samples ),
                        param.t0, param.tf,
                        MemoryBuffer::Allocator<float, 8>(param.alloc)
                    );
                    break;
                
                case Type::R64: 
                    sum<Int>(
                        typename Int::matrix_t( (double*)param.out, param.dim ),
                        param.n,
                        typename Int::matrix_span_t( (double*)param.in, param.dim, param.samples ),
                        param.t0, param.tf,
                        MemoryBuffer::Allocator<double, 8>(param.alloc)
                    );
                    break;
                case Type::C32: 
                    sum<Int>(
                        typename Int::matrix_t( (std::complex<float>*)param.out, param.dim ),
                        param.n,
                        typename Int::matrix_span_t( (std::complex<float>*)param.in, param.dim, param.samples ),
                        param.t0, param.tf,
                        MemoryBuffer::Allocator<std::complex<float>, 8>(param.alloc)
                    );
                    break;
                case Type::C64: 
                    sum<Int>(
                        typename Int::matrix_t( (std::complex<double>*)param.out, param.dim ),
                        param.n,
                        typename Int::matrix_span_t( (std::complex<double>*)param.in, param.dim, param.samples ),
                        param.t0, param.tf,
                        MemoryBuffer::Allocator<std::complex<double>, 8>(param.alloc)
                    );
                    break;

                default: throw std::runtime_error("magnus.sum only supports dtypes float32, float64, complex64, complex128.");
            }

        }

    }

}

#endif