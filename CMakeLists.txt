cmake_minimum_required(VERSION 3.16)

project(magnus LANGUAGES CXX)

option(MAGNUS_ENABLE_JAX_FFI "Build JAX FFI registration hooks" ON)
option(MAGNUS_ENABLE_OPENMP "Enable OpenMP support when available" ON)

find_package(BLAS REQUIRED)
find_package(TBB REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)

if(MAGNUS_ENABLE_OPENMP)
    find_package(OpenMP)
endif()

set(MAGNUS_GL_MAX_ORDER 64 CACHE STRING "Maximum generated Gauss-Legendre order")

set(MAGNUS_PACKAGE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/magnus")
set(MAGNUS_GENERATED_TYPING_FILE "${CMAKE_CURRENT_BINARY_DIR}/_generated_typing.pyi")

add_custom_target(magnus_package_stage
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_CURRENT_SOURCE_DIR}/src/api/magnus"
        "${MAGNUS_PACKAGE_DIR}"
    COMMENT "Staging magnus Python package"
)

add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/gl-nodes.cpp

    COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/src/util/gen_nodes.py
        ${MAGNUS_GL_MAX_ORDER}
        ${CMAKE_CURRENT_BINARY_DIR}/gl-nodes.cpp

    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/util/gen_nodes.py
    VERBATIM
)

add_library(magnus_core STATIC
    src/util/gausslegendre.cpp
    src/composer/backends.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/gl-nodes.cpp
)

target_sources(magnus_core PRIVATE
    src/linalg/backends.hpp
    src/linalg/blas.hpp
    src/linalg/fixed.hpp
    src/linalg/manual.hpp
    src/linalg/matrix.hpp
    src/linalg/spacecurve.hpp
    src/integration/backends.hpp
    src/integration/boole.hpp
    src/integration/default.hpp
    src/integration/integrate.hpp
    src/integration/simpson.hpp
    src/integration/trapezoid.hpp
)

set_source_files_properties(
    ${CMAKE_CURRENT_BINARY_DIR}/gl-nodes.cpp
    PROPERTIES
    GENERATED TRUE
)

target_compile_features(magnus_core PUBLIC cxx_std_23)
target_compile_options(magnus_core PUBLIC -O3 -march=native)
target_include_directories(magnus_core PUBLIC src)
target_link_libraries(magnus_core PUBLIC BLAS::BLAS TBB::tbb)

if(OpenMP_CXX_FOUND)
    target_link_libraries(magnus_core PUBLIC OpenMP::OpenMP_CXX)
endif()

set_target_properties(magnus_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_executable(magnus_typing_gen
    src/api/typing_gen.cpp
)

target_include_directories(magnus_typing_gen PRIVATE src)
target_link_libraries(magnus_typing_gen PRIVATE magnus_core)
target_compile_features(magnus_typing_gen PRIVATE cxx_std_23)
target_compile_options(magnus_typing_gen PRIVATE
    -O3
    -march=native
)

add_custom_command(
    OUTPUT ${MAGNUS_GENERATED_TYPING_FILE}
    COMMAND $<TARGET_FILE:magnus_typing_gen> ${MAGNUS_GENERATED_TYPING_FILE}
    DEPENDS magnus_typing_gen magnus_package_stage
    VERBATIM
)

add_custom_target(magnus_generated_typing DEPENDS ${MAGNUS_GENERATED_TYPING_FILE})

set(MAGNUS_PYBIND11_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/extern/pybind11"
    CACHE PATH "Path to the pybind11 git submodule"
)

if(NOT EXISTS "${MAGNUS_PYBIND11_DIR}/CMakeLists.txt")
    message(FATAL_ERROR
        "pybind11 submodule not found at ${MAGNUS_PYBIND11_DIR}. "
        "Run: git submodule update --init --recursive"
    )
endif()

set(PYBIND11_FINDPYTHON ON CACHE BOOL "Use CMake FindPython for pybind11" FORCE)
add_subdirectory("${MAGNUS_PYBIND11_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/extern/pybind11" EXCLUDE_FROM_ALL)

pybind11_add_module(magnus_python
    src/api/pybind.cpp
)

if(MAGNUS_ENABLE_JAX_FFI)
    set(MAGNUS_JAX_INCLUDE_DIR "" CACHE PATH "Path to JAX's bundled XLA FFI include directory")

    if(NOT MAGNUS_JAX_INCLUDE_DIR)
        execute_process(
            COMMAND "${Python3_EXECUTABLE}" "-c" "from jax import ffi; print(ffi.include_dir())"
            RESULT_VARIABLE MAGNUS_JAX_INCLUDE_RESULT
            OUTPUT_VARIABLE MAGNUS_JAX_INCLUDE_DIR
            ERROR_VARIABLE MAGNUS_JAX_INCLUDE_ERROR
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )

        if(NOT MAGNUS_JAX_INCLUDE_RESULT EQUAL 0)
            message(FATAL_ERROR
                "MAGNUS_ENABLE_JAX_FFI=ON requires JAX to be importable by "
                "${Python3_EXECUTABLE}.\n${MAGNUS_JAX_INCLUDE_ERROR}"
            )
        endif()
    endif()

    if(NOT EXISTS "${MAGNUS_JAX_INCLUDE_DIR}/xla/ffi/api/ffi.h")
        message(FATAL_ERROR
            "Could not find xla/ffi/api/ffi.h under MAGNUS_JAX_INCLUDE_DIR="
            "${MAGNUS_JAX_INCLUDE_DIR}"
        )
    endif()

    target_sources(magnus_python PRIVATE src/api/jax_ffi.cpp)
    target_include_directories(magnus_python PRIVATE "${MAGNUS_JAX_INCLUDE_DIR}")
    target_compile_definitions(magnus_python PRIVATE MAGNUS_ENABLE_JAX_FFI=1)
endif()

target_link_libraries(magnus_python PRIVATE magnus_core)
target_compile_features(magnus_python PRIVATE cxx_std_23)
target_compile_options(magnus_python PRIVATE
    -O3
    -march=native
)
set_target_properties(magnus_python PROPERTIES OUTPUT_NAME _core)
add_dependencies(magnus_python magnus_generated_typing)

add_custom_command(TARGET magnus_python POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${MAGNUS_GENERATED_TYPING_FILE}"
        "${MAGNUS_PACKAGE_DIR}/_generated_typing.pyi"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:magnus_python>
        "${MAGNUS_PACKAGE_DIR}/$<TARGET_FILE_NAME:magnus_python>"
)
