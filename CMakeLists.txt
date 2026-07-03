cmake_minimum_required(VERSION 3.16)

project(magnus LANGUAGES CXX)

option(MAGNUS_ENABLE_JAX_FFI "Build JAX FFI registration hooks" ON)

find_package(BLAS REQUIRED)
find_package(TBB REQUIRED)

add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/gl-nodes.o

    COMMAND ${CMAKE_COMMAND} -E chdir
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_OBJCOPY}
            -I binary
            -O elf64-x86-64
            -B i386:x86-64
            gl-nodes.bin
            ${CMAKE_CURRENT_BINARY_DIR}/gl-nodes.o
    COMMAND ${CMAKE_OBJCOPY}
            --add-section
            .note.GNU-stack=/dev/null
            --set-section-flags
            .note.GNU-stack=contents,readonly
            ${CMAKE_CURRENT_BINARY_DIR}/gl-nodes.o

    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/gl-nodes.bin
)

add_library(magnus_core STATIC
    src/gausslegendre.cpp
    src/backendcomposer.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/gl-nodes.o
)

set_source_files_properties(
    ${CMAKE_CURRENT_BINARY_DIR}/gl-nodes.o
    PROPERTIES
    EXTERNAL_OBJECT TRUE
    GENERATED TRUE
)

target_compile_features(magnus_core PUBLIC cxx_std_23)
target_compile_options(magnus_core PUBLIC -O3 -march=native)
target_include_directories(magnus_core PUBLIC src)
target_link_libraries(magnus_core PUBLIC BLAS::BLAS TBB::tbb)
set_target_properties(magnus_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)

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
    src/python.cpp
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

    target_sources(magnus_python PRIVATE src/jax_ffi.cpp)
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

add_custom_command(TARGET magnus_python POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_CURRENT_SOURCE_DIR}/demo/magnus"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:magnus_python>
        "${CMAKE_CURRENT_SOURCE_DIR}/demo/magnus/$<TARGET_FILE_NAME:magnus_python>"
)
