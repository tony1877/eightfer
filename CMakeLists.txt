cmake_minimum_required(VERSION 3.24)
project(eightfer VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

option(EIGHTFER_CUDA "Build ggml's CUDA backend" ON)

set(E8_LLAMA_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp)
if (NOT EXISTS ${E8_LLAMA_DIR}/ggml/CMakeLists.txt)
    message(FATAL_ERROR "ggml submodule missing. Run: git submodule update --init --depth 1")
endif()

# ggml (tensor library, CPU + CUDA kernels) from the pinned llama.cpp submodule.
# Static, no tests/examples; only targets we link get built.
set(BUILD_SHARED_LIBS OFF)
set(GGML_CUDA ${EIGHTFER_CUDA} CACHE BOOL "" FORCE)
add_subdirectory(${E8_LLAMA_DIR}/ggml ${CMAKE_BINARY_DIR}/ggml EXCLUDE_FROM_ALL)

if (MSVC)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/utf-8>)
endif()

add_library(e8_io STATIC src/io/async_reader.cpp)
target_include_directories(e8_io PUBLIC src)

add_library(e8_model STATIC
    src/model/gguf_file.cpp
    src/model/qwen35.cpp
    src/model/qwen4exp.cpp
    src/model/gguf_set.cpp
    src/util/mmap.cpp
    src/model/safetensors.cpp
    src/util/json.cpp
    src/runtime/spec.cpp
    src/kernels/q4k_small.cpp
)
target_include_directories(e8_model PUBLIC src)
target_link_libraries(e8_model PUBLIC ggml ggml-cpu)

add_executable(eightfer
    src/main.cpp
    src/cli/info.cpp
    src/cli/logits.cpp
    src/cli/pack.cpp
    src/cli/gen.cpp
    src/cli/ppl.cpp
    src/cli/selftest.cpp
    src/bench/bench.cpp
    src/bench/bench_mem.cpp
    src/bench/bench_gemv.cpp
    src/bench/bench_pcie.cpp
    src/bench/bench_disk.cpp
    src/sys/sysinfo.cpp
)
target_include_directories(eightfer PRIVATE src)
target_compile_definitions(eightfer PRIVATE EIGHTFER_VERSION="${PROJECT_VERSION}")
target_link_libraries(eightfer PRIVATE e8_io e8_model ggml)
if (WIN32)
    target_link_libraries(eightfer PRIVATE shell32)  # CommandLineToArgvW
endif()
