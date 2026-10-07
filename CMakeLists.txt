cmake_minimum_required(VERSION 3.24)
project(eightfer VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)  # llama.cpp's targets; ours switch to 20 below
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

# ggml (tensor library, CPU + CUDA kernels), libllama (tokenizer, vocab only) and llama-common (Jinja chat templates,
# reasoning / tool-call parsing for the server) from the pinned llama.cpp submodule. Static; none of llama.cpp's
# tools, tests or examples; only targets we link get built.
set(BUILD_SHARED_LIBS OFF)
set(GGML_CUDA ${EIGHTFER_CUDA} CACHE BOOL "" FORCE)
# CUDA graphs: llama.cpp's top-level CMake defaults them ON, ggml alone defaults OFF. Without them every decode
# step pays one launch per kernel (thousands per token on WDDM).
set(GGML_CUDA_GRAPHS ON CACHE BOOL "" FORCE)
foreach (opt LLAMA_BUILD_TESTS LLAMA_BUILD_TOOLS LLAMA_BUILD_EXAMPLES LLAMA_BUILD_SERVER LLAMA_BUILD_APP LLAMA_BUILD_UI
             LLAMA_OPENSSL LLAMA_SUBPROCESS LLAMA_LLGUIDANCE)
    set(${opt} OFF CACHE BOOL "" FORCE)
endforeach()
set(LLAMA_BUILD_COMMON ON CACHE BOOL "" FORCE)
set(GGML_LLAMAFILE OFF CACHE BOOL "" FORCE)  # llama.cpp turns it on; it does not build under MSVC with AVX512_BF16
add_subdirectory(${E8_LLAMA_DIR} ${CMAKE_BINARY_DIR}/llama.cpp EXCLUDE_FROM_ALL)
set(CMAKE_CXX_STANDARD 20)

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
    src/kernels/moe_cpu.cpp
)
target_include_directories(e8_model PUBLIC src)
target_link_libraries(e8_model PUBLIC ggml ggml-cpu)

add_executable(eightfer
    src/main.cpp
    src/cli/info.cpp
    src/cli/logits.cpp
    src/cli/pack.cpp
    src/cli/gen.cpp
    src/cli/dump.cpp
    src/cli/decode.cpp
    src/server/server.cpp
    src/cli/ppl.cpp
    src/cli/selftest.cpp
    src/bench/bench.cpp
    src/bench/bench_mem.cpp
    src/bench/bench_gemv.cpp
    src/bench/bench_pcie.cpp
    src/bench/bench_disk.cpp
    src/sys/sysinfo.cpp
)
target_include_directories(eightfer PRIVATE src ${E8_LLAMA_DIR}/vendor/cpp-httplib ${E8_LLAMA_DIR}/vendor)
target_compile_definitions(eightfer PRIVATE EIGHTFER_VERSION="${PROJECT_VERSION}")
target_link_libraries(eightfer PRIVATE e8_io e8_model ggml llama-common llama cpp-httplib)
if (WIN32)
    target_link_libraries(eightfer PRIVATE shell32)  # CommandLineToArgvW
endif()
