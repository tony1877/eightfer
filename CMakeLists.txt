cmake_minimum_required(VERSION 3.24)
project(shoehorn VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)  # llama.cpp's targets; ours switch to 20 below
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

option(SHOEHORN_CUDA "Build ggml's CUDA backend" ON)

set(E8_LLAMA_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp)
if (NOT EXISTS ${E8_LLAMA_DIR}/ggml/CMakeLists.txt)
    message(FATAL_ERROR "ggml submodule missing. Run: git submodule update --init --depth 1")
endif()

# ggml (tensor library, CPU + CUDA kernels), libllama (tokenizer, vocab only) and llama-common (Jinja chat templates,
# reasoning / tool-call parsing for the server) from the pinned llama.cpp submodule. Static; none of llama.cpp's
# tools, tests or examples; only targets we link get built.
set(BUILD_SHARED_LIBS OFF)
set(GGML_CUDA ${SHOEHORN_CUDA} CACHE BOOL "" FORCE)
# CUDA graphs: llama.cpp's top-level CMake defaults them ON, ggml alone defaults OFF. Without them every decode
# step pays one launch per kernel (thousands per token on WDDM).
set(GGML_CUDA_GRAPHS ON CACHE BOOL "" FORCE)
# Flash-attention kernels for the KV types we serve with (ggml's default list lacks mixed q8_0 K / q4_0 V, which
# would otherwise be converted to F16 on every attention call)
set(GGML_CUDA_FA_QUANTS "q4_0-q4_0;q8_0-q8_0;q8_0-q4_0;f16-f16;bf16-bf16" CACHE STRING "" FORCE)
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

add_executable(shoehorn
    src/main.cpp
    src/cli/info.cpp
    src/cli/logits.cpp
    src/cli/pack.cpp
    src/cli/gen.cpp
    src/cli/dump.cpp
    src/cli/multitest.cpp
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
target_include_directories(shoehorn PRIVATE src ${E8_LLAMA_DIR}/vendor/cpp-httplib ${E8_LLAMA_DIR}/vendor)
target_compile_definitions(shoehorn PRIVATE SHOEHORN_VERSION="${PROJECT_VERSION}")
target_link_libraries(shoehorn PRIVATE e8_io e8_model ggml llama-common llama cpp-httplib)
if (WIN32)
    target_link_libraries(shoehorn PRIVATE shell32)  # CommandLineToArgvW
endif()

# shoehorn serve /dashboard, /sensors and /models: the pages are embedded in the binary (regenerated when the HTML changes).
# MSVC caps one string literal at ~16 KB: adjacent raw literals of 8000 characters
function(e8_embed_html name)
  file(READ ${CMAKE_CURRENT_SOURCE_DIR}/src/server/${name}.html html)
  string(LENGTH "${html}" len)
  set(inc "")
  set(pos 0)
  while(pos LESS len)
    string(SUBSTRING "${html}" ${pos} 8000 part)
    string(APPEND inc "R\"E8DASH(${part})E8DASH\"\n")
    math(EXPR pos "${pos} + 8000")
  endwhile()
  file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/generated/${name}.inc "${inc}")
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/server/${name}.html)
endfunction()
e8_embed_html(dashboard)
e8_embed_html(sensors)
e8_embed_html(models)
target_include_directories(shoehorn PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/generated)