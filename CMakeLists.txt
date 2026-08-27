cmake_minimum_required(VERSION 3.20)
project(ml_inference_engine CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(FetchContent)

FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.3
)

FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.15.2
)

FetchContent_Declare(
    libnpy
    GIT_REPOSITORY https://github.com/llohse/libnpy.git
    GIT_TAG v1.0.1
)

FetchContent_MakeAvailable(nlohmann_json googletest libnpy)

enable_testing()

add_subdirectory(src)
add_subdirectory(tests)
