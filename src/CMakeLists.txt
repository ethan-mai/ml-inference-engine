add_library(engine STATIC
    engine.cpp
    npy_parser.cpp
)

target_include_directories(engine PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${libnpy_SOURCE_DIR}/include
)

target_link_libraries(engine PUBLIC
    nlohmann_json::nlohmann_json
)
