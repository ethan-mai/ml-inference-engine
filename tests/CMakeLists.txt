include(GoogleTest)

add_executable(engine_tests
    smoke_test.cpp
    npy_parser_test.cpp
)

target_link_libraries(engine_tests PRIVATE
    engine
    GTest::gtest_main
)

gtest_discover_tests(engine_tests
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)
