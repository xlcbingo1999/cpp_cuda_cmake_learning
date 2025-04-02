set(CMAKE_CXX_STANDARD 17)

add_executable(fast_float_test fast_float_test.cc)
target_link_libraries(fast_float_test PUBLIC fast_float)