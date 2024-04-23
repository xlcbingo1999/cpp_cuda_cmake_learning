set(CMAKE_CXX_STANDARD 20)

add_executable(atomic atomic.cc)
target_link_libraries(atomic PUBLIC pthread)
