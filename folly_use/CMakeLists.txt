set(CMAKE_CXX_STANDARD 20)

set(CMAKE_CXX_STANDARD_REQUIRED ON)


add_executable(small_vector small_vector/small_vector.cc)
target_link_libraries(small_vector PUBLIC folly)