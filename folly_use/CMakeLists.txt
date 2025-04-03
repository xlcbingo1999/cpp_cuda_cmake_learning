set(CMAKE_CXX_STANDARD 20)

set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(engine_src small_vector move_only)
foreach(var IN LISTS engine_src)
    message (STATUS "add ${var}")
    add_executable(${var} ${var}/${var}.cc)
    target_link_libraries(${var} PUBLIC folly)
endforeach()