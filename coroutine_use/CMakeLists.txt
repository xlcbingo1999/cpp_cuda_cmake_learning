set(CMAKE_CXX_STANDARD 20)

set(engine_src chat combine_vector)
foreach(var IN LISTS engine_src)
    message (STATUS "add ${var}")
    add_executable(${var} ${var}/${var}.cc)

    target_compile_options(${var} PRIVATE -fcoroutines)
endforeach()