set(CMAKE_CXX_STANDARD 20)

set(engine_src chat combine_vector functional_seq)
foreach(var IN LISTS engine_src)
    message (STATUS "add ${var}")
    set(CMAKE_BUILD_TYPE Debug)
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g3 -O0")
    add_executable(${var} ${var}/${var}.cc)

    target_compile_options(${var} PRIVATE -fcoroutines)
endforeach()