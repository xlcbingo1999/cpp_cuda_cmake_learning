set(CMAKE_CXX_STANDARD 20)

set(single_srcs chat combine_vector functional_seq lazy_example)
foreach(var IN LISTS single_srcs)
    set(CMAKE_BUILD_TYPE Debug)
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g3 -O0")
    add_executable(${var} ${var}/${var}.cc)

    target_link_libraries(${var} PUBLIC pthread)
    target_compile_options(${var} PRIVATE -fcoroutines)
    
endforeach()

add_subdirectory(task)