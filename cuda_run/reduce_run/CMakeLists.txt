set(targets reduce_0 reduce_1 reduce_2 reduce_3)

foreach(var IN LISTS targets)
    add_executable(${var} ${var}.cu)
    target_link_libraries(${var} PRIVATE CUDA::cudart ${CUDA_cublas_LIBRARY})
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        target_compile_options(${var} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
    endif()
endforeach()