add_subdirectory(glfw)
add_subdirectory(tomlplusplus)
add_subdirectory(json)
add_subdirectory(fast_float)

# 设置 FastFloat 的包含路径
set(FASTFLOAT_INCLUDE_DIR
    ${CMAKE_CURRENT_SOURCE_DIR}/fast_float/include
    CACHE INTERNAL "FastFloat include directories"
)

add_subdirectory(folly)

include(mono/find_mono.cmake)