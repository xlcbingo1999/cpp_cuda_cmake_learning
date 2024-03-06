# add_executable(main src/main.cc include/) # test的时候需要把include变成链接库才可以
# target_include_directories(main PUBLIC include)
# target_link_libraries(main PUBLIC glfw)
# target_compile_features(main PUBLIC cxx_std_17)

aux_source_directory(./src engine_src)

message("source file: ${engine_src}")

foreach(var in ${engine_src})
    message("sub source file: ${var}")
endforeach(var in ${engine_src})

# GLOB是把目录下的所有文件路径丢在变量里面，RECURSE可以进行递归
file(GLOB_RECURSE engine_header_files ./include/*.h)

add_library(${engine_name} STATIC ${engine_src} ${engine_header_files}) # 构建成一个静态链接库 这里是一个依赖
# target_include_directories 如果这里设置了Private, 只能是在编译${engine_name}的时候才能使用, 但是现在需要让子项目test也要用, 所以PUBLIC
target_include_directories(${engine_name} PUBLIC include) 
# target_link_libraries 第三方库引入
target_link_libraries(${engine_name} PUBLIC glfw tomlplusplus_tomlplusplus mono)

target_precompile_headers(${engine_name} PUBLIC ./include/pch.h)
# target_compile_features 如果这里设置了Private, 只会让${engine_name}用cxx17编译, 子项目test也要用，所以用public
target_compile_features(${engine_name} PUBLIC cxx_std_17)
