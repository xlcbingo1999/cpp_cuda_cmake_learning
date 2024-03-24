# 一次性把项目中的所有源文件(.cc)用一个polymorphism_src去配置
aux_source_directory(./src polymorphism_src)

# 一次性将所有的头文件(.h)用一个polymorphism_header_files去配置
file(GLOB_RECURSE polymorphism_header_files ./include/*.h)

# 构建一个动态库
add_library(libpolymorphism SHARED ${polymorphism_src} ${polymorphism_header_files}) # 构建成一个静态链接库 这里是一个依赖
target_include_directories(libpolymorphism PUBLIC ./include)

add_executable(polymorphism main.cc)
target_link_libraries(polymorphism PUBLIC libpolymorphism)