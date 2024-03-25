# 一次性把项目中的所有源文件(.cc)用一个polymorphism_src去配置
# aux_source_directory(./src polymorphism_src)

# 一次性将所有的头文件(.h)用一个polymorphism_header_files去配置
# file(GLOB_RECURSE polymorphism_C_src ./src/*.c)
file(GLOB_RECURSE polymorphism_src ./src/*.cc)
file(GLOB_RECURSE polymorphism_inl_files ./src/*.inl)
file(GLOB_RECURSE polymorphism_header_files ./include/*.h)

# 构建一个动态库
add_library(libpolymorphism SHARED ${polymorphism_src} ${polymorphism_header_files} ${polymorphism_inl_files}) # 构建成一个静态链接库 这里是一个依赖
target_include_directories(libpolymorphism PUBLIC ./include)

add_executable(polymorphism main.cc)
target_link_libraries(polymorphism PUBLIC libpolymorphism)
# -Wall: 生成所有警告信息
# -Werror: 
# -Wstrict-prototypes:
# -Wmissing-prototypes: 


# -rpath=../lib : 设置运行时动态库
# O0: 默认选项，目的: 减少编译时间;生成可调试代码(可打断)
# O/O1: 在不影响编译时间的情况下, 使用优化算法降低代码大小
# O2: 牺牲编译速度, 用于提高目标代码的执行速度
# O3：采用向量化算法, 提高算法的并行执行程度, 利用现代CPU的流水线和cache机制等
# Os: 目标是提高目标代码的运行速度，在-O2的基础上尽量降低目标代码的大小
# Ofast: 在O3的基础上, 加入语言特定的优化（比如加速浮点数计算的向量化优化--ffast-math）
# Og: 选择和-g选项不冲突的优化选项
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -O3 -g")
