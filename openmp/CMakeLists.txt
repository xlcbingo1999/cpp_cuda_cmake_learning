# 第一步: 获取依赖的OpenMP
find_package(OpenMP)
# 第二步: 从include和src中将所有的构建文件提取出来
file(GLOB_RECURSE openmp_inc ./include/*.h)
file(GLOB_RECURSE openmp_src ./src/*.cc)
# 第三步: 用所有的代码文件构建动态链接库
add_library(libmyopenmp SHARED ${openmp_src} ${openmp_inc})
# 第四步: 需要指定target的include文件夹, 用于让其他库可以使用
target_include_directories(libmyopenmp PUBLIC ./include)
# 第五步: 链接其他需要的动态依赖库
target_link_libraries(libmyopenmp PRIVATE OpenMP::OpenMP_CXX) # 连接OpenMP

# 第六步, 构建main.cc的内容
add_executable(openmp main.cc)
# 第七步, 链接刚构建的其他仓库
target_link_libraries(openmp PUBLIC libmyopenmp)