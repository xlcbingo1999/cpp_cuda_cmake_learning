
add_library(hello SHARED ./hello/hello.cc)
add_library(world SHARED ./world/world.cc)
add_executable(helloworld main.cc)


target_include_directories(hello PUBLIC ./hello) 
target_include_directories(world PUBLIC ./world)

# PRIVATE: 只能让./hello里面的头文件的符号给world库用
# INTERFACE: 只能让./hello里面的头文件的符号给world库的上级用, world自己是不能用的
target_link_libraries(world PUBLIC hello) 
target_link_libraries(helloworld PUBLIC world)
