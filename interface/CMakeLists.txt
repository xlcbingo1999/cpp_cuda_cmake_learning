add_library(add SHARED ./src/add.cc)
add_library(sub SHARED ./src/sub.cc)
add_executable(interface main.cc)

target_include_directories(add INTERFACE ./include)
target_include_directories(sub INTERFACE ./include)
target_link_libraries(sub INTERFACE add)
target_link_libraries(interface PUBLIC sub)