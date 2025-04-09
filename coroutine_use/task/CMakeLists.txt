aux_source_directory(./src task_src)

file (GLOB_RECURSE task_header ./include/*.h)

add_library(libtask SHARED ${task_src} ${task_header})
target_include_directories(libtask PUBLIC ./include)

add_executable(task main.cc)
target_link_libraries(task PUBLIC libtask)

target_compile_options(task PUBLIC -fcoroutines)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++20 -Wall -O3 -g")
