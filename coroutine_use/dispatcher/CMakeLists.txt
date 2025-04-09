aux_source_directory(./src dispatcher_src)

file (GLOB_RECURSE dispatcher_header ./include/*.h)

add_library(libdispatcher SHARED ${dispatcher_src} ${dispatcher_header})
target_include_directories(libdispatcher PUBLIC ./include)

add_executable(dispatcher main.cc)
target_link_libraries(dispatcher PUBLIC pthread)
target_link_libraries(dispatcher PUBLIC libdispatcher)

target_compile_options(dispatcher PUBLIC -fcoroutines)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++20 -Wall -O3 -g")
