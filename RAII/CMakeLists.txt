aux_source_directory(./src RAII_src)

file (GLOB_RECURSE RAII_header ./include/*.h)

add_library(libRAII SHARED ${RAII_src} ${RAII_header})
target_include_directories(libRAII PUBLIC ./include)
target_link_libraries(libRAII PUBLIC pthread)

add_executable(RAII main.cc)
target_link_libraries(RAII PUBLIC libRAII)