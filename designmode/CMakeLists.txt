file(GLOB_RECURSE designmode_src ./src/*.cc)
file(GLOB_RECURSE designmode_include ./include/.h)

add_library(libdesignmode SHARED ${designmode_src} ${designmode_include})
target_include_directories(libdesignmode PUBLIC ./include)
target_link_libraries(libdesignmode PUBLIC pthread)

add_executable(designmode main.cc)
target_link_libraries(designmode PUBLIC libdesignmode)