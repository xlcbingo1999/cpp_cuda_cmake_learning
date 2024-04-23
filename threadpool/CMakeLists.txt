file(GLOB_RECURSE header_files ./*.h)

add_executable(threadpool run.cc ${header_files})
target_link_libraries(threadpool PUBLIC pthread)