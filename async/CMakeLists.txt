add_executable(async async.cc)
target_link_libraries(async PUBLIC pthread)

add_executable(ptr ptr.cc)