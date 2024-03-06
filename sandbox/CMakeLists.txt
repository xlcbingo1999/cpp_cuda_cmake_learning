add_executable(sandbox main.cc)
target_link_libraries(sandbox PRIVATE ${engine_name})

# 必须把动态链接库拷贝到项目目录下, 所以用custom_command去做一个文件的拷贝
# 用cmake -E去拷贝方法会好一些
add_custom_command(TARGET sandbox
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${mono_so} $<TARGET_FILE_DIR:sandbox>
)