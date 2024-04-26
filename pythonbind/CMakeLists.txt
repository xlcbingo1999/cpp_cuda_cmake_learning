cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

project(pythonbind LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

find_package(PythonInterp REQUIRED) # 需要找到PythonInterp

# 从当前目录里面执行cmake module, 主要是UseCython和FindCython
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake-cython)

include(UseCython) # 引入cython

# 编译cyx文件为CXX文件
set_source_files_properties(example.pyx PROPERTIES CYTHON_IS_CXX TRUE)

# 创建一个python的模块
cython_add_module(example example.pyx example.cc)

# 获取头文件的路径
target_include_directories(example PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

enable_testing()

add_test(
    NAME
        example_test
    COMMAND
        ${CMAKE_COMMAND} -E env ACCOUNT_MODULE_PATH=$<TARGET_FILE_DIR:example>
        ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test.py
)