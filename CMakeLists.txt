cmake_minimum_required(VERSION 3.16)

project(
    engine
    LANGUAGES CXX
    VERSION 0.1.0
)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(engine_name engine)

add_subdirectory(3rdlibs)
add_subdirectory(engine)
add_subdirectory(sandbox)
# add_subdirectory(interface)
add_subdirectory(helloworld)
add_subdirectory(polymorphism)
add_subdirectory(RAII)
add_subdirectory(socket)
add_subdirectory(vector)
add_subdirectory(pthread_atomic_volatile)
add_subdirectory(async)
add_subdirectory(threadpool)

# 先把测试关掉
option(ENGINE_BUILD_TEST "should build unittest" ON)

if (ENGINE_BUILD_TEST)
    include(CTest)
    enable_testing()
    add_subdirectory(test)
endif()