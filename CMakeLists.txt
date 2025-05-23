cmake_minimum_required(VERSION 3.16)

project(
    engine
    LANGUAGES CUDA CXX
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
add_subdirectory(openmp)
add_subdirectory(stream)
add_subdirectory(bound)

# add_subdirectory(pythonbind) # 有问题
add_subdirectory(designmode)
add_subdirectory(cuda_run)
add_subdirectory(crtp)
add_subdirectory(fast_float_test)
add_subdirectory(folly_use)
add_subdirectory(coroutine_use)

# 先把测试关掉
option(ENGINE_BUILD_TEST "should build unittest" OFF)

if (ENGINE_BUILD_TEST)
    include(CTest)
    enable_testing()
    add_subdirectory(test)
endif()