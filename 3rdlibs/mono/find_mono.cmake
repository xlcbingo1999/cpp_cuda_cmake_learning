set(MONO_PATH "" CACHE PATH "C# mono frmaework path")

# TODO
add_library(mono SHARED IMPORTED GLOBAL)

set(mono_so "${MONO_PATH}/lib/libmono-2.0.so" CACHE INTERNAL "the mono so")

message("MONO_PATH: ${MONO_PATH}; mono_so: ${mono_so}")

target_include_directories(mono INTERFACE "${MONO_PATH}/include/mono-2.0")
set_target_properties(mono 
    PROPERTIES 
    IMPORTED_LOCATION "${MONO_PATH}/lib/libmono-2.0.so"
    IMPORTED_IMPLIB "${MONO_PATH}/lib/libmono-2.0.a"
) 
target_compile_definitions(mono INTERFACE MONO_PATH="${MONO_PATH}") # 编译时传递给编译器, 以宏的形式传递给C++代码