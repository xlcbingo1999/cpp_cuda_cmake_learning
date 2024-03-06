
macro(AddTest target_name)
    add_executable(${target_name} ./${target_name}.cc)
    add_test(NAME ${target_name} COMMAND $<TARGET_FILE:${target_name}>)
    target_link_libraries(${target_name} PRIVATE engine)
    target_precompile_headers(${target_name} REUSE_FROM engine)
endmacro(AddTest target_name)

AddTest(cgmath)
AddTest(geom)


# # 独立的测试写法，可以用宏去替代
# add_executable(cgmath ./cgmath.cc)
# add_test(NAME cgmath COMMAND $<TARGET_FILE:cgmath>)
# target_link_libraries(cgmath PRIVATE engine)
# target_precompile_headers(cgmath REUSE_FROM engine)

# add_executable(geom ./geom.cc)
# add_test(NAME geom COMMAND $<TARGET_FILE:geom>)
# target_link_libraries(geom PRIVATE engine)
# target_precompile_headers(geom REUSE_FROM engine)