add_executable(stream stream.cc)
target_link_libraries(stream -Wl,--no-whole-archive nlohmann_json::nlohmann_json)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -Wall -O3 -g")
