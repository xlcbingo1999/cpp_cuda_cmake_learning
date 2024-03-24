rm -rf ./build
cmake -S . -B build
cmake --build build
./build/$1/$1