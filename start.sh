rm -rf ./build
cmake -S . -B build
cmake --build build --verbose
./build/$1/$1