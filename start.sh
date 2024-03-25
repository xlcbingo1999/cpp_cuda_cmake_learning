rm -rf ./build
cmake -S . -B build -DMONO_PATH=/usr
cmake --build build --verbose
# ./build/$1/$1
cd ./build && ctest -C