rm -rf ./build
cmake -S . -B build -DMONO_PATH=/usr
cmake --build build --verbose -j 32
# ./build/$1/$1
cd ./build && ctest -C