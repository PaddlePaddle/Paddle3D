mkdir build
cd build
cmake .. -DCMAKE_TENSORRT_PATH=/Path/To/TensorRT
make -j$(nproc)
