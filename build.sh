cmake -B b  -DUSE_CUDA=off -DUSE_TENSORRT=on -DUSE_IOBINDING=on
cmake --build b
