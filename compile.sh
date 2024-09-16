#!/bin/bash

# Compile ParSpaRyser.cpp
g++ -std=c++11 -fopenmp -O3 ParSpaRyser.cpp -o executable_cpp
if [ $? -eq 0 ]; then
    echo "Compiled ParSpaRyser.cpp successfully to executable_cpp"
else
    echo "Failed to compile ParSpaRyser.cpp"
    exit 1
fi

# Compile ParSpaRyserCuda.cu
nvcc -Xcompiler -fopenmp -o executable_cuda ParSpaRyserCuda.cu
if [ $? -eq 0 ]; then
    echo "Compiled ParSpaRyserCuda.cu successfully to executable_cuda"
else
    echo "Failed to compile ParSpaRyserCuda.cu"
    exit 1
fi

# chmod +x compile.sh
# ./compile.sh