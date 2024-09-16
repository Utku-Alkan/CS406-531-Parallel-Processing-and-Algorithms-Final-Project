# ParSpaRyser Compilation and Execution Guide

This project involves the design, implementation, and performance evaluation of a parallel computing solution to compute sparse matrix permanents using OpenMP and CUDA. 
This guide provides instructions on how to compile and run the `ParSpaRyser.cpp` and `ParSpaRyserCuda.cu` files to create their respective executables and execute them.

## Prerequisites

Before you begin, ensure that you have the following installed on your system:

- **g++**: The GNU C++ compiler
- **nvcc**: The NVIDIA CUDA Compiler

## Files in the Repository

- `ParSpaRyser.cpp`: The C++ source file.
- `ParSpaRyserCuda.cu`: The CUDA source file.
- `compile.sh`: The shell script to compile the source files.
- `README.md`: This documentation file.

## Compilation Instructions

Follow the steps below to compile the source files and generate the executables:

1. **Make the `compile.sh` script executable**:
    chmod +x compile.sh

2. **Run the `compile.sh` script**:
    ./compile.sh


## Input File Format

The input matrix file should be in `.mat` format, containing an `n x n` matrix. The structure of the file is as follows:

- **First Row:** Contains two integers:
  - The number of rows/columns of the matrix (`n`).
  - The number of non-zero elements in the matrix.
- **Subsequent Rows:** Each row corresponds to a non-zero element in the matrix and contains three values:
  - `i`: Row index of the non-zero element (integer, starting from 0 up to `n-1`).
  - `j`: Column index of the non-zero element (integer, starting from 0 up to `n-1`).
  - `val`: The value of the non-zero element (real number).

### Example .mat File

```
3 4
0 0 1.5
0 2 2.3
1 1 3.7
2 0 4.2
```

## Execution Instructions

Once you have successfully compiled the executables, you can run them using the following commands:

    ./executable_cpp matrix_file_name no_threads

    ./executable_cuda matrix_file_name no_GPUs

## Expected Output

permanent   execution_time

Example output:

2.403738e+06    5.5
