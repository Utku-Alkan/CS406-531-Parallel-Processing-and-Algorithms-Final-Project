#include <iostream>
#include <cmath>
#include <omp.h>
#include <time.h>
#include <fstream>
#include <sstream>
#include <stdlib.h>
#include <cstring>
#include <string>
#include <vector>
#include <bitset>
#include <algorithm>
#include <iomanip>
using namespace std;

#define CudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            std::cerr << "Fatal error: " << msg << " (" << cudaGetErrorString(__err) << ") at " << __FILE__ << ":" << __LINE__ << std::endl; \
            std::cerr << "Aborting..." << std::endl; \
            exit(1); \
        } \
    } while (0)

// Helper function to count nonzero elements in a column
int countNonZerosInColumn(const double* matrix, int n, int col) {
    int count = 0;
    for (int row = 0; row < n; ++row) {
        if (matrix[row * n + col] != 0) {
            ++count;
        }
    }
    return count;
}

// Function to sort the matrix columns based on the number of nonzero elements
void sortMatrixColumnsByNonZeros(double* matrix, int n) {
    // Vector to store pairs of (nonzero count, column index)
    std::vector<std::pair<int, int>> nonZeroCounts(n);
    
    // Count nonzero elements in each column
    for (int col = 0; col < n; ++col) {
        int count = countNonZerosInColumn(matrix, n, col);
        nonZeroCounts[col] = {count, col};
    }
    
    // Sort columns by nonzero counts (ascending order)
    std::sort(nonZeroCounts.begin(), nonZeroCounts.end());
    
    // Create a new matrix to store the sorted columns
    std::vector<double> sortedMatrix(n * n, 0);
    
    // Reorder columns according to sorted indices
    for (int newCol = 0; newCol < n; ++newCol) {
        int originalCol = nonZeroCounts[newCol].second;
        for (int row = 0; row < n; ++row) {
            sortedMatrix[row * n + newCol] = matrix[row * n + originalCol];
        }
    }
    
    // Copy sorted matrix back to the original matrix
    std::copy(sortedMatrix.begin(), sortedMatrix.end(), matrix);
}

struct CRS {
    int* rptrs;
    int* columns;
    double* rvals;
    int numRows;
    int numNonZeros;
};


CRS readCRS(const std::string& filename) {
    CRS crs;
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        exit(1);
    }

    file >> crs.numRows >> crs.numNonZeros;

    crs.rptrs = new int[crs.numRows + 1]();
    crs.columns = new int[crs.numNonZeros];
    crs.rvals = new double[crs.numNonZeros];

    int row, col;
    double value;
    int index = 0;
    int currRow=-1;
    while (file >> row >> col >> value) {

        if(currRow != row)
        {
          crs.rptrs[row+1] += crs.rptrs[row];
          currRow = row;
        }
       
        crs.rptrs[row+1]++;
        crs.columns[index] = col;
        crs.rvals[index] = value;
      
        ++index;
    }
    crs.rptrs[crs.numRows] = crs.numNonZeros;  // last element points to the end

    file.close();
    return crs;
}

struct CCS {
    int* cptrs;
    int* rows;
    double* cvals;
    int numCols;
    int numNonZeros;
};

CCS readCCS(const std::string& filename) {
    CCS ccs;
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        exit(1);
    }

    int n;
    file >> n >> ccs.numNonZeros;
    ccs.numCols = n;

    // Read the entire matrix into a double array
    std::vector<double> matrix(n * n, 0);
    int row, col;
    double value;
    while (file >> row >> col >> value) {
        matrix[row * n + col] = value;
    }
    file.close();

    // Sort the columns of the matrix
    sortMatrixColumnsByNonZeros(matrix.data(), n);

    // Allocate memory for CCS structure
    ccs.cptrs = new int[ccs.numCols + 1]();
    ccs.rows = new int[ccs.numNonZeros];
    ccs.cvals = new double[ccs.numNonZeros];

    // Convert the sorted matrix to CCS format
    int index = 0;
    for (int col = 0; col < n; ++col) {
        ccs.cptrs[col] = index;
        for (int row = 0; row < n; ++row) {
            double val = matrix[row * n + col];
            if (val != 0) {
                ccs.rows[index] = row;
                ccs.cvals[index] = val;
                ++index;
            }
        }
    }
    ccs.cptrs[n] = ccs.numNonZeros;  // last element points to the end

    return ccs;
}

void PrintMatrix(double* matrix, int size){
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            cout <<std::setw(2)<< matrix[i*size+j] << " "; 
        }
        cout << endl;
    }

    cout << endl;
}

void freeCCS(CCS& ccs);
void freeCRS(CRS& crs);

__global__ void ParSpaRyserCudaKernel(int* cptrs, int* rows, double* cvals, int N, double*p, double*x, long long start, long long end, long long chunkSize){

    double myX[39];

    for (int i = 0; i < N; i++) {
        myX[i] = x[i];
    }

    double myP = 0;

    int threadId = threadIdx.x + blockIdx.x * blockDim.x;
    long long myStart = start + threadId * chunkSize;
    long long myEnd = min(start + ((threadId+1) * chunkSize), end);

    long long myStartReal = myStart-1;
    long long gray = myStartReal ^ (myStartReal >> 1);

    long long tempGray = gray; 
    for (int i = 0; i < (N-1); i++) 
    {
        if (tempGray%2) 
        {
            for (int p = cptrs[i]; p < cptrs[i+1]; p++) 
            {
                myX[rows[p]] += cvals[p];
            }
        }
        tempGray >>=1;
    } 

    double prod = 1;
    int zeroNum = 0;
    for (int i = 0; i < N; i++)
    {
        if(myX[i])
        {
            prod *= myX[i];
        }
        else
        {
            zeroNum++;
        }
    }

    long long g = myStart;
    int j;

    
    long long GrayCode_g; // init 
    int s;


    for (;g < myEnd;g++)
    {

        GrayCode_g = g ^ (g >> 1);
        // gray = (g-1) ^ ((g-1) >> 1);
        j = __ffsll(GrayCode_g ^ gray) - 1; // REVISED (NO +1)           
        s = 2*((GrayCode_g&(1ULL<<j))>>j)-1;
        gray = GrayCode_g;




        for (int i = cptrs[j]; i < cptrs[j+1]; i++) { // REVISED (till cptrs[j+1]) 
            if (!myX[rows[i]]) {
                zeroNum--;
                myX[rows[i]] += s*cvals[i]; 
                prod *= myX[rows[i]];
            }else{
                prod /= myX[rows[i]];
                myX[rows[i]] += s*cvals[i]; 
                if (!myX[rows[i]]) {
                    zeroNum++;
                }else{
                    prod *= myX[rows[i]];
                }
            }
        }

        if(!zeroNum) {
            myP += (g%2 ? -1 : 1) * prod;
        }
    }

    p[threadId] = myP;
}

double ParSpaRyserCuda(int* cptrs, int* rows, double* cvals, double* matrix, int N, int numBlocks, int blockSize, int numNonzeros){
    cudaSetDevice(0);
    
    double p = 1;
    double x[N];
    double rowSum;

    for (int i = 0; i < N; i++) {

        rowSum = 0;

        for (int j = 0; j < N; j++) {
            rowSum += matrix[i*N + j];
        }

        x[i] = matrix[(i*N) + N - 1] - (rowSum / 2);
        p *= x[i];
    }

    double *d_cvals;
    int *d_cptrs, *d_rows;
    double *d_x, *d_p;
    double *h_p = new double[numBlocks * blockSize];

    cudaMalloc(&d_cptrs, (N+1) * sizeof(int));
    cudaMalloc(&d_rows, numNonzeros * sizeof(int));
    cudaMalloc(&d_cvals, numNonzeros * sizeof(double));
    cudaMalloc(&d_p, numBlocks * blockSize * sizeof(double));
    cudaMalloc(&d_x, N * sizeof(double));

    cudaMemcpy(d_cptrs, cptrs, (N+1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rows, rows, numNonzeros * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cvals, cvals, numNonzeros * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x, N * sizeof(double), cudaMemcpyHostToDevice);

    long long int start = 1;
    long long int end = (1LL << (N-1));
    int numThreads = numBlocks * blockSize;
    long long int chunkSize = (end - start) / numThreads + 1;

    ParSpaRyserCudaKernel<<<numBlocks, blockSize>>>(d_cptrs, d_rows, d_cvals, N, d_p, d_x, 1, (1LL << (N-1)), chunkSize);
    
    cudaDeviceSynchronize();

    cudaMemcpy(h_p, d_p, numBlocks * blockSize * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_cptrs);
    cudaFree(d_rows);
    cudaFree(d_cvals);
    cudaFree(d_x);
    cudaFree(d_p);

    for (int i = 0; i < numBlocks * blockSize; i++) {
        p += h_p[i];
    }

    delete[] h_p;

    return((4 * (N & 1) - 2) * p);
}

double ParSpaRyserCudaTwoGpu(int* cptrs, int* rows, double* cvals, double* matrix, int N, int numBlocks, int blockSize, int numNonzeros){
    //timeTaken = omp_get_wtime();
    double p = 1;
    double p1 = 0;
    double p2 = 0;
    double x[N];
    double rowSum;

    for (int i = 0; i < N; i++) {

        rowSum = 0;

        for (int j = 0; j < N; j++) {
            rowSum += matrix[i*N + j];
        }

        x[i] = matrix[(i*N) + N - 1] - (rowSum / 2);
        p *= x[i];
    }
    //double times[2];
    //double laststart=33;
    #pragma omp parallel for num_threads(2) //lastprivate(laststart)
    for(int gpuCounter = 0; gpuCounter < 2; gpuCounter++){
        cudaSetDevice(gpuCounter);   
        double *d_cvals;
        int *d_cptrs, *d_rows;
        double *d_x, *d_p;
        double *h_p = new double[numBlocks * blockSize];

        cudaMalloc(&d_cptrs, (N+1) * sizeof(int));
        cudaMalloc(&d_rows, numNonzeros * sizeof(int));
        cudaMalloc(&d_cvals, numNonzeros * sizeof(double));
        cudaMalloc(&d_p, numBlocks * blockSize * sizeof(double));
        cudaMalloc(&d_x, N * sizeof(double));

        cudaMemcpy(d_cptrs, cptrs, (N+1) * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_rows, rows, numNonzeros * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_cvals, cvals, numNonzeros * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_x, x, N * sizeof(double), cudaMemcpyHostToDevice);

        long long int start = 1;
        long long int end = (1LL << (N-1));
        int numThreads = numBlocks * blockSize;
        long long int chunkSize = (end - start) / numThreads + 1;
        /* if(omp_get_thread_num())
            timeTaken = omp_get_wtime()-timeTaken;
        double startTime = omp_get_wtime(); */
        if(gpuCounter == 0){
            ParSpaRyserCudaKernel<<<numBlocks, blockSize>>>(d_cptrs, d_rows, d_cvals, N, d_p, d_x, start, (end/2), chunkSize);
        }else{
            ParSpaRyserCudaKernel<<<numBlocks, blockSize>>>(d_cptrs, d_rows, d_cvals, N, d_p, d_x, end/2, end, chunkSize);
        }
        //int tidd = omp_get_thread_num();
        cudaDeviceSynchronize();
        //double endTime = omp_get_wtime();
        //times[tidd] = endTime - startTime;
        //laststart = omp_get_wtime();

        //printf("GPU ID: %d synchronizes in %f seconds\n", gpuCounter, endTime - startTime);
        
        cudaMemcpy(h_p, d_p, numBlocks * blockSize * sizeof(double), cudaMemcpyDeviceToHost);

        cudaFree(d_cptrs);
        cudaFree(d_rows);
        cudaFree(d_cvals);
        cudaFree(d_x);
        cudaFree(d_p);

        if(gpuCounter == 0){
            for (int i = 0; i < numBlocks * blockSize; i++) {
                p1 += h_p[i];
            }
        }else{
            for (int i = 0; i < numBlocks * blockSize; i++) {
                p2 += h_p[i];
            }
        }
        delete[] h_p;
    }
    p += (p1+p2);
    //double result = ((4 * (N & 1) - 2) * p);
    //double lastEnd = omp_get_wtime();
    //cout<<"Real 2GPU time = "<<min(times[0],times[1])+lastEnd-laststart<<endl;
    //timeTaken = omp_get_wtime() - laststart + min(times[0],times[1]) + timeTaken;
    return ((4 * (N & 1) - 2) * p);
}

double ParSpaRyserCudaFourGpu(int* cptrs, int* rows, double* cvals, double* matrix, int N, int numBlocks, int blockSize, int numNonzeros){
    //timeTaken = omp_get_wtime();
    double p = 1;
    double p1 = 0;
    double p2 = 0;
    double p3 = 0;
    double p4 = 0;
    double x[N];
    double rowSum;

    for (int i = 0; i < N; i++) {

        rowSum = 0;

        for (int j = 0; j < N; j++) {
            rowSum += matrix[i*N + j];
        }

        x[i] = matrix[(i*N) + N - 1] - (rowSum / 2);
        p *= x[i];
    }
    //double times[2];
    //double laststart=33;
    #pragma omp parallel for num_threads(4) //lastprivate(laststart)
    for(int gpuCounter = 0; gpuCounter < 4; gpuCounter++){
        cudaSetDevice(gpuCounter);   
        double *d_cvals;
        int *d_cptrs, *d_rows;
        double *d_x, *d_p;
        double *h_p = new double[numBlocks * blockSize];

        cudaMalloc(&d_cptrs, (N+1) * sizeof(int));
        cudaMalloc(&d_rows, numNonzeros * sizeof(int));
        cudaMalloc(&d_cvals, numNonzeros * sizeof(double));
        cudaMalloc(&d_p, numBlocks * blockSize * sizeof(double));
        cudaMalloc(&d_x, N * sizeof(double));

        cudaMemcpy(d_cptrs, cptrs, (N+1) * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_rows, rows, numNonzeros * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_cvals, cvals, numNonzeros * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_x, x, N * sizeof(double), cudaMemcpyHostToDevice);

        long long int start = 1;  
        long long int end = (1LL << (N-1));  
        int numThreads = numBlocks * blockSize;
        long long int chunkSize = (end - start) / numThreads + 1;
        /* if(omp_get_thread_num())
            timeTaken = omp_get_wtime()-timeTaken;*/
        double startTime = omp_get_wtime(); 
        if(gpuCounter == 0){
            ParSpaRyserCudaKernel<<<numBlocks, blockSize>>>(d_cptrs, d_rows, d_cvals, N, d_p, d_x, start, 3*end/8, chunkSize);
        }else if(gpuCounter==1){
            ParSpaRyserCudaKernel<<<numBlocks, blockSize>>>(d_cptrs, d_rows, d_cvals, N, d_p, d_x, 3*end/8, 6*end/8, chunkSize);
        }else if(gpuCounter==2){
            ParSpaRyserCudaKernel<<<numBlocks, blockSize>>>(d_cptrs, d_rows, d_cvals, N, d_p, d_x, 6*end/8, 7*end/8, chunkSize);
        }else if(gpuCounter==3){
            ParSpaRyserCudaKernel<<<numBlocks, blockSize>>>(d_cptrs, d_rows, d_cvals, N, d_p, d_x, 7*end/8, end, chunkSize);
        }
        //int tidd = omp_get_thread_num();
        cudaDeviceSynchronize();
        double endTime = omp_get_wtime();
        //times[tidd] = endTime - startTime;
        //laststart = omp_get_wtime();

        //printf("GPU ID: %d synchronizes in %f seconds\n", gpuCounter, endTime - startTime);
        
        cudaMemcpy(h_p, d_p, numBlocks * blockSize * sizeof(double), cudaMemcpyDeviceToHost);

        cudaFree(d_cptrs);
        cudaFree(d_rows);
        cudaFree(d_cvals);
        cudaFree(d_x);
        cudaFree(d_p);

        if(gpuCounter == 0){
            for (int i = 0; i < numBlocks * blockSize; i++) {
                p1 += h_p[i];
            }
        }else if(gpuCounter==1){
            for (int i = 0; i < numBlocks * blockSize; i++) {
                p2 += h_p[i];
            }
        }else if(gpuCounter==2){
            for (int i = 0; i < numBlocks * blockSize; i++) {
                p3 += h_p[i];
            }
        }else if(gpuCounter==3){
            for (int i = 0; i < numBlocks * blockSize; i++) {
                p4 += h_p[i];
            }
        }
        delete[] h_p;
    }
    p += (p1+p2+p3+p4);
    //double result = ((4 * (N & 1) - 2) * p);
    //double lastEnd = omp_get_wtime();
    //cout<<"Real 2GPU time = "<<min(times[0],times[1])+lastEnd-laststart<<endl;
    //timeTaken = omp_get_wtime() - laststart + min(times[0],times[1]) + timeTaken;
    return ((4 * (N & 1) - 2) * p);
}



int main(int argc, char *argv[]){

    std::string filename = argv[1];
    int gpuCount = stoi(argv[2]);

    if ((gpuCount <= 0) || (gpuCount==3) || (gpuCount>4))
    {
        cout<<"Invalid GPU count:"<<gpuCount<<endl;
        exit(1);
    }

    CRS crs = readCRS(filename);
    CCS ccs = readCCS(filename);

    int n,nonzeros;
    double* matrix;
    
    
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open the file: " << filename << std::endl;
        return 1;
    }
    file >> n >> nonzeros;
    matrix = new double[n*n];
    memset(matrix, 0, sizeof(double) * n * n);

    for (int i = 0; i < nonzeros; ++i) {
        int row_id, col_id;
	      double nnz_value;
        file >> row_id >> col_id >> nnz_value;
	      matrix[(row_id * n) + col_id] = nnz_value;
    }
    file.close();  


    sortMatrixColumnsByNonZeros(matrix, n);
    double start,end,perm_spaRyser_Cuda;
    int gridsize = 16384;
    int blocksize = 256;
    if(gpuCount==1)
    {
        start = omp_get_wtime();
        perm_spaRyser_Cuda = ParSpaRyserCuda(ccs.cptrs, ccs.rows, ccs.cvals, matrix, n, gridsize, blocksize, nonzeros);
        end = omp_get_wtime();
    }
    else if(gpuCount == 2)
    {
        start = omp_get_wtime();
        perm_spaRyser_Cuda = ParSpaRyserCudaTwoGpu(ccs.cptrs, ccs.rows, ccs.cvals, matrix, n, gridsize, blocksize, nonzeros);
        end = omp_get_wtime();
    }
    else if(gpuCount == 4)
    {
        start = omp_get_wtime();
        perm_spaRyser_Cuda = ParSpaRyserCudaFourGpu(ccs.cptrs, ccs.rows, ccs.cvals, matrix, n, gridsize, blocksize, nonzeros);
        end = omp_get_wtime();
    }
    
    cout << perm_spaRyser_Cuda << "\t"<<end-start<<endl;
    //printf("%f seconds with gridSize: %d blockSize: %d\n\n", end - start, gridsize, blocksize);  
        
            
        
    delete[] matrix;
    freeCRS(crs);
    freeCCS(ccs);

    return 0;
}


void freeCRS(CRS& crs) {
    delete[] crs.rptrs;
    delete[] crs.columns;
    delete[] crs.rvals;
}

void freeCCS(CCS& ccs) {
    delete[] ccs.cptrs;
    delete[] ccs.rows;
    delete[] ccs.cvals;
}