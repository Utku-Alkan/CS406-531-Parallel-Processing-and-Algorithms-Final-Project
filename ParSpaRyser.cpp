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
        return crs;
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
        return ccs;
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

double ParSpaRyser(int* cptrs, int* rows, double* cvals, double* matrix,
 int N, int numThreads, long long start, long long end, int numChunks, int dynChSize) {
    
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
    //long long numChunks = 4096;
    long long chunkSize = (end - start) / numChunks + 1;

    #pragma omp parallel num_threads(numThreads)
    { 
        double myX[N];
        #pragma omp for schedule(guided, dynChSize)
        for (int chunkID = 0; chunkID < numChunks; chunkID++)
        {
            
            for (int i = 0; i < N; i++) {
                myX[i] = x[i];
            }

            double myP = 0;

            int threadId = omp_get_thread_num();
            long long myStart = start + chunkID * chunkSize;
            long long myEnd = min(start + ((chunkID+1) * chunkSize), end);

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
            /*long long GrayCode_g; // init 
            long long grayXor;
            for (unsigned long long g = myStart; g <= myEnd;)
            {
                GrayCode_g = g^ (g >> 1);
                grayXor = GrayCode_g ^ gray;
                
            } */
            long long g = myStart;
            int j;

            
            long long GrayCode_g; // init 
            // long long grayXor;
            int s;
            for (;g < myEnd;g++)
            {

                GrayCode_g = g ^ (g >> 1);
                // gray = (g-1) ^ ((g-1) >> 1);
                j = __builtin_ctzll(GrayCode_g ^ gray); // REVISED (NO +1)           
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

            #pragma omp atomic
                p += myP;
        }
    }
    return (p*(4*(N%2)-2));
}

double ParSpaRyserG(int* cptrs, int* rows, double* cvals, double* matrix, int N, int numThreads, long long start, long long end, int numChunks);

double ParSpaRyserD(int* cptrs, int* rows, double* cvals, double* matrix, int N, int numThreads, long long start, long long end, int numChunks);

int main(int argc, char *argv[]){

    std::string filename = argv[1];
    int threadCount = stoi(argv[2]);

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
    double start,end;
    int numChunks,dynChSize;
    int numThreads = threadCount;
    numChunks = 8192;

    start = omp_get_wtime();
    double perm_spaRyser = ParSpaRyserD(ccs.cptrs, ccs.rows, ccs.cvals, matrix, n,numThreads, 1, (1LL << (n-1)), numChunks);
    end = omp_get_wtime();
    cout << perm_spaRyser << "\t"<<end-start<<endl;

    
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

double ParSpaRyserG(int* cptrs, int* rows, double* cvals, double* matrix,
 int N, int numThreads, long long start, long long end, int numChunks) {
    
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
    //long long numChunks = 4096;
    long long chunkSize = (end - start) / numChunks + 1;

    #pragma omp parallel num_threads(numThreads)
    { 
        double myX[N];
        #pragma omp for schedule(guided, 1)
        for (int chunkID = 0; chunkID < numChunks; chunkID++)
        {
            
            for (int i = 0; i < N; i++) {
                myX[i] = x[i];
            }

            double myP = 0;

            int threadId = omp_get_thread_num();
            long long myStart = start + chunkID * chunkSize;
            long long myEnd = min(start + ((chunkID+1) * chunkSize), end);

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
            /*long long GrayCode_g; // init 
            long long grayXor;
            for (unsigned long long g = myStart; g <= myEnd;)
            {
                GrayCode_g = g^ (g >> 1);
                grayXor = GrayCode_g ^ gray;
                
            } */
            long long g = myStart;
            int j;

            
            long long GrayCode_g; // init 
            // long long grayXor;
            int s;
            for (;g < myEnd;g++)
            {

                GrayCode_g = g ^ (g >> 1);
                // gray = (g-1) ^ ((g-1) >> 1);
                j = __builtin_ctzll(GrayCode_g ^ gray); // REVISED (NO +1)           
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

            #pragma omp atomic
                p += myP;
        }
    }
    return (p*(4*(N%2)-2));
}

double ParSpaRyserD(int* cptrs, int* rows, double* cvals, double* matrix,
 int N, int numThreads, long long start, long long end, int numChunks) {
    
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
    //long long numChunks = 4096;
    long long chunkSize = (end - start) / numChunks + 1;

    #pragma omp parallel num_threads(numThreads)
    { 
        double myX[N];
        #pragma omp for schedule(dynamic, 1)
        for (int chunkID = 0; chunkID < numChunks; chunkID++)
        {
            
            for (int i = 0; i < N; i++) {
                myX[i] = x[i];
            }

            double myP = 0;

            int threadId = omp_get_thread_num();
            long long myStart = start + chunkID * chunkSize;
            long long myEnd = min(start + ((chunkID+1) * chunkSize), end);

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
            /*long long GrayCode_g; // init 
            long long grayXor;
            for (unsigned long long g = myStart; g <= myEnd;)
            {
                GrayCode_g = g^ (g >> 1);
                grayXor = GrayCode_g ^ gray;
                
            } */
            long long g = myStart;
            int j;

            
            long long GrayCode_g; // init 
            // long long grayXor;
            int s;
            for (;g < myEnd;g++)
            {

                GrayCode_g = g ^ (g >> 1);
                // gray = (g-1) ^ ((g-1) >> 1);
                j = __builtin_ctzll(GrayCode_g ^ gray); // REVISED (NO +1)           
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

            #pragma omp atomic
                p += myP;
        }
    }
    return (p*(4*(N%2)-2));
}