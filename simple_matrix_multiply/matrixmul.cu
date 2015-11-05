/*
 * Copyright 1993-2006 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO USER:
 *
 * This source code is subject to NVIDIA ownership rights under U.S. and
 * international Copyright laws.
 *
 * This software and the information contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a Non-Disclosure Agreement.  Any reproduction or
 * disclosure to any third party without the express written consent of
 * NVIDIA is prohibited.
 *
 * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR
 * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
 * OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
 * OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE
 * OR PERFORMANCE OF THIS SOURCE CODE.
 *
 * U.S. Government End Users.  This source code is a "commercial item" as
 * that term is defined at 48 C.F.R. 2.101 (OCT 1995), consisting  of
 * "commercial computer software" and "commercial computer software
 * documentation" as such terms are used in 48 C.F.R. 12.212 (SEPT 1995)
 * and is provided to the U.S. Government only as a commercial end item.
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the
 * source code with only those rights set forth herein.
 */

/* Matrix multiplication: C = A * B.
 * Host code.

 * Modified: Naga Kandasamy
 * Date: 11/3/2015
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

/* includes, kernels. */
#include "matrixmul_kernel.cu"


extern "C" void computeGold(float*, const float*, const float*, unsigned int, unsigned int, unsigned int);
Matrix AllocateDeviceMatrix(const Matrix);
Matrix AllocateMatrix(int, int, int);
void CopyToDeviceMatrix(Matrix, const Matrix);
void CopyFromDeviceMatrix(Matrix, const Matrix);
void MatrixMulOnDevice(const Matrix, const Matrix, Matrix);
int checkResults(float *, float *, int, float);

int 
main(int argc, char** argv) {
    Matrix  M, N, P;                /* Matrices for the program. */

	/* Number of elements in the solution matrix. Assuming square matrices, */
	unsigned int size_elements = WP * HP;
	srand(2012);
    M  = AllocateMatrix(MATRIX_SIZE, MATRIX_SIZE, 1);   /* Generate the matrices. */
    N  = AllocateMatrix(MATRIX_SIZE, MATRIX_SIZE, 1);
    P  = AllocateMatrix(MATRIX_SIZE, MATRIX_SIZE, 0);

	MatrixMulOnDevice(M, N, P);         	/* Multiply matrices on the device. */
    
    /* compute the matrix multiplication on the CPU for comparison. */
	Matrix reference = AllocateMatrix(MATRIX_SIZE, MATRIX_SIZE, 0);
	computeGold(reference.elements, M.elements, N.elements, HM, WM, WN);
		
	/* Check if the device result is equivalent to the expected solution. */
	int status = checkResults(reference.elements, P.elements, size_elements, 0.0001f);
	printf("Test %s\n", (1 == status) ? "PASSED" : "FAILED");
	
    /* Free host matrices. */
	free(M.elements); M.elements = NULL;
	free(N.elements); N.elements = NULL;
	free(P.elements); P.elements = NULL;
	return 0;
}

void 
MatrixMulOnDevice(const Matrix M, const Matrix N, Matrix P)
{
	/* Allocate device memory. */
	Matrix d_M = AllocateDeviceMatrix(M);
	Matrix d_N = AllocateDeviceMatrix(N);
	Matrix d_P = AllocateDeviceMatrix(P);

	/* Copy matrices to device memory. */
	CopyToDeviceMatrix(d_M, M);
	CopyToDeviceMatrix(d_N, N);

	/* Set up execution grid. */
	dim3 threads(MATRIX_SIZE, MATRIX_SIZE);
	dim3 grid(d_M.width/threads.x, d_N.height/threads.y);

	/* Launch kernel. */
	matrixMul<<<grid, threads>>>(d_P.elements, d_M.elements, d_N.elements);

	/* Check if kernel execution generated an error. */
	cudaError_t err = cudaGetLastError();
	if(cudaSuccess != err) {
		fprintf(stderr, "Kernel execution failed: %s.\n", cudaGetErrorString(err));
		exit(-1);
	}

	/* Copy result from device to host. */
	CopyFromDeviceMatrix(P, d_P);

	/* Clean up memory on the GPU. */
	cudaFree(d_M.elements);
	cudaFree(d_N.elements);
	cudaFree(d_P.elements);
}

Matrix 
AllocateDeviceMatrix(const Matrix M)    /* Allocate matrix on device. */
{
	Matrix Mdevice = M;
	int size = M.width * M.height * sizeof(float);
	cudaMalloc((void**)&Mdevice.elements, size);
	return Mdevice;
}

/* Allocate a matrix of dimensions height*width
   If init == 0, initialize to all zeroes.  
   If init == 1, perform random initialization.
   */
Matrix 
AllocateMatrix(int height, int width, int init)
{
	Matrix M;
	M.width = M.pitch = width; M.height = height;
	int size = M.width * M.height;
	M.elements = (float*) malloc(size*sizeof(float));

	for(unsigned int i = 0; i < M.height * M.width; i++){
		M.elements[i] = (init == 0) ? (0.0f) : (rand()/(float)RAND_MAX);
	}

	return M;
}	

void 
CopyToDeviceMatrix(Matrix Mdevice, const Matrix Mhost)      /* Copy to device. */
{
	int size = Mhost.width * Mhost.height * sizeof(float);
	Mdevice.height = Mhost.height;
	Mdevice.width = Mhost.width;
	Mdevice.pitch = Mhost.pitch;
	cudaMemcpy(Mdevice.elements, Mhost.elements, size, cudaMemcpyHostToDevice);
}

void 
CopyFromDeviceMatrix(Matrix Mhost, const Matrix Mdevice)    /* Copy to host. */
{
	int size = Mdevice.width * Mdevice.height * sizeof(float);
	cudaMemcpy(Mhost.elements, Mdevice.elements, size, cudaMemcpyDeviceToHost);
}

int 
checkResults(float *reference, float *gpu_result, int num_elements, float threshold)
{
    int checkMark = 1;
    for(int i = 0; i < num_elements; i++)
        if(fabsf((reference[i] - gpu_result[i])/reference[i]) > threshold){
            checkMark = 0;
            break;
        }

    return checkMark;
}
