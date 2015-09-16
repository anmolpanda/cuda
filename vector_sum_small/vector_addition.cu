/* Host side code that calls a GPU kernel to perform vector addition on the GPU using a single thread block. 
	We restrict the size of the vector to be up to 512 elements which is the maximum thread block size on this 
	GPU.

	Author: Naga Kandasamy
	Date last modified: 10/20/2014
*/


// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>

// includes, project
#include <cutil.h>

// includes, kernels
#include <vector_addition_kernel.cu>

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void run_test(void);
void compute_on_device(float *, float *, float *, int);
extern "C" void compute_gold( float *, float *, float *, int);

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int 
main( int argc, char** argv) 
{
	run_test();
	return 0;
}

////////////////////////////////////////////////////////////////////////////////
//! Perform vector addition on the CPU and the GPU and compare results for correctness
////////////////////////////////////////////////////////////////////////////////
void run_test(void) {
	int num_elements = NUM_ELEMENTS;
	float diff;
	int i; 

	// Obtain the vector length
	int vector_length = sizeof(float) * num_elements;

    // Allocate memory on the CPU for the input vectors A and B, and the output vector C
	float *A = (float *)malloc(vector_length);
	float *B = (float *)malloc(vector_length);
	float *gold_result = (float *)malloc(vector_length); // The result vector computed on the CPU
	float *gpu_result = (float *)malloc(vector_length); // The result vector computed on the GPU
	
	// Randomly generate input data. Initialize the input data to be integer values between 0 and 100 
	for(i = 0; i < num_elements; i++){
		A[i] = floorf(100*(rand()/(float)RAND_MAX));
     	B[i] = floorf(100*(rand()/(float)RAND_MAX));
	}

	// Compute the reference solution on the CPU
	compute_gold(A, B, gold_result, num_elements);
    
	// Compute the result vector on the GPU 
	compute_on_device(A, B, gpu_result, num_elements);

	// Compute the differences between the CPU and GPU results
	diff = 0.0;
   for(i = 0; i < num_elements; i++)
		diff = diff + abs(gold_result[i] - gpu_result[i]);

	printf("Difference between the CPU and GPU result: %f. \n", diff);
   
	// cleanup memory
	free(A);
	free(B);
	free(gold_result);
	free(gpu_result);
	
	return;
}

// Transfer vectors A and B from the CPU to the GPU, setup grid and thread dimentions, excute kernel function, and copy result vector back to the CPU
void compute_on_device(float *A_on_host, float *B_on_host, float *gpu_result, int num_elements){
	float *A_on_device = NULL;
	float *B_on_device = NULL;
	float *C_on_device = NULL; 

	// Allocate space on the GPU for vectors A and B, and copy the contents of the vectors to the GPU
	cudaMalloc((void**)&A_on_device, num_elements*sizeof(float));
	cudaMemcpy(A_on_device, A_on_host, num_elements*sizeof(float), cudaMemcpyHostToDevice);

	cudaMalloc((void**)&B_on_device, num_elements*sizeof(float));
	cudaMemcpy(B_on_device, B_on_host, num_elements*sizeof(float), cudaMemcpyHostToDevice);

	// Allocate space for the result vector on the GPU
	cudaMalloc((void**)&C_on_device, num_elements*sizeof(float));
	
 	// Set up the execution grid on the GPU 
	dim3 thread_block(num_elements, 1, 1); // Set the number of threads in the thread block
	dim3 grid(1,1);
	vector_addition_kernel<<<grid, thread_block>>>(A_on_device, B_on_device, C_on_device, num_elements); // Launch the kernel with a single thread block

	// Copy the result vector back from the GPU and store 
	cudaMemcpy(gpu_result, C_on_device, num_elements*sizeof(float), cudaMemcpyDeviceToHost);
	
	// Free memory on the GPU
	cudaFree(A_on_device);
	cudaFree(B_on_device);
	cudaFree(C_on_device);
}
     
