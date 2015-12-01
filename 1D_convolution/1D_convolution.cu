/* 
	GPU code for 1D convolution.
	Author: N. Kandasamy
	Date created: 06/02/2013
	Date last modified: 11/10/2015
*/

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <string.h>
#include <math.h>
#include <float.h>

#define THREAD_BLOCK_SIZE 512
#define MAX_KERNEL_WIDTH 15 // Limit on the maximum width of the kernel
#define KERNEL_WIDTH 7 // The actual width of the kernel. The width is usually an odd number

/* Includes, kernels. */
__constant__ float kernel_c[MAX_KERNEL_WIDTH]; // Allocation for the kernel in GPU constant memory
#include "1D_convolution_kernel.cu"

void run_test(int);
void compute_on_device(float *, float *, float *, int, int);
void check_for_error(char *);
void compute_gold( float *, float *, float *, int, int);
void print_result(float *, int);


int 
main( int argc, char** argv) 
{
	if(argc != 2){
		printf("Usage: 1D_convolution <num elements> \n");
		exit(0);	
	}
	int num_elements = atoi(argv[1]);
	run_test(num_elements);
	return 0;
}

void 
run_test(int num_elements) 
{
	float diff;
	int i; 

	// Obtain the vector length
	int vector_length = sizeof(float) * num_elements;

    // Allocate memory on the CPU for the input vector N and the convolved output vectors
	float *N = (float *)malloc(vector_length);
	float *gold_result = (float *)malloc(vector_length); // The result vector computed on the CPU
	float *gpu_result = (float *)malloc(vector_length); // The result vector computed on the GPU
	
	// Randomly generate input data. Initialize the input data to be integer values between 0 and 10 
	for(i = 0; i < num_elements; i++){
		N[i] = floorf(10*(rand()/(float)RAND_MAX));
	}

	// Generate the convolution mask and initialize it 
	int kernel_width = KERNEL_WIDTH;
	float *kernel = (float *)malloc(sizeof(float)*kernel_width);
	for(i = 0; i < kernel_width; i++){
			  kernel[i] = floorf(5*(rand()/(float)RAND_MAX));
	}

	printf("Calculating convolution result on the CPU. \n");
	compute_gold(N, gold_result, kernel, num_elements, kernel_width);

	// Compute the result vector on the GPU 
	compute_on_device(N, gpu_result, kernel, num_elements, kernel_width);

	// Compute the differences between the CPU and GPU results
	diff = 0.0;
	for(i = 0; i < num_elements; i++)
		diff = diff + fabs(gold_result[i] - gpu_result[i]);

	printf("Difference between the CPU and GPU result: %f. \n", diff);
   
	// cleanup memory
	free(N);
	free(kernel);
	free(gold_result);
	free(gpu_result);
	
	return;
}

// Calculate the convolution on the CPU
void 
compute_gold(float *N, float *result, float *kernel, int num_elements, int kernel_width)
{	  
    int i, j;
    float sum;

    for(i = 0; i < num_elements; i++){
        sum = 0.0;
        int N_start_point = i - (kernel_width/2);
        for(j = 0; j < kernel_width; j++){
            if((N_start_point + j >= 0) && (N_start_point + j < num_elements))
                sum += N[N_start_point + j]*kernel[j];
        }
        result[i] = sum;
    }
} 


// Convolve on the GPU
void 
compute_on_device(float *N_on_host, float *gpu_result, float *kernel_on_host, int num_elements, int kernel_width)
{
    float *N_on_device = NULL;
	float *kernel_on_device = NULL;
	float *result_on_device = NULL;

	// Allocate space on the GPU for vector N and copy the contents of the vector to the GPU
	cudaMalloc((void**)&N_on_device, num_elements*sizeof(float));
	cudaMemcpy(N_on_device, N_on_host, num_elements*sizeof(float), cudaMemcpyHostToDevice);

	// Allocate space on the GPU global mmeory for the kernel and copy over
	cudaMalloc((void**)&kernel_on_device, kernel_width*sizeof(float));
	cudaMemcpy(kernel_on_device, kernel_on_host, kernel_width*sizeof(float), cudaMemcpyHostToDevice);

	// Allocate space for the result vector on the GPU
	cudaMalloc((void**)&result_on_device, num_elements*sizeof(float));
	
 	// Set up the execution grid on the GPU
	dim3 thread_block(THREAD_BLOCK_SIZE, 1, 1); // Set the number of threads in the thread block
	// Determine the number of thread blocks needed. 
	// NOTE: I use a 1D grid but that won't work when the number of elements is very large due to limits on grid dimensions
   // For very large numbers of elements, use a 2D grid	
	int num_thread_blocks = ceil((float)num_elements/(float)THREAD_BLOCK_SIZE); 	
	printf("Setting up a (%d x 1) execution grid. \n", num_thread_blocks);
	dim3 grid(num_thread_blocks, 1);
	
	printf("Performing convolution on the GPU. The kernel is stored in global memory. \n");
	
	convolution_kernel_v1<<<grid, thread_block>>>(N_on_device, result_on_device, kernel_on_device, num_elements, kernel_width);
	cudaThreadSynchronize();
	
	check_for_error("KERNEL FAILURE");

	printf("Performing convolution on the GPU. The kernel is stored in constant memory. \n");
	// We copy the mask to GPU constant memory in an attempt to improve the performance
	cudaMemcpyToSymbol(kernel_c, kernel_on_host, kernel_width*sizeof(float)); 	
	
	convolution_kernel_v2<<<grid, thread_block>>>(N_on_device, result_on_device, num_elements, kernel_width);
	cudaThreadSynchronize();
	
	check_for_error("KERNEL FAILURE");


	printf("Performing tiled convolution on the GPU. The kernel is stored in constant memory. \n");
	cudaMemcpyToSymbol(kernel_c, kernel_on_host, kernel_width*sizeof(float)); 	
	
	convolution_kernel_tiled<<<grid, thread_block>>>(N_on_device, result_on_device, num_elements, kernel_width);
	cudaThreadSynchronize();
	
	check_for_error("KERNEL FAILURE");


	// Copy the convolved vector back from the GPU and store 
	cudaMemcpy(gpu_result, result_on_device, num_elements*sizeof(float), cudaMemcpyDeviceToHost);
	
	// Free memory on the GPU
	cudaFree(N_on_device);
	cudaFree(result_on_device);
	cudaFree(kernel_on_device);
}
  
void 
check_for_error(char *msg){
	cudaError_t err = cudaGetLastError();
	if(cudaSuccess != err){
		printf("CUDA ERROR: %s (%s). \n", msg, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}

// Helper function to print the result 
void 
print_result(float *result, int num_elements)
{
    for(int i = 0; i < num_elements; i++)
        printf("%f ", result[i]);
    printf("\n");
}
