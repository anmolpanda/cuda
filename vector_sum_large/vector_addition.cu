/* This code illustrates the use of the GPU to perform vector addition on arbirarily large vectors. 
    Author: Naga Kandasamy, 10/24/2015
*/  
  
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <string.h>
#include <math.h>
#include <float.h>
  
// includes, kernels
#include "vector_addition_kernel.cu"
  
#define THREAD_BLOCK_SIZE 128
#define NUM_THREAD_BLOCKS 240
  
////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void run_test (int);
void compute_on_device (float *, float *, float *, int);
void check_for_error (char *);
extern "C" void compute_gold (float *, float *, float *, int);

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int 
main (int argc, char **argv) 
{
    if (argc != 2){
        printf ("Usage: vector_addition <num elements> \n");
        exit (0);
    }

    int num_elements = atoi (argv[1]);
    run_test (num_elements);
	
    return 0;
}

void 
run_test (int num_elements)                 /* Perform vector addition on the CPU and the GPU. */
{
    float diff;
    int i;
		 			 
    /* Allocate memory on the CPU for the input vectors A and B, and the output vector C. */
     int vector_length = sizeof (float) * num_elements;
     float *A = (float *) malloc (vector_length);
     float *B = (float *) malloc (vector_length);
     float *gold_result = (float *) malloc (vector_length);	/* The result vector computed on the CPU. */
     float *gpu_result = (float *) malloc (vector_length);	/* The result vector computed on the GPU. */
			 
     /* Randomly generate input data. Initialize the input data to be integer values between 0 and 100. */ 
     for (i = 0; i < num_elements; i++){
         A[i] = floorf (100 * (rand () / (float) RAND_MAX));
         B[i] = floorf (100 * (rand () / (float) RAND_MAX));
     } 
				
     /* Compute the reference solution on the CPU. */
     printf ("Adding vectors on the CPU \n \n");
     compute_gold (A, B, gold_result, num_elements);
					 	  
     /* Compute the result vector on the GPU. */ 
     compute_on_device (A, B, gpu_result, num_elements);
  
     /* Compute the differences between the CPU and GPU results. */
     diff = 0.0;
     for (i = 0; i < num_elements; i++)
         diff += abs (gold_result[i] - gpu_result[i]);
	
     printf ("Difference between the CPU and GPU result: %f. \n", diff);
  
		  
     free (A); free (B);
     free (gold_result); free (gpu_result);
	
     return;
}


/* Host side code. Transfer vectors A and B from the CPU to the GPU, setup grid and thread dimensions, 
   execute the kernel function, and copy the result vector back to the CPU */
void 
compute_on_device (float *A_on_host, float *B_on_host, float *gpu_result, int num_elements)
{
    float *A_on_device = NULL;
    float *B_on_device = NULL;
    float *C_on_device = NULL;
	
    /* Allocate space on the GPU for vectors A and B, and copy the contents of the vectors to the GPU. */
    cudaMalloc ((void **) &A_on_device, num_elements * sizeof (float));
    cudaMemcpy (A_on_device, A_on_host, num_elements * sizeof (float), cudaMemcpyHostToDevice);
	
    cudaMalloc ((void **) &B_on_device, num_elements * sizeof (float));
    cudaMemcpy (B_on_device, B_on_host, num_elements * sizeof (float), cudaMemcpyHostToDevice);
    
    /* Allocate space for the result vector on the GPU. */
    cudaMalloc ((void **) &C_on_device, num_elements * sizeof (float));
  
    /* Set up the execution grid on the GPU. */
    int num_thread_blocks = NUM_THREAD_BLOCKS;
    dim3 thread_block (THREAD_BLOCK_SIZE, 1, 1);	// Set the number of threads in the thread block
    printf ("Setting up a (%d x 1) execution grid. \n", num_thread_blocks);
    dim3 grid (num_thread_blocks, 1);
  
	
    printf ("Adding vectors on the GPU. \n");
  
    /* Launch the kernel with multiple thread blocks. The kernel call is non-blocking. */
    vector_addition_kernel <<< grid, thread_block >>> (A_on_device, B_on_device, C_on_device, num_elements);	 
    cudaThreadSynchronize (); // Force the CPU to bolck here and wait for the GPU to complete
    check_for_error ("KERNEL FAILURE");
  
    // Copy the result vector back from the GPU and store 
    cudaMemcpy (gpu_result, C_on_device, num_elements * sizeof (float), cudaMemcpyDeviceToHost);
  
		  
    cudaFree (A_on_device); cudaFree (B_on_device); cudaFree (C_on_device);
} 

void 
check_for_error (char *msg)                 /* Checks for errors when executing the kernel. */
{
    cudaError_t err = cudaGetLastError ();
    if (cudaSuccess != err){
        printf ("CUDA ERROR: %s (%s). \n", msg, cudaGetErrorString (err));
        exit (EXIT_FAILURE);
    }
}


