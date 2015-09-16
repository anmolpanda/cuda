#ifndef _VECTOR_ADDITION_KERNEL_H_
#define _VECTOR_ADDITION_KERNEL_H_

// For simplicity, we're fixing the problem size to 512 elements.
#define NUM_ELEMENTS 512


__global__ void vector_addition_kernel(float *A, float *B, float *C, int num_elements){
 
	int thread_id = threadIdx.x; // Obtain the index of the thread within the thread block

	if(thread_id >= num_elements) 
		return; 
	
	C[thread_id] = A[thread_id] + B[thread_id];
	return; 
}

#endif // #ifndef _VECTOR_ADDITION_KERNEL_H
