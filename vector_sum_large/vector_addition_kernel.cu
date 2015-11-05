#ifndef _VECTOR_ADDITION_KERNEL_H_
#define _VECTOR_ADDITION_KERNEL_H_

__global__ void vector_addition_kernel(float *A, float *B, float *C, int num_elements){
	
    /* Obtain the index of the thread within the thread block, corresponding to the tile index. */
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x; 	
    int stride = blockDim.x * gridDim.x;                        /* Compute the stride length. */
		  
    while(thread_id < num_elements){
        C[thread_id] = A[thread_id] + B[thread_id];
        thread_id += stride;
    }
		  
    return; 
}



#endif // #ifndef _VECTOR_ADDITION_KERNEL_H
