#ifndef _CONVOLUTION_KERNEL_H_
#define _CONVOLUTION_KERNEL_H_

// The kernel is stored in GPU  global memory in this kernel implementation 
__global__ void convolution_kernel_v1(float *N, float *result, float *kernel, int num_elements, int kernel_width){
		  int i = blockIdx.x * blockDim.x + threadIdx.x; // Obtain the index of the thread within the grid
		  float sum = 0.0;
		  int j;

		  if(i >= num_elements)
					 return;

		  int N_start_point = i - (kernel_width/2); 
		  for(j = 0; j < kernel_width; j++){
					 if((j + N_start_point >= 0) && (j + N_start_point < num_elements))
								sum += N[j + N_start_point]*kernel[j];
		  }
		  result[i] = sum;
}

// The kernel is stored in GPU constant memory in kernel_c
__global__ void convolution_kernel_v2(float *N, float *result, int num_elements, int kernel_width){
		  int i = blockIdx.x * blockDim.x + threadIdx.x; // Obtain the index of the thread within the grid
		  float sum = 0.0;
		  int j;

		  if(i >= num_elements)
					 return;

		  int N_start_point = i - (kernel_width/2); 
		  for(j = 0; j < kernel_width; j++){
					 if((j + N_start_point >= 0) && (j + N_start_point < num_elements))
								sum += N[j + N_start_point]*kernel_c[j];
		  }
		  result[i] = sum;
}

// Tiled convolution kernel. The kernel is stored in GPU constant memory in kernel_c
__global__ void convolution_kernel_tiled(float *N, float *result, int num_elements, int kernel_width){
		  __shared__ float N_s[THREAD_BLOCK_SIZE + MAX_KERNEL_WIDTH - 1];
		  
		  int i = blockIdx.x * blockDim.x + threadIdx.x; // Obtain the index of the thread within the grid

		  int half_width = kernel_width/2;
					 
		  // Load the left halo elements from the previous tile. The number of halo elements will be half_width
		  int left_halo_index = (blockIdx.x - 1)*blockDim.x + threadIdx.x;
		  if(threadIdx.x >= (blockDim.x - half_width)){
					 N_s[threadIdx.x - (blockDim.x - half_width)] = (left_halo_index < 0) ? 0 : N[left_halo_index];
		  }
	
		  // Load the center elements for the tile
		  if(i < num_elements)
					 N_s[half_width + threadIdx.x] = N[i];
		  else 
					 N_s[half_width + threadIdx.x] = 0.0;

		  // Load the right halo elements from the next tile. The number of halo elements will again be half_width
		  int right_halo_index = (blockIdx.x + 1)*blockDim.x + threadIdx.x;
		  if(threadIdx.x < half_width){
					 N_s[threadIdx.x + (blockDim.x + half_width)] = (right_halo_index >= num_elements) ? 0 : N[right_halo_index];
		  }
		  __syncthreads();

		  float sum = 0.0;
		  int j;
		  for(j = 0; j < kernel_width; j++)
					 sum += N_s[j + threadIdx.x]*kernel_c[j];
		  
		  result[i] = sum;
}




#endif // #ifndef _CONVOLUTION_KERNEL_H_
