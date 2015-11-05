/* Matrix multiplication: P = M * N.
 * Device code.
 */

#ifndef _MATRIXMUL_KERNEL_H_
#define _MATRIXMUL_KERNEL_H_

#include "matrixmul.h"

__global__ void
matrixMul(float* P, float* M, float* N)
{
	// thread id
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	// initialize variable to hold temporary product
	float P_temp = 0;

	// calculate temporary product
	for (int k = 0; k < WM; ++k){
		float M_element = M[ty * WM + k];
		float N_element = N[k * WN + tx];
		P_temp += M_element * N_element;
	}

	// store product
	P[ty * WN + tx] = P_temp;
}

#endif // #ifndef _MATRIXMUL_KERNEL_H_
