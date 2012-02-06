/*
 * test_cudatypes.c
 *
 *  Created on: Feb 5, 2010
 *      Author: henmak
 */
#include <mex.h>

#include "mlcuda.h"

#define DEBUG 1
#if DEBUG
#define PRINTF(...) mexPrintf(__VA_ARGS__)
#else
#define PRINTF(...)
#endif
void mexFunction(int nlhs, mxArray *plhs[],
		int nrhs, const mxArray *prhs[])
{
	lcudaMatrix_8u matrix;
	lcudaArray_8u array;
	lcuda8u* mMatrix;
	int mm, mn;
	lcuda8u* mArray;
	int am, an;
	int i;

	if (nrhs != 2)
	{
		mxErrMsgTxt("The number of input arguments must be 2, one matrix and one array");
	}
	matrix = mlcudaMxToMatrix_8u(prhs[0]);
	array = mlcudaMxToArray_8u(prhs[1]);

	PRINTF("Matrix dimensions %d, %d\n", matrix.width, matrix.height);
	mMatrix = (lcuda8u*)mxGetData(prhs[0]);
	for (i = 0; i < matrix.width*matrix.height; ++i) {
		PRINTF("%d\t", mMatrix[i]);
		if ((1+i) % matrix.width == 0) {
			PRINTF("\n");
		}
	}
	PRINTF("Array length %d\n", array.length);
	PRINTF("Array dimensions %d, %d\n", array.width, array.height);
	mArray = (lcuda8u*)mxGetData(prhs[1]);
	for (i = 0; i < array.width*array.height; ++i) {
			PRINTF("%d\t", mArray[i]);
			if ((1+i) % array.width == 0) {
				PRINTF("\n");
			}
		}

	plhs[0] = mxCreateNumericMatrix(matrix.width, matrix.height, mxUINT8_CLASS, mxREAL);
	lcudaCpyFromMatrix_8u(matrix, (lcuda8u*)mxGetData(plhs[0]));
	plhs[1] = mxCreateNumericMatrix(array.length, 1, mxUINT8_CLASS, mxREAL);
	lcudaCpyFromArray_8u(array, (lcuda8u*)mxGetData(plhs[1]));

	lcudaFreeArray_8u(array);
	lcudaFreeMatrix_8u(matrix);

}
