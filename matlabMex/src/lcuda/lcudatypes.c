/**
 * \file lcudatypes.c
 * \ingroup lcudatypes
 *
 * \author Henrik Mäkitaavola
 */


#include "lcudatypes.h"
#include <stdio.h>

lcudaMatrix_8u lcudaAllocMatrix_8u(int width, int height)
{
	lcudaMatrix_8u matrix;

	matrix.data = nppiMalloc_8u_C1(width, height, (int*)&matrix.pitch);

	matrix.width = width;
	matrix.height = height;

	return matrix;
}

void lcudaFreeMatrix_8u(lcudaMatrix_8u matrix) {
	nppiFree(matrix.data);
}

void lcudaCpyToMatrix_8u(lcuda8u *data, lcudaMatrix_8u matrix) {
	cudaMemcpy2D( matrix.data, matrix.pitch, data, matrix.width,
			matrix.width, matrix.height, cudaMemcpyHostToDevice);
}

void lcudaCpyFromMatrix_8u(lcudaMatrix_8u matrix, lcuda8u *data) {
	cudaMemcpy2D(data, matrix.width,  matrix.data, matrix.pitch,
			matrix.width, matrix.height, cudaMemcpyDeviceToHost);
}

lcudaMatrix_8u lcudaCloneMatrix_8u(lcudaMatrix_8u matrix) {
	lcudaMatrix_8u newMatrix = lcudaAllocMatrix_8u(matrix.width, matrix.height);
	NppiSize size = {matrix.width, matrix.height};
	nppiCopy_8u_C1R(matrix.data, matrix.pitch, newMatrix.data, newMatrix.pitch, size);
	return newMatrix;
}



lcudaArray_8u lcudaAllocArray_8u(int length) {
	lcudaArray_8u array;
	array.length = length;
	cudaMalloc((void **)&array.data, length * sizeof(lcuda8u));

	return array;
}

void lcudaCpyToArray_8u(lcuda8u *data, lcudaArray_8u array) {
	cudaMemcpy(array.data, data, array.length, cudaMemcpyHostToDevice);
}

void lcudaCpyFromArray_8u(lcudaArray_8u array, lcuda8u *data) {
	cudaMemcpy(data, array.data, array.length, cudaMemcpyDeviceToHost);
}

void lcudaFreeArray_8u(lcudaArray_8u array) {
	cudaFree(array.data);
}

