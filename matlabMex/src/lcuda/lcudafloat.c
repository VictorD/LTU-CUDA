/**
 * \file lcudatypes.c
 * \ingroup lcudatypes
 *
 * \author Henrik Mäkitaavola
 */


#include "lcudafloat.h"
#include <mex.h>
lcudaMatrix lcudaAllocMatrix(int width, int height)
{
	lcudaMatrix matrix;
    cudaMallocPitch((void **)&matrix.data, (size_t*)&matrix.pitch, width * sizeof(lcudaFloat), height);
	matrix.width = width;
	matrix.height = height;


	return matrix;
}

void lcudaFreeMatrix(lcudaMatrix matrix) {
	nppiFree(matrix.data);
}

void lcudaCpyToMatrix(lcudaFloat *data, lcudaMatrix matrix) {
	cudaMemcpy2D(matrix.data, matrix.pitch, data, matrix.width * sizeof(lcudaFloat),
			matrix.width * sizeof(lcudaFloat), matrix.height, cudaMemcpyHostToDevice);
}

void lcudaCpyFromMatrix(lcudaMatrix matrix, lcudaFloat *data) {
	cudaMemcpy2D(data, matrix.width * sizeof(lcudaFloat),  matrix.data, matrix.pitch,
			matrix.width * sizeof(lcudaFloat), matrix.height, cudaMemcpyDeviceToHost);
}

lcudaMatrix lcudaCloneMatrix(lcudaMatrix matrix) {
	lcudaMatrix newMatrix = lcudaAllocMatrix(matrix.width, matrix.height);
	NppiSize size = {matrix.width, matrix.height};
	nppiCopy_32f_C1R(matrix.data, matrix.pitch, newMatrix.data, newMatrix.pitch, size);
	return newMatrix;
}



lcudaArray lcudaAllocArray(int length) {
	lcudaArray array;
	array.length = length;
	cudaMalloc((void **)&array.data, length * sizeof(lcudaFloat));

	return array;
}

void lcudaCpyToArray(lcudaFloat *data, lcudaArray array) {
	cudaMemcpy(array.data, data, array.length * sizeof(lcudaFloat), cudaMemcpyHostToDevice);
}

void lcudaCpyFromArray(lcudaArray array, lcudaFloat *data) {
	cudaMemcpy(data, array.data, array.length * sizeof(lcudaFloat), cudaMemcpyDeviceToHost);
}

void lcudaFreeArray(lcudaArray array) {
	cudaFree(array.data);
}

