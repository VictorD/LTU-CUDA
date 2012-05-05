#ifndef _LCUDA_BLANK_MATRIX_TYPE_H_
#define _LCUDA_BLANK_MATRIX_TYPE_H_

#include "ltucuda.cuh"
#include "HostMatrix.h"
#include "kernels/fill.cuh"

template <typename T>
class BlankImageMatrix : public HostMatrix<T> {
public:
	BlankImageMatrix(int width, int height) : HostMatrix(width, height) { }
	void fill(T color) {
		fillMatrix(this->devicePtr, this->pitch, this->width, this->height, color);
		exitOnError("createPaddedArray: thrust::fill");
	}

};

#endif