#ifndef _LCUDA_MATRIX_TYPE_H_
#define _LCUDA_MATRIX_TYPE_H_

#include "MatrixBase.h"

template <typename T> class HostMatrix : MatrixBase<T> {
public:
	HostMatrix(int width, int height) : MatrixBase(width, height) {} 
};

#endif