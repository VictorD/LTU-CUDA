#ifndef _LCUDA_MATLAB_BRIDGE_H_
#define _LCUDA_MATLAB_BRIDGE_H_

#include "../ltucuda.cuh"
#ifdef _CHAR16T
#define CHAR16_T
#endif
#include <mex.h>

cudaImage imageFromMXArray(const mxArray *mx);
cudaImage imageFromMXStruct(const mxArray *mx);
unsigned char mlcudaGetImageDataType(const mxArray* mx);

mxArray* imageToMXArray(cudaImage image);
mxArray* imageToMXStruct(cudaImage image);

#endif