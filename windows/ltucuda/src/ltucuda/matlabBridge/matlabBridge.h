#ifndef _LCUDA_MATLAB_BRIDGE_H_
#define _LCUDA_MATLAB_BRIDGE_H_

#include "../ltucuda.cuh"
#ifdef _CHAR16T
#define CHAR16_T
#endif
#include <mex.h>

cudaImage cudaImageFromMX(const mxArray *mx);
cudaPaddedImage cudaPaddedImageFromStruct(const mxArray *mx);
unsigned char mlcudaGetImageDataType(const mxArray* mx);

mxArray* mxArrayFromLCudaMatrix(cudaPaddedImage padded);
mxArray* mxStructFromLCudaMatrix(cudaPaddedImage padded);

#endif