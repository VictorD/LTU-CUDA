#ifndef _LCUDA_MATLAB_BRIDGE_H_
#define _LCUDA_MATLAB_BRIDGE_H_


#include "../ltucuda.cuh"
#include <vector>
using namespace std;

#ifdef _CHAR16T
#define CHAR16_T
#endif

#include <mex.h>
#include "../morphology/morphology.cuh"



cudaImage imageFromMXArray(const mxArray *mx);
cudaImage imageFromMXStruct(const mxArray *mx);
unsigned char mlcudaGetImageDataType(const mxArray* mx);

mxArray* imageToMXArray(cudaImage image);
mxArray* imageToMXStruct(cudaImage image);

vector<morphMask> morphMaskFromMXStruct(const mxArray *mx);

#endif