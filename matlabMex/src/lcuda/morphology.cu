/*
 * lcudamorph.c
 *
 *  Created on: Feb 6, 2010
 *      Author: henmak
 *      Modified by: vicdan-8 November 2011
 */

#include <cuda.h>
#include <memory>
#include <iostream>
#include <cassert>
#include <stdio.h>
#include <string.h>

#include <thrust/device_ptr.h>
#include <thrust/fill.h>

#include "morphology.cuh"
#include "sharedmem.cuh"
#include "mlcuda.h"


// 1.0f uint = 1065353216
#define FLOAT_MAX 1 

enum vhgwDirection {
    HORIZONTAL,
    VERTICAL
};

enum morphOperation {
    ERODE,
    DILATE
};

template<class dataType, morphOperation MOP>
__device__ inline dataType minMax(const dataType a, const dataType b) {
    return (MOP == ERODE) ? min(a,b) : max(a,b);
}

// Simple macro for minMax
#define MINMAX(op,dst, newVal) dst = minMax<dataType, op>((dataType)dst, (dataType)newVal);


texture<float, 2, cudaReadModeElementType> srcTex;

/* 
 * Generally one does not include .cu files like this, but here it allows the templated functions to be called,
 * without having to construct wrappers for all the various combinations of dataType/morphOp.
 *
 * Note: Drawback of this is that morphology.cu has to be updated for changes in the below includes to be reflected in the object file.
 */
// 3x3
#include "cu/lcuda3x3.cu"

// vHGW (1D SE)
#include "cu/lcudavhgw.cu"

// Generic
#include "cu/lcudaGenericKernel.cu"

template <class dataType, morphOperation MOP>
static void _dispatchMorphOp(const dataType* pSrc, Npp32s nSrcStep, dataType* pDst, Npp32s nDstStep, NppiSize srcROI, const Npp8u * pMask, 
                                   const float* maskHeight, NppiSize maskSize, NppiPoint anchor, NppiSize borderSize, char isFlat, int seBinary) {

    int offsetX = (borderSize.width)/4 - (maskSize.width)/2;
    int offsetY = (borderSize.height)/4 - (maskSize.height)/2;

    // Steps are specified in bytes. Pointer aritmetic below requires we compensate for size of <dataType>.
    nSrcStep /= sizeof(dataType);
    nDstStep /= sizeof(dataType);

    int srcBorderOffset = (nSrcStep * offsetY + offsetX);

    char processed = 0;
    if (isFlat) {
        // Custom fast 3x3
        if (maskSize.height == 3 && maskSize.width == 3) {
            _global3x3<dataType, MOP>(pSrc + srcBorderOffset, nSrcStep, pDst, nDstStep, srcROI, pMask, seBinary);
            processed = 1;
        } 
        // Vertical vHGW
        else if (maskSize.height == 1) {
            PRINTF("Vertical Erosion: SE Size (%dx%d)\n", maskSize.width, maskSize.height); 
            PRINTF("Erosion: Offset (%d,%d)\n", offsetX, offsetY);

            _globalVHGW<dataType, MOP, VERTICAL>(pSrc + srcBorderOffset, nSrcStep, pDst, nDstStep, srcROI, maskSize.width, borderSize);
            processed = 1;
        // Horizontal vHGW
        } else if (maskSize.width == 1) {
            PRINTF("Horizontal Erosion: SE Size (%dx%d)\n", maskSize.width, maskSize.height); 
            PRINTF("Erosion: Offset (%d,%d)\n", offsetX, offsetY);

            _globalVHGW<dataType, MOP, HORIZONTAL>(pSrc + srcBorderOffset, nSrcStep, pDst, nDstStep, srcROI, maskSize.height,  borderSize);
            processed = 1;
        }
    }

    // Non-flat and other arbitrary SE
    if (!processed) {
        PRINTF("Generic!\n");
        _globalGeneric<dataType, MOP>(pSrc + srcBorderOffset, nSrcStep, pDst, nDstStep, srcROI, pMask, maskSize, maskHeight, borderSize, anchor);
    }

    // Block until async kernels calls have been executed.
    cudaThreadSynchronize();
}

/* 
 * Public functions to perform erosion or dilation.
 */ 
void performDilation(const lcudaFloat * pSrc, Npp32s nSrcStep, lcudaFloat * pDst, Npp32s nDstStep, NppiSize srcROI, 
                    const Npp8u * pMask, const float* maskHeight, NppiSize maskSize, NppiPoint anchor, 
                        NppiSize borderSize, char isFlat, int seBinary) {
    _dispatchMorphOp<lcudaFloat, DILATE>(pSrc, nSrcStep, pDst, nDstStep, srcROI, pMask, maskHeight, maskSize, anchor, borderSize, isFlat, seBinary);
}

void performErosion(const lcudaFloat * pSrc, Npp32s nSrcStep, lcudaFloat * pDst, Npp32s nDstStep, NppiSize srcROI, 
                    const Npp8u * pMask, const float* maskHeight, NppiSize maskSize, NppiPoint anchor, 
                        NppiSize borderSize, char isFlat, int seBinary) {
    _dispatchMorphOp<lcudaFloat, ERODE>(pSrc, nSrcStep, pDst, nDstStep, srcROI, pMask, maskHeight, maskSize, anchor, borderSize, isFlat, seBinary);
}

EXTERN void performErosion_8u(const Npp8u * pSrc, Npp32s nSrcStep, Npp8u * pDst, Npp32s nDstStep, NppiSize srcROI, 
                   const Npp8u * pMask, const float * maskHeight, NppiSize maskSize, NppiPoint anchor, NppiSize borderSize, char isFlat, int seBinary) {
    _dispatchMorphOp<lcuda8u, ERODE>(pSrc, nSrcStep, pDst, nDstStep, srcROI, pMask, maskHeight, maskSize, anchor, borderSize, isFlat, seBinary);
}

EXTERN void performDilation_8u(const Npp8u * pSrc, Npp32s nSrcStep, Npp8u * pDst, Npp32s nDstStep, NppiSize srcROI, 
                   const Npp8u * pMask, const float * maskHeight, NppiSize maskSize, NppiPoint anchor, NppiSize borderSize, char isFlat, int seBinary) {
    _dispatchMorphOp<lcuda8u, DILATE>(pSrc, nSrcStep, pDst, nDstStep, srcROI, pMask, maskHeight, maskSize, anchor, borderSize, isFlat, seBinary);
}

template <class matrixType, class dataType>
static void _lcudaCopyBorder(matrixType src, matrixType dst, int color, int offsetX, int offsetY) {
    PRINTF("SRC: %d %d\n", src.width, src.height);
    PRINTF("DST: %d %d\n", dst.width, dst.height);
    PRINTF("Offsets x %d, y %d\n", offsetX, offsetY);

    int realPitch = dst.pitch / sizeof(dataType);

    thrust::device_ptr<dataType> dev_ptr(dst.data);
    thrust::fill(dev_ptr, dev_ptr + (dst.height-1)*realPitch + dst.width, color);

    dataType *data = dst.data + offsetY * realPitch + offsetX;
    cudaMemcpy2D(data, dst.pitch, src.data, src.pitch, src.width*sizeof(dataType), src.height, cudaMemcpyDeviceToDevice);
}

void lcudaCopyBorder(lcudaMatrix src, lcudaMatrix dst, int color, int offsetX, int offsetY) {
    _lcudaCopyBorder<lcudaMatrix, lcudaFloat>(src, dst, color, offsetX, offsetY);
}

EXTERN void lcudaCopyBorder_8u(lcudaMatrix_8u src, lcudaMatrix_8u dst, int color, int offsetX, int offsetY) {
    _lcudaCopyBorder<lcudaMatrix_8u, lcuda8u>(src, dst, color, offsetX, offsetY);
}
