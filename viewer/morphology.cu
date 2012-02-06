/*
 * lcudamorph.c
 *
 *  Created on: Feb 6, 2010
 *      Author: henmak
 *      Modified by vicdan-8 2011
 */ 

#include <thrust/device_ptr.h>
#include <thrust/fill.h>

#include <cuda.h>
#include <memory>
#include <iostream>
#include <cassert>
#include <stdio.h>
#include <string.h>

#include "morphology.cuh"


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
// vHGW (1D SE)
#include "kernels/lcudavhgw.cu"

// 3x3
#include "kernels/lcuda3x3.cu"

// Generic
#include "kernels/lcudaGeneric.cu"

template <class dataType, morphOperation MOP>
static void _dispatchMorphOp(const dataType* pSrc, int nSrcStep, dataType* pDst, int nDstStep, 
                                rect2d srcROI, morphMask mask, rect2d border) {

    int offsetX = border.width-mask.width/2;
    int offsetY = border.height-mask.height/2;
    PRINTF("OffsetX : %d, OffsetY : %d\n", offsetX, offsetY);

    // Steps are specified in bytes. Pointer aritmetic below requires we compensate for size of <dataType>.
    nSrcStep /= sizeof(dataType);
    nDstStep /= sizeof(dataType); 
 
    int srcBorderOffset = (nSrcStep * offsetY + offsetX);
    PRINTF("Border offset: %d\n", srcBorderOffset);
    char processed = 0;
    if (mask.isFlat) {
        // Custom fast 3x3
        if (mask.height == 3 && mask.width == 3) {
            _global3x3<dataType, MOP>(pSrc + srcBorderOffset, nSrcStep, 
                            pDst, nDstStep, srcROI, mask);
            processed = 1;
        }
        // Vertical vHGW
        else if (mask.width == 1) { 
            PRINTF("Vertical Erosion: SE Size (%dx%d)\n", mask.width, mask.height); 
            PRINTF("Erosion: Offset (%d,%d)\n", offsetX, offsetY);
            _globalVHGW<dataType, MOP, VERTICAL>(pSrc + srcBorderOffset, nSrcStep, 
                            pDst, nDstStep, srcROI, mask.height, border);
            processed = 1;
        // Horizontal vHGW
        } else if (mask.height == 1) {
            PRINTF("Horizontal Erosion: SE Size (%dx%d)\n", mask.width, mask.height); 
            PRINTF("Erosion: Offset (%d,%d)\n", offsetX, offsetY);

            _globalVHGW<dataType, MOP, HORIZONTAL>(pSrc + srcBorderOffset, nSrcStep, 
                            pDst, nDstStep, srcROI, mask.width, border);
            processed = 1;
        }
    }

    // Non-flat or arbitrary SE
    if (!processed) { 
        PRINTF("Generic!\n");
        _globalGeneric<dataType, MOP>(pSrc + srcBorderOffset, nSrcStep, pDst, nDstStep, srcROI, mask, border);
    } 
 
    // Block until async kernels calls have been executed.
    cudaThreadSynchronize();
}
 
/*    
 * Public functions to perform erosion or dilation. 
 */ 
EXTERN void performDilation(const float * pSrc, int nSrcStep, float *pDst, int nDstStep, rect2d srcROI, 
                        morphMask mask, rect2d border) {
    _dispatchMorphOp<float, DILATE>(pSrc, nSrcStep, pDst, nDstStep, srcROI, mask, border);
}

EXTERN void performErosion(const float * pSrc, int nSrcStep, float *pDst,
                               int nDstStep, rect2d srcROI, morphMask mask, rect2d border) {
    PRINTF("mask is height: %d, width :%d, isFlat: %d\n", mask.height, mask.width, mask.isFlat);
    _dispatchMorphOp<float, ERODE>(pSrc, nSrcStep, pDst, nDstStep, srcROI, mask, border);
} 

// Linear strel creation methods
EXTERN morphMask createArbitraryMask(unsigned char *data, float* surface, int width, int height, point2d anchor, int isFlat) {
    int pitch;
    unsigned char *dev_mask_data;
    cudaMallocPitch((void **)&dev_mask_data, (size_t*)&pitch, width, height);
    exitOnError("createArbitraryMask: alloc");
    cudaMemcpy2D(dev_mask_data, pitch, data, width, width, height, cudaMemcpyHostToDevice);
    exitOnError("createArbitraryMask: copy");
    morphMask mask = {width,height, dev_mask_data, pitch, surface, anchor, 1/*defaultBinaryValue*/, isFlat};
    return mask;
}

EXTERN morphMask createFlat3x3Mask(unsigned char *data) {
    point2d anchor = {1,1};
    morphMask mask = createArbitraryMask(data, NULL, 3,3,anchor, 1/*isFlat*/);
    mask.binaryValue = 256*data[8] + 128*data[7] + 64*data[6] + 
                       32*data[5] +  16*data[4] +  8*data[3] + 
                        4*data[2] +   2*data[1] +    data[0];
    exitOnError("createFlat3x3Mask: copy");
    return mask;
}

EXTERN morphMask _createFlatLineMask(int width, int height) {
    point2d anchor = {1,1};
    morphMask mask = {width,height, NULL, 0, NULL, anchor, 1, 1};
    return mask;
}

EXTERN morphMask createFlatHorizontalLineMask(int width) {
    return _createFlatLineMask(width, 1);
}

EXTERN morphMask createFlatVerticalLineMask(int height) {
    return _createFlatLineMask(1,height);
}

EXTERN float* copyImageToDevice(float *hostImage, rect2d image, int *pitch) {
    float *dev;
    cudaMallocPitch((void **)&dev, (size_t*)pitch, image.width * sizeof(float), image.height);
    exitOnError("copyImageToDevice: alloc");
 
    cudaMemcpy2D(dev, *pitch, hostImage, image.width * sizeof(float),
    			image.width * sizeof(float), image.height, cudaMemcpyHostToDevice);
    exitOnError("copyImageToDevice: copy");
    return dev;
}

EXTERN float* copyImageToHost(float *device_data, rect2d image, int pitch) {
    float *host;
    int bytesNeeded = image.width*image.height*sizeof(float);
    mallocHost((void**)&host,bytesNeeded, PINNED, false);
    cudaMemcpy2D(host, image.width * sizeof(float), device_data, pitch, image.width*sizeof(float), image.height, cudaMemcpyDeviceToHost);
    exitOnError("copyImageToHost: copy");
    return host;
}

// Add border padding
EXTERN float* createPaddedArray(rect2d borderSize, rect2d imgSize, float defaultValue, int *pitch) {
    float *padded;
    int paddedHeight = imgSize.height + 2*borderSize.height;
    int paddedWidth = imgSize.width + 2*borderSize.width;

    cudaMallocPitch((void **)&padded, (size_t*)pitch, paddedWidth * sizeof(float), paddedHeight);
    exitOnError("createPaddedArray: alloc");

    thrust::device_ptr<float> dev_ptr(padded);
    thrust::fill(dev_ptr, dev_ptr + (paddedHeight-1) * (*pitch/sizeof(float)) + paddedWidth, 255.0f);
    exitOnError("createPaddedArray: thrust::fill");
    return padded;
}
