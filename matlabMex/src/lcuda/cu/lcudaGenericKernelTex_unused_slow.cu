#include <cuda.h>
#include <npp.h>
#include <nppi.h>
#include <memory>
#include <iostream>
#include <cassert>
#include <stdio.h>

#include "mlcuda.h"
#include "lcudaextern.h"

template <class dataType, morphOperation MOP>
__device__ dataType _processPixel(const int x, const int y, int width, int height, 
                                    const Npp8u *pMask, const float* maskHeight, NppiSize maskSize) {

    dataType result = (MOP == ERODE) ? 255 : 0;

    for (int j = 0; j < maskSize.height; ++j)
    {
        for (int i = 0; i < maskSize.width; ++i)
        {
            if (pMask[j*maskSize.width+i] > 0) {
                dataType bestval = tex2D(srcTex, x+i, y+j);
                dataType nonFlatWeight = maskHeight[j*maskSize.width + i];
                if (MOP == ERODE)
                    bestval -= nonFlatWeight;
                else
                    bestval += nonFlatWeight; 

                SET_IF_BETTER(MOP, result, bestval);
            }
        }
    }
    return result;
}

template <class dataType, morphOperation MOP> 
__global__ void genericKernel(size_t offsetX, size_t offsetY, dataType *result, int resultStep, unsigned int width, unsigned int height, 
                                const Npp8u *pMask, const float* maskHeight, NppiSize maskSize, NppiSize borderSize, NppiPoint anchor) {
    const int y = __mul24(blockIdx.y, blockDim.y) + threadIdx.y;
    const int x = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

    if (y < height-borderSize.height) {
        if (x < width-borderSize.width) {
            result[y* resultStep + x] = _processPixel<dataType, MOP>(x+offsetX, y+offsetY, width, height, pMask, maskHeight, maskSize);
        }
    }
}

template <class dataType, morphOperation MOP> 
NppStatus _globalGenericTex(size_t offsetX, size_t offsetY, dataType *result, int resultStep, NppiSize oSizeROI, 
                            const unsigned char *pMask, NppiSize maskSize, const float* maskHeight, NppiSize borderSize, NppiPoint anchor) {

    const unsigned int width = oSizeROI.width + borderSize.width;
    const unsigned int height = oSizeROI.height + borderSize.height;

    dim3 gridSize((width+16-1)/16, (height+16-1)/16);
    dim3 blockSize(16,16);

    /*unsigned int maskNum = maskSize.width * maskSize.height;
    float hostMask[maskNum];
    cudaMemcpy((void*)hostMask, (const void*)maskHeight, maskNum * sizeof(float), cudaMemcpyDeviceToHost);
    printf("maskNum: %d\n", maskNum);
    for(int i = 0; i < maskNum; i++) printf("maskHeight %d : %f\n", i, hostMask[i]);*/

    genericKernel<dataType, MOP><<<gridSize,blockSize>>>(offsetX, offsetY, result, resultStep, width, height, pMask, maskHeight, maskSize, borderSize, anchor);

    cudaUnbindTexture(srcTex);

#if 1 // DEBUG_ON
    // check for error
    cudaError_t error2 = cudaGetLastError();
    if(error2 != cudaSuccess)
    {
        // print the CUDA error message
        printf("CUDA error: %s\n", cudaGetErrorString(error2));
    }
#endif
    return NPP_SUCCESS;
}

