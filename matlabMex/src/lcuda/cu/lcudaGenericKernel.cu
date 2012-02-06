
template <class dataType, morphOperation MOP>
__device__ inline dataType _processPixel(const int x, const int y, const dataType *img, int imgStep, 
                                    const Npp8u *pMask, const float* maskHeight, NppiSize maskSize) {

    dataType result = (MOP == ERODE) ? 255 : 0;
    dataType newval;

    int i,j, maskPos;
    for (j = 0; j < maskSize.height; ++j)
    {
        maskPos = __umul24(j,maskSize.width);
        for (i = 0; i < maskSize.width; ++i)
        {
            if (pMask[maskPos] > 0) {
                if (MOP == ERODE) {
                    newval = img[y*imgStep + x + (j*imgStep + i)] - maskHeight[maskPos];
                    if (newval < result)
                        result = newval;
                } else {
                    newval = img[y*imgStep + x + (j*imgStep + i)] + maskHeight[maskPos];
                    if (newval > result)
                        result = newval;
                }
            }
            maskPos++;
        }
    }

    return result;
}

template <class dataType, morphOperation MOP> 
__global__ void genericKernel(const dataType *img, int imgStep, int shStep, dataType *result, int resultStep, unsigned int width, 
                                  unsigned int height, const lcuda8u *pMask, const float* maskHeight, NppiSize maskSize) {
    const int y = __mul24(blockIdx.y, blockDim.y) + threadIdx.y;
    const int x = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

    // Copy SE into shared memory for faster access
    extern __shared__ lcuda8u shMask[];
    int i;
    int currentIndex = __umul24(shStep,threadIdx.x);
    for(i = 0; i < shStep; ++i) {
        shMask[currentIndex] = pMask[currentIndex];
        currentIndex++;
    }
    __syncthreads();

    if (y < height && x < width) {
        result[y * resultStep + x] = _processPixel<dataType, MOP>(x, y, img, imgStep, shMask, maskHeight, maskSize);
    }
}

template <class dataType, morphOperation MOP> 
NppStatus _globalGeneric(const dataType *img, Npp32s imgStep, dataType *result, Npp32s resultStep, NppiSize oSizeROI, 
                            const Npp8u *pMask, NppiSize maskSize, const float* maskHeight, NppiSize borderSize, NppiPoint anchor) {

    const unsigned int width = oSizeROI.width;
    const unsigned int height = oSizeROI.height;

    dim3 gridSize((width+16-1)/16, (height+16-1)/16);
    dim3 blockSize(16,16);

    int reqSharedMemSize = (maskSize.width)*(maskSize.height);
    int maskBitsCpyPerThread = (maskSize.width*maskSize.height+16-1)/16;

    genericKernel<dataType, MOP><<<gridSize,blockSize, reqSharedMemSize>>>(img, imgStep, maskBitsCpyPerThread, result, resultStep, 
                                                                               width, height, pMask, maskHeight, maskSize);

#if 1 // DEBUG_ON
    // check for error
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        // print the CUDA error message
        printf("CUDA error: %s\n", cudaGetErrorString(error));
    }
#endif
    return NPP_SUCCESS;
}

EXTERN NppStatus lcudaErodeGeneric_8u(const unsigned char * img, Npp32s imgStep, unsigned char * result, Npp32s resultStep,
                                        NppiSize oSizeROI, const Npp8u *pMask, NppiSize maskSize,
                                             const float *maskHeight, NppiSize borderSize, NppiPoint anchor) {
    return _globalGeneric<unsigned char, ERODE>(img, imgStep, result, resultStep, oSizeROI, pMask, maskSize, maskHeight, borderSize, anchor);
}

EXTERN NppStatus lcudaErodeGeneric(const float * img, Npp32s imgStep, float * result, Npp32s resultStep,
                                        NppiSize oSizeROI, const Npp8u *pMask, NppiSize maskSize, 
                                            const float *maskHeight, NppiSize borderSize, NppiPoint anchor) {
    return _globalGeneric<float, ERODE>(img, imgStep, result, resultStep, oSizeROI, pMask, maskSize, maskHeight, borderSize, anchor);
}
