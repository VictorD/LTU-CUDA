
template <class dataType, morphOperation MOP>
__device__ /*inline*/ dataType _processPixel(const int x, const int y, const dataType *img, int imgStep, morphMask mask) {
    dataType result = (MOP == ERODE) ? 255 : 0;
    dataType newval;
    int h,w;
    for (h = 0; h < mask.height; ++h) {
        for (w = 0; w < mask.width; ++w) {
            if (mask.data[h*mask.pitch+w] > 0) {
                if (MOP == ERODE) {
                    newval = img[y*imgStep + x + (h*imgStep + w)];//- surfaceHeights[maskPos]; TODO: Update this device pointer.
                    if (newval < result)
                        result = newval;
                } else {
                    newval = img[y*imgStep + x + (h*imgStep + w)];// + surfaceHeights[maskPos];
                    if (newval > result)
                        result = newval;
                }
            }
        }
    }

    return result;
}

template <class dataType, morphOperation MOP> 
__global__ void genericKernel(const dataType *img, int imgStep, int bytesPerThread, dataType *result, int resultStep, unsigned int width, 
                                  unsigned int height, morphMask mask) {
    const int x = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
    const int y = __mul24(blockIdx.y, blockDim.y) + threadIdx.y;

    if (y < height && x < width) {
        result[y * resultStep + x] = _processPixel<dataType, MOP>(x, y, img, imgStep, mask);
    }
}

template <class dataType, morphOperation MOP> 
int _globalGeneric(const dataType *img, int imgStep, dataType *result, int resultStep, rect2d roi, morphMask mask, rect2d border) {
    const unsigned int width = roi.width;
    const unsigned int height = roi.height;
    dim3 gridSize((width+16-1)/16, (height+16-1)/16);
    dim3 blockSize(16,16);

    int smemBytes = (mask.width)*(mask.height);
    int bytesPerThread = (smemBytes+16-1)/16;

    // Note: Assumes SE side length is odd.
    // Even numbered width/height (i.e. 4x4) needs img offset (imgStep+1) so that anchor is at (1,1)
    genericKernel<dataType, MOP><<<gridSize,blockSize, smemBytes>>>(img, imgStep, bytesPerThread, 
                                                                        result, resultStep, width, height, mask);
#if 1
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
        printf("CUDA error: %s\n", cudaGetErrorString(error));
#endif
    return LCUDA_SUCCESS;
}
