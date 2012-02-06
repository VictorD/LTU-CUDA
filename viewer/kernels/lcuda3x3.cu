

/*
 * Binary packed integer values for common 3x3 kernels.
 */
#define CROSS 186
#define HOLLOW_CROSS 170
#define SLASH 84
#define BACKSLASH 273
#define VERTICAL_LINE 146
#define HORIZONTAL_LINE 56

/*
 * Kernel macros with code common to multiple kernels.
 */
#define KERNEL_VAR_INIT_AND_CHECK \
	const int x = __umul24(blockIdx.x, blockDim.x) + threadIdx.x; \
	const int y = __umul24(blockIdx.y, blockDim.y) + threadIdx.y; \
    if (y >= height || x >= width)                                \
        return;                                                   \
                                                                  \
    const dataType *imgCol = (dataType*)img + y * imgStep + x;


template <class dataType, morphOperation MOP> 
__global__ void _backslashKernel(const dataType *img, int imgStep, dataType *result, 
                                    int resultStep, unsigned int width, unsigned int height) {
    KERNEL_VAR_INIT_AND_CHECK;

    dataType bestval;
    dataType dr_vs_ul;

    if (MOP == ERODE) {
        dr_vs_ul = min((dataType)imgCol[1+imgStep], (dataType)imgCol[-1-imgStep]);
        bestval  = min(dr_vs_ul, imgCol[0]);
    } else {
        dr_vs_ul = max((dataType)imgCol[1+imgStep], (dataType)imgCol[-1-imgStep]);
        bestval  = max(dr_vs_ul, imgCol[0]);
    }

    result[y * resultStep + x] = bestval;
}

template <class dataType, morphOperation MOP> 
__global__ void _horizontalKernel(const dataType *img, int imgStep, dataType *result, 
                                    int resultStep, unsigned int width, unsigned int height) {
    KERNEL_VAR_INIT_AND_CHECK;

    dataType bestval;
    dataType l_vs_r;

    if (MOP == ERODE) {
        l_vs_r = min((dataType)imgCol[1], (dataType)imgCol[-1]);
        bestval = min(l_vs_r, imgCol[0]);
    } else {
        l_vs_r  = max((dataType)imgCol[1], (dataType)imgCol[-1]);
        bestval = max(l_vs_r, imgCol[0]);
    }

    result[y * resultStep + x] = bestval;
}

template <class dataType, morphOperation MOP> 
__global__ void _verticalKernel(const dataType *img, int imgStep, dataType *result, 
                                    int resultStep, unsigned int width, unsigned int height) {
    KERNEL_VAR_INIT_AND_CHECK;

    dataType bestval;
    dataType u_vs_d;

    if (MOP == ERODE) {
        u_vs_d  = min((dataType)imgCol[imgStep], (dataType)imgCol[-imgStep]);
        bestval = min(u_vs_d, imgCol[0]);
    } else {
        u_vs_d  = max((dataType)imgCol[imgStep], (dataType)imgCol[-imgStep]);
        bestval = max(u_vs_d, imgCol[0]);
    }

    result[y * resultStep + x] = bestval;
}

template <class dataType, morphOperation MOP> 
__global__ void _slashKernel(const dataType *img, int imgStep, dataType *result, 
                                    int resultStep, unsigned int width, unsigned int height) {
    KERNEL_VAR_INIT_AND_CHECK;

    dataType bestval;
    dataType dl_vs_ur;

    if (MOP == ERODE) {
        dl_vs_ur = min((dataType)imgCol[1-imgStep], (dataType)imgCol[-1+imgStep]);
        bestval  = min(dl_vs_ur, imgCol[0]);
    } else {
        dl_vs_ur = max((dataType)imgCol[1-imgStep], (dataType)imgCol[-1+imgStep]);
        bestval  = max(dl_vs_ur, imgCol[0]);
    }

    result[y * resultStep + x] = bestval;
}

template <class dataType, morphOperation MOP> 
__global__ void _crossKernel(const dataType *img, int imgStep, dataType *result, 
                                    int resultStep, unsigned int width, unsigned int height) {

    KERNEL_VAR_INIT_AND_CHECK;

    dataType bestval;
    dataType u_vs_d;
    dataType l_vs_r;

    if (MOP == ERODE) {
        l_vs_r  = min((dataType)imgCol[1], (dataType)imgCol[-1]);
        u_vs_d  = min((dataType)imgCol[+imgStep], (dataType)imgCol[-imgStep]);
        bestval = min(u_vs_d, l_vs_r);
        bestval = min(bestval, imgCol[0]);
    } else {
        l_vs_r  = max((dataType)imgCol[1], (dataType)imgCol[-1]);
        u_vs_d  = max((dataType)imgCol[imgStep], (dataType)imgCol[-imgStep]);
        bestval = max(u_vs_d, l_vs_r);
        bestval = max(bestval, imgCol[0]);
    }

    result[y * resultStep + x] = bestval;
}

template <class dataType, morphOperation MOP> 
__global__ void _hollowCrossKernel(const dataType *img, int imgStep, dataType *result, 
                                    int resultStep, unsigned int width, unsigned int height) {

    KERNEL_VAR_INIT_AND_CHECK;

    dataType bestval;
    dataType u_vs_d;
    dataType l_vs_r;

    if (MOP == ERODE) {
        l_vs_r  = min((dataType)imgCol[1], (dataType)imgCol[-1]);
        u_vs_d  = min((dataType)imgCol[imgStep], (dataType)imgCol[-imgStep]);
        bestval = min(u_vs_d, l_vs_r);
    } else {
        l_vs_r  = max((dataType)imgCol[1], (dataType)imgCol[-1]);
        u_vs_d  = max((dataType)imgCol[imgStep], (dataType)imgCol[-imgStep]);
        bestval = max(u_vs_d, l_vs_r);
    }

    result[y * resultStep + x] = bestval;
}

template <class dataType, morphOperation MOP> 
__global__ void _generic3x3Kernel(const dataType *img, int imgStep, dataType *result, 
                                    int resultStep, unsigned int width, unsigned int height,
                                        const unsigned char *pMask, unsigned int maskStep) {
	const int x = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	const int y = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
    if (y >= height || x >= width)
        return;

    const dataType *imgCol = (dataType*)img + y * imgStep + x;

          dataType bestval   = (pMask[0] > 0)            ? imgCol[-imgStep-1] : 255;
    const dataType up        = (pMask[1] > 0)            ? imgCol[-imgStep]   : 255;
    const dataType upRight   = (pMask[2] > 0)            ? imgCol[-imgStep+1] : 255;
    const dataType left      = (pMask[maskStep] > 0)     ? imgCol[-1]         : 255;
    const dataType center    = (pMask[maskStep+1] > 0)   ? imgCol[0]          : 255;
    const dataType right     = (pMask[maskStep+2] > 0)   ? imgCol[1]          : 255;
    const dataType downLeft  = (pMask[2*maskStep] > 0)   ? imgCol[imgStep-1]  : 255;
    const dataType down      = (pMask[2*maskStep+1] > 0) ? imgCol[imgStep]    : 255;
    const dataType downRight = (pMask[2*maskStep+2] > 0) ? imgCol[imgStep+1]  : 255;

    MINMAX(MOP,bestval, up);
    MINMAX(MOP,bestval, upRight);

    MINMAX(MOP,bestval, left);
    MINMAX(MOP,bestval, center);
    MINMAX(MOP,bestval, right);

    MINMAX(MOP,bestval, downLeft);
    MINMAX(MOP,bestval, down);
    MINMAX(MOP,bestval, downRight);

    result[y * resultStep + x] = bestval;
}

template <class dataType, morphOperation MOP> 
int _global3x3(const dataType *img, int imgStep, dataType *result, int resultStep, rect2d oSizeROI, morphMask mask) {
    const unsigned int width = oSizeROI.width;
    const unsigned int height = oSizeROI.height;

    dim3 gridSize((width+16-1)/16, (height+16-1)/16);
    dim3 blockSize(16,16);
    // Anchor at (1,1)
    int offset = 1*imgStep + 1;

    PRINTF("BinaryValue :%d\n", mask.binaryValue);
    switch(mask.binaryValue/*-mask.binaryValue+1*/) {
        case CROSS:
            _crossKernel<dataType, MOP><<<gridSize,blockSize>>>(img + offset, imgStep,result, resultStep, width, height);
        break;
        case SLASH:
            _slashKernel<dataType, MOP><<<gridSize,blockSize>>>(img + offset, imgStep,result, resultStep, width, height);
        break;
        case BACKSLASH: 
            _backslashKernel<dataType, MOP><<<gridSize,blockSize>>>(img + offset, imgStep,result, resultStep, width, height);
        break;
        case HOLLOW_CROSS: 
            _hollowCrossKernel<dataType, MOP><<<gridSize,blockSize>>>(img + offset, imgStep,result, resultStep, width, height);
        break;
        case VERTICAL_LINE:
            PRINTF("VERTICAL\n");
            _verticalKernel<dataType, MOP><<<gridSize,blockSize>>>(img + offset, imgStep,result, resultStep, width, height);
        break;
        case HORIZONTAL_LINE:
            PRINTF("hOZ\n");
            _horizontalKernel<dataType, MOP><<<gridSize,blockSize>>>(img + offset, imgStep,result, resultStep, width, height);
        break;
        default: {
            _generic3x3Kernel<dataType, MOP><<<gridSize,blockSize>>>(img + offset, imgStep,result, resultStep, 
                                                                        width, height, mask.data, mask.pitch);
        }
    }

#if 1 // DEBUG_ON
    // check for error
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        // print the CUDA error message
        PRINTF("CUDA error: %s\n", cudaGetErrorString(error));
    }
#endif

    return LCUDA_SUCCESS;
}
