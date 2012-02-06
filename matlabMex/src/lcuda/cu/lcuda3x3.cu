

/*
 * Binary packed integer values for common 3x3 kernels.
 */
#define CROSS 186
#define HOLLOW_CROSS 170
#define SLASH 84
#define BACKSLASH 273
#define VERTICAL_LINE 56
#define HORIZONTAL_LINE 146

/*
 * Kernel macros with code common to multiple kernels.
 */
#define KERNEL_VAR_INIT_AND_CHECK \
	const int y = __umul24(blockIdx.y, blockDim.y) + threadIdx.y; \
	const int x = __umul24(blockIdx.x, blockDim.x) + threadIdx.x; \
    if (y >= height || x >= width) \
        return; \
\
    const dataType *imgCol = (dataType*)img + y * imgStep;


template <class dataType, morphOperation MOP> 
__global__ void _backslashKernel(const dataType *img, int imgStep, dataType *result, 
                                    int resultStep, unsigned int width, unsigned int height) {
    KERNEL_VAR_INIT_AND_CHECK;

    dataType bestval;
    dataType dr_vs_ul;

    if (MOP == ERODE) {
        dr_vs_ul = min((dataType)imgCol[x+1+imgStep], (dataType)imgCol[x-1-imgStep]);
        bestval  = min(dr_vs_ul, imgCol[x]);
    } else {
        dr_vs_ul = max((dataType)imgCol[x+1+imgStep], (dataType)imgCol[x-1-imgStep]);
        bestval  = max(dr_vs_ul, imgCol[x]);
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
        l_vs_r  = min((dataType)imgCol[x+imgStep], (dataType)imgCol[x-imgStep]);
        bestval = min(l_vs_r, imgCol[x]);
    } else {
        l_vs_r  = max((dataType)imgCol[x+imgStep], (dataType)imgCol[x-imgStep]);
        bestval = max(l_vs_r, imgCol[x]);
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
        u_vs_d = min((dataType)imgCol[x+1], (dataType)imgCol[x-1]);
        bestval = min(u_vs_d, imgCol[x]);
    } else {
        u_vs_d  = max((dataType)imgCol[x+1], (dataType)imgCol[x-1]);
        bestval = max(u_vs_d, imgCol[x]);
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
        dl_vs_ur = min((dataType)imgCol[x+1-imgStep], (dataType)imgCol[x-1+imgStep]);
        bestval  = min(dl_vs_ur, imgCol[x]);
    } else {
        dl_vs_ur = max((dataType)imgCol[x+1-imgStep], (dataType)imgCol[x-1+imgStep]);
        bestval  = max(dl_vs_ur, imgCol[x]);
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
        u_vs_d  = min((dataType)imgCol[x+1], (dataType)imgCol[x-1]);
        l_vs_r  = min((dataType)imgCol[x+imgStep], (dataType)imgCol[x-imgStep]);
        bestval = min(u_vs_d, l_vs_r);
        bestval = min(bestval, imgCol[x]);
    } else {
        u_vs_d  = max((dataType)imgCol[x+1], (dataType)imgCol[x-1]);
        l_vs_r  = max((dataType)imgCol[x+imgStep], (dataType)imgCol[x-imgStep]);
        bestval = max(u_vs_d, l_vs_r);
        bestval = max(bestval, imgCol[x]);
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
        u_vs_d  = min((dataType)imgCol[x+1], (dataType)imgCol[x-1]);
        l_vs_r  = min((dataType)imgCol[x+imgStep], (dataType)imgCol[x-imgStep]);
        bestval = min(u_vs_d, l_vs_r);
    } else {
        u_vs_d  = max((dataType)imgCol[x+1], (dataType)imgCol[x-1]);
        l_vs_r  = max((dataType)imgCol[x+imgStep], (dataType)imgCol[x-imgStep]);
        bestval = max(u_vs_d, l_vs_r);
    }

    result[y * resultStep + x] = bestval;
}

template <class dataType, morphOperation MOP> 
__global__ void _generic3x3Kernel(const dataType *img, int imgStep, dataType *result, 
                                    int resultStep, unsigned int width, unsigned int height, const Npp8u *pMask) {

    KERNEL_VAR_INIT_AND_CHECK;

    dataType bestval         = (pMask[0] > 0) ? imgCol[x-1-imgStep] : 255;
    const dataType left      = (pMask[1] > 0) ? imgCol[x - imgStep] : 255;
    const dataType downLeft  = (pMask[2] > 0) ? imgCol[x+1-imgStep] : 255;
    const dataType up        = (pMask[3] > 0) ? imgCol[x-1]         : 255;
    const dataType center    = (pMask[4] > 0) ? imgCol[x]           : 255;
    const dataType down      = (pMask[5] > 0) ? imgCol[x+1]         : 255;
    const dataType upRight   = (pMask[6] > 0) ? imgCol[x-1+imgStep] : 255;;
    const dataType right     = (pMask[7] > 0) ? imgCol[x + imgStep] : 255;;
    const dataType downRight = (pMask[8] > 0) ? imgCol[x+1+imgStep] : 255;;

    MINMAX(MOP,bestval, up);
    MINMAX(MOP,bestval, left);
    MINMAX(MOP,bestval, downLeft);

    MINMAX(MOP,bestval, center);
    MINMAX(MOP,bestval, down);

    MINMAX(MOP,bestval, upRight);
    MINMAX(MOP,bestval, right);
    MINMAX(MOP,bestval, downRight);

    result[y * resultStep + x] = bestval;
}

template <class dataType, morphOperation MOP> 
NppStatus _global3x3(const dataType *img, Npp32s imgStep, dataType *result, Npp32s resultStep, NppiSize oSizeROI, const Npp8u *pMask, int seBinary) {
    const unsigned int width = oSizeROI.width;
    const unsigned int height = oSizeROI.height;

    dim3 gridSize((width+32-1)/32, (height+16-1)/16);
    dim3 blockSize(32,16);
    // Anchor at (1,1)
    int offset = 1*imgStep + 1; 

    switch(seBinary) {
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
            _verticalKernel<dataType, MOP><<<gridSize,blockSize>>>(img + offset, imgStep,result, resultStep, width, height);
        break;
        case HORIZONTAL_LINE:
            _horizontalKernel<dataType, MOP><<<gridSize,blockSize>>>(img + offset, imgStep,result, resultStep, width, height);
        break;
        default: {
            _generic3x3Kernel<dataType, MOP><<<gridSize,blockSize>>>(img + offset, imgStep,result, resultStep, width, height, pMask);
        }
    }

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

/*
EXTERN NppStatus lcudaErode3x3_8u(const unsigned char * img, Npp32s imgStep, unsigned char * result, 
                                Npp32s resultStep, NppiSize oSizeROI, const Npp8u *pMask, int seBinary) {
    return _global3x3<unsigned char, ERODE>(img, imgStep, result, resultStep, oSizeROI, pMask, seBinary);
}

EXTERN NppStatus lcudaErode3x3(const float * img, Npp32s imgStep, float * result, Npp32s resultStep,
                                        NppiSize oSizeROI, const Npp8u *pMask, int seBinary) {
    return _global3x3<float, ERODE>(img, imgStep, result, resultStep, oSizeROI, pMask, seBinary);
}

EXTERN NppStatus lcudaDilate3x3_8u(const unsigned char * img, Npp32s imgStep, unsigned char * result, 
                                Npp32s resultStep, NppiSize oSizeROI, const Npp8u *pMask, int seBinary) {
    return _global3x3<unsigned char, DILATE>(img, imgStep, result, resultStep, oSizeROI, pMask, seBinary);
}

EXTERN NppStatus lcudaDilate3x3(const float * img, Npp32s imgStep, float * result, Npp32s resultStep,
                                        NppiSize oSizeROI, const Npp8u *pMask, int seBinary) {
    return _global3x3<float, DILATE>(img, imgStep, result, resultStep, oSizeROI, pMask, seBinary);
}*/
