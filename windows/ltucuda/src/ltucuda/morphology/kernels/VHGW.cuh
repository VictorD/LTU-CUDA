#ifndef _CUDA_KERNELS_H_
#define _CUDA_KERNELS_H_

#include "morphology.cuh"

template <class dataType, morphOperation MOP>
__global__ void _horizontalVHGWKernel(const dataType *img, int imgStep, dataType *result, 
                                    int resultStep, unsigned int maxSteps, unsigned int width, unsigned int height,
                                        unsigned int size, rect2d borderSize) {
	const unsigned int step   = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	const unsigned int y      = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
    const unsigned int startx = __umul24(step,size);

    if (y >= height || startx > width)
        return;

    const dataType *lineIn = img+y*imgStep;
    dataType *lineOut      = result+y*resultStep;
    const unsigned int center  = startx + (size-1);

    dataType minarray[512];
    minarray[size-1] = lineIn[center];

    dataType nextMin;
    unsigned int k;
    if (MOP == ERODE) {
        for(k=1;k<size; ++k) {
            nextMin = lineIn[center-k];
            minarray[size-1-k] = min(minarray[size-k], nextMin);

            nextMin = (center+k < width+size-1) ? lineIn[center+k] : 255;
            minarray[size-1+k] = min(minarray[size+k-2], nextMin);
        }
    } else {
        for(k=1;k<size; ++k) {
            nextMin = lineIn[__umul24(center-k,imgStep)];
            minarray[size-1-k] = max(minarray[size-k], nextMin);
        
            nextMin = lineIn[__umul24(center+k,imgStep)];
            minarray[size-1+k] = max(minarray[size+k-2], nextMin);
        }
    }

    int diff = width - startx;
    if (diff > 0) {
        lineOut += startx;
        lineOut[0] = minarray[0];

        for(k=1; k < size-1; ++k) {
            if (diff > k) {
                lineOut[k] = minMax<dataType, MOP>(minarray[k], minarray[k+size-1]);
            }
        }

        if (diff > size-1) {
            lineOut[size-1] = minarray[2*(size-1)];
        }
    }
}

template <class dataType, morphOperation MOP>
__global__ void _verticalVHGWKernel(const dataType *img, int imgStep, dataType *result, 
                                    int resultStep, unsigned int maxSteps, unsigned int width, unsigned int height,
                                        unsigned int size, rect2d borderSize) {
	const unsigned int x      = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	const unsigned int step   = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
    const unsigned int starty = __umul24(step,size);

    if (x >= width || starty > height)
        return;

    const dataType *lineIn = img+x;
    dataType *lineOut      = result+x;

    const unsigned int center  = starty + (size-1);

    dataType minarray[512];
    minarray[size-1] = lineIn[center*imgStep];

    dataType nextMin;
    unsigned int k;
    if (MOP == ERODE) {
        for(k=1;k<size; ++k) {
            nextMin = lineIn[(center-k)*imgStep];
            minarray[size-1-k] = min(minarray[size-k], nextMin);

            nextMin = (center+k < height+size-1) ? lineIn[(center+k)*imgStep] : 255;
            minarray[size-1+k] = min(minarray[size+k-2], nextMin);
        }
    } else {
        for(k=1;k<size; ++k) {
            nextMin = lineIn[__umul24(center-k,imgStep)];
            minarray[size-1-k] = max(minarray[size-k], nextMin);
        
            nextMin = lineIn[__umul24(center+k,imgStep)];
            minarray[size-1+k] = max(minarray[size+k-2], nextMin);
        }
    }

    int diff = height - starty;
    if (diff > 0) {
        lineOut += starty*resultStep;
        lineOut[0] = minarray[0];

        for(k=1; k < size-1; ++k) {
            if (diff > k) {
                lineOut[k*resultStep] = minMax<dataType, MOP>(minarray[k], minarray[k+size-1]);
            }
        }

        if (diff > size-1) {
            lineOut[(size-1)*resultStep] = minarray[2*(size-1)];
        }
    }
}

template <class dataType, morphOperation MOP>
__global__ void _diagonalBackslashVHGWKernel(const dataType *img, int imgStep, dataType *result, int resultStep, unsigned int maxSteps, unsigned int width, unsigned int height,
                                        unsigned int size, rect2d borderSize) {
	const int x      = __umul24(blockIdx.x, blockDim.x) + threadIdx.x -(size-1)/2;

	const unsigned int step   = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
    const unsigned int starty = __umul24(step,size);

    if (x >= width+(size-1) || starty > height)
        return;

    const dataType *lineIn	  = img+x;
    const unsigned int center = starty + (size-1);

    dataType minarray[512];
    minarray[size-1] = lineIn[center*imgStep];

    dataType nextMin;
    unsigned int k;
    if (MOP == ERODE) {
        for(k=1;k<size; ++k) {
			int minIndex = (center-k)*imgStep-k;
            nextMin = (minIndex > 0) ? lineIn[minIndex] : 255;
            minarray[size-1-k] = min(minarray[size-k], nextMin);

            nextMin = lineIn[(center+k)*imgStep+k];
            minarray[size-1+k] = min(minarray[size+k-2], nextMin);
        }
    }

	dataType *lineOut = result + starty*resultStep + x-(size-1);

	if (x >= width+(size-1))
		return;
	else {
		for(k=0; k < size; ++k) {
			if (x+k < size-1)
				continue;
			else if (x+k >= width+(size-1) || starty+k >= height)
				break;

			lineOut[k*resultStep+k] = minMax<dataType, MOP>(minarray[k], minarray[k+size-1]);
		}
	}
}

template <class dataType, morphOperation MOP>
__global__ void _diagonalSlashVHGWKernel(const dataType *img, int imgStep, dataType *result, int resultStep, unsigned int maxSteps, 
                                        unsigned int width, unsigned int height, unsigned int size, rect2d borderSize) {
    const int x      = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	const unsigned int step   = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
    const int starty = __umul24(step,size);

    if (x >= width+(size-1) || starty > height)
        return;

    const dataType *lineIn	  = img+x;
	dataType *lineOut = result + starty*resultStep + x;
    const unsigned int center = starty + (size-1);

    dataType minarray[512];
    minarray[size-1] = lineIn[center*imgStep];

    dataType nextMin;
    int k;
    if (MOP == ERODE) {
        for(k=1;k<size; ++k) {
			int minIndex = (center-k)*imgStep+k;
            nextMin = (minIndex > 0) ? lineIn[minIndex] : 255;
            minarray[size-1-k] = min(minarray[size-k], nextMin);

            nextMin = lineIn[(center+k)*imgStep-k];
            minarray[size-1+k] = min(minarray[size+k-2], nextMin);
        }
    }



//	if (x >= width+(size-1)/2 || starty >= height)
		//return;
	
	for(k=0; k < size; ++k) {
		if (x-k < width && starty+k < height) {
			lineOut[k*resultStep-k] = minMax<dataType, MOP>(minarray[k], minarray[k+size-1]);
		}
    }
}
 
template <class dataType, morphOperation MOP>
int _globalVHGW(const dataType * img, int imgStep, dataType * result, int resultStep, rect2d oSizeROI,morphMask mask, rect2d borderSize) {
    const unsigned int width = oSizeROI.width;
    const unsigned int height = oSizeROI.height;
    unsigned int steps;
	unsigned int size;	

	PRINTF("width %d, height %d\n", width, height);
    PRINTF("Border (w: %d , h: %d)\n", borderSize.width, borderSize.height);

	switch(mask.direction) {
		case VERTICAL: {
				size = mask.height;
				steps = (height+size-1)/size;
				dim3 gridSize((width+128-1)/128, (steps+2-1)/2);
				dim3 blockSize(128, 2);
				_verticalVHGWKernel<dataType, MOP><<<gridSize,blockSize>>>(img, imgStep,result, resultStep, steps, width, height, size, borderSize);
			}
			break;
		case HORIZONTAL: {
				size = mask.width;
				steps = (width+size-1)/size;
				dim3 gridSize((steps+128-1)/128, (height+2-1)/2);
				dim3 blockSize(128, 2);
				_horizontalVHGWKernel<dataType, MOP><<<gridSize, blockSize>>>(img, imgStep,result, resultStep, steps, width, height, size, borderSize);
			}
			break;
		case DIAGONAL_SLASH: {
				size = mask.height;
				PRINTF("MASK SIZE IS: %d\n", size);
				steps = (height+5*(size-1))/size;
				dim3 gridSize((width+5*(size-1)+128-1)/128, (steps+2-1)/2);
				dim3 blockSize(128, 2);
				_diagonalSlashVHGWKernel<dataType, MOP><<<gridSize,blockSize>>>(img, imgStep,result, resultStep, steps, width, height, size, borderSize);
		}
		break;
		default: { 
				size = mask.height;
				PRINTF("MASK SIZE IS: %d\n", size);
				steps = (height+size-1)/size;
				dim3 gridSize((width+2*(size-1)+128-1)/128, (steps+2-1)/2);
				dim3 blockSize(128, 2);
				_diagonalBackslashVHGWKernel<dataType, MOP><<<gridSize,blockSize>>>(img, imgStep,result, resultStep, steps, width, height, size, borderSize);
			}
			break;
    }

    // check for error
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        // print the CUDA error message and exit
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    return LCUDA_SUCCESS;
}

#endif
