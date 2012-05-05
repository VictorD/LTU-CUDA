#ifndef MORPHOLOGYHELPER_H_
#define MORPHOLOGYHELPER_H_

#include "../ltucuda.cuh"
#include <cuda_runtime_api.h>

enum maskType {
	THREE_BY_THREE,
	VHGW,
	GENERIC
};

enum vhgwDirection {
    HORIZONTAL,
    VERTICAL,
	DIAGONAL_BACKSLASH,
	DIAGONAL_SLASH
};

struct morphMask {
    unsigned int width;
    unsigned int height;
    unsigned char *data;
    unsigned int pitch;
    float *surfaceHeights;
    point2d anchor;
    unsigned char isFlat;
	maskType type;
	vhgwDirection direction;
	int binaryValue;
};


void performErosion(const float * pSrc, int nSrcStep, float *pDst, int nDstStep, rect2d srcROI, 
                        morphMask mask, rect2d borderSize);

void performDilation(const float * pSrc, int nSrcStep, float *pDst, int nDstStep, rect2d srcROI, 
                        morphMask mask, rect2d borderSize);

morphMask createTBTMask(unsigned char *data);
morphMask createVHGWMask(int sideLength, vhgwDirection direction);
morphMask createArbitraryMask(unsigned char *data, float* surface, int width, int height, point2d anchor, int isFlat);

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


#endif /* MORPHOLOGYHELPER_H_ */
