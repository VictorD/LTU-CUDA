/**
 * \defgroup lcudamorph LTU Cuda Morphological operations
 * \ingroup lcuda
 *
 * Morphological operations in the LTU Cuda library.
 *
 */
/*@{*/

#include "pinnedmem.cuh"

#ifndef MORPHOLOGYHELPER_H_
#define MORPHOLOGYHELPER_H_

#ifdef __cplusplus
#define EXTERN extern "C"
#else
#define EXTERN
#endif

#define LCUDA_SUCCESS 1
#define LCUDA_FAIL 0

#define PRINTF(...) //printf(__VA_ARGS__)

typedef struct {
    int width;
    int height;
} rect2d;

typedef struct {
    int x;
    int y;
} point2d;

typedef struct {
    unsigned int width;
    unsigned int height;    
    unsigned int pitch;
    float *data;
} cudaImage;

typedef struct {
    unsigned int width;
    unsigned int height;
    unsigned char *data;
    unsigned int pitch;
    float *surfaceHeights;
    point2d anchor;
    int binaryValue;
    unsigned char isFlat;
} morphMask;

EXTERN void performErosion(const float * pSrc, int nSrcStep, float *pDst, int nDstStep, rect2d srcROI, 
                        morphMask mask, rect2d borderSize);

EXTERN void performDilation(const float * pSrc, int nSrcStep, float *pDst, int nDstStep, rect2d srcROI, 
                        morphMask mask, rect2d borderSize);

EXTERN morphMask createFlat3x3Mask(unsigned char *data);
EXTERN morphMask createFlatHorizontalLineMask(int width);
EXTERN morphMask createFlatVerticalLineMask(int height);
EXTERN morphMask createArbitraryMask(unsigned char *data, float* surface, int width, int height, 
    point2d anchor, int isFlat);

EXTERN float* copyImageToDevice(float *hostImage, rect2d image, int *pitch);
EXTERN float* copyImageToHost(float *device_data, rect2d image, int pitch);
EXTERN float* createPaddedArray(rect2d borderSize, rect2d imgSize, float defaultValue, int *pitch);

#endif /* MORPHOLOGYHELPER_H_ */
