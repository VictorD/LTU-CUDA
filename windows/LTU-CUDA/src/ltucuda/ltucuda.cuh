#ifndef _LTU_CUDA_BASE_H_
#define _LTU_CUDA_BASE_H_

#include "pinnedmem.cuh"

#ifdef __cplusplus
#define EXTERN extern "C"
#else
#define EXTERN
#endif

#define LCUDA_SUCCESS 1
#define LCUDA_FAIL 0

#define PRINTF(...) printf(__VA_ARGS__)
#define FLOAT_MAX 1 

typedef struct {
    int width;
    int height;
} rect2d;

typedef struct {
    int x;
    int y;
} point2d;

typedef struct {
    int width;
	int height;
    unsigned int pitch;
    float *data;
} cudaImage;

typedef struct {
	cudaImage image;
	rect2d border;
} cudaPaddedImage;

void allocImageOnDevice(cudaImage &image);
void fillImageOnDevice(cudaImage &image, const float value);

void copyImageToDevice(float *hostImage, cudaImage &image);
float* copyImageToHost(cudaImage &image);

cudaImage createImage(int width, int height, float defaultValue);
cudaPaddedImage createPaddedImage(rect2d border, rect2d size, float defaultValue);
cudaPaddedImage padImage(cudaImage image, rect2d border, int borderColor);


float *getBorderOffsetImagePtr(cudaPaddedImage padded);

int getPitch(cudaImage image);
int getPitch(cudaPaddedImage padded);

float *getData(cudaImage image);
float *getData(cudaPaddedImage padded);

rect2d getBorder(cudaPaddedImage padded);

void exitOnError(const char *whereAt);

#endif /* LTU_CUDA_BASE_H_ */
