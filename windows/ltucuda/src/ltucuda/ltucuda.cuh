#ifndef _LTU_CUDA_BASE_H_
#define _LTU_CUDA_BASE_H_

// Needed for compatibility MATLAB <-> VS2010
#ifdef _CHAR16T
#define CHAR16_T
#endif

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



void deviceAllocImage(cudaImage &image);
void deviceFillImage(cudaImage &image, const float value);
void copyHostToImage(cudaImage &image, float *data);

cudaImage cloneImage(cudaImage image);

void deviceAllocImageWithData(cudaImage &image, float *data);
float* copyImageToHost(cudaImage &image);
void copyImageToHost(cudaImage &from, float* to);
void copyDeviceToImage(cudaImage &to, float *from);
void copyPaddedToImage(cudaPaddedImage &src, cudaImage &dst);

cudaImage createImage(int width, int height);
cudaPaddedImage createPaddedImage(rect2d border, int width, int height, float defaultValue);
cudaPaddedImage createPaddedFromImage(cudaImage image, rect2d border, float defaultValue);

float *getBorderOffsetImagePtr(cudaPaddedImage padded);

int getPitch(cudaImage image);
int getPitch(cudaPaddedImage padded);

float *getData(cudaImage image);
float *getData(cudaPaddedImage padded);

rect2d getBorder(cudaPaddedImage padded);
rect2d getNoBorderSize(cudaPaddedImage padded);

void exitOnError(const char *whereAt);

#endif /* LTU_CUDA_BASE_H_ */