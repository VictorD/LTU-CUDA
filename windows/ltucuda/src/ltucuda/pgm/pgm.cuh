#ifndef LCUDA_PGM_H_
#define LCUDA_PGM_H_

#include "../ltucuda.cuh"

float* loadPGM(const char *filename, int *width, int *height);
void savePGM(const char *filename, float* imageData, int width, int height);

cudaImage loadImageToDevice(const char *filename);
cudaPaddedImage loadPaddedImageToDevice(const char *filename, rect2d border, float defaultValue);


#endif
