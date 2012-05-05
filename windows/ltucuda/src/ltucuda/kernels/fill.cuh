#ifndef _LTU_CUDA_BORDER_FILL_
#define _LTU_CUDA_BORDER_FILL_

#include "../ltucuda.cuh"

void fillImage(cudaPaddedImage padded, float defaultValue);
void thrustFillImage(cudaImage &image, const float value);

#endif