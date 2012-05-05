#ifndef _TRANSPOSE_KERNEL_HEADER_
#define _TRANSPOSE_KERNEL_HEADER_

#include "../ltucuda.cuh"

#define TILE_DIM 32
#define BLOCK_ROWS 8
#define NUM_REPS 100

void transposeImage(float *dev_in, float *dev_out, rect2d image);
cudaImage createTransposedImage(cudaImage input);
#endif