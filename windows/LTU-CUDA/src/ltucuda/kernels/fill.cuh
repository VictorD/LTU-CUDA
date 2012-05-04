#ifndef _LTU_CUDA_BORDER_FILL_
#define _LTU_CUDA_BORDER_FILL_

#include "../ltucuda.cuh"

void fillImageBorder(cudaPaddedImage padded, float defaultValue);

#endif