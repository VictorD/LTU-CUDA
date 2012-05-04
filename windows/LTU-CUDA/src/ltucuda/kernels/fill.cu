#include "fill.cuh"

__global__ void borderFillKernel(float *data, int pitch, int width, int height, float value) {
	int x   = blockIdx.x * blockDim.x + threadIdx.x;
	int y   = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < width && y < height) {
		data[y*pitch + x] = value;
	}

}

/*
template <typename T>
__global__ void initMatrix(T *matrix, int width, int height, T val) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = idx; i < width * height; i += gridDim.x * blockDim.x) {
        matrix[i]=val;
    }
}*/

void fillImageBorder(cudaPaddedImage padded, float defaultValue) {
	int steps = (padded.image.width+256)/256;
	dim3 gridSize(steps, padded.image.height);
	dim3 blockSize(256, 1);
	borderFillKernel<<<gridSize, blockSize>>>(padded.image.data, padded.image.pitch/sizeof(float), padded.image.width, padded.image.height, defaultValue);
}