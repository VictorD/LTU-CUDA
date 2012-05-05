#include "fill.cuh"
#include <thrust/device_ptr.h>
#include <thrust/fill.h>

__global__ void borderFillKernel(float *data, int pitch, int width, int height, float value) {
	int x   = blockIdx.x * blockDim.x + threadIdx.x;
	int y   = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < width && y < height) {
		data[y*pitch + x] = value;
	}

}

void fillImage(cudaPaddedImage padded, float defaultValue) {
	int steps = (padded.image.width+256)/256;
	dim3 gridSize(steps, padded.image.height);
	dim3 blockSize(256, 1);
	borderFillKernel<<<gridSize, blockSize>>>(padded.image.data, padded.image.pitch/sizeof(float), padded.image.width, padded.image.height, defaultValue);
}


void thrustFillImage(cudaImage &image, const float value) {
	// Fill array with defaultValue
    thrust::device_ptr<float> dev_ptr(image.data);
    thrust::fill(dev_ptr, dev_ptr + (image.height-1) * (image.pitch/sizeof(float)) + image.width, value);
    exitOnError("createPaddedArray: thrust::fill");
}