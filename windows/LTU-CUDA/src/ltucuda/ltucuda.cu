#include "ltucuda.cuh"
#include <memory>
#include <iostream>
#include <cassert>
#include <stdio.h>
#include <string.h>
#include <thrust/device_ptr.h>
#include <thrust/fill.h>

#include "pgm/pgm.cuh"
#include "kernels/fill.cuh"

void allocImageOnDevice(cudaImage &image) {
	cudaMallocPitch((void **)&image.data, (size_t*)&image.pitch, image.width * sizeof(float), image.height);
    exitOnError("copyImageToDevice: alloc");
}

void setImageDeviceData(cudaImage &image, float *data) {
	cudaMemcpy2D(image.data, image.pitch, data, image.width * sizeof(float), image.width * sizeof(float), image.height, cudaMemcpyHostToDevice);
    exitOnError("setImageDeviceData");
}

void fillImageOnDevice(cudaImage &image, const float value) {
	// Fill array with defaultValue
    thrust::device_ptr<float> dev_ptr(image.data);
    thrust::fill(dev_ptr, dev_ptr + (image.height-1) * (image.pitch/sizeof(float)) + image.width, value);
    exitOnError("createPaddedArray: thrust::fill");
}

void copyImageToDevice(float *data, cudaImage &image) {
	allocImageOnDevice(image);
    setImageDeviceData(image, data);
}

float* copyImageToHost(cudaImage &image) {
    float *host;
    int bytesNeeded = image.width*image.height*sizeof(float);
    mallocHost((void**)&host,bytesNeeded, PINNED, false);

    cudaMemcpy2D(host, image.width * sizeof(float), image.data, image.pitch, image.width*sizeof(float), image.height, cudaMemcpyDeviceToHost);
    exitOnError("copyImageToHost: copy");
    return host;
}

cudaImage createImage(int width, int height, float defaultValue) {
	cudaImage image;
	image.width  = width;
    image.height = height;

	allocImageOnDevice(image);
	//fillImageOnDevice(image, defaultValue);
/*
    thrust::device_ptr<float> dev_ptr(image.data);
    thrust::fill(dev_ptr, dev_ptr + (image.height-1) * (image.pitch/sizeof(float)) + image.width, defaultValue);
    exitOnError("createPaddedArray: thrust::fill");*/

    return image;
}

// Add border padding
cudaPaddedImage createPaddedImage(rect2d border, rect2d size, float defaultValue) {
	cudaPaddedImage result;
	result.image = createImage(size.width + 2*border.width, size.height + 2*border.height, defaultValue);
	result.border = border;
	//fillImageOnDevice(result.image, defaultValue);
	fillImageBorder(result, defaultValue);
    return result;
}

cudaPaddedImage padImage(cudaImage image, rect2d border, int borderColor) {
	rect2d imageSize = {image.width, image.height};
	cudaPaddedImage padded = createPaddedImage(border, imageSize, borderColor);
	float *noBorderPtr = getBorderOffsetImagePtr(padded);
    cudaMemcpy2D(noBorderPtr, getPitch(padded), image.data, image.pitch, image.width*sizeof(float), image.height, cudaMemcpyDeviceToDevice);
    exitOnError("copy image data into padded array!");
	return padded;
}

int getPitch(cudaImage image) {
	return image.pitch;
}

int getPitch(cudaPaddedImage padded) {
	return getPitch(padded.image);
}

float *getData(cudaImage image) {
	return image.data;
}

float *getData(cudaPaddedImage padded) {
	return getData(padded.image);
}

rect2d getBorder(cudaPaddedImage padded) {
	return padded.border;
}

float *getBorderOffsetImagePtr(cudaPaddedImage padded) {
	return getData(padded) + getBorder(padded).height * getPitch(padded)/sizeof(float) + getBorder(padded).width;
}


/*
 * exitOnError: Show the error message and terminate the application.
 */ 
void exitOnError(const char *whereAt) {
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
        {
            // print the CUDA error message and exit
            printf("CUDA error at %s: %s\n", whereAt, cudaGetErrorString(error));
            exit(-1);
        }
}

extern void showMemUsage();


