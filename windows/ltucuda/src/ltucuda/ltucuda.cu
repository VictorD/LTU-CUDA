#include "ltucuda.cuh"
#include <memory>
#include <iostream>
#include <cassert>
#include <stdio.h>
#include <string.h>
#include "pgm/pgm.cuh"
#include "kernels/fill.cuh"
#include "pinnedmem.cuh"
#ifdef _CHAR16T
#define CHAR16_T
#endif
#include <mex.h>


void deviceToDevice(cudaImage &dst, float *data) {
	cudaMemcpy2D(dst.data, dst.pitch, data, dst.width * sizeof(float), dst.width * sizeof(float), dst.height, cudaMemcpyDeviceToDevice);
    exitOnError("setImageDeviceData");
}

cudaImage cloneImage(cudaImage image) {
	cudaImage newImg;
	newImg.width = image.width;
	newImg.height = image.height;
	allocImageOnDevice(newImg);
	deviceToDevice(newImg, image.data);
	return newImg;
}

cudaPaddedImage allocPaddedImageOnDevice(cudaImage image, rect2d border, float defaultValue) {
	rect2d imageSize = {image.width, image.height};
	cudaPaddedImage padded = createPaddedImage(border, imageSize, defaultValue);
	cudaMemcpy2D(getBorderOffsetImagePtr(padded), padded.image.pitch, image.data, image.width * sizeof(float), image.width * sizeof(float), image.height, cudaMemcpyHostToDevice);
    exitOnError("loadPaddedImageToDevice: copy");
	return padded;
}

cudaPaddedImage createPaddedImage(rect2d border, rect2d size, float defaultValue) {
	cudaPaddedImage result;
	result.image = createImage(size.width + 2*border.width, size.height + 2*border.height, defaultValue);
	result.border = border;
	fillImage(result, defaultValue);
    return result;
}

cudaImage createImage(int width, int height, float defaultValue) {
	cudaImage image;
	image.width  = width;
    image.height = height;
	allocImageOnDevice(image);
    return image;
}

void allocImageOnDevice(cudaImage &image) {
	cudaMallocPitch((void **)&image.data, (size_t*)&image.pitch, image.width * sizeof(float), image.height);
    exitOnError("copyImageToDevice: alloc");
}

void setImageDeviceData(cudaImage &image, float *data) {
	cudaMemcpy2D(image.data, image.pitch, data, image.width * sizeof(float), image.width * sizeof(float), image.height, cudaMemcpyHostToDevice);
    exitOnError("setImageDeviceData");
}


void copyImageToDevice(float *data, cudaImage &image) {
	allocImageOnDevice(image);
    setImageDeviceData(image, data);
}


float* copyImageToHost(cudaImage &image) {
    float *host;
    int bytesNeeded = image.width*image.height*sizeof(float);
    mallocHost((void**)&host,bytesNeeded, PINNED, false);
	exitOnError("mallocHost Error");
	//mexPrintf("\nFetching image data from %016llX\n", image.data);
    cudaMemcpy2D(host, image.width * sizeof(float), image.data, image.pitch, image.width*sizeof(float), image.height, cudaMemcpyDeviceToHost);
    exitOnError("copyImageToHost");
    return host;
}

void copyImageToHost(cudaImage &matrix, float* data) {
	cudaMemcpy2D(data, matrix.width * sizeof(float),  matrix.data, matrix.pitch, matrix.width * sizeof(float), matrix.height, cudaMemcpyDeviceToHost);

    exitOnError("copyImageToHost: copy");
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
            //exit(-1);
			system("pause");//mexCallMATLAB("pause");
        }
}

extern void showMemUsage();

