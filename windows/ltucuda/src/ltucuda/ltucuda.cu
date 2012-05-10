#include "ltucuda.cuh"
#include "pgm/pgm.cuh"
#include "kernels/fill.cuh"
#include "pinnedmem.cuh"
#include <memory>
#include <iostream>
#include <cassert>
#include <stdio.h>
#include <string.h>

// Needed for compatibility MATLAB <-> VS2010
#ifdef _CHAR16T
#define CHAR16_T
#endif
#include <mex.h>

cudaImage cloneImage(cudaImage image) {
	cudaImage newImg;
	newImg.width = image.width;
	newImg.height = image.height;
	deviceAllocImage(newImg);
	copyDeviceToImage(newImg, image.data);
	return newImg;
}

cudaPaddedImage createPaddedImage(rect2d border, int width, int height, float fillVal) {
	cudaPaddedImage result;
	result.image = createImage(width + 2*border.width, height + 2*border.height);
	result.border = border;
	fillImage(result, fillVal);
    return result;
}

cudaImage createImage(int width, int height) {
	cudaImage image;
	image.width  = width;
    image.height = height;
	deviceAllocImage(image);
    return image;
}

void deviceAllocImage(cudaImage &image) {
	cudaMallocPitch((void **)&image.data, (size_t*)&image.pitch, image.width * sizeof(float), image.height);
    exitOnError("deviceAllocImage");
}

void copyHostToImage(cudaImage &image, float *data) {
	cudaMemcpy2D(image.data, image.pitch, data, image.width * sizeof(float), image.width * sizeof(float), image.height, cudaMemcpyHostToDevice);
    exitOnError("copyHostToImage");
}

void copyDeviceToImage(cudaImage &dst, float *data) {
	cudaMemcpy2D(dst.data, dst.pitch, data, dst.width * sizeof(float), dst.width * sizeof(float), dst.height, cudaMemcpyDeviceToDevice);
    exitOnError("copyHostToImage");
}

void copyPaddedToImage(cudaPaddedImage &src, cudaImage &dst) {
	cudaMemcpy2D(dst.data, dst.pitch, getBorderOffsetImagePtr(src), getPitch(src.image), dst.width * sizeof(float), dst.height, cudaMemcpyDeviceToDevice);
    exitOnError("copyHostToImage");
}

void deviceAllocImageWithData(cudaImage &image, float *data) {
	deviceAllocImage(image);
    copyHostToImage(image, data);
}

float* copyImageToHost(cudaImage &image) {
    float *host;
    int bytesNeeded = image.width*image.height*sizeof(float);
    mallocHost((void**)&host,bytesNeeded, PINNED, false);

	copyImageToHost(image, host);
    return host;
}

void copyImageToHost(cudaImage &matrix, float* data) {
	cudaMemcpy2D(data, matrix.width * sizeof(float),  matrix.data, matrix.pitch, matrix.width * sizeof(float), matrix.height, cudaMemcpyDeviceToHost);
    exitOnError("copyImageToHost:");
}

cudaPaddedImage createPaddedFromImage(cudaImage image, rect2d border, float borderColor) {
	cudaPaddedImage padded = createPaddedImage(border, image.width, image.height, borderColor);
    cudaMemcpy2D(getBorderOffsetImagePtr(padded), getPitch(padded), getData(image), getPitch(image), image.width*sizeof(float), image.height, cudaMemcpyDeviceToDevice);
    exitOnError("deviceAllocPadded");
	return padded;
}



/*
 *    Getters 
 * 
 *  *********************************************************
 */
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

rect2d getNoBorderSize(cudaPaddedImage padded) {
	rect2d imgSize = { padded.image.width - 2*padded.border.width, padded.image.height - 2*padded.border.height };
	return imgSize;
}

float *getBorderOffsetImagePtr(cudaImage image) {
	return getData(image);
}

float *getBorderOffsetImagePtr(cudaPaddedImage padded) {
	return getData(padded) + getBorder(padded).height * getPitch(padded)/sizeof(float) + getBorder(padded).width;
}

void exitOnError(const char *whereAt) {
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
        {
            printf("CUDA error at %s: %s\n", whereAt, cudaGetErrorString(error));
            exit(-1);
        }
}