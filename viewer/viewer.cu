#include <iostream>
#include <memory>
#include <cassert>
#include <thrust/device_ptr.h>
#include <thrust/fill.h>
#include "morphology.cuh"
#include "pgm.cuh"
#include <sys/time.h>

float* copyImageToDevice(float *hostImage, rect2d image, int *pitch) {
    float *dev;
    cudaMallocPitch((void **)&dev, (size_t*)pitch, image.width * sizeof(float), image.height);
    exitOnError("copyImageToDevice: alloc");
 
    cudaMemcpy2D(dev, *pitch, hostImage, image.width * sizeof(float),
    			image.width * sizeof(float), image.height, cudaMemcpyHostToDevice);
    exitOnError("copyImageToDevice: copy");
    return dev;
}

float* copyImageToHost(float *device_data, rect2d image, int pitch) {
    float *host;
    int bytesNeeded = image.width*image.height*sizeof(float);
    mallocHost((void**)&host,bytesNeeded, PINNED, false);
    cudaMemcpy2D(host, image.width * sizeof(float), device_data, pitch, image.width*sizeof(float), image.height, cudaMemcpyDeviceToHost);
    exitOnError("copyImageToHost: copy");
    return host;
}

// Add border padding
float* createPaddedArray(rect2d borderSize, rect2d imgSize, float defaultValue, int *pitch) {
    float *padded;
    int paddedHeight = imgSize.height + 2*borderSize.height;
    int paddedWidth = imgSize.width + 2*borderSize.width;

    cudaMallocPitch((void **)&padded, (size_t*)pitch, paddedWidth * sizeof(float), paddedHeight);
    exitOnError("createPaddedArray: alloc");

    thrust::device_ptr<float> dev_ptr(padded);
    thrust::fill(dev_ptr, dev_ptr + (paddedHeight-1) * (*pitch/sizeof(float)) + paddedWidth, 255.0f);
    exitOnError("createPaddedArray: thrust::fill");
    return padded;
}

int doStuff( int argc, const char* argv[] )
{
    rect2d image;
    float *host_in = loadPGM("../images/hmap.pgm", &image.width, &image.height);
    int pitch_in;
    timeval start, stop, result;
    float *dev_in = copyImageToDevice(host_in, image, &pitch_in);

    gettimeofday(&start, NULL);

    unsigned char maskData[9] = {0,0,0,1,1,1,0,0,0};
    morphMask mask1 = createFlatHorizontalLineMask(43);
    morphMask mask2 = createFlatVerticalLineMask(43);
    
    // Create padded array on device, filled with 255.0f
    rect2d border = {mask1.width/2,mask2.height/2};
    int padded_pitch;
    int padded_pitch2;
    float *padded_dev_in = createPaddedArray(border, image, 255.0f, &padded_pitch);
    float *padded_dev_in2 = createPaddedArray(border, image, 255.0f, &padded_pitch2);

    float *data = padded_dev_in + border.height * padded_pitch/sizeof(float) + border.width;
    cudaMemcpy2D(data, padded_pitch, dev_in, pitch_in, image.width*sizeof(float), image.height, cudaMemcpyDeviceToDevice);
    exitOnError("copy image data into padded array!");

    // Erode
    float *data2 = padded_dev_in2 + border.height * padded_pitch2/sizeof(float) + border.width;    
    performErosion(padded_dev_in, padded_pitch, data2, padded_pitch2, image, mask1, border);
    performErosion(padded_dev_in2, padded_pitch2, dev_in, pitch_in, image, mask2, border);

    // Copy resulting image from device to host
    float *host_out = copyImageToHost(dev_in, image, pitch_in);

    gettimeofday(&stop, NULL);

    savePGM("fileTest_out.pgm", host_out, image.width, image.height);

    timersub(&stop, &start, &result);
    printf("Erosion took: %d microsec\n", result.tv_sec*1000000 + result.tv_usec);
    // Free memory
    cudaFree(dev_in);
    cudaFree(mask1.data);
    cudaFree(padded_dev_in);
    freeHost(host_in, REGULAR);
    freeHost(host_out, PINNED);
}
