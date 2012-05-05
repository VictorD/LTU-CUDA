#include "pgm.cuh"
#include "pgmb_io.hpp"
#include "../pinnedmem.cuh"

#include <iostream> // cout, cerr
#include <fstream> // ifstream
#include <sstream> // stringstream
#include <vector>
using namespace std;

cudaImage loadImageToDevice(const char *filename) {
	printf("oyyoyoyobo\n\n\n\n");
	cudaImage image;
    float *host_in = loadPGM(filename, &image.width, &image.height);
	copyImageToDevice(host_in, image);
	freeHost(host_in, REGULAR);
	return image;
}

cudaPaddedImage loadPaddedImageToDevice(const char *filename, rect2d border, float defaultValue) {
	cudaImage image;
    float *data = loadPGM(filename, &image.width, &image.height);
	rect2d imageSize = {image.width, image.height};

	cudaPaddedImage padded = createPaddedImage(border, imageSize, defaultValue);
	cudaMemcpy2D(getBorderOffsetImagePtr(padded), padded.image.pitch, data, image.width * sizeof(float), image.width * sizeof(float), image.height, cudaMemcpyHostToDevice);
    exitOnError("loadPaddedImageToDevice: copy");
	freeHost(data, REGULAR);
	return padded;
}

float* loadPGM(const char *filename, int *width, int *height) {
	unsigned char maxGray = 0;
	unsigned char *image;
	int w,h;

	if (pgmb_read(filename, w, h, maxGray, &image)) {
		system("pause");
		exit(-1);
	}

	printf("Width: %d, Height: %d, MaxGray: %d\n", w, h, maxGray);

	float *floatImage = new float[w*h];

	for(int i = 0; i < w; ++i)
		for(int j = 0; j < h; ++j)
			floatImage[w*j+i] = image[w*j+i];

	*width = w;
	*height = h;
	return floatImage;
}


void savePGM(const char *filename, float* image, int width, int height) {
	// Not using pgmb_io due to bug in pgmb_write... grayscale off by 1 for some grayscale levels...
	int i,j,maxintensity;
	ofstream fp;
	maxintensity = 255;

    fp.open(filename, ios::binary);
    fp << "P5" << endl;
    fp << "#Created 1/2/02" << endl;  // Or some other comment
    fp << width << " " << height << endl;
    fp << maxintensity << endl;      // Almost always equals 255
    for (i=0;i<width;i++)
       for (j=0;j<height;j++) 
          fp << (unsigned char) image[i*height+j]; 
}