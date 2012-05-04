#include "pgmTest.h"
#include "pgm.cuh"
#include "cudaBase.h"
#include <cassert>
#include <stdio.h>

void testPGM() {
	printf("Testing PGM load and save... ");
	int width,height;
	float *imageIn = loadPGM("tests/pgmIn.pgm", &width, &height);

	savePGM("tests/pgmOut.pgm", imageIn, width, height);

	int savedWidth, savedHeight;
	float *imageSaved = loadPGM("tests/pgmOut.pgm", &savedWidth, &savedHeight);

	assert(savedWidth == width && savedHeight == height);

	int match = 0;
	for(int i = 0; i < width; i++)
		for(int j = 0; j < height; j++)
			if (imageIn[j*width+i] == imageSaved[j*width+i])
				match++;

	assert(match == width*height);

	printf("OK!\n");

	freeHost(imageIn, REGULAR);
}