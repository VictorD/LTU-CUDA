#ifndef _IMG_MATRIX_PGM__H_
#define _IMG_MATRIX_PGM__H_

#include "MatrixBase.h"
#include "pgm/pgm.cuh"
#include "pgmb_io.hpp"

template <typename T> class PGMLoader { 
public:
	static MatrixBase<unsigned char> matrixFromPGM(std::string filename) {
		int width,height;
	    unsigned char *data = _loadPGM(filename, &width, &height);
		MatrixBase<unsigned char> image(width, height, data);
		return image;
	}

private:
	PGMLoader() { }

	static unsigned char * _loadPGM(std::string filename, int *width, int *height) {
		unsigned char maxGray = 0;
		unsigned char *image;
		int w,h;

		if (pgmb_read(filename, w, h, maxGray, &image)) {
			system("pause");
			exit(-1);
		}

		printf("Width: %d, Height: %d, MaxGray: %d\n", w, h, maxGray);

		unsigned char *tmp = new unsigned char[w*h];

		for(int i = 0; i < w; ++i)
			for(int j = 0; j < h; ++j)
				tmp[w*j+i] = image[w*j+i];

		*width = w;
		*height = h;
		return tmp;
	}
}; 


#endif