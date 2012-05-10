#include <memory>
#include "matlabBridge.h"
#include "../pinnedmem.cuh"

unsigned char mlcudaGetImageDataType(const mxArray* mx) {
    return ((unsigned char*)mxGetData( mxGetField(mx, 0, "dataType") ))[0];
}

cudaImage allocImage() {
	cudaImage tmp;
	tmp.width = 2048;
	tmp.height = 2048;
	tmp.allocWidth = tmp.width;
	tmp.allocHeight = tmp.height;
	deviceAllocImage(tmp);
	return tmp;
}

void copyMXArrayToImage(cudaImage &image, const mxArray *mx) {

	image.width  = (int) mxGetM(mx);
	image.height = (int) mxGetN(mx);
	if (!mxIsSingle(mx)) {
		printf(" %d %d", image.width, image.height);
		mexErrMsgTxt("The input image must be of type single!");
	}

	copyHostToImage(image, (float*)mxGetData(mx));
	//deviceAllocImageWithData(tmp, (float*)mxGetData(mx));
}

cudaImage imageFromMXStruct(const mxArray *mx) {
	if (!mxIsStruct(mx)) {
		mexErrMsgTxt("The input image must be a struct");
	}

	cudaImage im;
	im.data   = (float*) ((__int64*)mxGetData( mxGetField(mx, 0, "address") ))[0];
	im.width  = ((int*)mxGetData( mxGetField(mx, 0, "width") ))[0];
	im.height = ((int*)mxGetData( mxGetField(mx, 0, "height") ))[0];
	im.pitch  = ((int*)mxGetData( mxGetField(mx, 0, "pitch") ))[0];
	return im;
}

mxArray* imageToMXArray(cudaImage image) {
	mxArray* tmp = mxCreateNumericMatrix(image.width, image.height, mxSINGLE_CLASS, mxREAL);
	copyImageToHost(image, (float*)mxGetData(tmp));
	return tmp;
}

mxArray* imageToMXStruct(cudaImage image) {
	const char* fields[5] = { "address", "width", "height", "pitch", "dataType" };

	mxArray* ima  = mxCreateNumericMatrix(1,1, mxUINT64_CLASS, mxREAL);
	mxArray* imw  = mxCreateNumericMatrix(1,1, mxUINT32_CLASS, mxREAL);
	mxArray* imh  = mxCreateNumericMatrix(1,1, mxUINT32_CLASS, mxREAL);
	mxArray* imp  = mxCreateNumericMatrix(1,1, mxUINT32_CLASS, mxREAL);
	mxArray* imt  = mxCreateNumericMatrix(1,1, mxUINT8_CLASS, mxREAL);
	*(__int64*)mxGetData(ima) = (__int64)image.data;
	*(unsigned int*)mxGetData(imw) = image.width;
	*(unsigned int*)mxGetData(imh) = image.height;
	*(unsigned int*)mxGetData(imp) = image.pitch;
    *(unsigned char*)mxGetData(imt) = 1;

	mxArray* ims = mxCreateStructMatrix(1, 1, 5, fields);
	mxSetField(ims, 0, "address", ima);
	mxSetField(ims, 0, "width", imw);
	mxSetField(ims, 0, "height", imh);
	mxSetField(ims, 0, "pitch", imp);
    mxSetField(ims, 0, "dataType", imt);

	return ims;
}

bool isDiagonal(unsigned char* maskData, int width, int height);
bool isBackslash(unsigned char* maskData, int width, int height);

vector<morphMask> morphMaskFromMXStruct(const mxArray *mx) {
	if (!mxIsStruct(mx)) {
		mexErrMsgTxt("The input structure element must be a struct");
	}

	const mxArray *dataField = mxGetField(mx, 0, "data");
	const mxArray *heightField = mxGetField(mx, 0, "heights");
	unsigned char* maskData = (unsigned char*)mxGetData(dataField);

	int strelCount = ((int*)mxGetData( mxGetField(mx, 0, "num") ))[0];
	rect2d  *sizes = (rect2d*)mxGetData( mxGetField(mx, 0, "sizes") );
	int    *isFlat = ((int*)mxGetData( mxGetField(mx, 0, "isFlat") ));
	float *heights = (float*)mxGetData(heightField);

	vector<morphMask> result;
	for(int i = 0; i < strelCount; i++) {
		int width  = sizes[i].width;
		int height = sizes[i].height;

		morphMask m;
		if (height == 1) {
			//printf("hozmask %d\n", width);
			result.push_back(createVHGWMask(width,  HORIZONTAL));
		} else if (width == 1) {
			//printf("vertmask %d\n", height);
			result.push_back(createVHGWMask(height, VERTICAL));
		} else if (height == 3 && width == 3) {
			//printf("tbtMask\n");
			unsigned char *d = new unsigned char[9];
			memcpy(d, maskData, 9);
			result.push_back(createTBTMask(d));
		} else {
			if (height == width) {
				bool diag = isDiagonal(maskData, width, height);
				bool bs = isBackslash(maskData, width, height);
				if (diag) {
					result.push_back(createVHGWMask(width, DIAGONAL_BACKSLASH));
				} else if (bs) {
					result.push_back(createVHGWMask(width, DIAGONAL_SLASH));
				}
			}
		}
		maskData += width*height;
	}

	return result;
}

bool isDiagonal(unsigned char* maskData, int width, int height) {
	bool diagonal = true;
	for (int w = 0; w < width; w++) {
		for(int h = 0; h < height; h++) {
			if ((h == w && maskData[width*h+w] == 0) ||
				(h != w && maskData[width*h+w] == 1)) {
				diagonal = false;
			}
		}
	}
	return diagonal;
}

bool isBackslash(unsigned char* maskData, int width, int height) {
	bool bs = true;
	for (int w = 0; w < width; w++) {
		for(int h = 0; h < height; h++) {
			if ((h == w && maskData[width*h+(width-1-w)] == 0) ||
				(h != w && maskData[width*h+(width-1-w)] == 1)) {
				bs = false;
			}
		}
	}
	return bs;
}