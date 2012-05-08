#include "matlabBridge.h"
#include "../pinnedmem.cuh"

unsigned char mlcudaGetImageDataType(const mxArray* mx) {
    return ((unsigned char*)mxGetData( mxGetField(mx, 0, "dataType") ))[0];
}

cudaImage imageFromMXArray(const mxArray *mx) {
	if (!mxIsSingle(mx)) {
		mexErrMsgTxt("The input image must be of type single");
	}

	cudaImage tmp;
	tmp.width  = (int) mxGetM(mx);
	tmp.height = (int) mxGetN(mx);
	deviceAllocImageWithData(tmp, (float*)mxGetData(mx));
	return tmp;
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