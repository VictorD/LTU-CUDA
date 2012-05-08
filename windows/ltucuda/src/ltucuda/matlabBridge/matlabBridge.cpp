#include "matlabBridge.h"
#include "../pinnedmem.cuh"


unsigned char mlcudaGetImageDataType(const mxArray* mx) {
    return ((unsigned char*)mxGetData( mxGetField(mx, 0, "dataType") ))[0];
}

cudaImage cudaImageFromMX(const mxArray *mx) {
	if (!mxIsSingle(mx)) {
		mexErrMsgTxt("The input image must be of type single");
	}

	cudaImage tmp;
	tmp.width = (int) mxGetM(mx);
	tmp.height = (int) mxGetN(mx);
	tmp.data = (float*)mxGetData(mx);
	return tmp;
}

cudaPaddedImage cudaPaddedImageFromStruct(const mxArray *mx) {
	if (!mxIsStruct(mx)) {
		mexErrMsgTxt("The input image must be a struct");
	}

	cudaImage im;
	im.data   = (float*) ((size_t*)mxGetData( mxGetField(mx, 0, "address") ))[0];
	im.width  = ((int*)mxGetData( mxGetField(mx, 0, "width") ))[0];
	im.height = ((int*)mxGetData( mxGetField(mx, 0, "height") ))[0];
	im.pitch  = ((int*)mxGetData( mxGetField(mx, 0, "pitch") ))[0];

	cudaPaddedImage padded;
	padded.image = im;
	padded.border.width  = ((int*)mxGetData( mxGetField(mx, 0, "borderWidth") ))[0];
	padded.border.height = ((int*)mxGetData( mxGetField(mx, 0, "borderHeight")))[0];
	return padded;
}

mxArray* mxArrayFromLCudaMatrix(cudaPaddedImage padded) {
	rect2d noborder = getNoBorderSize(padded);
	int width = noborder.width;
	int height = noborder.height;
	mxArray* tmp = mxCreateNumericMatrix(width, height, mxSINGLE_CLASS, mxREAL);
	cudaMemcpy2D((float*)mxGetData(tmp), width * sizeof(float),  getBorderOffsetImagePtr(padded), padded.image.pitch, width * sizeof(float), height, cudaMemcpyDeviceToHost);
	return tmp;
}

mxArray* mxStructFromLCudaMatrix(cudaPaddedImage padded) {
	const char* fields[7] = { "address", "width", "height", "borderWidth", "borderHeight", "pitch", "dataType" };

	mxArray* ima = mxCreateNumericMatrix(1,1, mxUINT64_CLASS, mxREAL);
	mxArray* imw = mxCreateNumericMatrix(1,1, mxUINT32_CLASS, mxREAL);
	mxArray* imh = mxCreateNumericMatrix(1,1, mxUINT32_CLASS, mxREAL);
	mxArray* imp = mxCreateNumericMatrix(1,1, mxUINT32_CLASS, mxREAL);
	mxArray* imbw = mxCreateNumericMatrix(1,1, mxUINT32_CLASS, mxREAL);
	mxArray* imbh = mxCreateNumericMatrix(1,1, mxUINT32_CLASS, mxREAL);
	mxArray* imt = mxCreateNumericMatrix(1,1, mxUINT8_CLASS, mxREAL);
	*(size_t*)mxGetData(ima) = (size_t)padded.image.data;
	*(unsigned int*)mxGetData(imw) = padded.image.width;
	*(unsigned int*)mxGetData(imh) = padded.image.height;
	*(unsigned int*)mxGetData(imp) = padded.image.pitch;
	*(unsigned int*)mxGetData(imbw) = padded.border.width;
	*(unsigned int*)mxGetData(imbh) = padded.border.height;
    *(unsigned char*)mxGetData(imt) = 1;

	mxArray* ims = mxCreateStructMatrix(1, 1, 7, fields);
	mxSetField(ims, 0, "address", ima);
	mxSetField(ims, 0, "width", imw);
	mxSetField(ims, 0, "height", imh);
	mxSetField(ims, 0, "borderWidth", imbw);
	mxSetField(ims, 0, "borderHeight", imbh);
	mxSetField(ims, 0, "pitch", imp);
    mxSetField(ims, 0, "dataType", imt);

	return ims;
}