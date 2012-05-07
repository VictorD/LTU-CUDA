#include "matlabBridge.h"
#include "../pinnedmem.cuh"
unsigned char mlcudaGetImageDataType(const mxArray* mx) {
    return ((unsigned char*)mxGetData( mxGetField(mx, 0, "dataType") ))[0];
}

cudaImage cudaImageFromMX(const mxArray *mx) {
	cudaImage tmp;

	if (!mxIsSingle(mx)) {
		mexErrMsgTxt("The input image must be of type single");
	}

	tmp.width = mxGetM(mx);
	tmp.height = mxGetN(mx);
	tmp.data = (float*)mxGetData(mx);
	return tmp;
}

cudaPaddedImage cudaPaddedImageFromStruct(const mxArray *mx) {
	if (!mxIsStruct(mx)) {
		mexErrMsgTxt("The input image must be a struct");
	}

	cudaPaddedImage padded;
	cudaImage im;
	im.data = (float*) ((unsigned long long*)mxGetData( mxGetField(mx, 0, "address") ))[0];
	im.width = ((int*)mxGetData( mxGetField(mx, 0, "width") ))[0];
	im.height = ((int*)mxGetData( mxGetField(mx, 0, "height") ))[0];
	im.pitch = ((int*)mxGetData( mxGetField(mx, 0, "pitch") ))[0];
	int borderWidth = ((int*)mxGetData( mxGetField(mx, 0, "borderWidth") ))[0];
	int borderHeight = ((int*)mxGetData( mxGetField(mx, 0, "borderHeight") ))[0];
	rect2d border = {borderWidth, borderHeight};
	padded.border = border;
	padded.image = im;
	return padded;
}

mxArray* mxArrayFromLCudaMatrix(cudaPaddedImage padded) {
	int width = padded.image.width - 2*padded.border.width;
	int height = padded.image.height - 2*padded.border.height;
	mxArray* tmp = mxCreateNumericMatrix(width, height, mxSINGLE_CLASS, mxREAL);
	//printf("width %d , height %d\n", width, height);
	//copyImageToHost(padded.image, (float*)mxGetData(tmp));
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
	  *(unsigned long long*)mxGetData(ima) = (unsigned long long)padded.image.data;
	  *(unsigned int*)mxGetData(imw) = padded.image.width;
	  *(unsigned int*)mxGetData(imh) = padded.image.height;
	  *(unsigned int*)mxGetData(imp) = padded.image.pitch;
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