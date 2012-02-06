/*
 * matcuda.c
 *
 *  Created on: Feb 5, 2010
 *      Author: henmak
 */

#define DEBUG 0
#if DEBUG
#define PRINTF(...) mexPrintf(__VA_ARGS__)
#else
#define PRINTF(...)
#endif

#include "mlcudafloat.h"

lcudaMatrix mlcudaMxToMatrix(const mxArray* mx) {
	if (!mxIsSingle(mx)) {
		printf("The input image must be of type single");
	}

	int width = mxGetM(mx);
	int height = mxGetN(mx);
	lcudaMatrix matrix = lcudaAllocMatrix(width, height);

	lcudaCpyToMatrix((lcudaFloat*)mxGetData(mx), matrix);
	return matrix;
}

mxArray* mlcudaMatrixToMx(lcudaMatrix matrix) {
	mxArray* output = mxCreateNumericMatrix(matrix.width, matrix.height,
			mxSINGLE_CLASS, mxREAL);
	lcudaCpyFromMatrix(matrix, (lcudaFloat*)mxGetData(output));
	return output;
}

lcudaArray mlcudaMxToArray(const mxArray* mx) {
	int width = mxGetM(mx);
	int height = mxGetN(mx);
	lcudaArray array = lcudaAllocArray(width*height);

	lcudaCpyToArray((lcudaFloat*)mxGetData(mx), array);
	array.width = width;
	array.height = height;
	return array;
}

mxArray* mlcudaArrayToMx(lcudaArray array) {
	mxArray* output = mxCreateNumericMatrix(array.length, 1, mxUINT8_CLASS, mxREAL);
	lcudaCpyFromArray(array, (lcudaFloat*)mxGetData(output));
	return output;
}

lcudaMatrix mlcudaStructToMatrix(const mxArray* mx) {
	if (!mxIsStruct(mx)) {
		mexErrMsgTxt("The input image must be a struct");
	}

	lcudaMatrix im;
	im.data = (lcudaFloat *) ((unsigned long*)mxGetData( mxGetField(mx, 0, "address") ))[0];
	im.width = ((int*)mxGetData( mxGetField(mx, 0, "width") ))[0];
	im.height = ((int*)mxGetData( mxGetField(mx, 0, "height") ))[0];
	im.pitch = ((int*)mxGetData( mxGetField(mx, 0, "pitch") ))[0];

	PRINTF("Address %d\n", im.data);
	PRINTF("Width %d\n", im.width);
	PRINTF("height %d\n", im.height);
	PRINTF("pitch %d\n", im.pitch);

	return im;
}

mxArray *mlcudaMatrixToStruct(lcudaMatrix im) {
	const char* fields[5] = { "address", "width", "height", "pitch", "dataType" };

	mxArray* ima = mxCreateNumericMatrix(1,1, mxUINT64_CLASS, mxREAL);
	mxArray* imw = mxCreateNumericMatrix(1,1, mxUINT32_CLASS, mxREAL);
	mxArray* imh = mxCreateNumericMatrix(1,1, mxUINT32_CLASS, mxREAL);
	mxArray* imp = mxCreateNumericMatrix(1,1, mxUINT32_CLASS, mxREAL);
	mxArray* imt = mxCreateNumericMatrix(1,1, mxUINT8_CLASS, mxREAL);
	  *(unsigned long*)mxGetData(ima) = (unsigned long)im.data;
	  *(unsigned int*)mxGetData(imw) = im.width;
	  *(unsigned int*)mxGetData(imh) = im.height;
	  *(unsigned int*)mxGetData(imp) = im.pitch;
      *(unsigned char*)mxGetData(imt) = 1;

	mxArray* ims = mxCreateStructMatrix(1, 1, 5, fields);
	mxSetField(ims, 0, "address", ima);
	mxSetField(ims, 0, "width", imw);
	mxSetField(ims, 0, "height", imh);
	mxSetField(ims, 0, "pitch", imp);
    mxSetField(ims, 0, "dataType", imt);

	return ims;
}

void mlcudaErodeDilate(lcudaMatrix im, lcudaStrel_8u lcudaSe, bool dilate) {
	if (dilate) {
		lcudaDilate(im, im, lcudaSe);
	} else {
		lcudaErode(im, im, lcudaSe);
	}
}

