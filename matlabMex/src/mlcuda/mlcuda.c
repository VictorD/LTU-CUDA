/*
 * matcuda.c
 *
 *  Created on: Feb 5, 2010
 *      Author: henmak
 */

#include "mlcuda.h"
#include <string.h>

lcudaMatrix_8u mlcudaMxToMatrix_8u(const mxArray* mx) {
	if (!mxIsUint8(mx)) {
		mexErrMsgTxt("The input image must be of uint8 type");
	}
	int width = mxGetM(mx);
	int height = mxGetN(mx);
	lcudaMatrix_8u matrix = lcudaAllocMatrix_8u(width, height);

	lcudaCpyToMatrix_8u((lcuda8u*)mxGetData(mx), matrix);
	return matrix;
}

mxArray* mlcudaMatrixToMx_8u(lcudaMatrix_8u matrix) {
	mxArray* output = mxCreateNumericMatrix(matrix.width, matrix.height,
			mxUINT8_CLASS, mxREAL);
	lcudaCpyFromMatrix_8u(matrix, (lcuda8u*)mxGetData(output));
	return output;
}

lcudaArray_8u mlcudaMxToArray_8u(const mxArray* mx) {
	int width = mxGetM(mx);
	int height = mxGetN(mx);
	lcudaArray_8u array = lcudaAllocArray_8u(width*height);

	lcudaCpyToArray_8u((lcuda8u*)mxGetData(mx), array);
	array.width = width;
	array.height = height;
	return array;
}

mxArray* mlcudaArrayToMx_8u(lcudaArray_8u array) {
	mxArray* output = mxCreateNumericMatrix(array.length, 1, mxUINT8_CLASS, mxREAL);
	lcudaCpyFromArray_8u(array, (lcuda8u*)mxGetData(output));
	return output;
}

lcudaMatrix_8u mlcudaStructToMatrix_8u(const mxArray* mx) {
	if (!mxIsStruct(mx)) {
		mexErrMsgTxt("The input image must be a struct");
	}

	lcudaMatrix_8u im;
	im.data = (lcuda8u *) ((unsigned long*)mxGetData( mxGetField(mx, 0, "address") ))[0];
	im.width = ((int*)mxGetData( mxGetField(mx, 0, "width") ))[0];
	im.height = ((int*)mxGetData( mxGetField(mx, 0, "height") ))[0];
	im.pitch = ((int*)mxGetData( mxGetField(mx, 0, "pitch") ))[0];


	PRINTF("Address %d\n", im.data);
	PRINTF("Width %d\n", im.width);
	PRINTF("height %d\n", im.height);
	PRINTF("pitch %d\n", im.pitch);

	return im;
}

mxArray *mlcudaMatrixToStruct_8u(lcudaMatrix_8u im) {
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
      *(unsigned char*)mxGetData(imt) = 0;

	mxArray* ims = mxCreateStructMatrix(1, 1, 5, fields);
	mxSetField(ims, 0, "address", ima);
	mxSetField(ims, 0, "width", imw);
	mxSetField(ims, 0, "height", imh);
	mxSetField(ims, 0, "pitch", imp);
    mxSetField(ims, 0, "dataType", imt);

	return ims;
}

lcudaStrel_8u mlcudaStructToStrel_8u(const mxArray* mx) {
	if (!mxIsStruct(mx)) {
		mexErrMsgTxt("The input structure element must be a struct");
	}

	lcudaStrel_8u se;
    const mxArray *dataField = mxGetField(mx, 0, "data");

	se.data = mlcudaMxToArray_8u(dataField);

	se.numStrels = ((int*)mxGetData( mxGetField(mx, 0, "num") ))[0];
	//se.sizes = (lcudaSize*) mxMalloc(se.numStrels * sizeof(lcudaSize));
	se.sizes = (lcudaSize*)mxGetData( mxGetField(mx, 0, "sizes") );
    se.isFlat = ((int*)mxGetData( mxGetField(mx, 0, "isFlat") ))[0];
	se.heights = mlcudaMxToArray( mxGetField(mx, 0, "heights"));


    // To speed up processing of 3x3 SE , we store them in binary packed form as well!
    se.binary = (int*)malloc(sizeof(int) * se.numStrels);
    memset(se.binary   , 0, sizeof(int)* se.numStrels);

    lcuda8u *seData = (lcuda8u*)mxGetData(dataField);
    int i;
    for(i = 0; i < se.numStrels; ++i) {
       int sw = se.sizes[i].width;
       int sh = se.sizes[i].height;

       if (sw == 3 && sh == 3) {
           se.binary[i] = 256*seData[8] + 128*seData[7] + 64*seData[6] + 32*seData[5] + 16*seData[4] + 8*seData[3] + 4*seData[2] + 2*seData[1] + seData[0];
       }

       seData += sw*sh;
    }

#if DEBUG
    int i;
	for (i = 0; i < se.numStrels; ++i) {
		//se.sizes[i].width = sizes[i*2];
		//se.sizes[i].height = sizes[i*2 + 1];
		PRINTF("Size (%d,%d)\n", se.sizes[i].width, se.sizes[i].height);
	}
#endif
	PRINTF("Nums: %d\n", se.numStrels);

	return se;
}

void mlcudaErodeDilate_8u(lcudaMatrix_8u im, lcudaStrel_8u lcudaSe, bool dilate) {
	if (dilate) {
		lcudaDilate_8u(im, im, lcudaSe);
	} else {
		lcudaErode_8u(im, im, lcudaSe);
	}
}

unsigned char mlcudaGetImageDataType(const mxArray* mx) {
    return ((unsigned char*)mxGetData( mxGetField(mx, 0, "dataType") ))[0];
}
