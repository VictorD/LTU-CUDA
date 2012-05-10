#include "ltucuda/matlabBridge/matlabBridge.h"
#include "ltucuda/morphology/morphology.cuh"
#include "ltucuda/kernels/transpose.cuh"
#include "ltucuda/pgm/pgm.cuh"
#include <math.h>


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
	if (nrhs != 2) {
		mexErrMsgTxt("The number of input arguments must be 2, one image struct and one cuda structure element");
	}

	const mxArray *mx = prhs[0];
	const mxArray *maskMX = prhs[1];
    char dataType = mlcudaGetImageDataType(mx);

    // UINT8
    /*if (dataType == 0) {
        mlcudaErodeDilate_8u(mlcudaStructToMatrix_8u(prhs[0]), se, false);
    } 
    // FLOAT
    else*/
	if (dataType == 1) {
		cudaImage image = imageFromMXStruct(mx);
		vector<morphMask> masks = morphMaskFromMXStruct(maskMX);

		if (masks.size() == 0) {
			printf("No valid strel element found! Aborting erosion\n");
			return;
		}


		rect2d border = {256,256}; // {mask.width , mask.height/2}
		cudaPaddedImage tmp = createPaddedFromImage(image, border, 255.0f);



		/* DIAGONAL */
		
		//morphMask diagMask = createVHGWMask(23, DIAGONAL_BACKSLASH);
		cudaImage *tmpPtr;
		cudaImage *src = &tmp.image;
		cudaImage *dst = &image;
		rect2d roi = {image.width, image.height};
/*		float *dst = getData(image);
		for(int i = 0; i < masks.size(); i++) {*/
			morphMask theMask = masks.at(0);
			performErosion(getData(*src), getPitch(tmp), getData(*dst), getPitch(image), roi, theMask, tmp.border);
			tmpPtr = src;
			src = dst;
			dst = tmpPtr;
		//}
		
		
		/* ROW FILTER */
		/*morphMask diagMask = createVHGWMask(pr[0], HORIZONTAL);
		performErosion(getData(tmp), getPitch(tmp), getData(image), getPitch(image), roi, diagMask, tmp.border);
		cudaFree(getData(tmp));*/
		
		/* ROW FILTER using transpose + COL filter + transpose */
		/*cudaImage flippedIn = createTransposedImage(tmp.image);

		rect2d fb = {tmp.border.height, tmp.border.width};
		cudaPaddedImage flippedOut;
		flippedOut.border = fb;
		flippedOut.image = createImage(flippedIn.width, flippedIn.height);

		rect2d flippedROI = getNoBorderSize(flippedOut);
		morphMask vertMask = createVHGWMask(pr[0], VERTICAL);
		performErosion(getData(flippedIn), getPitch(flippedIn), getBorderOffsetImagePtr(flippedOut), getPitch(flippedOut), flippedROI, vertMask, flippedOut.border);
		exitOnError("VHGW Horizontal Transpose Test");
		cudaFree(getData(flippedIn));

		rect2d maxSize = {tmp.image.width, tmp.image.height};
		transposeImage(flippedOut.image.data, tmp.image.data, maxSize);
		cudaFree(getData(flippedOut));

		copyPaddedToImage(tmp, image);
		cudaFree(getData(tmp));*/
    } else {
        printf("UNKNOWN. (This is bad)\n");
    }
	// Free memory
	//lcudaFreeStrel_8u(se);
}

