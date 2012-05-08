#include "ltucuda/matlabBridge/matlabBridge.h"
#include "ltucuda/morphology/morphology.cuh"
#include "ltucuda/kernels/transpose.cuh"
#include "ltucuda/pgm/pgm.cuh"
#include <math.h>


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
	if (nrhs != 2) {
		mexErrMsgTxt("The number of input arguments must be 2, one image struct and one cuda structure element");
	}

	//lcudaStrel_8u se = mlcudaStructToStrel_8u(prhs[1]);
	unsigned char *pr = (unsigned char *)mxGetData(prhs[1]);

    const mxArray *mx = prhs[0];
    char dataType = mlcudaGetImageDataType(mx);

    // UINT8
    /*if (dataType == 0) {
        mlcudaErodeDilate_8u(mlcudaStructToMatrix_8u(prhs[0]), se, false);
    } 
    // FLOAT
    else*/
	if (dataType == 1 && pr[0] > 0 && pr[0] < 512) {
		cudaImage image = imageFromMXStruct(mx);

		rect2d border = {256,256};
		cudaPaddedImage tmp = createPaddedFromImage(image, border, 255.0f);

		rect2d roi = {image.width, image.height};

		/* DIAGONAL */
		
		morphMask diagMask = createVHGWMask(pr[0], DIAGONAL_BACKSLASH);
		performErosion(getData(tmp), getPitch(tmp), getData(image), getPitch(image), roi, diagMask, tmp.border);
		
		
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

