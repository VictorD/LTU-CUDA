#include "ltucuda/matlabBridge/matlabBridge.h"
#include "ltucuda/pgm/pgm.cuh"

void mexFunction(int nlhs, mxArray *plhs[],
		int nrhs, const mxArray *prhs[])
{
	if (nrhs != 2) {
		mexErrMsgTxt("The number of input arguments must be 2, one image struct and one data array");
	}

    const mxArray *mx = prhs[0];
	const mxArray *mxData = prhs[1];

    if (mxIsSingle(mxData)) {
		cudaImage ptr = imageFromMXStruct(mx);
		copyMXArrayToImage(ptr, mxData);

		plhs[0] = imageToMXStruct(ptr);

    } /*else if (mxIsUint8(mx)) {
	    cudaImage image = cudaImageFrom8bitMX(const mx);
      	plhs[0] = cudaMatrixToStruct(image);

	} */ else {
        printf("UNKNOWN. (This is bad)\n");
    }
}
