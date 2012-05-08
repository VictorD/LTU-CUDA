#include "ltucuda/matlabBridge/matlabBridge.h"
#include "ltucuda/pgm/pgm.cuh"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    const mxArray *mx = prhs[0];

    char dataType = mlcudaGetImageDataType(mx);

    // UINT8
    /*if (dataType == 0) {
	    cudaPaddedImage padded = imageFromMXStruct(mx);
      	plhs[0] = mlcudaMatrixToMx_8u(im);

    } 
    // FLOAT
    else*/ 
	if (dataType == 1) {   
		cudaImage image = imageFromMXStruct(mx);
      	plhs[0] = imageToMXArray(image);
	} else {
        printf("ERROR: UNKNOWN DATA TYPE\n");
    }
}