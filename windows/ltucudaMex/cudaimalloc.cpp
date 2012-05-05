
#include <mex.h>
#include "ltucuda/ltucuda.cuh"

cudaImage lcudaImageFromMX(const mxArray *mx) {
	cudaImage tmp;

	if (!mxIsSingle(mx)) {
		mexErrMsgTxt("The input image must be of type single");
	}

	tmp.width = mxGetM(mx);
	tmp.height = mxGetN(mx);
	tmp.data = (float*)mxGetData(mx);
	return tmp;
}

mxArray* lcudaMatrixToStruct(cudaImage im) {
	return NULL;
}

void mexFunction(int nlhs, mxArray *plhs[],
		int nrhs, const mxArray *prhs[])
{
    const mxArray *mx = prhs[0];
    if (mxIsSingle(mx)) {
        cudaImage image = lcudaImageFromMX(mx);
      	plhs[0] = lcudaMatrixToStruct(image);

    } /*else if (mxIsUint8(mx)) {
	    cudaImage image = cudaImageFrom8bitMX(const mx);
      	plhs[0] = cudaMatrixToStruct(image);

	} */ else {
        printf("UNKNOWN. (This is bad)\n");
    }
}
