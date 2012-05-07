#include "ltucuda/matlabBridge/matlabBridge.h"
#include "ltucuda/pinnedmem.cuh"

void mexFunction(int nlhs, mxArray *plhs[],
		int nrhs, const mxArray *prhs[])
{
	/*size_t free = 0, total = 0;
    cudaError_t result = cudaMemGetInfo(&free, &total);

    mexPrintf("free memory in bytes %u (%u MB), total memory in bytes %u (%u MB). ", free, free/1024/1024, total, total/1024/1024);

    if( total > 0 )
        mexPrintf("%2.2f%% free\n", (100.0*free)/total );
    else
        mexPrintf("\n");
	*/

    const mxArray *mx = prhs[0];
    if (mxIsSingle(mx)) {
		cudaImage image = cudaImageFromMX(mx);
		rect2d border = {256,256}; // Extra large border for now
		cudaPaddedImage padded = allocPaddedImageOnDevice(image, border, 255.0f);

      	plhs[0] = mxStructFromLCudaMatrix(padded);

    } /*else if (mxIsUint8(mx)) {
	    cudaImage image = cudaImageFrom8bitMX(const mx);
      	plhs[0] = cudaMatrixToStruct(image);

	} */ else {
        printf("UNKNOWN. (This is bad)\n");
    }
}
