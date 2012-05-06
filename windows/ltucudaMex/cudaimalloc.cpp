#include "ltucuda/matlabBridge/matlabBridge.h"
#include "ltucuda/pgm/pgm.cuh"
void mexFunction(int nlhs, mxArray *plhs[],
		int nrhs, const mxArray *prhs[])
{
    const mxArray *mx = prhs[0];
    if (mxIsSingle(mx)) {
		cudaImage image = cudaImageFromMX(mx);
		rect2d border = {4,4}; // Extra large border for now
		cudaPaddedImage padded = allocPaddedImageOnDevice(image, border, 255.0f);

		/*cudaPaddedImage paddedAfter = cudaPaddedImageFromStruct(mxStructFromLCudaMatrix(padded));
		float *host_out = copyImageToHost(padded.image);
		savePGM("allocBeforeTest.pgm", host_out, padded.image.width, padded.image.height);
		host_out = copyImageToHost(paddedAfter.image);
		savePGM("allocAfterTest.pgm", host_out, paddedAfter.image.width, paddedAfter.image.height);
		*/
      	plhs[0] = mxStructFromLCudaMatrix(padded);

    } /*else if (mxIsUint8(mx)) {
	    cudaImage image = cudaImageFrom8bitMX(const mx);
      	plhs[0] = cudaMatrixToStruct(image);

	} */ else {
        printf("UNKNOWN. (This is bad)\n");
    }
}
