#include "mlcuda.h"

#if DOXYGEN
/**
 * \ingroup mexmlcuda
 * Gets a image from the device and copies it to the host memory so that it is
 * viewable in Matlab.
 *
 * \param cudaImage The structure with the information about the cuda image that
 * 					should be fetched from the device.
 */
image cudaimget(mexMlcudaImStruct cudaImage) {
#else
void mexFunction(int nlhs, mxArray *plhs[],
		int nrhs, const mxArray *prhs[])
#endif
{
    const mxArray *mx = prhs[0];

    char dataType = mlcudaGetImageDataType(mx);

    // UINT8
    if (dataType == 0) {
	    lcudaMatrix_8u im = mlcudaStructToMatrix_8u(mx);
      	plhs[0] = mlcudaMatrixToMx_8u(im);

    } 
    // FLOAT
    else if (dataType == 1) {   
        lcudaMatrix im = mlcudaStructToMatrix(mx);
      	plhs[0] = mlcudaMatrixToMx(im);

	} else {
        printf("UNKNOWN. (This is bad)\n");
    }
}
