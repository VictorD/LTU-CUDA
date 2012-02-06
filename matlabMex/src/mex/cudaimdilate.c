#include "mlcuda.h"

#if DOXYGEN
/**
 * \ingroup mexmlcuda
 * Dilates a image located on the device.
 * The dilated image result will be stored in the source image memory.
 *
 * \param cudaImage The image on the device that should be dilated.
 * \param se The structure element that should be applied to the image.
 *
 */
void cudaimdilate(mexMlcudaImStruct cudaImage, lcudaStrel_8u se) {
#else
void mexFunction(int nlhs, mxArray *plhs[],
		int nrhs, const mxArray *prhs[])
#endif
{
	if (nrhs != 2)
		mxErrMsgTxt("The number of input arguments must be 2, "
				" one image struct and "
				"one cuda structure element");

	lcudaStrel_8u se = mlcudaStructToStrel_8u(prhs[1]);

    char dataType = mlcudaGetImageDataType(prhs[0]);

    // UINT8
    if (dataType == 0) {
        mlcudaErodeDilate_8u(mlcudaStructToMatrix_8u(prhs[0]), se, true);
    } 
    // FLOAT
    else if (dataType == 1) {
	    mlcudaErodeDilate(mlcudaStructToMatrix(prhs[0]), se, true);
    } else {
        printf("UNKNOWN. (This is bad)\n");
    }

	// Free memory
	lcudaFreeStrel_8u(se);
}
