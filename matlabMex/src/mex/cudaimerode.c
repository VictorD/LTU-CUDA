#include "mlcuda.h"

#if DOXYGEN
/**
 * \ingroup mexmlcuda
 * Erodes a image located on the device.
 * The eroded image result will be stored in the source image memory.
 *
 * \param cudaImage The image on the device that should be eroded.
 * \param se The structure element that should be applied to the image.
 *
 */
void cudaimerode(mexMlcudaImStruct cudaImage, lcudaStrel_8u se) {
#else
void mexFunction(int nlhs, mxArray *plhs[],
		int nrhs, const mxArray *prhs[]) {
#endif
	if (nrhs != 2)
		mexErrMsgTxt("The number of input arguments must be 2, "
				" one image struct and "
				"one cuda structure element");

	lcudaStrel_8u se = mlcudaStructToStrel_8u(prhs[1]);

    char dataType = mlcudaGetImageDataType(prhs[0]);

    // UINT8
    if (dataType == 0) {
        mlcudaErodeDilate_8u(mlcudaStructToMatrix_8u(prhs[0]), se, false);
    } 
    // FLOAT
    else if (dataType == 1) {
	    mlcudaErodeDilate(mlcudaStructToMatrix(prhs[0]), se, false);
    } else {
        printf("UNKNOWN. (This is bad)\n");
    }

	// Free memory
	lcudaFreeStrel_8u(se);
}

