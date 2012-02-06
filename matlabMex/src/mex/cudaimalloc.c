#include "mlcuda.h"

#if DOXYGEN
/**
 * \ingroup mexmlcuda
 * Allocates and copies a given image on the device.
 *
 * \param image The image that should be allocated on the device, must be of uint8
 * 				type.
 * \return A structure with the information about the image on the device.
 */
mexMlcudaImStruct cudaimalloc(image) {
#else
void mexFunction(int nlhs, mxArray *plhs[],
		int nrhs, const mxArray *prhs[])
#endif
{
    const mxArray *mx = prhs[0];
    if (mxIsSingle(mx)) {
        lcudaMatrix im = mlcudaMxToMatrix(mx);
      	plhs[0] = mlcudaMatrixToStruct(im);

    } else if (mxIsUint8(mx)) {
	    lcudaMatrix_8u im = mlcudaMxToMatrix_8u(mx);
      	plhs[0] = mlcudaMatrixToStruct_8u(im);

	} else {
        printf("UNKNOWN. (This is bad)\n");
    }
}
