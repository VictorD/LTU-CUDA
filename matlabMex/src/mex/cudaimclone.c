#include "mlcuda.h"

#if DOXYGEN
/**
 * \ingroup mexmlcuda
 * Clones a image on the device.
 *
 * \param cudaImage A mex cuda image structure containing the information about the image that should be cloned.
 * \return A structure with the information about the cloned image on the device.
 */
mexMlcudaImStruct cudaimclone(mexMlcudaImStruct cudaImage) {
#else
void mexFunction(int nlhs, mxArray *plhs[],
#endif
		int nrhs, const mxArray *prhs[])
{
	lcudaMatrix_8u im = mlcudaStructToMatrix_8u(prhs[0]);
	plhs[0] = mlcudaMatrixToStruct_8u(lcudaCloneMatrix_8u(im));
}
