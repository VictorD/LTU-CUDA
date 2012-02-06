#include "mlcuda.h"

#if DOXYGEN
/**
 * \ingroup mexmlcuda
 * Frees the memory allocated by a image on the device.
 * For every cudaimmalloc and cudaimclone call one call to this function should
 * be made to prevent memory leaks.
 *
 * \param cudaImage The structure with the information about the cuda image that
 * 					should be freed.
 */
void cudaimfree(mexMlcudaImStruct cudaImage) {
#else
void mexFunction(int nlhs, mxArray *plhs[],
		int nrhs, const mxArray *prhs[])
#endif
{
	lcudaFreeMatrix_8u(mlcudaStructToMatrix_8u(prhs[0]));
}
