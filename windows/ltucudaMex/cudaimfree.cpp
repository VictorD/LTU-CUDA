#include "ltucuda/matlabBridge/matlabBridge.h"
#include "ltucuda/pinnedmem.cuh"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
	if (nrhs == 0) {
		cudaDeviceReset();
	} else {
		const mxArray *mx = prhs[0];
		cudaImage image = imageFromMXStruct(mx);
		cudaFree(image.data);
	}
}
