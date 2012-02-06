/*
 * reconstructcuda.c
 *
 *  Created on: Mar 10, 2010
 *      Author: henmak
 */

#include "mlcuda.h"

void mexFunction(int nlhs, mxArray *plhs[],
		int nrhs, const mxArray *prhs[]) {
	reconcuda(nlhs, plhs, nrhs, prhs);
}
