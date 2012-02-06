/*
 * vhgwCPU.c
 * 
 * CPU Test of van Herk Gil-Werman algorithm
 * used as a reference for the GPU implementation.
 *
 * Created: Jul 13, 2011  
 * Author:  Victor Danell 
 *
 */

#include <string.h>
#include "mlcuda.h"


#define VMIN(a,b) (a<b) ? a:b
#define VMAX(a,b) (a>b) ? a:b

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
	if (nrhs != 2)
		mxErrMsgTxt("The number of input arguments must be 2, "
				" one image struct and "
				"one structure element");

    if (!mxIsUint8(prhs[0])) {
		mxErrMsgTxt("The input image must be of uint8 type");
	}

    unsigned int width = mxGetM(prhs[0]);
	unsigned int height = mxGetN(prhs[0]);
	unsigned char *img = (unsigned char*)mxGetData(prhs[0]);
    unsigned char *result = malloc(width*height*sizeof(unsigned char));
	unsigned int size = (int) mxGetScalar(prhs[1]);

    unsigned int hsize = size / 2;
    unsigned int steps = (width - 2*hsize)/size; // int division?
    unsigned char buffer[width];
    unsigned int startmin;
    unsigned char minarray[2*size-1];
    unsigned int i,j,k;
    unsigned char *lines, *lined;
    unsigned char minval;
    unsigned int startx,starty;

    //printf("Starting loop height:\n");
    for(i = 0; i < height; i++) {
        lines = img + i * width;
        lined = result + i * width;

        //printf("Fill pixel buffer: %d\n", img[0]);
        for(j = 0; j < width; j++) {
            buffer[j] = *(lines + j);
        }
        //printf("sizeof buffer: %d" , sizeof(buffer));

        //printf("Init minarray:\n");
        for(j = 0; j < steps; j++) {
            startmin = (j+1) * size - 1;
            minarray[size-1] = buffer[startmin];
            for(k=1;k<size;k++) {
                minarray[size-1-k] = VMIN(minarray[size-k],buffer[startmin-k]);
                minarray[size-1+k] = VMIN(minarray[size+k-2],buffer[startmin+k]);
            }

            startx = hsize+j*size;
            lined[startx] = minarray[0];
            lined[startx+size-1] = minarray[2*size-2];
            for(k=1;k<size-1;k++) {
                minval = VMIN(minarray[k], minarray[k+size-1]);
                lined[startx+k] = minval;
            }
        }
    }
    //printf("Done done done!\n");

   
    plhs[0] = mxCreateNumericMatrix(width, height, mxUINT8_CLASS, mxREAL);

    double *output = mxGetPr(plhs[0]);
    memcpy(output, result, width*height*sizeof(unsigned char));    
}

