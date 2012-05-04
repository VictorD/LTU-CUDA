#include <thrust/device_ptr.h>
#include <thrust/fill.h>

#include <cuda.h>
#include <memory>
#include <iostream>
#include <cassert>
#include <stdio.h> 
#include <string.h>
      
#include "kernels/generic.cuh"
#include "kernels/VHGW.cuh"
#include "kernels/3x3.cuh"
#include "morphology.cuh"

template <class dataType, morphOperation MOP>
static void _dispatchMorphOp(const dataType* pSrc, int nSrcStep, dataType* pDst, int nDstStep, rect2d srcROI, morphMask mask, rect2d border) {

    int offsetX = border.width-mask.width/2;
    int offsetY = border.height-mask.height/2;
    PRINTF("OffsetX : %d, OffsetY : %d\n", offsetX, offsetY);

    // Steps are specified in bytes. Pointer aritmetic below requires we compensate for size of <dataType>.
    nSrcStep /= sizeof(dataType);
    nDstStep /= sizeof(dataType);
     
    int srcBorderOffset = (nSrcStep * offsetY + offsetX);
    PRINTF("Border offset: %d\n", srcBorderOffset);
	switch(mask.type) { 
		case THREE_BY_THREE: {
			PRINTF("3x3 kernel!\n");
			_global3x3<dataType, MOP>(pSrc + srcBorderOffset, nSrcStep, pDst, nDstStep, srcROI, mask);
		}  
		break; 
		case VHGW: { 
            PRINTF("Erosion: Offset (%d,%d)\n", offsetX, offsetY);
            _globalVHGW<dataType, MOP>(pSrc + srcBorderOffset, nSrcStep, pDst, nDstStep, srcROI, mask, border);
		}   
		break;   
		default: {
			PRINTF("Generic!\n");
			_globalGeneric<dataType, MOP>(pSrc + srcBorderOffset, nSrcStep, pDst, nDstStep, srcROI, mask, border);
			PRINTF("Generic done :D\n");
		}
		break;   
    }           
    // Block until async kernel call has been executed.
    cudaThreadSynchronize();
}
 
/*    
 * Public functions to perform erosion or dilation. 
 */ 
void performDilation(const float * pSrc, int nSrcStep, float *pDst, int nDstStep, rect2d srcROI, morphMask mask, rect2d border) {
    _dispatchMorphOp<float, DILATE>(pSrc, nSrcStep, pDst, nDstStep, srcROI, mask, border);
}

void performErosion(const float * pSrc, int nSrcStep, float *pDst, int nDstStep, rect2d srcROI, morphMask mask, rect2d border) {
    PRINTF("mask is height: %d, width :%d, isFlat: %d\n", mask.height, mask.width, mask.isFlat);
    _dispatchMorphOp<float, ERODE>(pSrc, nSrcStep, pDst, nDstStep, srcROI, mask, border);
} 

unsigned char* copyMaskDataToDevice(int &pitch, unsigned char *data, int width, int height) {
	unsigned char *dev_mask_data;
    cudaMallocPitch((void **)&dev_mask_data, (size_t*)&pitch, width, height);
    exitOnError("createMask: alloc");
    cudaMemcpy2D(dev_mask_data, pitch, data, width, width, height, cudaMemcpyHostToDevice);
    exitOnError("createMask: copy");
	return dev_mask_data;
}
// Linear strel creation methods
morphMask createArbitraryMask(unsigned char *data, float* surface, int width, int height, point2d anchor, int isFlat) {
    int pitch;
	unsigned char *dev_mask_data = copyMaskDataToDevice(pitch, data, width, height);
    morphMask mask = {width,height, dev_mask_data, pitch, surface, anchor, isFlat, GENERIC};
    return mask;
}

morphMask createTBTMask(unsigned char *data) {
	for(int i = 0; i < 9; i++)
		printf("data [%d] = %d", i, data[i]);
	printf("---");
    point2d anchor = {1,1};
    int pitch;
	unsigned char *dev_mask_data = copyMaskDataToDevice(pitch, data, 3, 3);
    int binValue = 256*data[8] + 128*data[7] + 64*data[6] + 
					32*data[5] +  16*data[4] +  8*data[3] + 
					 4*data[2] +   2*data[1] +    data[0];
	morphMask tbt;
	tbt.anchor = anchor;
	tbt.data = dev_mask_data;
	tbt.pitch = pitch;
	tbt.isFlat = true;
	tbt.type = THREE_BY_THREE;
	tbt.binaryValue = binValue;
	tbt.width = 3;
	tbt.height = 3;

    return tbt;
}

morphMask createVHGWMask(int sideLength, vhgwDirection direction) {
	//point2d anchor = {1,1};
	morphMask vhgw;
	vhgw.width  = (direction == VERTICAL)   ? 1 : sideLength;
	vhgw.height = (direction == HORIZONTAL) ? 1: sideLength;
	vhgw.direction = direction;
	vhgw.isFlat = 1;
	vhgw.type = VHGW;
    return vhgw;
}