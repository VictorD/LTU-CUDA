/*
 * lcudamorph.c
 *
 *  Created on: Jul 16, 2011
 *      Author: vicdan
 */
#include "lcudamorphfloat.cuh"
#include <thrust/device_ptr.h>
#include <thrust/fill.h>
#include <string.h>

#include "mlcuda.h"

static void lcudaErodeDilate(lcudaMatrix src, lcudaMatrix dst, lcudaStrel_8u se, int dilate) {
	lcudaMatrix paddedSrc;
	lcudaMatrix paddedSrc2;
	int i;
	NppiPoint anchor;
	NppiSize srcROI = { src.width, src.height};
	NppiSize maskSize;
	NppiSize borderSize;

	int borderColor;

	if (se.numStrels == 0) {
		return;
	}

	if (dilate) {
		borderColor = 0;
	} else {
		borderColor = 1.0f;
	}
	// Padded images.
	// Padd the image with respect to the largest structure element.
	borderSize.width = se.sizes[0].width;
	borderSize.height = se.sizes[0].height;

	for (i = 1; i < se.numStrels; ++i) {
		if (borderSize.width < se.sizes[i].width) {
			borderSize.width = se.sizes[i].width;
		}
		if (borderSize.height < se.sizes[i].height) {
			borderSize.height = se.sizes[i].height;
		}
	}

    // Need 1.5x SE size border at the ends.
    borderSize.width *=2;
    borderSize.height *=2;

	paddedSrc = lcudaAllocMatrix(src.width + 2 * borderSize.width, 
					    src.height + 2 * borderSize.height);


	lcudaCopyBorder(src, paddedSrc, borderColor, borderSize.width/4, borderSize.height/4);
	paddedSrc2 = lcudaCloneMatrix(paddedSrc);
	lcudaMatrix* esrc = &paddedSrc;
	lcudaMatrix* edst = &paddedSrc2;

	int seDataPos = 0;

	for (i = 0; i < se.numStrels-1; ++i) {
		lcudaMatrix* etmp;
		maskSize.width = se.sizes[i].width;
		maskSize.height = se.sizes[i].height;
		anchor.x = (maskSize.width - borderSize.width/2) / 2;
		anchor.y = (maskSize.height - borderSize.height/2) / 2;

		if (dilate) {
			   /* nppiDilate_8u_C1R(esrc->data, esrc->pitch,
							        edst->data + (edst->pitch * ((borderSize.height-1)/2))+((borderSize.width-1)/2), edst->pitch,
							            srcROI, se.data.data+seDataPos, maskSize, anchor);
                */
            performDilation(esrc->data, esrc->pitch, 
	            edst->data + (borderSize.width)/4 + (edst->pitch/sizeof(float) * (borderSize.height/4)),
		            edst->pitch, srcROI, se.data.data+seDataPos, se.heights.data+seDataPos, maskSize, anchor, borderSize, se.isFlat, se.binary[i]);
   		} else {          
			performErosion(esrc->data, esrc->pitch, 
				edst->data + (borderSize.width)/4 + (edst->pitch/sizeof(float) * (borderSize.height/4)),
					edst->pitch, srcROI, se.data.data+seDataPos, se.heights.data+seDataPos, maskSize, anchor, borderSize, se.isFlat, se.binary[i]);
		}
		etmp = esrc;
		esrc = edst;
		edst = etmp;
		seDataPos += maskSize.width*maskSize.height;
	}
	maskSize.width = se.sizes[se.numStrels-1].width;
	maskSize.height = se.sizes[se.numStrels-1].height;
	anchor.x = (maskSize.width - borderSize.width/2) / 2;
	anchor.y = (maskSize.height - borderSize.height/2) / 2;

	if (dilate) {
		performDilation(esrc->data, esrc->pitch, dst.data, dst.pitch, srcROI, 
                    se.data.data+seDataPos, se.heights.data+seDataPos, maskSize, anchor, borderSize, se.isFlat, se.binary[se.numStrels-1]);
	} else {
		performErosion(esrc->data, esrc->pitch, dst.data, dst.pitch, srcROI, 
                    se.data.data+seDataPos, se.heights.data+seDataPos, maskSize, anchor, borderSize, se.isFlat, se.binary[se.numStrels-1]);
	}

	lcudaFreeMatrix(paddedSrc);
	lcudaFreeMatrix(paddedSrc2);
}
void lcudaErode(lcudaMatrix src, lcudaMatrix dst, lcudaStrel_8u se) {
	lcudaErodeDilate(src, dst, se, 0);
}

void lcudaDilate(lcudaMatrix src, lcudaMatrix dst, lcudaStrel_8u se) {
	lcudaErodeDilate(src, dst, se, 1);
}

void lcudaFreeStrel(lcudaStrel se) {
	lcudaFreeArray(se.data);
}
