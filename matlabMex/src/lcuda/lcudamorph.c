/*
 * lcudamorph.c
 *
 *  Created on: Feb 6, 2010
 *      Author: henmak
 */
#include "lcudamorph.h"
#include "mlcuda.h"

static void lcudaErodeDilate_8u(lcudaMatrix_8u src, lcudaMatrix_8u dst, lcudaStrel_8u se, int dilate) {
	lcudaMatrix_8u paddedSrc;
	lcudaMatrix_8u paddedSrc2;
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
		borderColor = 0xFF;
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

	paddedSrc = lcudaAllocMatrix_8u(src.width + borderSize.width - 1, 
					    src.height + borderSize.height - 1);

	lcudaCopyBorder_8u(src, paddedSrc, borderColor, borderSize.width/4, borderSize.height/4);
	paddedSrc2 = lcudaCloneMatrix_8u(paddedSrc);
	lcudaMatrix_8u* esrc = &paddedSrc;
	lcudaMatrix_8u* edst = &paddedSrc2;

	int seDataPos = 0;


	for (i = 0; i < se.numStrels-1; ++i) {
		lcudaMatrix_8u* etmp;
		maskSize.width = se.sizes[i].width;
		maskSize.height = se.sizes[i].height;
		anchor.x = (maskSize.width - borderSize.width/2) / 2;
		anchor.y = (maskSize.height - borderSize.height/2) / 2;

		if (dilate) {
			performDilation_8u(esrc->data, esrc->pitch, 
				edst->data + (borderSize.width)/4 + (edst->pitch * (borderSize.height/4)),
					edst->pitch, srcROI, se.data.data+seDataPos, se.heights.data + seDataPos, maskSize, anchor, borderSize, se.isFlat,  se.binary[i]);
   		} else {          
			performErosion_8u(esrc->data, esrc->pitch, 
				edst->data + (borderSize.width)/4 + (edst->pitch * (borderSize.height/4)),
					edst->pitch, srcROI, se.data.data+seDataPos, se.heights.data + seDataPos, maskSize, anchor, borderSize, se.isFlat,  se.binary[i]);
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
		performDilation_8u(esrc->data, esrc->pitch, dst.data, dst.pitch, srcROI, 
                    se.data.data+seDataPos, se.heights.data+seDataPos, maskSize, anchor, borderSize, se.isFlat,  se.binary[se.numStrels-1]);
                        /*esrc->data, esrc->pitch, dst.data, dst.pitch, srcROI, se.data.data+seDataPos, maskSize, anchor);*/
	} else {
		performErosion_8u(esrc->data, esrc->pitch, dst.data, dst.pitch, srcROI, 
                    se.data.data+seDataPos, se.heights.data+seDataPos, maskSize, anchor, borderSize, se.isFlat,  se.binary[se.numStrels-1]);
	}

	lcudaFreeMatrix_8u(paddedSrc);
	lcudaFreeMatrix_8u(paddedSrc2);
}
void lcudaErode_8u(lcudaMatrix_8u src, lcudaMatrix_8u dst, lcudaStrel_8u se) {
	lcudaErodeDilate_8u(src, dst, se, 0);
}

void lcudaDilate_8u(lcudaMatrix_8u src, lcudaMatrix_8u dst, lcudaStrel_8u se) {
	lcudaErodeDilate_8u(src, dst, se, 1);
}

void lcudaFreeStrel_8u(lcudaStrel_8u se) {
	lcudaFreeArray_8u(se.data);
}
