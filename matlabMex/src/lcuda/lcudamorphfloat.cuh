/**
 * \defgroup lcudamorph LTU Cuda Morphological operations
 * \ingroup lcuda
 *
 * Morphological operations in the LTU Cuda library.
 *
 */
/*@{*/


/**
 * \file lcudamorph.h
 *
 * Header file for the LTU Cuda morphological operations.
 *
 * \author Henrik Mäkitaavola
 */

#ifndef LCUDAMORPHFLOAT_H_
#define LCUDAMORPHFLOAT_H_
#include "lcudafloat.h"
#include "lcudamorph.h"
#include "lcudaextern.h"
#include "morphology.cuh"

/**
 * \struct lcudaStrel_8u
 * A morphological structure element.
 * The data member contains all the structure elements in a array row by row and
 * structure element by structure element.
 *
 * Ex:
 * 		We want to apply 2 structure elements to a image:
 *
 * 		| 0 1 0 1 | \n
 * 		| 1 1 1 1 | \n
 * 		| 0 1 0 1 | \n
 * 		| 1 1 1 1 | \n
 *
 * 			&
 *
 * 		| 1 0 1 | \n
 * 		| 0 1 0 | \n
 * 		| 1 0 1 | \n
 *
 * 		Then the data array would contain : \n
 * 		<-First element-----------------> <-Second element-> \n
 * 		[ 0 1 0 1 1 1 1 1 0 1 0 1 1 1 1 1 1 0 1 0 1 0 1 0 1 ] \n
 * 		The sizes array would contain: \n
 * 		[ { 4, 4} , {3, 3} ] \n
 * 		and the numStrels would be equal to 2. \n
 */
typedef struct {
	lcudaArray data; ///< Pointer to the data of the structure elements.
	lcudaSize* sizes; ///< Array of the sizes of the structure elements. The sizes should be listed in the same order as the structure elements.
	int numStrels; ///< The number of structure elements stored in the struct.
} lcudaStrel;

/**
 * Erodes a 8bit matrix.
 * The matrix will be padded with zeroes with respect to the largest sub
 * structure elements in the input structure element.
 *
 * \param src The source matrix the erosion should be applied to.
 * \param dst A matrix where the result should be stored.
 * \param se The structuring element that the source matrix should be eroded with.
 */
EXTERN void lcudaErode(lcudaMatrix src, lcudaMatrix dst, lcudaStrel_8u se);

/**
 * Dilate a 8bit matrix.
 * The matrix will be padded with 0xFF with respect to the largest sub
 * structure elements in the input structure element.
 *
 * \param src The source matrix the dilation should be applied to.
 * \param dst A matrix where the result should be stored.
 * \param se The structuring element that the source matrix should be dilated with.
 */
EXTERN void lcudaDilate(lcudaMatrix src, lcudaMatrix dst, lcudaStrel_8u se);

/**
 * Frees the memory allocated by a cuda structure element on the device.
 *
 * \param se The structure element who's memory should be freed.
 */
EXTERN void lcudaFreeStrel(lcudaStrel se);

#include <mex.h>
EXTERN void reconcuda(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]);

/*}@*/
#endif /* LCUDAMORPH_H_ */
